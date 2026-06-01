import logging
import math
import threading

from python.zrt.hardware import HardwareSpec
from python.zrt.ir import OpNode
from python.zrt.ir import TensorMeta, DType
from python.zrt.simulator import OpSimulator, SimResult

_DEFAULT_BLOCK_DIM = 64

logger = logging.getLogger(__name__)

try:
    from cost_model.infer.performance_predict import PerformancePredict

    _COST_MODEL_AVAILABLE = True
except ImportError:
    PerformancePredict = None
    _COST_MODEL_AVAILABLE = False


class LookupSimulator(OpSimulator):
    name = "lookup"
    priority = 1

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.lock = threading.Lock()
        self.caller = None

    def init_once(self, hw: "HardwareSpec"):
        # 双重锁检测，保证 cost model 调用器只被初始化一次
        if self.initialized:
            return
        with self.lock:
            if self.initialized:
                return
            try:
                if PerformancePredict is not None:
                    self.caller = PerformancePredict(_HW_ENV_DICT.get(hw.name, {}))
                    self.initialized = True
            except Exception as e:
                self.caller = None
                logger.warning(f"init cost model caller failed: {e}")

    def can_simulate(self, node: "OpNode", hw: "HardwareSpec") -> bool:
        if not _COST_MODEL_AVAILABLE or hw.name not in _HW_ENV_DICT or node.category == "communication":
            return False
        return True

    def simulate(self, node: "OpNode", hw: "HardwareSpec") -> SimResult:
        # 初始化 cost model 调用器，必须知道硬件型号
        self.init_once(hw)

        if self.caller is None:
            logger.warning(f"call cost model for {node}(op_short={node.op_short}) failed, cost model caller is none")
            return _build_sim_result(node, hw, node.annotations.get("latency_us", 0), "roofline", 0.3)

        inputs = [_convert_tensor(input_tensor) for input_tensor in node.inputs]
        outputs = [_convert_tensor(output_tensor) for output_tensor in node.outputs]
        params = (inputs, outputs, node.attrs, _DEFAULT_BLOCK_DIM)

        try:
            cost_model_op_type = _get_cost_model_op_type(node)
            result_code, msg, result, _ = self.caller.op_performance_predict(cost_model_op_type, params, {})
            if result_code != 0 or msg:
                logger.warning(
                    f"call cost model for {node}(op_short={node.op_short}) failed, "
                    f"result code: {result_code}, msg: {msg}, result: {result}, op_type: {cost_model_op_type}")
                latency_us = node.annotations.get("latency_us", 0)
                confidence = 0.3
            else:
                logger.debug(
                    f"call cost model for {node}(op_short={node.op_short}) success, "
                    f"result code: {result_code}, msg: {msg}, result: {result}, op_type: {cost_model_op_type}")
                latency_us = result["predict_time"]
                confidence = 0.8

            return _build_sim_result(node, hw, latency_us, self.name, confidence)
        except Exception as e:
            logger.warning(f"call cost model for {node}(op_short={node.op_short}) failed, error message: {e}")
            return _build_sim_result(node, hw, node.annotations.get("latency_us", 0), "roofline", 0.3)


_HW_ENV_DICT: dict[str, dict[str, str]] = {
    "Ascend 910B": {
        "soc_version": "Ascend910B3",
        "cann_version": "7.0.0",
        "pta_version": "1.11.0",
        "predict_type": "task_duration"
    },
    "Ascend 910C": {
        "soc_version": "Ascend910C1",
        "cann_version": "8.0.RC2",
        "pta_version": "1.11.0",
        "predict_type": "task_duration"
    },
    "Ascend 910C SuperPod": {
        "soc_version": "Ascend910C1",
        "cann_version": "8.0.RC2",
        "pta_version": "1.11.0",
        "predict_type": "task_duration"
    },
    "Ascend 910D": {  # pending
        "soc_version": "Ascend910_9591",
        "cann_version": "7.3.0",
        "pta_version": "1.11.0",
        "predict_type": "task_duration"
    }
}

_DTYPE_TO_TORCH_STR: dict[DType, str] = {
    DType.FP32: "torch.float32",
    DType.FP16: "torch.float16",
    DType.BF16: "torch.bfloat16",
    DType.FP8_E4M3: "torch.float8_e4m3fn",
    DType.FP8_E5M2: "torch.float8_e5m2fn",
    DType.INT8: "torch.int8",
    DType.INT32: "torch.int32",
    DType.INT64: "torch.int64",
    DType.UINT8: "torch.uint8",
    DType.BOOL: "torch.bool",
    DType.INT4: "torch.int8_quant",  # cost model supported
    DType.UNKNOWN: "torch.bfloat16",
}

_TO_COST_MODEL_OP_TYPE: dict[str, str] = {
    "mm": "Matmul",
    "bmm": "BatchMatMul",
    "flash_attn": "FlashAttention",
    "sdpa": "FlashAttention",
    "floor_divide": "FloorDiv",
}


def _convert_tensor(tensor: "TensorMeta") -> dict:
    return {
        "dtype": _DTYPE_TO_TORCH_STR.get(tensor.dtype, "torch.bfloat16"),
        "format": "",
        "name": "",
        "origin_dtype": _DTYPE_TO_TORCH_STR.get(tensor.dtype, "torch.bfloat16"),
        "origin_format": "",
        "origin_shape": list(tensor.shape),
        "shape": list(tensor.shape),
        "size": int(math.prod(tensor.shape)),
    }


def _get_cost_model_op_type(node: "OpNode") -> str:
    # 特殊算子通过映射，其余算子转大驼峰风格
    op_short = node.op_short
    if not op_short:
        op_short = node.op_type
    if op_short in _TO_COST_MODEL_OP_TYPE:
        return _TO_COST_MODEL_OP_TYPE[op_short]
    return "".join(word.title() for word in op_short.split("_"))


def _get_primary_dtype(node: "OpNode") -> DType:
    if node.category != "compute" or not node.inputs:
        if node.outputs:
            return node.outputs[0].dtype
        return DType.BF16
    quant_act = node.annotations.get("quant_act")
    if quant_act and quant_act not in ("bf16", "fp16", "fp32"):
        normalized = "fp8_e4m3" if quant_act == "fp8" else quant_act
        try:
            return DType(normalized)
        except ValueError:
            pass
    return node.inputs[0].dtype


def _calculate_hw_util(node: "OpNode", hw: "HardwareSpec", latency_us: "float") -> float:
    dtype = _get_primary_dtype(node)
    flops = node.annotations.get("flops", 0)
    peak = hw.peak_flops(dtype)  # ops/s
    hw_util = 0.0
    if peak > 0 and latency_us > 0:
        actual_rate = flops / (latency_us * 1e-6)
        hw_util = min(1.0, actual_rate / peak)
    return hw_util


def _build_sim_result(node, hw, latency_us, backend, confidence) -> SimResult:
    # 当 node.annotations 无赋值时，全部使用默认值 0，confidence设置为 0
    hw_utilization = _calculate_hw_util(node, hw, latency_us)
    return SimResult(
        op_node_id=node.id,
        latency_us=latency_us,
        compute_us=node.annotations.get("compute_us", 0),
        memory_us=node.annotations.get("memory_us", 0),
        flops=node.annotations.get("flops", 0),
        read_bytes=node.annotations.get("read_bytes", 0),
        write_bytes=node.annotations.get("write_bytes", 0),
        arithmetic_intensity=node.annotations.get("arithmetic_intensity", 0),
        bound=node.annotations.get("bound", "memory"),
        hw_utilization=hw_utilization,
        backend=backend,
        confidence=confidence if hw_utilization != 0 else 0,
    )
