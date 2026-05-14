from __future__ import annotations

import concurrent.futures
import itertools
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Generator

import pandas as pd
from tqdm import tqdm

from zrt.hardware.registry import load as load_hw
from zrt.training.io.config_loader import _parse_model, _parse_system, _parse_strategy
from zrt.training.search.estimator import estimate, pareto_frontier
from zrt.training.search.report import report_summary
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.report import TrainingReport
from zrt.training.spec.strategy import (
    CPKind,
    MuonConfig,
    OptKind,
    PPSched,
    RecomputePolicy,
    Strategy,
    TPOverlap,
)
from zrt.training.spec.system import GPU, SystemSpec

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parent.parent / "configs" / "models"


def _load_model_spec(model_name: str) -> ModelSpec:
    import yaml
    path = _MODELS_DIR / f"{model_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Model config not found: {path}. "
            f"Available: {[p.stem for p in _MODELS_DIR.glob('*.yaml')]}"
        )
    with open(path, encoding="utf-8") as f:
        return _parse_model(yaml.safe_load(f))


def _make_system_from_config(config: Dict) -> SystemSpec:
    hw_name = config.get("hw", "nvidia_h100_sxm")
    hw = load_hw(hw_name)

    gpus_per_node = config.get("gpus_per_node", 8)
    if "world_size" in config:
        nodes = config["world_size"] // gpus_per_node
    else:
        nodes = config.get("nodes", 1)

    return SystemSpec(
        gpu=GPU(
            name=hw.name,
            flops_bf16=hw.compute.bf16_tflops,
            flops_fp8=hw.compute.fp8_tops or hw.compute.bf16_tflops * 2,
            hbm_gb=hw.memory.capacity_gb,
            hbm_bw_gbps=hw.memory.hbm_bandwidth_gbps,
            cube_tflops=hw.compute.cube_bf16_tflops,
            vector_tflops=hw.compute.vector_bf16_tflops,
            overlap_ratio=dict(hw.compute.overlap_ratio),
        ),
        interconnect=hw.interconnect,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        host_mem_gb=config.get("host_mem_gb", 256.0),
    )


def _make_strategy_from_config(config: Dict) -> Strategy:
    recompute = RecomputePolicy()
    rc_str = config.get("recompute", "none")
    if rc_str == "selective":
        recompute.per_layer = {"moe": {"attn"}, "dense": {"attn"}}
    elif rc_str == "full":
        recompute.per_layer = {"moe": {"full"}, "dense": {"full"}}

    muon_config = None
    opt_str = config.get("optimizer", "adam")
    if opt_str == "muon":
        muon_config = MuonConfig(rotation=config.get("muon_rotation", True))

    pp_schedule = PPSched(config.get("pp_schedule", "1f1b"))
    vpp_chunks = config.get("vpp_chunks", 1)
    if pp_schedule != PPSched.INTERLEAVED:
        vpp_chunks = 1

    return Strategy(
        tp=config.get("tp", 1),
        cp=config.get("cp", 1),
        pp=config.get("pp", 1),
        ep=config.get("ep", 1),
        dp=config.get("dp", 1),
        micro_batch=config.get("micro_batch", 1),
        global_batch=config.get("global_batch", 0),
        pp_schedule=pp_schedule,
        vpp_chunks=vpp_chunks,
        zero_stage=config.get("zero_stage", 0),
        recompute=recompute,
        optimizer=OptKind(opt_str),
        muon_config=muon_config,
        tp_overlap=TPOverlap(config.get("tp_overlap", "none")),
        ep_overlap=config.get("ep_overlap", False),
        cp_kind=CPKind(config.get("cp_kind", "none")),
        dualbatch=config.get("dualbatch", False),
        dp_overlap_in_bubble=config.get("dp_overlap_in_bubble", True),
    )


@dataclass
class TrainingConfigManager:
    param_grid: Dict[str, List[Any]]
    save_all_result: bool = False
    output_path: str = ""

    def __post_init__(self):
        model = self.param_grid.get("model", "unknown")
        if isinstance(model, list):
            model = model[0] if model else "unknown"
        world_sizes = self.param_grid.get("world_size", [1])
        max_world = max(world_sizes) if isinstance(world_sizes, list) else world_sizes
        self.output_path = os.path.join("output", "training_search", f"{model}_ws_{max_world}")

    def _get_divisors(self, n: int) -> List[int]:
        return [i for i in range(1, n + 1) if n % i == 0]

    def _expand_auto_values_optimized(self, grid: Dict[str, List[Any]], world_size: int) -> None:
        """
        核心优化：基于已知维度的最小值，动态裁剪并收紧 auto 变量的选择范围。
        避免盲目扩展成全量 world_size 的约数，将 auto 并行搜索空间缩减 90% 以上。
        
        注意：EP 不占用额外 rank，world_size = TP*CP*PP*DP
        """
        rank_keys = ["tp", "cp", "pp", "dp"]
        all_keys = ["tp", "cp", "pp", "ep", "dp"]

        # 确保所有并行维度都有默认值
        for key in all_keys:
            if key not in grid:
                grid[key] = [1]

        # 1. 整理出哪些键是固定值或具体范围，哪些键被设为了 "auto"
        explicit_rank_keys = []
        auto_rank_keys = []
        auto_ep = False
        
        for key in all_keys:
            vals = grid[key]
            is_auto = vals == "auto" or vals == ["auto"] or "auto" in vals
            if key == "ep":
                auto_ep = is_auto
            elif is_auto:
                auto_rank_keys.append(key)
            else:
                if key in rank_keys:
                    explicit_rank_keys.append(key)

        if not auto_rank_keys and not auto_ep:
            return

        # 2. 计算已知 rank 维度的乘积最小值（不包括 EP）
        min_explicit_prod = 1
        for key in explicit_rank_keys:
            vals = [v for v in grid[key] if v != "auto"]
            if vals:
                min_explicit_prod *= min(vals)

        # 3. 动态推导 auto rank 变量所允许的最大上限边界值
        max_allowed_val = world_size // min_explicit_prod

        # 提取当前合法边界内所有符合物理整除条件的约数
        all_divisors = self._get_divisors(world_size)
        optimized_divisors = [d for d in all_divisors if d <= max_allowed_val]

        # 4. 回填 rank 参数网格（tp, cp, pp, dp）
        for key in auto_rank_keys:
            vals = grid[key]
            if isinstance(vals, list):
                clean_vals = [v for v in vals if v != "auto"]
                grid[key] = sorted(list(set(clean_vals + optimized_divisors)))
            else:
                grid[key] = optimized_divisors

        # 5. EP 特殊处理：EP 独立扩展，不参与 rank 计算
        if auto_ep:
            vals = grid["ep"]
            if isinstance(vals, list):
                clean_vals = [v for v in vals if v != "auto"]
                # EP 可以是任意值，通常基于 num_experts 的约束
                grid["ep"] = sorted(list(set(clean_vals + optimized_divisors)))
            else:
                grid["ep"] = optimized_divisors

    def get_valid_parallel_combos(self, grid: Dict[str, List[Any]], target_ws: int) -> List[Tuple[int, ...]]:
        parallel_keys = ["tp", "cp", "pp", "ep", "dp"]
        p_grids = [grid.get(k, [1]) for k in parallel_keys]

        valid_combos = []
        for tp, cp, pp, ep, dp in itertools.product(*p_grids):
            # EP 不占用额外 rank，world_size = tp*cp*pp*dp
            if tp * cp * pp * dp == target_ws:
                valid_combos.append((tp, cp, pp, ep, dp))
        return valid_combos

    def count_total_configs(self) -> int:
        grid = {k: (v if isinstance(v, list) else [v]) for k, v in self.param_grid.items()}
        world_sizes = grid.get("world_size", [1])
        target_ws = world_sizes[0]
        self._expand_auto_values_optimized(grid, target_ws)

        parallel_keys = ["tp", "cp", "pp", "ep", "dp"]
        valid_p_count = len(self.get_valid_parallel_combos(grid, target_ws))

        other_keys = [k for k in grid.keys() if k not in parallel_keys]
        other_combinations = 1
        for k in other_keys:
            other_combinations *= len(grid[k])

        return other_combinations * valid_p_count

    def generate_static_configs_stream(self) -> Generator[Dict[str, Any], None, None]:
        grid = {k: (v if isinstance(v, list) else [v]) for k, v in self.param_grid.items()}
        world_sizes = grid.get("world_size", [1])
        if len(world_sizes) > 1:
            raise ValueError("Only single world_size is supported when using 'auto' parallel strategy")
        target_ws = world_sizes[0]

        # 激活精简版网格扩展
        self._expand_auto_values_optimized(grid, target_ws)

        parallel_keys = ["tp", "cp", "pp", "ep", "dp"]
        other_keys = [k for k in grid.keys() if k not in parallel_keys]

        valid_parallel_combos = self.get_valid_parallel_combos(grid, target_ws)
        logger.info(f"Dynamic Boundary Auto Optimization: {len(valid_parallel_combos)} paths verified.")

        other_grids = [grid[k] for k in other_keys]
        for other_vals in itertools.product(*other_grids):
            base_config = dict(zip(other_keys, other_vals))

            for p_vals in valid_parallel_combos:
                config = base_config.copy()
                config.update(dict(zip(parallel_keys, p_vals)))
                yield config

    def generate_static_configs(self) -> List[Dict[str, Any]]:
        return list(self.generate_static_configs_stream())


def run_training_task_wrapper(config: Dict) -> Optional[Dict]:
    from zrt.training.ir.builders import build_graph
    from zrt.training.models.flops import op_cost as _op_cost

    model_name = config.get("model", "deepseek_v3_2")
    try:
        model = _load_model_spec(model_name)
        strategy = _make_strategy_from_config(config)
        system = _make_system_from_config(config)
        strategy.validate(model, system)
    except Exception as e:
        return {"status": "error", "config": config, "type": "validation_error", "message": str(e)}

    try:
        graph = build_graph(model, strategy)
        op_costs = {op.name: _op_cost(op, model) for op in graph.ops}
        report = estimate(model, system, strategy, graph=graph)

        return {
            "status": "success",
            "config": config,
            "report": report,
            "graph": graph,
            "model": model,
            "system": system,
            "strategy": strategy,
            "op_costs": op_costs,
        }
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return {"status": "error", "config": config, "type": "runtime_error"}


def execute_training_tasks_batched(
        manager: TrainingConfigManager,
        workers: int,
        batch_size: int = 1000,
) -> Generator[List[Dict], None, None]:
    """
    分批执行训练任务，每完成 batch_size 个配置后 yield 一批结果，
    避免内存累积导致崩溃。
    """
    batch_results = []
    error_count = 0
    total_yielded = 0

    total_configs = manager.count_total_configs()
    logger.info(f"Total configurations to traverse: {total_configs}, batch_size={batch_size}")

    config_generator = manager.generate_static_configs_stream()
    MAX_QUEUE_SIZE = workers * 3
    futures_map = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _ in range(MAX_QUEUE_SIZE):
            try:
                cfg = next(config_generator)
                fut = executor.submit(run_training_task_wrapper, cfg)
                futures_map[fut] = cfg
            except StopIteration:
                break

        with tqdm(total=total_configs, desc="Evaluating configs", unit="config") as pbar:
            while futures_map:
                done, _ = concurrent.futures.wait(futures_map.keys(), return_when=concurrent.futures.FIRST_COMPLETED)

                for fut in done:
                    cfg = futures_map.pop(fut)
                    pbar.update(1)

                    res = fut.result()
                    if not res:
                        continue

                    if res["status"] == "success":
                        batch_results.append(res)
                    elif res["status"] == "error":
                        error_count += 1

                    if len(batch_results) >= batch_size:
                        total_yielded += len(batch_results)
                        logger.info(f"Yielding batch: {len(batch_results)} results (total yielded: {total_yielded})")
                        yield batch_results
                        batch_results = []

                while len(futures_map) < MAX_QUEUE_SIZE:
                    try:
                        cfg = next(config_generator)
                        fut = executor.submit(run_training_task_wrapper, cfg)
                        futures_map[fut] = cfg
                    except StopIteration:
                        break

    if batch_results:
        total_yielded += len(batch_results)
        logger.info(f"Yielding final batch: {len(batch_results)} results (total yielded: {total_yielded})")
        yield batch_results

    logger.info(f"Completed: {total_yielded} success, {error_count} errors")


def format_results(reports: List[TrainingReport], configs: List[Dict]) -> pd.DataFrame:
    rows = []
    for cfg, report in zip(configs, reports):
        d = {
            "model": cfg.get("model", "unknown"),
            "world_size": cfg.get("world_size", 1),
            "tp": cfg.get("tp", 1),
            "cp": cfg.get("cp", 1),
            "pp": cfg.get("pp", 1),
            "ep": cfg.get("ep", 1),
            "dp": cfg.get("dp", 1),
            "zero_stage": cfg.get("zero_stage", 0),
            "pp_schedule": cfg.get("pp_schedule", "1f1b"),
            "recompute": cfg.get("recompute", "none"),
            "optimizer": cfg.get("optimizer", "adam"),
            "step_time_ms": round(report.step_time_ms, 3),
            "pipeline_time_ms": round(report.pipeline_time_ms, 3),
            "mfu": round(report.mfu, 4),
            "hfu": round(report.hfu, 4),
            "bubble_fraction": round(report.bubble_fraction, 4),
            "tokens_per_sec": round(report.tokens_per_sec, 1),
        }
        if report.memory:
            d["memory_gb"] = round(report.memory.total / 1e9, 2)
        rows.append(d)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mfu", ascending=False)

    cols = ["model", "world_size", "tp", "cp", "pp", "ep", "dp", "zero_stage", "pp_schedule",
            "recompute", "optimizer", "step_time_ms", "pipeline_time_ms", "mfu", "hfu",
            "bubble_fraction", "tokens_per_sec", "memory_gb"]
    df = df[[c for c in cols if c in df.columns]]
    return df


def save_results(df: pd.DataFrame, frontier_results: List[Dict], output_path: str, mode: str = "write"):
    from zrt.training.io.excel_exporter import export_estimate_excel
    os.makedirs(output_path, exist_ok=True)

    csv_path = os.path.join(output_path, "results_summary.csv")
    if mode == "append" and os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path} (mode={mode})")

    xlsx_dir = os.path.join(output_path, "xlsx")
    os.makedirs(xlsx_dir, exist_ok=True)

    for i, result in enumerate(frontier_results, 1):
        cfg = result["config"]
        tp = cfg.get("tp", 1)
        cp = cfg.get("cp", 1)
        pp = cfg.get("pp", 1)
        ep = cfg.get("ep", 1)
        dp = cfg.get("dp", 1)
        xlsx_name = f"config_{i}_tp{tp}_cp{cp}_pp{pp}_dp{dp}_ep{ep}.xlsx"
        xlsx_path = os.path.join(xlsx_dir, xlsx_name)

        export_estimate_excel(
            report=result["report"],
            graph=result["graph"],
            model=result["model"],
            system=result["system"],
            strategy=result["strategy"],
            op_costs=result["op_costs"],
            output_path=xlsx_path,
        )
        logger.info(f"Excel saved to: {xlsx_path}")

    print("=" * 60)
    print("Training Search Results")
    print("=" * 60)
    print(f"Pareto frontier: {len(frontier_results)} configs")
    print(f"Excel reports: {xlsx_dir}")
    print()
    print("Top 5 configs by MFU:")
    print(df.head(5).to_string())
    print("=" * 60)

    if frontier_results:
        print("\nTop1 Result:")
        print(report_summary(frontier_results[0]["report"]))


def run_training_search_parallel(
        param_grid: Dict[str, List[Any]],
        workers: int = 8,
        save_all_result: bool = False,
        batch_size: int = 500,
) -> pd.DataFrame:
    manager = TrainingConfigManager(
        param_grid=param_grid,
        save_all_result=save_all_result,
    )

    start_time = time.time()
    logger.info(f"Starting batched stream-search with {workers} workers, batch_size={batch_size}...")

    all_results = []
    frontier_results = []
    is_first_batch = True

    for batch in execute_training_tasks_batched(manager, workers, batch_size=batch_size):
        all_results.extend(batch)

        reports = [r["report"] for r in batch]
        batch_frontier = pareto_frontier(reports)
        batch_frontier_results = [batch[reports.index(rp)] for rp in batch_frontier]
        frontier_results.extend(batch_frontier_results)

        batch_df = format_results(batch_frontier, [r["config"] for r in batch_frontier_results])
        save_mode = "write" if is_first_batch else "append"
        save_results(batch_df, batch_frontier_results, manager.output_path, mode=save_mode)
        is_first_batch = False

        del batch
        del reports
        del batch_frontier
        del batch_frontier_results

    elapsed = time.time() - start_time
    logger.info(f"Search completed in {elapsed:.2f} seconds, total results: {len(all_results)}")

    if not all_results:
        logger.error("No valid configurations found.")
        return pd.DataFrame()

    all_reports = [r["report"] for r in all_results]
    final_frontier_reports = pareto_frontier(all_reports)
    final_frontier_results = [all_results[all_reports.index(rp)] for rp in final_frontier_reports]

    final_df = format_results(final_frontier_reports, [r["config"] for r in final_frontier_results])
    save_results(final_df, final_frontier_results, manager.output_path, mode="write")

    return final_df


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # 注意：global_batch 必须 >= micro_batch * dp 的最大值
    # 当 world_size=8192, tp=1, cp=1, pp=1 时，dp 最大可达 8192
    # 因此 global_batch 应设置较大值如 65536
    training_param_grid = {
        "model": ["deepseek_v3_2"],
        "hw": ["nvidia_h100_sxm"],
        "world_size": [8192],
        "tp": [1, 2, 4, 8, 16],
        "cp": "auto",
        "pp": [1, 2, 4, 8],
        "ep": [256],
        "dp": "auto",
        "micro_batch": [1,16, 32,],
        "global_batch": [512, 1024, 2048, 4096, 8192, 65536],
        "seq_len": [8192, 65536, 131072, 262144, 524288, 1048576],
        "zero_stage": [1, 2, 3],
        "pp_schedule": ["dualpipe", "dualpipe_v"],
        "vpp_chunks": [1, 2],
        "cp_kind": ["ulysses", "ring"],
        "tp_overlap": ["coc", ],
        "ep_overlap": [True],
        "dualbatch": [True],
        "dp_overlap_in_bubble": [True],
        "recompute": ["selective"],
        "optimizer": ["adam"],
    }

    df = run_training_search_parallel(param_grid=training_param_grid, workers=32, batch_size=1000)
