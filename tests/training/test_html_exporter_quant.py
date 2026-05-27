from __future__ import annotations

from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.io.html_exporter import export_estimate_html
from zrt.training.ir.builders import build_graph
from zrt.training.models.flops import op_cost
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.report import TrainingReport
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import GPU, SystemSpec


def _system() -> SystemSpec:
    link = LinkSpec(type="test", bandwidth_gbps=1000, latency_us=1)
    return SystemSpec(
        gpu=GPU(
            name="test-gpu",
            flops_bf16=100,
            flops_fp8=400,
            flops_fp4=800,
            hbm_gb=80,
            hbm_bw_gbps=1_000_000,
        ),
        host_mem_gb=1024,
        interconnect=InterconnectSpec(intra_node=link, inter_node=link),
        nodes=1,
        gpus_per_node=1,
    )


def test_html_export_serializes_quant_cast_meta(tmp_path):
    model = ModelSpec(
        hidden=128,
        ffn=256,
        num_heads=4,
        num_kv_heads=4,
        head_dim=32,
        vocab=1000,
        seq_len=64,
        layers=[LayerKind.MOE],
        num_experts=8,
        moe_ffn=256,
        top_k=2,
        moe_act_dtype=Dtype.FP8_E4M3,
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
        routed_expert_weight_dtype=Dtype.FP4,
    )
    system = _system()
    strategy = Strategy()
    graph = build_graph(model, strategy)
    assert any(op.kind == "cast" for op in graph.ops)

    op_costs = {op.name: op_cost(op, model, system) for op in graph.ops}
    out = tmp_path / "quant.html"

    export_estimate_html(
        report=TrainingReport(step_time_ms=1.0),
        graph=graph,
        model=model,
        system=system,
        strategy=strategy,
        op_costs=op_costs,
        output_path=out,
    )

    html = out.read_text(encoding="utf-8")
    assert "const DATA = JSON.parse(" in html
    assert "fp8_e4m3" in html
    assert "Dtype.FP8_E4M3" not in html
