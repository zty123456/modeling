"""Demo: Chrome Trace JSON export from PP pipeline scheduling.

Exercises three export modes:
  1. export_stitched       — pipeline grid (stage x microbatch)
  2. export_per_stage      — per-op detail from DAGScheduler
  3. export_combined       — both views in one file (offset pids)

Output files:
  demo_trace/stitched.json       pipeline grid view
  demo_trace/per_stage.json      per-op detail
  demo_trace/combined.json       combined (offset pids)
"""
from __future__ import annotations

import json
import os

from python.zrt.executor.scheduler import DAGScheduler, Timeline, ScheduledOp
from python.zrt.executor.pp_stitcher import PPStitcher, PPStitchedTimeline, GridTask
from python.zrt.executor.chrome_trace import ChromeTraceExporter
from zrt.pipeline import run_trace_phases
from zrt.transform.analysis import estimate_training_from_graphs
import python.zrt.hardware.registry as hw_registry


def _build_mock_timelines(pp: int = 4) -> list[Timeline]:
    """Build mock per-stage Timelines with realistic TP/EP comm ops."""
    timelines: list[Timeline] = []
    for s in range(pp):
        ops: list[ScheduledOp] = []
        t = 0.0

        # --- Forward ---
        # QKV projection (compute)
        ops.append(ScheduledOp(f"qkv_proj", 0, "compute", t, t + 40, 40, "matmul", "compute", "fwd"))
        t += 40
        # TP all_reduce after QKV (comm on stream 1)
        ops.append(ScheduledOp(f"tp_ar_qkv", 1, "comm", t, t + 8, 8, "comm.all_reduce", "communication", "fwd"))
        t += 8
        # Attention compute
        ops.append(ScheduledOp(f"attn_core", 0, "compute", t, t + 25, 25, "attention", "compute", "fwd"))
        t += 25
        # O projection
        ops.append(ScheduledOp(f"o_proj", 0, "compute", t, t + 30, 30, "matmul", "compute", "fwd"))
        t += 30
        # TP all_reduce after O (comm)
        ops.append(ScheduledOp(f"tp_ar_o", 1, "comm", t, t + 8, 8, "comm.all_reduce", "communication", "fwd"))
        t += 8
        # FFN gate + up
        ops.append(ScheduledOp(f"ffn_gate_up", 0, "compute", t, t + 45, 45, "matmul", "compute", "fwd"))
        t += 45
        # EP all_to_all on MoE path (comm stream 2)
        ops.append(ScheduledOp(f"ep_a2a_fwd", 2, "comm", t, t + 12, 12, "comm.all_to_all", "communication", "fwd"))
        t += 12
        # FFN down
        ops.append(ScheduledOp(f"ffn_down", 0, "compute", t, t + 30, 30, "matmul", "compute", "fwd"))
        t += 30

        # --- Backward ---
        # FFN down bwd
        ops.append(ScheduledOp(f"ffn_down_bwd", 0, "compute", t, t + 60, 60, "matmul", "compute", "bwd"))
        t += 60
        # EP all_to_all bwd
        ops.append(ScheduledOp(f"ep_a2a_bwd", 2, "comm", t, t + 12, 12, "comm.all_to_all", "communication", "bwd"))
        t += 12
        # FFN gate_up bwd
        ops.append(ScheduledOp(f"ffn_gate_up_bwd", 0, "compute", t, t + 55, 55, "matmul", "compute", "bwd"))
        t += 55
        # TP all_reduce bwd
        ops.append(ScheduledOp(f"tp_ar_bwd", 1, "comm", t, t + 8, 8, "comm.all_reduce", "communication", "bwd"))
        t += 8
        # O proj bwd
        ops.append(ScheduledOp(f"o_proj_bwd", 0, "compute", t, t + 45, 45, "matmul", "compute", "bwd"))
        t += 45
        # Attn bwd
        ops.append(ScheduledOp(f"attn_bwd", 0, "compute", t, t + 50, 50, "matmul", "compute", "bwd"))
        t += 50
        # TP all_reduce bwd
        ops.append(ScheduledOp(f"tp_ar_bwd2", 1, "comm", t, t + 8, 8, "comm.all_reduce", "communication", "bwd"))
        t += 8
        # QKV bwd
        ops.append(ScheduledOp(f"qkv_proj_bwd", 0, "compute", t, t + 55, 55, "matmul", "compute", "bwd"))
        t += 55

        tl = Timeline(scheduled_ops=ops, graph_name=f"stage_{s}", phase="fwd+bwd")
        timelines.append(tl)

    return timelines


def _build_stitched_pipeline(timelines: list[Timeline]) -> PPStitchedTimeline:
    """Build PP stitched timeline from per-stage Timelines."""
    pp = len(timelines)
    stage_fwd = {s: tl.phase_latency("fwd") for s, tl in enumerate(timelines)}
    stage_bwd = {s: tl.phase_latency("bwd") for s, tl in enumerate(timelines)}

    stitcher = PPStitcher(
        stage_fwd_us=stage_fwd,
        stage_bwd_us=stage_bwd,
        pp=pp, M=6,
        p2p_latency_us=3,
        schedule="1f1b",
    )
    return stitcher.stitch_from_timelines(timelines)


def demo_trace_export():
    """Run all four export modes and print validation output."""
    out_dir = "demo_trace"
    os.makedirs(out_dir, exist_ok=True)

    print("Building mock data ...")
    timelines = _build_mock_timelines(pp=4)
    stitched = _build_stitched_pipeline(timelines)

    exporter = ChromeTraceExporter(time_unit="us")

    # ── Mode 1: pipeline grid only ────────────────────────────────────────
    path1 = os.path.join(out_dir, "stitched.json")
    doc1 = exporter.export_stitched(stitched, path1)
    data1 = json.loads(doc1)
    ev1 = data1["traceEvents"]
    n_grid = sum(1 for e in ev1 if e["ph"] == "X")
    n_inst = sum(1 for e in ev1 if e["ph"] == "i")
    print(f"\n[1] stitched.json  -> {path1}")
    print(f"    {n_grid} complete events (grid tasks)")
    print(f"    {n_inst} instant events (phase markers)")
    _sample(ev1, 5)

    # ── Mode 2: per-stage detail with microbatch expansion ────────────────
    path2 = os.path.join(out_dir, "per_stage.json")
    doc2 = exporter.export_per_stage(timelines, path2, M=stitched.M, pp_stitched=stitched)
    data2 = json.loads(doc2)
    ev2 = data2["traceEvents"]
    n_ops = sum(1 for e in ev2 if e["ph"] == "X")
    print(f"\n[2] per_stage.json  -> {path2}")
    print(f"    {n_ops} ops across {len(timelines)} stages x {stitched.M} mbs")
    # Count by stage
    by_stage = {}
    for e in ev2:
        if e["ph"] == "X":
            by_stage[e["pid"]] = by_stage.get(e["pid"], 0) + 1
    for pid, count in sorted(by_stage.items()):
        print(f"    stage {pid}: {count} ops")
    # Check mb presence
    mbs = {e.get("args", {}).get("mb") for e in ev2 if e["ph"] == "X"}
    print(f"    microbatch ids in args: {sorted(mbs)}")
    _sample(ev2, 5)

    # ── Mode 3: combined (offset pids) ────────────────────────────────────
    path3 = os.path.join(out_dir, "combined.json")
    doc3 = exporter.export_combined(stitched, timelines, path3)
    data3 = json.loads(doc3)
    ev3 = data3["traceEvents"]
    # Check pid ranges
    pids = sorted(set(e["pid"] for e in ev3 if "pid" in e))
    print(f"\n[3] combined.json   -> {path3}")
    print(f"    {len(ev3)} total events")
    print(f"    pids: {min(pids)} .. {max(pids)} (grid=0..{stitched.pp-1}, detail={stitched.pp}..{2*stitched.pp-1})")
    _sample(ev3, 5)


def _sample(events: list[dict], n: int = 3):
    """Print a few sample events."""
    complete = [e for e in events if e.get("ph") == "X"]
    print(f"    Sample events:")
    for e in complete[:n]:
        name = e["name"]
        pid = e.get("pid", "?")
        tid = e.get("tid", "?")
        ts = e["ts"]
        dur = e["dur"]
        args = e.get("args", {})
        extra = " ".join(f"{k}={v}" for k, v in args.items())
        print(f"      pid={pid} tid={tid} {name:40s} ts={ts:>8.0f} dur={dur:>6.0f}  {extra}")


if __name__ == "__main__":
    # Step 1: 抓训练图
    result = run_trace_phases(
        model_id="hf_models/deepseek_v4",
        num_layers=4,
        phases=("train_forward", "train_backward"),
    )

    # Step 2: 训练建模（复用已抓的 OpGraph，无需重抓）
    hw = hw_registry.load("nvidia_h100_sxm")
    report = estimate_training_from_graphs(
        forward_graph=result.graphs["train_forward"],
        backward_graph=result.graphs["train_backward"],
        hw_spec=hw,
        tp=8, pp=4, dp=2,
        total_params=671e9,
        num_layers_full=61,
        output_dir="output/deepseek_v4/1f1b",
        pp_schedule="1f1b",
        vpp_chunks=1
    )
    print("1f1b")
    print(report.summary())

    report = estimate_training_from_graphs(
        forward_graph=result.graphs["train_forward"],
        backward_graph=result.graphs["train_backward"],
        hw_spec=hw,
        tp=8, pp=4, dp=2,
        total_params=671e9,
        num_layers_full=61,
        output_dir="output/deepseek_v4/interleaved",
        pp_schedule="interleaved",
        vpp_chunks=2
    )
    print("interleaved")
    print(report.summary())

    report = estimate_training_from_graphs(
        forward_graph=result.graphs["train_forward"],
        backward_graph=result.graphs["train_backward"],
        hw_spec=hw,
        tp=8, pp=4, dp=2,
        total_params=671e9,
        num_layers_full=61,
        output_dir="output/deepseek_v4/dualpipe",
        pp_schedule="dualpipe",
        vpp_chunks=1
    )
    print("dualpipe")
    print(report.summary())

    report = estimate_training_from_graphs(
        forward_graph=result.graphs["train_forward"],
        backward_graph=result.graphs["train_backward"],
        hw_spec=hw,
        tp=8, pp=4, dp=2,
        total_params=671e9,
        num_layers_full=61,
        output_dir="output/deepseek_v4/dualpipev",
        pp_schedule="dualpipev",
        vpp_chunks=2
    )
    print("dualpipeV")
    print(report.summary())

    report = estimate_training_from_graphs(
        forward_graph=result.graphs["train_forward"],
        backward_graph=result.graphs["train_backward"],
        hw_spec=hw,
        tp=8, pp=4, dp=2,
        total_params=671e9,
        num_layers_full=61,
        output_dir="output/deepseek_v4/zerobubble",
        pp_schedule="zb",
        vpp_chunks=1
    )
    print("zerobubble")
    print(report.summary())

