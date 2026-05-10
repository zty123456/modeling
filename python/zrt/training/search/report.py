"""Report formatters — JSON, dict, summary string."""

from __future__ import annotations

import json
from pathlib import Path

from zrt.training.search.estimator import Report


def report_to_dict(report: Report) -> dict:
    """Convert Report to a JSON-serializable dict."""
    d = {
        "step_time_ms": round(report.step_time_ms, 3),
        "pipeline_time_ms": round(report.pipeline_time_ms, 3),
        "mfu": round(report.mfu, 4),
        "hfu": round(report.hfu, 4),
        "total_flops": report.total_flops,
        "warnings": report.warnings,
        "config_summary": report.config_summary,
        "schedule_name": report.schedule_name,
        "bubble_fraction": round(report.bubble_fraction, 4),
        "warmup_ms": round(report.warmup_ms, 3),
        "steady_ms": round(report.steady_ms, 3),
        "cooldown_ms": round(report.cooldown_ms, 3),
        "dp_exposed_ms": round(report.dp_exposed_ms, 3),
        "optimizer_time_ms": round(report.optimizer_time_ms, 3),
        "optimizer_comm_ms": round(report.optimizer_comm_ms, 3),
        "warmup_fwd_ms": round(report.warmup_fwd_ms, 3),
        "warmup_bwd_ms": round(report.warmup_bwd_ms, 3),
        "steady_fwd_ms": round(report.steady_fwd_ms, 3),
        "steady_bwd_ms": round(report.steady_bwd_ms, 3),
        "cooldown_fwd_ms": round(report.cooldown_fwd_ms, 3),
        "cooldown_bwd_ms": round(report.cooldown_bwd_ms, 3),
        "steady_fwd_per_mb_ms": round(report.steady_fwd_per_mb_ms, 3),
        "steady_bwd_per_mb_ms": round(report.steady_bwd_per_mb_ms, 3),
        "steady_per_mb_ms": round(report.steady_per_mb_ms, 3),
        "compute_time_ms": round(report.compute_time_ms, 3),
        "exposed_comm_ms": round(report.exposed_comm_ms, 3),
        "tp_exposed_ms": round(report.tp_exposed_ms, 3),
        "cp_exposed_ms": round(report.cp_exposed_ms, 3),
        "ep_exposed_ms": round(report.ep_exposed_ms, 3),
        "pp_exposed_ms": round(report.pp_exposed_ms, 3),
        "hidden_comm_ms": round(report.hidden_comm_ms, 3),
        "dp_hidden_ms": round(report.dp_hidden_ms, 3),
        "tp_hidden_ms": round(report.tp_hidden_ms, 3),
        "ep_hidden_ms": round(report.ep_hidden_ms, 3),
        "total_comm_volume_ms": round(report.total_comm_volume_ms, 3),
        "tokens_per_sec": round(report.tokens_per_sec, 1),
        "effective_params": report.effective_params,
        "flops_per_token": report.flops_per_token,
    }

    if report.memory is not None:
        d["memory"] = report.memory.to_gb()

    if report.per_stage:
        stages = []
        for i, st in enumerate(report.per_stage):
            stages.append({
                "stage": i,
                "fwd_ms": round(st.fwd * 1000, 3),
                "bwd_ms": round(st.bwd * 1000, 3),
                "total_ms": round((st.fwd + st.bwd) * 1000, 3),
            })
        d["per_stage"] = stages

    return d


def report_to_json(report: Report, path: str | Path) -> None:
    """Write Report to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(report_to_dict(report), f, indent=2, default=str)


def report_summary(report: Report) -> str:
    """Human-readable single-config summary string."""
    lines = []
    lines.append("=" * 60)
    lines.append("Training Estimation Report")
    lines.append("=" * 60)

    cs = report.config_summary
    lines.append(f"  Model:    {cs.get('model', 'N/A')}")
    lines.append(f"  System:   {cs.get('system', 'N/A')}")
    lines.append(f"  Strategy: {cs.get('strategy', 'N/A')}")
    lines.append(f"  Microbatches: {cs.get('num_microbatches', 'N/A')}")
    lines.append("")

    lines.append(f"  Step time:  {report.step_time_ms:.1f} ms")
    lines.append(f"  Schedule:   {report.schedule_name}")
    lines.append(f"  Bubble:     {report.bubble_fraction:.1%}")
    lines.append(f"  MFU:        {report.mfu:.1%}")
    lines.append(f"  HFU:        {report.hfu:.1%}")

    if report.flops_per_token > 0:
        lines.append(f"  FLOPs/token: {report.flops_per_token:.2e}")
    if report.tokens_per_sec > 0:
        lines.append(f"  Tokens/s:    {report.tokens_per_sec:.0f}")

    if report.warmup_ms > 0 or report.steady_ms > 0:
        lines.append(f"  Step time breakdown:")
        lines.append(f"    Phase      | FWD (ms)  | BWD (ms)  | Total (ms)")
        lines.append(f"    Warmup     | {report.warmup_fwd_ms:9.2f} | {report.warmup_bwd_ms:9.2f} | {report.warmup_ms:9.2f}")
        lines.append(f"    Steady     | {report.steady_fwd_ms:9.2f} | {report.steady_bwd_ms:9.2f} | {report.steady_ms:9.2f}")
        lines.append(f"    Cooldown   | {report.cooldown_fwd_ms:9.2f} | {report.cooldown_bwd_ms:9.2f} | {report.cooldown_ms:9.2f}")
        
        total_fwd = report.warmup_fwd_ms + report.steady_fwd_ms + report.cooldown_fwd_ms
        total_bwd = report.warmup_bwd_ms + report.steady_bwd_ms + report.cooldown_bwd_ms
        total_phase = report.warmup_ms + report.steady_ms + report.cooldown_ms
        lines.append(f"    TOTAL      | {total_fwd:9.2f} | {total_bwd:9.2f} | {total_phase:9.2f}")

        if report.dp_exposed_ms > 0:
            lines.append(f"    DP AR exposed:   {report.dp_exposed_ms:.2f} ms")
        if report.optimizer_time_ms > 0:
            lines.append(f"    Optimizer:       {report.optimizer_time_ms:.2f} ms")
        if report.optimizer_comm_ms > 0:
            lines.append(f"    Opt comm:        {report.optimizer_comm_ms:.2f} ms")

        if report.steady_per_mb_ms > 0:
            mb_count = int(report.steady_ms / report.steady_per_mb_ms) if report.steady_per_mb_ms > 0 else 0
            lines.append(f"    Per-microbatch (steady phase):")
            lines.append(
                f"    FWD: {report.steady_fwd_per_mb_ms:.2f} ms | BWD: {report.steady_bwd_per_mb_ms:.2f} ms | Total: {report.steady_per_mb_ms:.2f} ms")
            lines.append(f"    (averaged over {mb_count} microbatches in steady phase)")

    # Compute / comm breakdown
    if report.compute_time_ms > 0 or report.exposed_comm_ms > 0:
        lines.append(f"  Compute / Comm breakdown:")
        lines.append(f"    Pipeline time:  {report.pipeline_time_ms:.2f} ms")
        lines.append(f"    Compute:        {report.compute_time_ms:.2f} ms")
        lines.append(f"    Exposed comm:   {report.exposed_comm_ms:.2f} ms")
        if report.tp_exposed_ms > 0:
            lines.append(f"      TP (RS/AG):   {report.tp_exposed_ms:.2f} ms")
        if report.cp_exposed_ms > 0:
            lines.append(f"      CP (A2A):     {report.cp_exposed_ms:.2f} ms")
        if report.ep_exposed_ms > 0:
            lines.append(f"      EP (A2A):     {report.ep_exposed_ms:.2f} ms")
        if report.pp_exposed_ms > 0:
            lines.append(f"      PP (P2P):     {report.pp_exposed_ms:.2f} ms")
        if report.dp_exposed_ms > 0:
            lines.append(f"      DP (AR/RS):   {report.dp_exposed_ms:.2f} ms")
        lines.append(f"    Hidden comm:    {report.hidden_comm_ms:.2f} ms")
        if report.dp_hidden_ms > 0:
            lines.append(f"      DP hidden:    {report.dp_hidden_ms:.2f} ms")
        if report.tp_hidden_ms > 0:
            lines.append(f"      TP hidden:    {report.tp_hidden_ms:.2f} ms")
        if report.ep_hidden_ms > 0:
            lines.append(f"      EP hidden:    {report.ep_hidden_ms:.2f} ms")
        lines.append(f"    Total comm vol: {report.total_comm_volume_ms:.2f} ms")
        exposed_pct = report.exposed_comm_ms / report.pipeline_time_ms * 100 if report.pipeline_time_ms > 0 else 0
        lines.append(f"    Exposed ratio:  {exposed_pct:.1f}% of pipeline")

    if report.memory is not None:
        gb = report.memory.to_gb()
        lines.append(f"  Memory:")
        lines.append(f"    weights:     {gb['weights_gb']:.2f} GB")
        lines.append(f"    grads:       {gb['grads_gb']:.2f} GB")
        lines.append(f"    opt_state:   {gb['opt_state_gb']:.2f} GB")
        lines.append(f"    activations: {gb['activations_gb']:.2f} GB")
        lines.append(f"    comm_buf:    {gb['comm_buffers_gb']:.2f} GB")
        lines.append(f"    TOTAL:       {gb['total_gb']:.2f} GB")

    if report.per_stage:
        lines.append(f"  Per-stage times:")
        for i, st in enumerate(report.per_stage):
            lines.append(
                f"    Stage {i}: fwd={st.fwd*1000:.2f}ms  bwd={st.bwd*1000:.2f}ms  "
                f"total={((st.fwd+st.bwd)*1000):.2f}ms"
            )

    if report.warnings:
        lines.append(f"  Warnings:")
        for w in report.warnings:
            lines.append(f"    - {w}")

    lines.append("=" * 60)
    return "\n".join(lines)