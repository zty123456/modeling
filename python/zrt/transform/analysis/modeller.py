"""Training modeller: main entry point for training performance estimation.

Two usage modes:

1. **From an existing OpGraph** (lower-level)::

       from python.zrt.transform.analysis import estimate_training
       report = estimate_training(graph, ctx)

2. **From a model directory** (end-to-end, captures graph then estimates)::

       from python.zrt.transform.analysis import model_training
       report = model_training(
           model_id="hf_models/deepseek_v3_2",
           num_layers=4,
           seq_len=128,
           hw_spec=my_hardware_spec,
           total_params=671e9,
           tp=8, pp=4, dp=2,
           micro_batch=1, global_batch=8192,
       )
       print(report.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


@dataclass
class TrainingReport:
    """Training performance estimation report."""

    # Config summary
    config_summary: str = ""

    # Timing metrics
    step_time_ms: float = 0.0
    per_stage_ms: float = 0.0

    # Efficiency metrics
    mfu: float = 0.0  # Model FLOPs Utilization

    # FLOPs breakdown
    training_flops: float = 0.0
    forward_flops: float = 0.0
    backward_flops: float = 0.0

    # Memory breakdown (per GPU)
    memory_breakdown: dict[str, float] = field(default_factory=dict)

    # Pipeline metrics
    warmup_steps: int = 0
    cooldown_steps: int = 0
    steady_steps: int = 0
    bubble_fraction: float = 0.0

    # Model info
    total_params: int = 0

    def to_dict(self) -> dict:
        """Convert report to JSON-serializable dict."""
        return {
            "config_summary": self.config_summary,
            "step_time_ms": self.step_time_ms,
            "per_stage_ms": self.per_stage_ms,
            "mfu": self.mfu,
            "training_flops": self.training_flops,
            "forward_flops": self.forward_flops,
            "backward_flops": self.backward_flops,
            "memory_breakdown_gb": {
                k: v / 1e9 for k, v in self.memory_breakdown.items()
            },
            "warmup_steps": self.warmup_steps,
            "cooldown_steps": self.cooldown_steps,
            "steady_steps": self.steady_steps,
            "bubble_fraction": self.bubble_fraction,
            "total_params": self.total_params,
        }

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "Training Estimation Report",
            "=" * 40,
            f"Config: {self.config_summary}",
            "",
            "Timing:",
            f"  Step time: {self.step_time_ms:.2f} ms",
            f"  Per-stage: {self.per_stage_ms:.2f} ms",
            "",
            "Efficiency:",
            f"  MFU: {self.mfu:.1%}",
            "",
            "FLOPs:",
            f"  Training: {self.training_flops/1e12:.2f} TFLOPs",
            f"  Forward: {self.forward_flops/1e12:.2f} TFLOPs",
            f"  Backward: {self.backward_flops/1e12:.2f} TFLOPs",
            "",
            "Memory (per GPU):",
        ]
        for k, v in self.memory_breakdown.items():
            lines.append(f"  {k}: {v/1e9:.2f} GB")
        lines.extend([
            "",
            "Pipeline:",
            f"  Warmup steps: {self.warmup_steps}",
            f"  Steady steps: {self.steady_steps}",
            f"  Cooldown steps: {self.cooldown_steps}",
            f"  Bubble fraction: {self.bubble_fraction:.1%}",
            "",
            f"Total params: {self.total_params/1e9:.2f}B",
        ])
        return "\n".join(lines)


def estimate_training(
    graph: "OpGraph",
    ctx: "TransformContext",
) -> TrainingReport:
    """Estimate training performance metrics.

    This function runs training-specific analysis passes on the graph
    and returns a comprehensive training performance report.

    Parameters
    ----------
    graph : OpGraph
        The computation graph (typically a forward pass graph)
    ctx : TransformContext
        Transform context with training configuration (ctx.training must be set)

    Returns
    -------
    TrainingReport
        Training performance estimation report

    Examples
    --------
    >>> from python.zrt.transform.context import TransformContext, TrainingConfig
    >>> ctx = TransformContext(
    ...     hw_spec=my_hw,
    ...     training=TrainingConfig(optimizer="adam", zero_stage=1, ...),
    ... )
    >>> report = estimate_training(graph, ctx)
    >>> print(report.summary())
    """
    from python.zrt.transform.pipeline import build_training_pipeline

    # Run the full training pipeline (inference passes + training-specific)
    pipe = build_training_pipeline()
    g = pipe.run(graph, ctx)

    # Extract metrics from graph metadata
    pipeline_metrics = g.metadata.get("pipeline_metrics")
    memory_breakdown = g.metadata.get("memory_breakdown")

    # Build config summary
    parallel = ctx.parallel
    training = ctx.training
    config_parts = []
    if parallel.tp > 1:
        config_parts.append(f"TP{parallel.tp}")
    if parallel.pp > 1:
        config_parts.append(f"PP{parallel.pp}")
    if parallel.ep > 1:
        config_parts.append(f"EP{parallel.ep}")
    if parallel.dp > 1:
        config_parts.append(f"DP{parallel.dp}")
    if training:
        config_parts.append(f"ZeRO-{training.zero_stage}")
        config_parts.append(f"{training.optimizer}")
        config_parts.append(f"micro{training.micro_batch}")

    config_summary = "-".join(config_parts) if config_parts else "default"

    # Build report
    report = TrainingReport(
        config_summary=config_summary,
        step_time_ms=pipeline_metrics.step_time_ms if pipeline_metrics else 0.0,
        per_stage_ms=pipeline_metrics.per_stage_ms if pipeline_metrics else 0.0,
        mfu=pipeline_metrics.mfu if pipeline_metrics else 0.0,
        training_flops=g.metadata.get("training_flops", 0.0),
        forward_flops=g.metadata.get("forward_flops", 0.0),
        backward_flops=g.metadata.get("backward_flops", 0.0),
        memory_breakdown=memory_breakdown.to_dict() if memory_breakdown else {},
        warmup_steps=pipeline_metrics.warmup_steps if pipeline_metrics else 0,
        cooldown_steps=pipeline_metrics.cooldown_steps if pipeline_metrics else 0,
        steady_steps=pipeline_metrics.steady_steps if pipeline_metrics else 0,
        bubble_fraction=pipeline_metrics.bubble_fraction if pipeline_metrics else 0.0,
        total_params=g.metadata.get("total_params", 0),
    )

    return report


def model_training(
    model_id: str,
    num_layers: int = 4,
    batch_size: int = 1,
    seq_len: int = 128,
    total_params: int | None = None,
    hidden: int = 7168,
    num_layers_full: int | None = None,
    hw_spec: "HardwareSpec | None" = None,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    dp: int = 1,
    zero_stage: int = 1,
    optimizer: str = "adam",
    micro_batch: int = 1,
    global_batch: int = 32,
    output_dir: str | None = None,
) -> TrainingReport:
    """Capture a computation graph from a model and estimate training performance.

    This is the main end-to-end entry point. It chains graph capture
    (``run_trace_phases``) → IR conversion → ``estimate_training`` in a
    single call.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID or local path (e.g. ``"hf_models/deepseek_v3_2"``).
    num_layers : int
        Number of layers to trace (keep small for speed; results are scaled).
    batch_size, seq_len : int
        Trace input dimensions.
    total_params : int or None
        Full model parameter count.  If provided, stored in graph metadata
        so the FLOPs pass uses the authoritative count.
    hidden : int
        Hidden dimension (for memory estimation).
    num_layers_full : int or None
        Total layers in the full model (for memory scaling).  Defaults to
        *num_layers* if not provided.
    hw_spec : HardwareSpec or None
        Target hardware.  Required for realistic latency / MFU estimates.
    tp, pp, ep, dp : int
        Parallelism dimensions.
    zero_stage : int
        ZeRO stage (0–3).
    optimizer : str
        Optimizer name (``"adam"``, ``"adamw"``, ``"muon"``).
    micro_batch, global_batch : int
        Batch configuration.
    output_dir : str or None
        Where to write trace outputs (Excel, JSON, ONNX).  None = temp dir.

    Returns
    -------
    TrainingReport
    """
    from python.zrt.graph import run_trace_phases
    from python.zrt.transform.context import ParallelConfig, TrainingConfig, TransformContext

    # 1. Capture both forward and backward phases
    _, phase_records = run_trace_phases(
        model_id=model_id,
        num_layers=num_layers,
        batch_size=batch_size,
        seq_len=seq_len,
        phases=("train_forward", "train_backward"),
        output_dir=output_dir,
    )

    # 2. Build shared metadata
    from python.zrt.ir.adapter import records_to_opgraph

    metadata: dict = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_layers": num_layers_full or num_layers,
        "num_layers_traced": num_layers,
        "hidden": hidden,
    }
    if total_params is not None:
        metadata["total_params"] = int(total_params)

    # 3. Build OpGraphs for each captured phase
    graphs = {}
    for phase in ("train_forward", "train_backward"):
        records = phase_records.get(phase, [])
        if records:
            graphs[phase] = records_to_opgraph(
                records=records,
                name=f"{model_id.replace('/', '_')}_{phase}",
                phase=phase,
                metadata={**metadata},
            )

    if "train_forward" not in graphs:
        raise ValueError("train_forward phase produced no records")

    # 4. Build context
    ctx = TransformContext(
        hw_spec=hw_spec,
        parallel=ParallelConfig(tp=tp, pp=pp, ep=ep, dp=dp),
        training=TrainingConfig(
            optimizer=optimizer,
            zero_stage=zero_stage,
            micro_batch=micro_batch,
            global_batch=global_batch,
        ),
    )

    # 5. Run each phase through the training pipeline
    from python.zrt.transform.pipeline import build_training_pipeline

    pipe = build_training_pipeline()
    results = {}
    for phase, graph in graphs.items():
        results[phase] = pipe.run(graph, ctx)

    # 6. Aggregate: forward graph carries FLOPs/memory metadata.
    #    Per-stage time = forward + backward.
    fwd = results["train_forward"]
    fwd_metrics = fwd.metadata.get("pipeline_metrics")
    per_stage_ms = fwd_metrics.per_stage_ms if fwd_metrics else 0.0

    bwd = results.get("train_backward")
    if bwd:
        bwd_metrics = bwd.metadata.get("pipeline_metrics")
        if bwd_metrics:
            per_stage_ms += bwd_metrics.per_stage_ms

    # 7. Build report
    memory_breakdown = fwd.metadata.get("memory_breakdown")

    # Recompute step time with combined per-stage time
    pp_val = ctx.parallel.pp
    num_microbatches = ctx.training.num_microbatches
    warmup_steps = max(0, pp_val - 1)
    cooldown_steps = max(0, pp_val - 1)
    steady_steps = max(0, num_microbatches - pp_val + 1)
    total_steps = warmup_steps + num_microbatches + cooldown_steps
    step_time_ms = per_stage_ms * total_steps
    bubble_fraction = (warmup_steps + cooldown_steps) / total_steps if total_steps > 0 else 0.0

    # MFU
    training_flops = fwd.metadata.get("training_flops", 0.0)
    world_size = ctx.parallel.total_devices
    step_time_sec = step_time_ms / 1000.0
    achieved_flops = training_flops / step_time_sec if step_time_sec > 0 else 0.0
    if hw_spec is not None:
        from python.zrt.ir.types import DType
        peak_flops_total = world_size * hw_spec.peak_flops(DType.BF16)
    else:
        peak_flops_total = 0.0
    mfu = achieved_flops / peak_flops_total if peak_flops_total > 0 else 0.0

    # Config summary
    parallel = ctx.parallel
    training = ctx.training
    config_parts = []
    if parallel.tp > 1:
        config_parts.append(f"TP{parallel.tp}")
    if parallel.pp > 1:
        config_parts.append(f"PP{parallel.pp}")
    if parallel.ep > 1:
        config_parts.append(f"EP{parallel.ep}")
    if parallel.dp > 1:
        config_parts.append(f"DP{parallel.dp}")
    if training:
        config_parts.append(f"ZeRO-{training.zero_stage}")
        config_parts.append(f"{training.optimizer}")
        config_parts.append(f"micro{training.micro_batch}")
    config_summary = "-".join(config_parts) if config_parts else "default"

    return TrainingReport(
        config_summary=config_summary,
        step_time_ms=step_time_ms,
        per_stage_ms=per_stage_ms,
        mfu=mfu,
        training_flops=fwd.metadata.get("training_flops", 0.0),
        forward_flops=fwd.metadata.get("forward_flops", 0.0),
        backward_flops=fwd.metadata.get("backward_flops", 0.0),
        memory_breakdown=memory_breakdown.to_dict() if memory_breakdown else {},
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
        steady_steps=steady_steps,
        bubble_fraction=bubble_fraction,
        total_params=fwd.metadata.get("total_params", 0),
    )
