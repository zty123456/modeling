from __future__ import annotations

from dataclasses import dataclass

from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import ModelSpec


FP8_DTYPES = {Dtype.FP8_E4M3, Dtype.FP8_E5M2}


@dataclass(frozen=True)
class WavePipelineResult:
    total_s: float
    exposed_comm_s: float
    hidden_comm_s: float
    comm_total_s: float


def infer_quant_variant(model: ModelSpec) -> str:
    if (
        model.routed_expert_weight_dtype is Dtype.FP4
        and model.effective_moe_act_dtype() in FP8_DTYPES
    ):
        return "w4a8"
    return "standard"


def resolve_mega_moe_waves(
    *,
    requested: int,
    hardware_waves: int,
    experts_per_rank: int,
) -> int:
    if experts_per_rank <= 1:
        return 1
    target = requested if requested > 0 else (hardware_waves if hardware_waves > 0 else 4)
    target = max(1, min(int(target), experts_per_rank))
    divisors = [d for d in range(1, experts_per_rank + 1) if experts_per_rank % d == 0]
    valid = [d for d in divisors if d <= target]
    return max(valid) if valid else 1


def simulate_wave_pipeline(
    *,
    waves: int,
    dispatch_s: float,
    compute_s: float,
    combine_s: float,
) -> WavePipelineResult:
    if waves <= 0:
        return WavePipelineResult(0.0, 0.0, 0.0, 0.0)

    dispatch_done: list[float | None] = [None] * waves
    compute_done: list[float | None] = [None] * waves
    combine_done: list[bool] = [False] * waves

    comm_cursor = 0.0
    compute_cursor = 0.0
    next_dispatch = 0
    next_compute = 0
    completed_combines = 0

    def schedule_ready_compute() -> None:
        nonlocal next_compute, compute_cursor
        while next_compute < waves and dispatch_done[next_compute] is not None:
            start = max(compute_cursor, dispatch_done[next_compute] or 0.0)
            compute_cursor = start + compute_s
            compute_done[next_compute] = compute_cursor
            next_compute += 1

    while completed_combines < waves:
        schedule_ready_compute()

        ready_combine = None
        for wave_id, done_at in enumerate(compute_done):
            if done_at is not None and not combine_done[wave_id] and done_at <= comm_cursor:
                ready_combine = wave_id
                break

        if ready_combine is not None:
            comm_cursor += combine_s
            combine_done[ready_combine] = True
            completed_combines += 1
            continue

        if next_dispatch < waves:
            comm_cursor += dispatch_s
            dispatch_done[next_dispatch] = comm_cursor
            next_dispatch += 1
            continue

        future_combines = [
            done_at
            for wave_id, done_at in enumerate(compute_done)
            if done_at is not None and not combine_done[wave_id] and done_at > comm_cursor
        ]
        if future_combines:
            comm_cursor = min(future_combines)
            continue

        schedule_ready_compute()

    total = max(comm_cursor, compute_cursor)
    comm_total = waves * (dispatch_s + combine_s)
    exposed = max(0.0, total - waves * compute_s)
    exposed = min(exposed, comm_total)
    hidden = max(0.0, comm_total - exposed)
    return WavePipelineResult(total, exposed, hidden, comm_total)
