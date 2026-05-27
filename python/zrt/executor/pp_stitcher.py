"""PP Pipeline Stitcher — topology-driven pipeline schedule from per-stage Timelines.

Replaces formula-based PipelineComposer with a stage × microbatch grid
approach.  Three edge types drive the schedule:

  Edge ①  F→B activation dependency   : G[s][m].fwd → G[s][m].bwd
  Edge ②  cross-stage P2P              : G[s][m].fwd → G[s+1][m].fwd
                                         G[s+1][m].bwd → G[s][m].bwd
  Edge ③  device-serial protocol       : depends on schedule (1F1B, VPP, DualPipe, ...)

Algorithm
---------
1. Build a list of ``GridTask`` for each cell in the grid.
2. Add the three edge types as dependencies between tasks.
3. Run list scheduling: each "device" is treated as a serial stream.
   ready_time = max(predecessor end times)
   start      = max(ready_time, device_free_time)
   end        = start + latency
4. Collect all task starts/ends into a global ``PPStitchedTimeline``.

TP/EP/CP characteristics are already embedded in the per-stage fwd/bwd
latencies from DAGScheduler — this stitcher only handles the PP pipeline
orchestration across microbatch × stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

from python.zrt.executor.scheduler import ScheduledOp, Timeline


# ── schedule types ────────────────────────────────────────────────────────────

class PPScheduleKind(Enum):
    ONE_F_ONE_B = auto()
    INTERLEAVED = auto()      # VPP
    DUALPIPE = auto()
    DUALPIPE_V = auto()       # DualPipeV: DualPipe + virtual stage splitting
    ZERO_BUBBLE = auto()      # ZB: split bwd into dx/dw, defer dw


# ── grid task ──────────────────────────────────────────────────────────────────

@dataclass
class GridTask:
    """A single scheduling unit in the stage × microbatch grid.

    Represents one execution block: fwd or bwd of one microbatch on one stage.
    """

    task_id: str              # e.g. "s0_m0_fwd", "s1_m2_bwd"
    stage_id: int
    mb_id: int
    phase: str                # "fwd" | "bwd" | "bwd_dx" | "bwd_dw"
    latency_us: float
    stream_id: int            # device id (same as stage_id by default)

    # Dependencies: task_ids that must finish before this task starts
    dependencies: list[str] = field(default_factory=list)

    # Scheduling result (filled by list scheduler)
    start_us: float = 0.0
    end_us: float = 0.0

    @property
    def category(self) -> str:
        return "communication" if "p2p" in self.phase else "compute"


@dataclass
class PPStitchedTimeline:
    """Complete multi-device pipeline timeline.

    Contains every task's start/end time across all stages and microbatches.
    """

    tasks: list[GridTask] = field(default_factory=list)
    pp: int = 1
    M: int = 1
    schedule_name: str = "1f1b"

    # Summary metrics
    step_time_us: float = 0.0
    warmup_us: float = 0.0
    steady_us: float = 0.0
    cooldown_us: float = 0.0
    bubble_us: float = 0.0
    bubble_fraction: float = 0.0
    p2p_overhead_us: float = 0.0

    def to_scheduled_ops(self, per_stage_timeline: Timeline | None = None) -> list[ScheduledOp]:
        """Convert grid tasks to ScheduledOp list for Chrome Trace export."""
        ops: list[ScheduledOp] = []
        for task in self.tasks:
            ops.append(ScheduledOp(
                node_id=task.task_id,
                stream_id=task.stream_id,
                stream_type="compute" if task.category == "compute" else "comm",
                start_us=task.start_us,
                end_us=task.end_us,
                latency_us=task.latency_us,
                op_type=f"pp.{task.phase}",
                category=task.category,
                phase=task.phase,
            ))
        return ops

    @property
    def total_latency_us(self) -> float:
        return self.step_time_us

    def summary(self) -> str:
        lines = [
            f"PPStitchedTimeline(schedule={self.schedule_name}, pp={self.pp}, M={self.M})",
            f"  step_time = {self.step_time_us:.1f} us  ({self.step_time_us / 1000:.3f} ms)",
            f"  warmup    = {self.warmup_us:.1f} us",
            f"  steady    = {self.steady_us:.1f} us",
            f"  cooldown  = {self.cooldown_us:.1f} us",
            f"  bubble    = {self.bubble_us:.1f} us  ({self.bubble_fraction:.2%})",
            f"  per_stage = {self.per_stage_us():.1f} us",
        ]
        return "\n".join(lines)

    def per_stage_us(self) -> float:
        """Bottleneck per-stage fwd+bwd time (µs)."""
        if not self.tasks:
            return 0.0
        per_stage: dict[int, float] = {}
        for t in self.tasks:
            per_stage[t.stage_id] = max(
                per_stage.get(t.stage_id, 0.0),
                t.end_us,
            )
        return max(per_stage.values(), default=0.0)


# ── edge builders (the three edge types) ──────────────────────────────────────

def _add_activation_dependency(tasks: dict[str, GridTask]) -> None:
    """Edge ①: within same stage+mb, fwd must finish before bwd starts."""
    for task in tasks.values():
        if task.phase == "bwd":
            fwd_id = _task_id(task.stage_id, task.mb_id, "fwd")
            if fwd_id in tasks:
                task.dependencies.append(fwd_id)


def _add_cross_stage_p2p(
    tasks: dict[str, GridTask],
    pp: int,
    M: int,
    p2p_latency_us: float,
    virtual_stages: list[int] | None = None,
) -> None:
    """Edge ②: inter-stage P2P links.

    Forward:  G[s][m].fwd → G[s+1][m].fwd
    Backward: G[s+1][m].bwd → G[s][m].bwd

    P2P transfer time is modelled by adding ``p2p_latency_us`` to the
    receiving task's latency — the destination GPU cannot begin real
    compute until the activation/gradient arrives over the wire.
    """
    for m in range(M):
        for s in range(pp - 1):
            fwd_src = _task_id(s, m, "fwd")
            fwd_dst = _task_id(s + 1, m, "fwd")
            if fwd_src in tasks and fwd_dst in tasks:
                tasks[fwd_dst].dependencies.append(fwd_src)
                tasks[fwd_dst].latency_us += p2p_latency_us

            bwd_src = _task_id(s + 1, m, "bwd")
            bwd_dst = _task_id(s, m, "bwd")
            if bwd_src in tasks and bwd_dst in tasks:
                tasks[bwd_dst].dependencies.append(bwd_src)
                tasks[bwd_dst].latency_us += p2p_latency_us


def _add_device_serial_1f1b(
    tasks: dict[str, GridTask],
    pp: int,
    M: int,
) -> None:
    """Edge ③ (1F1B): per-device 1F1B execution order.

    The device serial is split into TWO independent chains:

      Chain A (warmup forwards):
        F[0] → F[1] → ... → F[w-1]          (w = pp - s)

      Chain B (alternating + cooldown):
        B[0] → F[w] → B[1] → F[w+1] → ... → B[M-1]

    NO edge connects Chain A to Chain B.  When the warmup phase finishes
    and B[0] has all its data-dependency requirements satisfied (activation
    dep F[0]→B[0] ✓, cross-stage backward B[s+1][0]→B[s][0] ✓), the list
    scheduler will naturally place B[0] on the freed device.

    This separation is critical: connecting warmup→alternating would
    force ALL warmup forwards to complete before any backward, which
    destroys pipeline overlap — see Edge ② cross-stage backward.

    Example (pp=4, M=6):

      Stage 0 (w=4):
        Chain A:  F₀ → F₁ → F₂ → F₃
        Chain B:           B₀ → F₄ → B₁ → F₅ → B₂ → B₃ → B₄ → B₅

      Stage 3 (w=1):
        Chain A:  F₀
        Chain B:  B₀ → F₁ → B₁ → F₂ → B₂ → F₃ → B₃ → F₄ → B₄ → F₅ → B₅
    """

    def _warmup_tasks(s: int, w: int) -> list[tuple[int, str]]:
        return [(m, "fwd") for m in range(min(w, M))]

    def _alternating_tasks(s: int, w: int) -> list[tuple[int, str]]:
        """B[0]→F[w]→B[1]→F[w+1]→...→B[M-1] alternating + cooldown."""
        order: list[tuple[int, str]] = []
        b_idx = 0
        f_idx = w
        while f_idx < M:
            if b_idx < M:
                order.append((b_idx, "bwd"))
            order.append((f_idx, "fwd"))
            b_idx += 1
            f_idx += 1
        while b_idx < M:
            order.append((b_idx, "bwd"))
            b_idx += 1
        return order

    for s in range(pp):
        w = pp - s  # warmup count

        # Chain A: warmup forwards
        _chain_on_device(tasks, s, _warmup_tasks(s, w))

        # Chain B: alternating + cooldown  (no edge from chain A!)
        _chain_on_device(tasks, s, _alternating_tasks(s, w))


def _chain_on_device(
    tasks: dict[str, GridTask],
    stage_id: int,
    ordered: list[tuple[int, str]],
) -> None:
    """Chain ``ordered`` tasks serially on ``stage_id``: each → next."""
    prev: str | None = None
    for mb, phase in ordered:
        tid = _task_id(stage_id, mb, phase)
        if tid not in tasks:
            continue
        if prev is not None and prev not in tasks[tid].dependencies:
            tasks[tid].dependencies.append(prev)
        prev = tid


def _add_device_serial_dualpipe(
    tasks: dict[str, GridTask],
    pp: int,
    M: int,
) -> None:
    """Edge ③ (DualPipe): per-device execution order with dual-stream skip.

    Two parallel streams per device:
      Stream A:  F₀→F₁→B₀→F₂→B₁→F₃→B₂→...  (skipping pattern)
      Stream B:  B₀→B₁→F₀→B₂→F₁→B₃→F₂→...  (reversed, offset)

    Dependencies added per device:
      1. Forward chain: F[m] → F[m+1]  (serial microbatch order)
      2. Backward chain: B[m] → B[m+1]  (serial microbatch order)
      3. Dual skip: B[m] → F[m+2]       (anti-parallel hand-off after each bwd)

    Forward-chain and backward-chain make the per-device ordering explicit,
    preventing the list scheduler from constructing sub-optimal schedules
    when stage loads are imbalanced (the chain constrains microbatches to
    execute in natural order, giving DualPipe's anti-parallel skip a
    predictable base to work from).
    """
    for s in range(pp):
        # Forward chain: F[0] → F[1] → ... → F[M-1]
        _chain_on_device(
            tasks, s,
            [(m, "fwd") for m in range(M)],
        )
        # Backward chain: B[0] → B[1] → ... → B[M-1]
        _chain_on_device(
            tasks, s,
            [(m, "bwd") for m in range(M)],
        )
        # DualPipe skip: B[m] → F[m+2]  (anti-parallel stream hand-off)
        for m in range(M):
            bwd_id = _task_id(s, m, "bwd")
            if m + 2 < M:
                next_fwd_id = _task_id(s, m + 2, "fwd")
                if bwd_id in tasks and next_fwd_id in tasks:
                    tasks[next_fwd_id].dependencies.append(bwd_id)


def _add_device_serial_dualpipe_v(
    tasks: dict[str, GridTask],
    pp: int,
    vpp: int,
    M: int,
) -> None:
    """Edge ③ (DualPipeV with V>1): cross-vstage DualPipe hand-off.

    Each physical device holds virtual stages from both the forward
    pipeline (v < eff_pp/2) and the backward pipeline (v >= eff_pp/2).
    The two pipelines run concurrently on the device — the cross-vstage
    skip creates the anti-parallel hand-off:

      B_{B-first}[m] → F_{F-first}[m+2]

    No F-chain/B-chain are added; the list scheduler's stream_id constraint
    naturally serializes tasks on the same physical device.  The cross-vstage
    skip is the minimal edge needed to express the DualPipe bidirectional
    interleaving when V>1.
    """
    eff_pp = pp * vpp
    pivot = eff_pp // 2

    for dev in range(pp):
        fwd_v = next((v for v in range(eff_pp) if v % pp == dev and v < pivot), None)
        bwd_v = next((v for v in range(eff_pp) if v % pp == dev and v >= pivot), None)
        if fwd_v is None or bwd_v is None:
            continue
        for m in range(M):
            bwd_id = _task_id(bwd_v, m, "bwd")
            if m + 2 < M:
                next_fwd_id = _task_id(fwd_v, m + 2, "fwd")
                if bwd_id in tasks and next_fwd_id in tasks:
                    tasks[next_fwd_id].dependencies.append(bwd_id)


def _add_device_serial_zb(
    tasks: dict[str, GridTask],
    pp: int,
    M: int,
) -> None:
    """Edge ③ (ZeroBubble): split bwd into bwd_dx and bwd_dw on the grid.

    bwd_dw can be deferred to fill pipeline bubbles.
    Protocol per device: F₀→B_dx₀→F₁→B_dx₁→... ; B_dw tasks float independently.
    """
    for s in range(pp):
        for m in range(M):
            bwd_dx_id = _task_id(s, m, "bwd_dx")
            if m + 1 < M:
                next_fwd_id = _task_id(s, m + 1, "fwd")
                if bwd_dx_id in tasks and next_fwd_id in tasks:
                    tasks[next_fwd_id].dependencies.append(bwd_dx_id)


def _task_id(stage_id: int, mb_id: int, phase: str) -> str:
    return f"s{stage_id}_m{mb_id}_{phase}"


# ── list scheduler ────────────────────────────────────────────────────────────

def _list_schedule(tasks: dict[str, GridTask]) -> list[GridTask]:
    """List scheduling over grid tasks.

    Each device (stream) is serial.  Data dependencies across devices
    are respected via predecessor end-time checking.
    """
    device_free: dict[int, float] = {}
    finish: dict[str, float] = {}
    ready: list[GridTask] = []

    # Topological sort via Kahn's algorithm with in-degree tracking
    in_degree: dict[str, int] = {tid: len(t.dependencies) for tid, t in tasks.items()}
    for tid, deg in in_degree.items():
        if deg == 0:
            ready.append(tasks[tid])

    scheduled: list[GridTask] = []
    while ready:
        # Pick task with earliest possible start time (greedy)
        ready.sort(key=lambda t: max(
            device_free.get(t.stream_id, 0.0),
            max((finish.get(d, 0.0) for d in t.dependencies), default=0.0),
        ))
        task = ready.pop(0)

        # Data dependency: wait for all predecessors
        pred_done = max(
            (finish.get(d, 0.0) for d in task.dependencies),
            default=0.0,
        )
        # Resource constraint: wait for device to be free
        dev_free = device_free.get(task.stream_id, 0.0)

        task.start_us = max(pred_done, dev_free)
        task.end_us = task.start_us + task.latency_us

        finish[task.task_id] = task.end_us
        device_free[task.stream_id] = task.end_us
        scheduled.append(task)

        # Release dependents
        for tid, t in tasks.items():
            if task.task_id in t.dependencies:
                in_degree[tid] -= 1
                if in_degree[tid] == 0:
                    ready.append(t)

    if len(scheduled) < len(tasks):
        unscheduled = [tid for tid in tasks if tid not in finish]
        raise ValueError(
            f"Cycle detected in grid task dependencies: "
            f"{len(unscheduled)} task(s) with unresolved edges: {unscheduled[:10]}"
        )

    return scheduled


# ── grid builder ───────────────────────────────────────────────────────────────

def _build_task_id_set(tasks: list[GridTask]) -> dict[str, GridTask]:
    return {t.task_id: t for t in tasks}


# ── main stitcher ─────────────────────────────────────────────────────────────

class PPStitcher:
    """Topology-driven pipeline schedule from per-stage Timelines.

    Parameters
    ----------
    stage_fwd_us : dict[int, float]
        Per-stage forward latency (µs), keyed by stage_id.
    stage_bwd_us : dict[int, float]
        Per-stage backward latency (µs), keyed by stage_id.
    stage_bwd_dw_us : dict[int, float] | None
        Per-stage weight-gradient latency (µs), for ZeroBubble split.
        Defaults to 0 for each stage if not provided.
    pp : int
        Number of pipeline stages.
    M : int
        Number of microbatches.
    p2p_latency_us : float
        One-way P2P activation transfer time (µs) between adjacent stages.
    schedule : str
        Pipeline schedule: "1f1b", "interleaved", "dualpipe", "dualpipev", "zb".
    vpp_chunks : int
        Virtual pipeline stages per device (for VPP/interleaved and DualPipeV).
    """

    def __init__(
        self,
        stage_fwd_us: dict[int, float],
        stage_bwd_us: dict[int, float],
        *,
        pp: int,
        M: int,
        p2p_latency_us: float = 0.0,
        schedule: str = "1f1b",
        vpp_chunks: int = 1,
        stage_bwd_dw_us: dict[int, float] | None = None,
    ) -> None:
        self._stage_fwd = stage_fwd_us
        self._stage_bwd = stage_bwd_us
        self._stage_bwd_dw = stage_bwd_dw_us or {}
        self._pp = pp
        self._M = M
        self._p2p_us = p2p_latency_us
        self._schedule = schedule
        self._vpp = vpp_chunks

    # ── public API ────────────────────────────────────────────────────────

    def stitch(self) -> PPStitchedTimeline:
        """Build and schedule the stage × microbatch grid.

        Returns a PPStitchedTimeline with per-task timing and summary metrics.
        """
        if self._pp <= 1:
            return self._stitch_pp1()

        kind = self._resolve_schedule_kind()
        # DualPipeV with vpp<=1 degenerates to DualPipe (same as DualPipeVComposer)
        if kind == PPScheduleKind.DUALPIPE_V and self._vpp <= 1:
            kind = PPScheduleKind.DUALPIPE
        eff_pp = self._effective_pp(kind)
        tasks = self._build_grid(kind, eff_pp)
        task_map = _build_task_id_set(tasks)

        _add_activation_dependency(task_map)
        _add_cross_stage_p2p(task_map, eff_pp, self._M, self._p2p_us)

        self._add_device_serial(task_map, kind, eff_pp)

        scheduled = _list_schedule(task_map)

        return self._build_result(scheduled, kind, eff_pp)

    def stitch_from_timelines(
        self,
        timelines: list[Timeline],
    ) -> PPStitchedTimeline:
        """Convenience: build stage_fwd/stage_bwd from DAGScheduler Timelines.

        ``timelines[s]`` is the DAGScheduler output for stage ``s``.
        """
        stage_fwd: dict[int, float] = {}
        stage_bwd: dict[int, float] = {}

        for s, tl in enumerate(timelines):
            fwd_lat = tl.phase_latency("fwd")
            bwd_lat = tl.phase_latency("bwd")
            if fwd_lat == 0.0 and bwd_lat == 0.0:
                fwd_lat = tl.total_latency_us
            stage_fwd[s] = fwd_lat
            stage_bwd[s] = bwd_lat

        self._stage_fwd = stage_fwd
        self._stage_bwd = stage_bwd
        return self.stitch()

    # ── grid construction ─────────────────────────────────────────────────

    def _effective_pp(self, kind: PPScheduleKind) -> int:
        if kind in (PPScheduleKind.INTERLEAVED, PPScheduleKind.DUALPIPE_V) and self._vpp > 1:
            return self._pp * self._vpp
        return self._pp

    def _vstage_to_device(self, virtual_stage: int, kind: PPScheduleKind) -> int:
        if kind == PPScheduleKind.INTERLEAVED and self._vpp > 1:
            return virtual_stage % self._pp
        if kind == PPScheduleKind.DUALPIPE_V and self._vpp > 1:
            return virtual_stage % self._pp
        return virtual_stage

    def _build_grid(self, kind: PPScheduleKind, eff_pp: int) -> list[GridTask]:
        """Create GridTask for every cell in the stage × microbatch grid."""
        tasks: list[GridTask] = []

        if kind == PPScheduleKind.ZERO_BUBBLE:
            for s in range(self._pp):
                fwd_us = self._stage_fwd.get(s, 0.0)
                bwd_us = self._stage_bwd.get(s, 0.0)
                dw_us = self._stage_bwd_dw.get(s, 0.0)
                dx_us = max(0.0, bwd_us - dw_us)
                for m in range(self._M):
                    tasks.append(GridTask(
                        task_id=_task_id(s, m, "fwd"),
                        stage_id=s, mb_id=m, phase="fwd",
                        latency_us=fwd_us, stream_id=s,
                    ))
                    tasks.append(GridTask(
                        task_id=_task_id(s, m, "bwd_dx"),
                        stage_id=s, mb_id=m, phase="bwd_dx",
                        latency_us=dx_us, stream_id=s,
                    ))
                    tasks.append(GridTask(
                        task_id=_task_id(s, m, "bwd_dw"),
                        stage_id=s, mb_id=m, phase="bwd_dw",
                        latency_us=dw_us, stream_id=s,
                    ))
        elif kind == PPScheduleKind.INTERLEAVED and self._vpp > 1:
            for v in range(eff_pp):
                phys = self._vstage_to_device(v, kind)
                fwd_us = self._stage_fwd.get(phys, 0.0) / self._vpp
                bwd_us = self._stage_bwd.get(phys, 0.0) / self._vpp
                for m in range(self._M):
                    tasks.append(GridTask(
                        task_id=_task_id(v, m, "fwd"),
                        stage_id=v, mb_id=m, phase="fwd",
                        latency_us=fwd_us, stream_id=phys,
                    ))
                    tasks.append(GridTask(
                        task_id=_task_id(v, m, "bwd"),
                        stage_id=v, mb_id=m, phase="bwd",
                        latency_us=bwd_us, stream_id=phys,
                    ))
        elif kind == PPScheduleKind.DUALPIPE_V and self._vpp > 1:
            for v in range(eff_pp):
                phys = self._vstage_to_device(v, kind)
                fwd_us = self._stage_fwd.get(v % self._pp, 0.0) / self._vpp
                bwd_us = self._stage_bwd.get(v % self._pp, 0.0) / self._vpp
                for m in range(self._M):
                    tasks.append(GridTask(
                        task_id=_task_id(v, m, "fwd"),
                        stage_id=v, mb_id=m, phase="fwd",
                        latency_us=fwd_us, stream_id=phys,
                    ))
                    tasks.append(GridTask(
                        task_id=_task_id(v, m, "bwd"),
                        stage_id=v, mb_id=m, phase="bwd",
                        latency_us=bwd_us, stream_id=phys,
                    ))
        else:
            for s in range(self._pp):
                fwd_us = self._stage_fwd.get(s, 0.0)
                bwd_us = self._stage_bwd.get(s, 0.0)
                for m in range(self._M):
                    tasks.append(GridTask(
                        task_id=_task_id(s, m, "fwd"),
                        stage_id=s, mb_id=m, phase="fwd",
                        latency_us=fwd_us, stream_id=s,
                    ))
                    tasks.append(GridTask(
                        task_id=_task_id(s, m, "bwd"),
                        stage_id=s, mb_id=m, phase="bwd",
                        latency_us=bwd_us, stream_id=s,
                    ))

        return tasks

    def _add_device_serial(
        self,
        task_map: dict[str, GridTask],
        kind: PPScheduleKind,
        eff_pp: int,
    ) -> None:
        """Apply per-device serialization edges (Edge ③).

        For 1F1B/VPP (interleaved) without VPP, explicit device-serial
        edges prevent the list scheduler from reordering microbatches.

        For DualPipeV with V>1, explicit F/B chains and cross-vstage
        DualPipe skip edges are needed to create the bidirectional
        interleaving — P2P constraints alone cannot express the
        anti-parallel hand-off between F-first and B-first virtual
        stages on the same physical device.
        """
        if kind == PPScheduleKind.ONE_F_ONE_B:
            _add_device_serial_1f1b(task_map, eff_pp, self._M)
        elif kind == PPScheduleKind.INTERLEAVED:
            if self._vpp <= 1:
                _add_device_serial_1f1b(task_map, eff_pp, self._M)
        elif kind == PPScheduleKind.DUALPIPE:
            _add_device_serial_dualpipe(task_map, eff_pp, self._M)
        elif kind == PPScheduleKind.DUALPIPE_V:
            if self._vpp <= 1:
                _add_device_serial_dualpipe(task_map, eff_pp, self._M)
            else:
                _add_device_serial_dualpipe_v(task_map, self._pp, self._vpp, self._M)
        elif kind == PPScheduleKind.ZERO_BUBBLE:
            _add_device_serial_zb(task_map, eff_pp, self._M)

    # ── result assembly ───────────────────────────────────────────────────

    def _build_result(
        self,
        scheduled: list[GridTask],
        kind: PPScheduleKind,
        eff_pp: int,
    ) -> PPStitchedTimeline:
        """Compute summary metrics from scheduled tasks.

        Phases for 1F1B:
            warmup   = time until stage pp-1 starts its first bwd (mb=0)
            cooldown = time from stage 0 finishing last fwd (mb=M-1) until end
            steady   = step_time - warmup - cooldown

        Bubble is computed from: step_time - M * per_stage_bottleneck
        (the part that exceeds ideal serial execution).
        """
        if not scheduled:
            return PPStitchedTimeline(pp=self._pp, M=self._M)

        max_end = max(t.end_us for t in scheduled)
        min_start = min(t.start_us for t in scheduled)
        step_us = max_end - min_start

        per_stage = max(
            self._stage_fwd.get(s, 0.0) + self._stage_bwd.get(s, 0.0)
            for s in range(self._pp)
        )
        ideal_time = self._M * per_stage
        bubble_us = max(0.0, step_us - ideal_time)

        # Phase boundaries via key grid cells
        by_id = {t.task_id: t for t in scheduled}

        # warmup: time from start until last stage starts its first bwd (mb=0)
        last_stage = eff_pp - 1
        warmup_key = _task_id(last_stage, 0, "bwd")
        if warmup_key not in by_id and kind == PPScheduleKind.ZERO_BUBBLE:
            warmup_key = _task_id(last_stage, 0, "bwd_dx")
        warmup_end = by_id[warmup_key].start_us if warmup_key in by_id else min_start
        warmup_us = max(0.0, warmup_end - min_start)

        # cooldown: time from stage 0 starting its last bwd (mb=M-1) until end
        cooldown_key = _task_id(0, self._M - 1, "bwd")
        if cooldown_key not in by_id and kind == PPScheduleKind.ZERO_BUBBLE:
            cooldown_key = _task_id(0, self._M - 1, "bwd_dx")
        cooldown_start = by_id[cooldown_key].start_us if cooldown_key in by_id else max_end
        cooldown_us = max(0.0, max_end - cooldown_start)

        steady_us = max(0.0, step_us - warmup_us - cooldown_us)

        schedule_names: dict[PPScheduleKind, str] = {
            PPScheduleKind.ONE_F_ONE_B: "1f1b",
            PPScheduleKind.INTERLEAVED: "interleaved",
            PPScheduleKind.DUALPIPE: "dualpipe",
            PPScheduleKind.DUALPIPE_V: "dualpipev",
            PPScheduleKind.ZERO_BUBBLE: "zb",
        }

        return PPStitchedTimeline(
            tasks=scheduled,
            pp=self._pp,
            M=self._M,
            schedule_name=schedule_names.get(kind, "1f1b"),
            step_time_us=max_end - min_start,
            warmup_us=warmup_us,
            steady_us=steady_us,
            cooldown_us=cooldown_us,
            bubble_us=bubble_us,
            bubble_fraction=bubble_us / (max_end - min_start) if max_end > min_start else 0.0,
            p2p_overhead_us=0.0,
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _stitch_pp1(self) -> PPStitchedTimeline:
        """Handle pp=1: single stage, no pipeline."""
        fwd_us = self._stage_fwd.get(0, 0.0)
        bwd_us = self._stage_bwd.get(0, 0.0)

        tasks: list[GridTask] = []
        cursor = 0.0
        for m in range(self._M):
            fwd = GridTask(
                task_id=_task_id(0, m, "fwd"),
                stage_id=0, mb_id=m, phase="fwd",
                latency_us=fwd_us, stream_id=0,
            )
            fwd.start_us = cursor
            fwd.end_us = cursor + fwd_us

            bwd = GridTask(
                task_id=_task_id(0, m, "bwd"),
                stage_id=0, mb_id=m, phase="bwd",
                latency_us=bwd_us, stream_id=0,
                dependencies=[fwd.task_id],
            )
            cursor += fwd_us
            bwd.start_us = cursor
            bwd.end_us = cursor + bwd_us
            cursor += bwd_us

            tasks.extend([fwd, bwd])

        step = self._M * (fwd_us + bwd_us)
        return PPStitchedTimeline(
            tasks=tasks, pp=1, M=self._M, schedule_name="1f1b",
            step_time_us=step,
            steady_us=step,
        )

    def _resolve_schedule_kind(self) -> PPScheduleKind:
        s = self._schedule.lower()
        if s in ("1f1b", "one_f_one_b", "gpipe"):
            return PPScheduleKind.ONE_F_ONE_B
        if s in ("interleaved", "vpp", "i1f1b"):
            return PPScheduleKind.INTERLEAVED
        if s in ("dualpipe", "dp"):
            return PPScheduleKind.DUALPIPE
        if s in ("dualpipev", "dpv"):
            return PPScheduleKind.DUALPIPE_V
        if s in ("zb", "zero_bubble", "zerobubble"):
            return PPScheduleKind.ZERO_BUBBLE
        return PPScheduleKind.ONE_F_ONE_B


# ── factory ────────────────────────────────────────────────────────────────────

def stitch_pp_pipeline(
    stage_fwd_us: dict[int, float],
    stage_bwd_us: dict[int, float],
    *,
    pp: int,
    M: int,
    p2p_latency_us: float = 0.0,
    schedule: str = "1f1b",
    vpp_chunks: int = 1,
    stage_bwd_dw_us: dict[int, float] | None = None,
) -> PPStitchedTimeline:
    """One-shot PP pipeline stitch.

    Example
    -------
    >>> result = stitch_pp_pipeline(
    ...     stage_fwd_us={0: 100, 1: 80, 2: 120, 3: 90},
    ...     stage_bwd_us={0: 200, 1: 160, 2: 240, 3: 180},
    ...     pp=4, M=8, p2p_latency_us=5, schedule="1f1b",
    ... )
    >>> print(result.summary())
    """
    stitcher = PPStitcher(
        stage_fwd_us=stage_fwd_us,
        stage_bwd_us=stage_bwd_us,
        pp=pp, M=M,
        p2p_latency_us=p2p_latency_us,
        schedule=schedule,
        vpp_chunks=vpp_chunks,
        stage_bwd_dw_us=stage_bwd_dw_us,
    )
    return stitcher.stitch()