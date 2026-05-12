from __future__ import annotations

import itertools
import logging
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        nodes=config.get("nodes", 1),
        gpus_per_node=config.get("gpus_per_node", 8),
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
    neg_metrics: List[str] = field(default_factory=lambda: ["seq_len", "micro_batch"])
    thresholds: Dict[str, float] = field(default_factory=lambda: {"max_memory_gb": 80.0})
    black_list_dict: Dict = field(default_factory=dict)
    best_dict: Dict = field(default_factory=dict)
    lock: Any = None
    save_all_result: bool = False
    output_path: str = ""
    _fp_cache: Dict = field(default_factory=dict)
    _mv_cache: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.lock is None:
            self.lock = multiprocessing.get_context("spawn").Lock()

        model = self.param_grid.get("model", "unknown")
        if isinstance(model, list):
            model = model[0] if model else "unknown"
        world_sizes = self.param_grid.get("world_size", [1])
        max_world = max(world_sizes) if isinstance(world_sizes, list) else world_sizes
        threshold_str = "_".join([f"{k}_{v}" for k, v in self.thresholds.items()])
        self.output_path = os.path.join("output", "training_search", f"{model}_ws_{max_world}_{threshold_str}")

    def generate_static_configs(self) -> List[Dict[str, Any]]:
        grid = {k: (v if isinstance(v, list) else [v]) for k, v in self.param_grid.items()}
        keys, values = list(grid.keys()), list(grid.values())

        configs = []
        for combination in itertools.product(*values):
            config = dict(zip(keys, combination))
            world_size = config.get("world_size", 1)
            tp = config.get("tp", 1)
            cp = config.get("cp", 1)
            pp = config.get("pp", 1)
            dp = config.get("dp", 1)
            if world_size != tp * cp * pp * dp:
                continue
            configs.append(config)

        configs.sort(key=lambda x: [x.get(m, 0) for m in self.neg_metrics])
        logger.info(
            f"Config Stats | Grid: {len(list(itertools.product(*values)))} | Filtered: {len(configs)}"
        )
        return configs

    def is_pruned(self, config: Dict) -> bool:
        fp = self._fp_cache.get(id(config)) or self._get_fingerprint(config)
        curr_vals = self._mv_cache.get(id(config)) or self._get_metric_vals(config)

        if fp not in self.black_list_dict:
            return False
        for recorded_vals in self.black_list_dict[fp]:
            if all(c >= r for c, r in zip(curr_vals, recorded_vals)):
                return True
        return False

    def update_black_list(self, config: Dict, error_type: str):
        fp = self._get_fingerprint(config)
        vals = self._get_metric_vals(config)
        with self.lock:
            if fp not in self.black_list_dict:
                self.black_list_dict[fp] = []
            if vals not in self.black_list_dict[fp]:
                self.black_list_dict[fp].append(vals)

    def _get_fingerprint(self, config: Dict) -> Tuple:
        result = tuple(sorted((k, v) for k, v in config.items() if k not in self.neg_metrics))
        self._fp_cache[id(config)] = result
        return result

    def _get_metric_vals(self, config: Dict) -> Tuple:
        result = tuple(config.get(m, 0) for m in self.neg_metrics)
        self._mv_cache[id(config)] = result
        return result


def run_training_task_wrapper(config: Dict, manager: TrainingConfigManager) -> Optional[Dict]:
    from zrt.training.ir.builders import build_graph
    from zrt.training.models.flops import op_cost as _op_cost

    if manager.is_pruned(config):
        return None

    model_name = config.get("model", "deepseek_v3_2")
    model = _load_model_spec(model_name)
    strategy = _make_strategy_from_config(config)
    system = _make_system_from_config(config)

    try:
        strategy.validate(model, system)
    except ValueError:
        manager.update_black_list(config, "validation_error")
        return None

    try:
        graph = build_graph(model, strategy)
        op_costs: Dict[str, object] = {}
        for op in graph.ops:
            op_costs[op.name] = _op_cost(op, model)

        report = estimate(model, system, strategy, graph=graph)

        max_mem_gb = manager.thresholds.get("max_memory_gb", 80.0)
        if report.memory and report.memory.total / 1e9 > max_mem_gb:
            manager.update_black_list(config, "oom")
            return None

        return {
            "config": config,
            "report": report,
            "graph": graph,
            "model": model,
            "system": system,
            "strategy": strategy,
            "op_costs": op_costs,
        }
    except Exception as e:
        manager.update_black_list(config, "runtime_error")
        logger.warning(f"Evaluation failed: {e}")
        return None


def execute_training_tasks(
        configs: List[Dict],
        manager: TrainingConfigManager,
        workers: int,
) -> List[Dict]:
    results = []
    active_configs = [cfg for cfg in configs if not manager.is_pruned(cfg)]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_training_task_wrapper, cfg, manager): cfg for cfg in active_configs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating configs", unit="config"):
            result = future.result()
            if result:
                results.append(result)

    logger.info(f"Config Stats | Total: {len(configs)}, Valid: {len(results)}")
    return results


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
    df = df.sort_values("mfu", ascending=False)

    cols = ["model", "world_size", "tp", "cp", "pp", "ep", "dp", "zero_stage", "pp_schedule",
            "recompute", "optimizer", "step_time_ms", "pipeline_time_ms", "mfu", "hfu",
            "bubble_fraction", "tokens_per_sec", "memory_gb"]
    df = df[[c for c in cols if c in df.columns]]
    return df


def save_results(df: pd.DataFrame, frontier_results: List[Dict], output_path: str):
    from zrt.training.io.excel_exporter import export_estimate_excel

    if os.path.exists(output_path):
        for item in os.listdir(output_path):
            item_path = os.path.join(output_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    csv_path = os.path.join(output_path, "results_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")

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
        neg_metrics: List[str] = None,
        thresholds: Dict[str, float] = None,
        workers: int = 16,
        save_all_result: bool = False,
) -> pd.DataFrame:
    neg_metrics = neg_metrics or ["seq_len", "micro_batch"]
    thresholds = thresholds or {"max_memory_gb": 80.0}

    with multiprocessing.Manager() as mp_manager:
        manager = TrainingConfigManager(
            param_grid=param_grid,
            neg_metrics=neg_metrics,
            thresholds=thresholds,
            black_list_dict=mp_manager.dict(),
            best_dict=mp_manager.dict(),
            lock=mp_manager.Lock(),
            save_all_result=save_all_result,
        )

        configs = manager.generate_static_configs()

        start_time = time.time()
        logger.info(f"Starting parallel search with {workers} workers...")

        results = execute_training_tasks(configs, manager, workers)

        elapsed = time.time() - start_time
        logger.info(f"Search completed in {elapsed:.2f} seconds")

        if not results:
            logger.error("No valid configurations found.")
            return pd.DataFrame()

        reports = [r["report"] for r in results]
        frontier_reports = pareto_frontier(reports)
        frontier_results = [results[reports.index(rp)] for rp in frontier_reports]

        df = format_results(frontier_reports, [r["config"] for r in frontier_results])
        save_results(df, frontier_results, manager.output_path)

        return df


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    training_param_grid = {
        "model": ["deepseek_v3_2"],
        "hw": ["nvidia_h100_sxm"],
        "nodes": [8],
        "gpus_per_node": [8],
        "host_mem_gb": [2048.0],
        "world_size": [64],
        "tp": [1, 2, 4, 8],
        "cp": [1, 2, 4, 8],
        "pp": [1, 2, 4, 8],
        "ep": [64],
        "dp": [1, 2, 4, 8],
        "micro_batch": [1],
        "global_batch": [1024],
        "seq_len": [4096],
        "zero_stage": [1],
        "pp_schedule": ["1f1b"],
        "vpp_chunks": [1],
        "cp_kind": ["ulysses"],
        "tp_overlap": ["none"],
        "ep_overlap": [False],
        "dualbatch": [False],
        "dp_overlap_in_bubble": [True],
        "recompute": ["selective"],
        "optimizer": ["adam"],
    }

    df = run_training_search_parallel(param_grid=training_param_grid, workers=32)