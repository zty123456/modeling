"""Tests for spec-based training Excel export."""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

from openpyxl import load_workbook

from zrt.training.io.excel_exporter import export_estimate_excel
from zrt.training.search.training_search_util import _make_system_from_config
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.report import TrainingReport
from zrt.training.spec.strategy import Strategy


def test_estimate_excel_includes_comm_domain_sheet():
    model = ModelSpec(
        hidden=128,
        ffn=256,
        num_heads=8,
        num_kv_heads=8,
        head_dim=16,
        vocab=32000,
        seq_len=1024,
        layers=[LayerKind.DENSE, LayerKind.DENSE],
    )
    system = _make_system_from_config({
        "hw": "nvidia_gb300_nvl576",
        "world_size": 128,
    })
    strategy = Strategy(
        tp=4,
        cp=2,
        pp=2,
        ep=2,
        dp=8,
        micro_batch=1,
        global_batch=8,
    )

    output_dir = Path("output") / "test_estimate_excel_comm_domain"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        output_path = export_estimate_excel(
            report=TrainingReport(step_time_ms=100.0, mfu=0.1),
            graph=SimpleNamespace(ops=[]),
            model=model,
            system=system,
            strategy=strategy,
            op_costs={},
            output_path=output_dir / "estimate.xlsx",
        )

        wb = load_workbook(output_path, data_only=True)
        assert "Comm Domains" in wb.sheetnames
        ws = wb["Comm Domains"]

        headers = [ws.cell(row=1, column=i).value for i in range(1, 7)]
        assert headers == ["Metric", "EP", "PP", "DP", "TP", "CP"]

        rows = {
            ws.cell(row=i, column=1).value: [
                ws.cell(row=i, column=j).value for j in range(2, 7)
            ]
            for i in range(2, ws.max_row + 1)
        }
        assert rows["Group Size"] == [2, 2, 8, 4, 2]
        assert all(rows["Tier"])
        assert all("[" in str(v) for v in rows["Rank Sample"])
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
