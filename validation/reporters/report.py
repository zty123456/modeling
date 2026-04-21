"""
报告生成
"""
import json
from pathlib import Path
from typing import List
from validation.models import PredictionResult


def print_report(results: List[PredictionResult]):
    """打印验证报告"""
    print("\n" + "=" * 100)
    print("E2E Validation Report: Public Benchmark Data vs Model Predictions")
    print("=" * 100)

    total = len(results)
    passed = sum(1 for r in results if r.is_accurate)

    print(f"\nSummary: {passed}/{total} scenarios PASSED (±20% error tolerance)")

    print("\n" + "-" * 100)
    print(f"{'Scenario':<35} {'Measured (tok/s)':<20} {'Accuracy':<20}")
    print("-" * 100)

    for result in results:
        status = "PASS" if result.is_accurate else "FAIL"
        error_str = f"{result.throughput_error_pct:.1f}%" if result.throughput_error_pct else "N/A"

        print(f"{result.scenario_id:<35} {result.measured_throughput_tok_s:<20.1f} {status:<20} ({error_str})")

    print("\n" + "-" * 100)
    print("Detailed Results:\n")

    for result in results:
        print(f"\n[{result.scenario_id}]")
        print(f"  Hardware: {result.hardware}")
        print(f"  Model: {result.model_name}")
        if result.measured_throughput_tok_s:
            print(f"  Measured Throughput: {result.measured_throughput_tok_s:.1f} tok/s")
        if result.predicted_throughput_tok_s:
            print(f"  Predicted Throughput: {result.predicted_throughput_tok_s:.1f} tok/s")
        if result.throughput_error_pct is not None:
            print(f"  Throughput Error: {result.throughput_error_pct:.1f}%")
        else:
            print(f"  Throughput Error: N/A (using measured as baseline)")

        if result.predicted_total_memory_mb:
            print(f"  Memory Budget: {result.predicted_total_memory_mb:.1f} MB (feasible={result.memory_feasible})")

        if result.predicted_comm_latency_us:
            comm_ms = result.predicted_comm_latency_us / 1000.0
            print(f"  Est. Comm Latency: {comm_ms:.3f} ms")

        if result.predicted_compute_time_ms:
            print(f"  Est. Compute Time/Token: {result.predicted_compute_time_ms:.4f} ms")

        print(f"  Status: {'PASS' if result.is_accurate else 'FAIL'}")

    print("\n" + "=" * 100)


def export_report_json(results: List[PredictionResult], output_path: str = "validation_report.json"):
    """导出 JSON 报告"""
    report_data = {
        "timestamp": "2026-04-21",
        "summary": {
            "total_scenarios": len(results),
            "passed": sum(1 for r in results if r.is_accurate),
        },
        "results": [
            {
                "scenario_id": r.scenario_id,
                "model_name": r.model_name,
                "hardware": r.hardware,
                "measured_throughput_tok_s": r.measured_throughput_tok_s,
                "predicted_throughput_tok_s": r.predicted_throughput_tok_s,
                "throughput_error_pct": r.throughput_error_pct,
                "status": "PASS" if r.is_accurate else "FAIL",
            }
            for r in results
        ],
    }

    report_path = Path(output_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nReport saved: {report_path}")
