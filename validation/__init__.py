"""
验证模块 - 性能精度对标和报告生成

核心 API:
  - validate_scenario(scenario) -> PredictionResult
  - VALIDATION_SCENARIOS: 标准验证场景集合
  - print_report(results) / export_report_json(results)
"""

from validation.models import (
    HardwareConfig,
    ModelConfig,
    InferenceConfig,
    ValidationScenario,
    PredictionResult,
)
from validation.estimators import (
    estimate_memory_budget,
    estimate_compute_time,
    estimate_comm_latency,
)
from validation.validators import validate_scenario
from validation.reporters import print_report, export_report_json
from validation.scenarios import VALIDATION_SCENARIOS

__all__ = [
    # Models
    "HardwareConfig",
    "ModelConfig",
    "InferenceConfig",
    "ValidationScenario",
    "PredictionResult",
    # Estimators
    "estimate_memory_budget",
    "estimate_compute_time",
    "estimate_comm_latency",
    # Validators
    "validate_scenario",
    # Reporters
    "print_report",
    "export_report_json",
    # Scenarios
    "VALIDATION_SCENARIOS",
]
