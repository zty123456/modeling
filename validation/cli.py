"""
验证命令行接口
"""
import sys
import argparse
from validation.scenarios import VALIDATION_SCENARIOS
from validation.validators import validate_scenario
from validation.reporters import print_report, export_report_json


def main():
    parser = argparse.ArgumentParser(
        description="E2E Validation: Public Benchmark Data vs Model Predictions"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run single scenario (e.g., A100_Llama2_70B_TP4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_report.json",
        help="Output report path",
    )
    args = parser.parse_args()

    # 选择要验证的场景
    if args.scenario:
        scenarios = [s for s in VALIDATION_SCENARIOS if s.scenario_id == args.scenario]
        if not scenarios:
            print(f"Error: Scenario '{args.scenario}' not found")
            sys.exit(1)
    else:
        scenarios = VALIDATION_SCENARIOS

    # 验证每个场景
    results = [validate_scenario(s) for s in scenarios]

    # 打印报告
    print_report(results)

    # 导出 JSON 报告
    export_report_json(results, args.output)


if __name__ == "__main__":
    main()
