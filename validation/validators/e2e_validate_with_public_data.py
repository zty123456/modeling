"""
端到端验证脚本（已迁移到 validation 模块）

使用方法:
    python e2e_validate_with_public_data.py
    python e2e_validate_with_public_data.py --scenario A100_Llama2_70B_TP4
"""

# 转发到新模块
from validation.cli import main

if __name__ == "__main__":
    main()
