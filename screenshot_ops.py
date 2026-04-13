"""LLM Operator Screenshot — entry point.

Delegates all logic to the screenshot_ops package.

Direct usage (recommended)::

    python screenshot_ops.py <model_id> [--layers N] [--output FILE]
                             [--batch-size B] [--seq-len S]

    python screenshot_ops.py deepseek-ai/DeepSeek-V3-0324 --layers 4
    python screenshot_ops.py Qwen/Qwen3-8B --layers 4
    python screenshot_ops.py ./hf_models/deepseek_v3 --layers 4

Package usage (backward-compatible with remote"s --model flag)::

    python -m screenshot_ops.main --model v3
    python -m screenshot_ops.main --model v3.2
"""
from screenshot_ops.main import main  # noqa: F401

if __name__ == "__main__":
    main()
