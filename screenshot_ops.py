"""LLM Operator Screenshot — entry point.

Delegates all logic to the python.zrt.graph package.

Direct usage (recommended)::

    python screenshot_ops.py <model_id> [--layers N] [--output-dir DIR]
                             [--batch-size B] [--seq-len S]

    python screenshot_ops.py deepseek-ai/DeepSeek-V3-0324 --layers 4
    python screenshot_ops.py Qwen/Qwen3-8B --layers 4
    python screenshot_ops.py ./hf_models/deepseek_v3 --layers 4
"""
from python.zrt.graph.main import main  # noqa: F401

if __name__ == "__main__":
    main()
