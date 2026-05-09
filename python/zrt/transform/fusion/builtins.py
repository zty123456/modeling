"""Layer 1: Built-in common rules for standard PyTorch decompositions.

These rules cover the model-agnostic patterns: RMSNorm (custom-written
and ``nn.RMSNorm``), CrossEntropy, Dropout.  Model-specific rules go
into ``rules/<model>.yaml``.

Rich format: each rule includes ``MatchPattern`` (auto-derived from
op_sequences for legacy clarity), ``io_roles``, ``shape_derivation``,
and FLOPs / memory formulas — see ``docs/fusion_v2_rich_rules_zh.md``.
"""
from __future__ import annotations

from .registry import register_rule
from .rule import (
    DEFAULT_SKIP_OPS,
    IORole,
    MatchPattern,
    ModuleFusionRule,
    ShapeDerivation,
)


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm — covers the 6-op decomposition produced by both Llama-style
# custom RMSNorm classes and ``nn.RMSNorm`` (the latter uses
# ``add_.Scalar`` instead of ``add.Tensor`` for the eps).
#
# Matching: ordered_regex with regex alternations on the eps-add op,
# tolerating ``_to_copy`` / ``view`` / ``permute`` / etc. interleaving.
# ─────────────────────────────────────────────────────────────────────────────

_RMSNORM_PATTERN = MatchPattern(
    kind="ordered_regex",
    op_regexes=(
        r"aten\.pow\.Tensor_Scalar",
        r"aten\.mean\.dim",
        r"aten\.(add|add_)\.(Tensor|Scalar)",
        r"aten\.rsqrt\.default",
        r"aten\.mul\.Tensor",
        r"aten\.mul\.Tensor",
    ),
    skip_ops=DEFAULT_SKIP_OPS,
    min_ops=4,
    max_ops=20,
)


_RMSNORM_IO = (
    IORole(role="activation", source_kind="input",
           source_op_index=0, source_arg_index=0,
           shape_role="[B,S,H]"),
    IORole(role="weight", source_kind="input",
           source_op_index=-1, source_arg_index=0,
           shape_role="[H]"),
    IORole(role="output", source_kind="output",
           source_op_index=-1, source_arg_index=-1,
           shape_role="[B,S,H]"),
)


_RMSNORM_SHAPE = ShapeDerivation(
    batch_size="activation.shape[0]",
    seq_len="activation.shape[1]",
    hidden_in="activation.shape[-1]",
    hidden_out="output.shape[-1]",
)


# 2 mul + 1 add + 1 rsqrt + 2 mul ≈ 6 ops/elem; numel = B*S*H.
_RMSNORM_FLOPS = "6 * batch_size * seq_len * hidden_in"
_RMSNORM_MEMORY = "activation.bytes + weight.bytes + output.bytes"


# ─────────────────────────────────────────────────────────────────────────────
# CrossEntropy — log_softmax + nll_loss
# ─────────────────────────────────────────────────────────────────────────────

_CROSS_ENTROPY_PATTERN = MatchPattern(
    kind="ordered_regex",
    op_regexes=(
        r"aten\._log_softmax\.default",
        r"aten\.nll_loss_forward\.default",
    ),
    skip_ops=DEFAULT_SKIP_OPS,
    min_ops=2,
    max_ops=8,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dropout (training mode) — empty_like + bernoulli_ + div_ + mul
# ─────────────────────────────────────────────────────────────────────────────

_DROPOUT_PATTERN = MatchPattern(
    kind="ordered_regex",
    op_regexes=(
        r"aten\.empty_like\.default",
        r"aten\.bernoulli_\.float",
        r"aten\.div_\.Scalar",
        r"aten\.mul\.Tensor",
    ),
    skip_ops=DEFAULT_SKIP_OPS,
    min_ops=4,
    max_ops=8,
)


# ─────────────────────────────────────────────────────────────────────────────

def import_builtin_rules() -> None:
    """Register Layer-1 built-in rules.

    Called by ``platforms/__init__.py:load_platform_rules`` before any
    model-specific YAML / Python rules.
    """
    # RMSNorm — custom (Llama / Qwen / Mistral / DeepSeek style)
    register_rule(ModuleFusionRule(
        target_class=("RMSNorm", "LlamaRMSNorm", "Qwen2RMSNorm",
                      "Qwen3RMSNorm", "MistralRMSNorm",
                      "DeepseekV[0-9]+RMSNorm"),
        op_type="rms_norm",
        pattern=_RMSNORM_PATTERN,
        io_roles=_RMSNORM_IO,
        shape_derivation=_RMSNORM_SHAPE,
        flops_formula=_RMSNORM_FLOPS,
        memory_formula=_RMSNORM_MEMORY,
        priority=10,
    ))

    # nn.RMSNorm (PyTorch built-in)
    register_rule(ModuleFusionRule(
        target_class="nn.RMSNorm",
        op_type="rms_norm",
        pattern=_RMSNORM_PATTERN,
        io_roles=_RMSNORM_IO,
        shape_derivation=_RMSNORM_SHAPE,
        flops_formula=_RMSNORM_FLOPS,
        memory_formula=_RMSNORM_MEMORY,
        priority=10,
    ))

    # nn.CrossEntropyLoss
    register_rule(ModuleFusionRule(
        target_class="nn.CrossEntropyLoss",
        op_type="cross_entropy",
        pattern=_CROSS_ENTROPY_PATTERN,
        io_roles=(
            IORole(role="logits", source_kind="input",
                   source_op_index=0, source_arg_index=0,
                   shape_role="[B,S,V]"),
            IORole(role="output", source_kind="output",
                   source_op_index=-1, source_arg_index=-1,
                   shape_role="[]"),
        ),
        shape_derivation=ShapeDerivation(
            batch_size="logits.shape[0]",
            seq_len="logits.shape[1]",
            vocab_size="logits.shape[-1]",
        ),
        # log_softmax: ~3 ops/elem; nll_loss: 1 gather/elem.
        flops_formula="4 * batch_size * seq_len * vocab_size",
        memory_formula="logits.bytes + output.bytes",
        priority=10,
    ))

    # nn.Dropout (training mode)
    register_rule(ModuleFusionRule(
        target_class="nn.Dropout",
        op_type="dropout",
        pattern=_DROPOUT_PATTERN,
        annotations={"transparent": True},
        priority=10,
    ))
