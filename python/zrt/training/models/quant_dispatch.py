"""Shared per-component dtype bundle dispatch.

Both the spec-side ``resolve_op_dtypes`` (in ``quant.py``) and the graph-side
``graph_resolve_op_dtypes`` (in ``transform/analysis/quant.py``) delegate to
``dispatch()`` here — eliminating the line-for-line duplication of the five
bundle helpers.
"""
from __future__ import annotations

from zrt.training.spec.dtype import Dtype
from zrt.training.models.quant import OpDtypeBundle
from zrt.training.models.quant_protocol import QuantProfileLike


def _attention_bundle(p: QuantProfileLike) -> OpDtypeBundle:
    a = p.effective_attn_act_dtype()
    return OpDtypeBundle(
        in_act=a, weight=p.attn_weight_dtype, out_act=a,
        compute=p.attn_compute_dtype, grad_in=a, grad_weight=a, grad_act=a,
    )


def _routed_expert_bundle(p: QuantProfileLike) -> OpDtypeBundle:
    a = p.effective_moe_act_dtype()
    return OpDtypeBundle(
        in_act=a, weight=p.routed_expert_weight_dtype, out_act=a,
        compute=p.routed_expert_compute_dtype, grad_in=a, grad_weight=a,
        grad_act=a,
    )


def _shared_expert_bundle(p: QuantProfileLike) -> OpDtypeBundle:
    a = p.effective_moe_act_dtype()
    return OpDtypeBundle(
        in_act=a, weight=p.shared_expert_weight_dtype, out_act=a,
        compute=p.shared_expert_compute_dtype, grad_in=a, grad_weight=a,
        grad_act=a,
    )


def _embedding_norm_bundle() -> OpDtypeBundle:
    return OpDtypeBundle(
        in_act=Dtype.BF16, weight=Dtype.BF16, out_act=Dtype.BF16,
        compute=Dtype.BF16, grad_in=Dtype.BF16, grad_weight=Dtype.BF16,
        grad_act=Dtype.BF16,
    )


def _default_bundle(p: QuantProfileLike) -> OpDtypeBundle:
    return OpDtypeBundle(
        in_act=p.act_dtype, weight=p.param_dtype, out_act=p.act_dtype,
        compute=p.act_dtype, grad_in=p.act_dtype, grad_weight=p.act_dtype,
        grad_act=p.act_dtype,
    )


def dispatch(component: str, profile: QuantProfileLike) -> OpDtypeBundle:
    """Return the dtype bundle for a given component category and profile."""
    if component == "attention":
        return _attention_bundle(profile)
    if component == "routed_expert":
        return _routed_expert_bundle(profile)
    if component == "shared_expert":
        return _shared_expert_bundle(profile)
    if component in ("embedding", "norm"):
        return _embedding_norm_bundle()
    return _default_bundle(profile)
