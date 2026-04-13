"""Load any HuggingFace causal LM onto meta device for op-sequence tracing.

Supports:
  - HF Hub model IDs  (``deepseek-ai/DeepSeek-V3``, ``Qwen/Qwen3-8B``, …)
  - Local directories (``./hf_models/deepseek_v3``, …)
    * Standard model types (llama, qwen2, …): config.json only required
    * Custom architectures: need auto_map in config.json + modeling files

Compatibility patches applied here:
  - Deprecated transformers internals (is_torch_fx_available, etc.)
  - torch.autocast with device_type='meta' (transformers 4.50+ RoPE issue)
  - Generic MoE forward replacement (avoids .cpu().numpy() on meta tensors)
  - DeepSeek-V3.2 Indexer forward replacement
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Compatibility patches ──────────────────────────────────────────────────────

_KNOWN_AUTOCAST_DEVICES = frozenset({"cpu", "cuda", "xpu", "hpu", "mps", "xla"})


def apply_compat_patches() -> None:
    """Add deprecated transformers attrs expected by older model code.

    Also patches ``torch.autocast`` to accept ``device_type='meta'``.
    transformers 4.50+ passes the tensor's device type to autocast inside
    RoPE; meta tensors surface as ``'meta'``, which torch 2.x rejects.
    We remap unknown device types to ``'cpu'`` (no-op for meta tensors).
    """
    try:
        import transformers.utils.import_utils as _iu
        if not hasattr(_iu, "is_torch_fx_available"):
            _iu.is_torch_fx_available = lambda: True
    except ImportError:
        pass
    try:
        import transformers.utils as _tu
        if not hasattr(_tu, "is_torch_fx_available"):
            _tu.is_torch_fx_available = lambda: True
    except ImportError:
        pass
    try:
        import transformers.pytorch_utils as _pu
        if not hasattr(_pu, "is_torch_greater_or_equal_than_1_13"):
            _pu.is_torch_greater_or_equal_than_1_13 = True
    except ImportError:
        pass

    if not getattr(torch.amp.autocast, "_meta_device_safe", False):
        _orig_init = torch.amp.autocast.__init__

        def _safe_init(self, device_type: str, *args: Any, **kwargs: Any) -> None:
            if device_type not in _KNOWN_AUTOCAST_DEVICES:
                device_type = "cpu"
            _orig_init(self, device_type, *args, **kwargs)

        torch.amp.autocast.__init__ = _safe_init  # type: ignore[method-assign]
        torch.amp.autocast._meta_device_safe = True  # type: ignore[attr-defined]


# ── Config normalization ───────────────────────────────────────────────────────

def _normalize_config(config: Any) -> None:
    """Apply generic compatibility fixes to a PretrainedConfig in-place."""
    # rope_scaling: older modeling code reads 'type'; newer configs write 'rope_type'
    rs = getattr(config, "rope_scaling", None)
    if isinstance(rs, dict) and "rope_type" in rs and "type" not in rs:
        rs["type"] = rs["rope_type"]
    config._attn_implementation = "eager"


# ── Generic MoE patch ──────────────────────────────────────────────────────────

def _is_moe_module(module: nn.Module) -> bool:
    experts = getattr(module, "experts", None)
    return (
        isinstance(experts, nn.ModuleList)
        and any(e is not None for e in experts)
        and not getattr(module, "_meta_patched", False)
    )


def _returns_router_tuple(mod: nn.Module) -> bool:
    """True if this MoE forward returns (hidden, router_logits) tuple."""
    try:
        src = inspect.getsource(type(mod).forward)
        if any(pat in src for pat in ("router_logits", "aux_loss",
                                      "return hidden_states,")):
            return True
    except Exception:
        pass
    return hasattr(mod, "router") and not hasattr(mod, "gate")


def patch_moe_for_meta(model: nn.Module) -> None:
    """Replace MoE forwards with a meta-tensor-safe simplified version."""
    patched = 0
    for _, module in model.named_modules():
        if not _is_moe_module(module):
            continue
        module._meta_patched = True
        module.forward = _make_meta_moe_forward(module)
        patched += 1
    if patched:
        logger.info("Applied meta-tensor MoE patch to %d module(s).", patched)


def _make_meta_moe_forward(mod: nn.Module):
    _tuple_return = _returns_router_tuple(mod)

    def _forward(hidden_states: torch.Tensor, *args: Any, **kwargs: Any):
        try:
            result = _impl(hidden_states)
            return (result, None) if _tuple_return else result
        except Exception as exc:
            logger.debug("Meta MoE forward error (%s) — returning identity.", exc)
            return (hidden_states, None) if _tuple_return else hidden_states

    def _impl(hidden_states: torch.Tensor) -> torch.Tensor:
        orig = hidden_states
        bs, seq, h = orig.shape
        flat = orig.reshape(bs * seq, h)

        gate_weight: Optional[torch.Tensor] = None
        gate = getattr(mod, "gate", None)
        if gate is not None and callable(gate):
            try:
                gate_out = gate(orig)
                if isinstance(gate_out, (tuple, list)):
                    gate_weight = gate_out[1]
                else:
                    gate_weight = torch.softmax(gate_out.float(), dim=-1)[:, :1]
            except Exception:
                try:
                    gate_out = gate(flat)
                    if isinstance(gate_out, (tuple, list)):
                        gate_weight = gate_out[1]
                    else:
                        gate_weight = torch.softmax(gate_out.float(), dim=-1)[:, :1]
                except Exception as exc:
                    logger.debug("Gate forward failed (%s).", exc)

        first_expert = next((e for e in mod.experts if e is not None), None)
        if first_expert is None:
            return orig

        try:
            expert_out = first_expert(flat)
        except Exception:
            expert_out = first_expert(orig).reshape(bs * seq, -1)

        y = (expert_out * gate_weight[:, :1]
             if gate_weight is not None else expert_out)
        try:
            y = y.reshape(bs, seq, -1)
        except Exception:
            y = orig

        for attr in ("shared_experts", "shared_expert"):
            shared = getattr(mod, attr, None)
            if shared is not None and callable(shared):
                try:
                    y = y + shared(orig)
                except Exception as exc:
                    logger.debug("Shared expert failed (%s).", exc)
                break

        return y

    return _forward


# ── DeepSeek-V3.2 Indexer patch ───────────────────────────────────────────────

def _patch_indexer_forward(IndexerClass: type) -> None:
    """Patch DeepSeek-V3.2 Indexer to work on meta device."""
    def _indexer_forward_meta(self, x, qr, position_ids=None, attention_mask=None):
        bsz, seqlen, _ = x.size()
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.index_n_heads, self.index_head_dim)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.index_head_dim - self.rope_head_dim], dim=-1)
        q_nope = q.transpose(1, 2)
        k_nope = k_nope.unsqueeze(1).expand(-1, self.index_n_heads, -1, -1)
        scores = torch.matmul(q_nope, k_nope.transpose(2, 3))
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = torch.nn.functional.softmax(
            scores, dim=-1, dtype=torch.float32).to(x.dtype)
        topk_indices = scores.topk(min(self.index_topk, seqlen), dim=-1)[1]
        return topk_indices

    IndexerClass.forward = _indexer_forward_meta


# ── Public API ────────────────────────────────────────────────────────────────

def load_model(
    model_id: str,
    num_hidden_layers: int = 4,
) -> Tuple[nn.Module, Any]:
    """Load any HF causal LM onto meta device for op-sequence tracing.

    Parameters
    ----------
    model_id:
        HF Hub ID (``"deepseek-ai/DeepSeek-V3"``) **or** a local directory
        (``"./hf_models/deepseek_v3"``).
    num_hidden_layers:
        Number of transformer blocks to instantiate (2–4 is enough to see all
        distinct op patterns including dense + MoE layers).

    Returns
    -------
    (model, config)
        model  — on meta device, eval mode, MoE-patched.
        config — ``config._full_num_hidden_layers`` stores the original depth.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    apply_compat_patches()

    logger.info("Loading config from %s …", model_id)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    config._full_num_hidden_layers = getattr(config, "num_hidden_layers", None)
    config.num_hidden_layers = num_hidden_layers
    _normalize_config(config)

    logger.info("Instantiating %s on meta device (%d layers) …",
                type(config).__name__, num_hidden_layers)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()

    patch_moe_for_meta(model)

    # DeepSeek-V3.2 Indexer patch (if present)
    for _, module in model.named_modules():
        cls_name = type(module).__name__
        if "Indexer" in cls_name and hasattr(module, "wq_b"):
            _patch_indexer_forward(type(module))
            logger.debug("Patched Indexer: %s", cls_name)
            break

    return model, config
