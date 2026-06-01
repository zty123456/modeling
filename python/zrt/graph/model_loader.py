"""Load any HuggingFace causal LM for op-sequence tracing via FakeTensorMode.

FakeTensorMode creates lightweight "fake tensors" that track shapes, dtypes,
and strides without allocating real memory.  Compared to the old meta-device
approach this gives:

  - Correct stride propagation (contiguous / channels-last / etc.)
  - Proper device simulation (cpu / cuda) without the hardware
  - Better support for factory functions (zeros, ones, arange …)
  - Fewer compatibility patches needed

Loading strategy (tiered)
--------------------------
1. **Direct load** from *model_id* (HF Hub ID or local path) via
   ``AutoConfig`` + ``AutoModelForCausalLM.from_config``.

2. **Local-registry fallback** — when step 1 fails because:
   - The transformers version does not recognise the architecture
     (e.g. ``deepseek_v32`` not in the built-in registry), OR
   - The model file imports a symbol removed in the current transformers
     version (e.g. ``is_flash_attn_greater_or_equal_2_10`` in 5.x),
   …the loader looks up the model ID / model-type in the local registry
   (``compat._LOCAL_REGISTRY``) and retries from the matching
   ``hf_models/<model>/`` directory, which carries ``auto_map`` in its
   ``config.json`` and therefore works regardless of the built-in registry.

This mirrors the pattern used by vLLM and SGLang: they maintain an internal
architecture registry and fall back to bundled model implementations when
transformers support lags.

Supports:
  - HF Hub IDs  (``deepseek-ai/DeepSeek-V3``, ``Qwen/Qwen3-8B``, …)
  - Local directories (``./hf_models/deepseek_v3``, …)
    * Standard model types (llama, qwen2, …): config.json only required
    * Custom architectures: need auto_map in config.json + modeling files
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from python.zrt.graph.patches import (
    apply_compat_patches,
    is_moe_module as _is_moe_module,
    patch_for_training_capture,
    patch_hc_for_capture,
    patch_indexer_for_fake,
    patch_moe_for_fake,
    patch_moe_for_meta,
    patch_v4_inference_stubs,
)
from python.zrt.graph.layer_strategy import infer_layer_profile

logger = logging.getLogger(__name__)


# ── Config normalization ─────────────────────────────────────────────────────

def _normalize_config(config: Any) -> None:
    """Apply generic compatibility fixes to a PretrainedConfig in-place."""
    rs = getattr(config, "rope_scaling", None)
    if isinstance(rs, dict) and "rope_type" in rs and "type" not in rs:
        rs["type"] = rs["rope_type"]
    config._attn_implementation = "eager"


# ── Layer-type inference ─────────────────────────────────────────────────────

def infer_layer_types(config: Any) -> Dict[str, List[int]]:
    """Infer which transformer layers are dense vs. sparse (MoE) from config.

    Handles three architectures without hard-coding model names:

    * **DeepSeek-V3 / V3.2 style** — config has ``first_k_dense_replace`` and
      (optionally) ``moe_layer_freq``:
      layers ``[0, first_k_dense_replace)`` are dense; the remaining layers are
      MoE when ``layer_idx % moe_layer_freq == 0``, otherwise dense.
    * **Mixtral style** — config has ``num_local_experts`` (but no
      ``first_k_dense_replace``): all layers are treated as MoE.
    * **Standard dense models** — no MoE fields: all layers are dense.

    The layer count is taken from ``config._full_num_hidden_layers`` when
    available (set by :func:`load_model`), falling back to
    ``config.num_hidden_layers``.

    Returns
    -------
    {"dense": [layer_idx, ...], "sparse": [layer_idx, ...]}
        Each list is sorted; together they cover every layer index.
    """
    total: int = (
        getattr(config, "_full_num_hidden_layers", None)
        or getattr(config, "num_hidden_layers", 0)
    )

    first_k_dense: Optional[int] = getattr(config, "first_k_dense_replace", None)
    moe_layer_freq: int = getattr(config, "moe_layer_freq", 1) or 1
    has_local_experts: bool = getattr(config, "num_local_experts", None) is not None
    has_routed_experts: bool = getattr(config, "n_routed_experts", None) is not None

    dense: List[int] = []
    sparse: List[int] = []

    if first_k_dense is not None:
        for i in range(total):
            if i < first_k_dense:
                dense.append(i)
            elif i % moe_layer_freq == 0:
                sparse.append(i)
            else:
                dense.append(i)
    elif has_local_experts or has_routed_experts:
        sparse = list(range(total))
    else:
        dense = list(range(total))

    return {"dense": dense, "sparse": sparse}


def auto_target_layers(config: Any) -> List[int]:
    """Return the representative layer indices to trace for this config.

    Selects the **first dense layer** and the **first sparse (MoE) layer**
    (when the model has MoE layers).  For purely dense models returns ``[0]``.
    For all-MoE models returns ``[0]``.

    The returned list is sorted and deduplicated.
    """
    types = infer_layer_types(config)
    result: List[int] = []
    if types["dense"]:
        result.append(types["dense"][0])
    if types["sparse"]:
        result.append(types["sparse"][0])
    return sorted(set(result)) or [0]


# ── Error classification ──────────────────────────────────────────────────────

def _is_arch_error(msg: str) -> bool:
    """True if the error indicates an unrecognised / unsupported architecture."""
    msg_lower = msg.lower()
    return (
        "does not recognize this architecture" in msg
        or ("model type" in msg_lower and "does not" in msg_lower)
    )


def _is_import_compat_error(exc: Exception) -> bool:
    """True if the error is a missing-symbol import failure in a model file.

    These arise when a model file written for transformers 4.x tries to import
    a symbol that was removed in 5.x.  The version shims in ``compat.py``
    cover known cases, but a local-directory retry is cheaper than enumerating
    every possible removal.
    """
    if not isinstance(exc, (ImportError, AttributeError)):
        return False
    msg = str(exc)
    return "cannot import name" in msg or "has no attribute" in msg


# ── Tiered config loading ─────────────────────────────────────────────────────

def _load_config(model_id: str) -> Tuple[Any, str]:
    """Load ``PretrainedConfig``, falling back to the local registry on failure.

    Returns
    -------
    (config, effective_id)
        *effective_id* is the path/ID that was actually used (may differ from
        *model_id* when a local fallback was applied).
    """
    from transformers import AutoConfig
    from python.zrt.graph.compat import find_local_fallback

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return config, model_id
    except (ValueError, OSError) as exc:
        if not _is_arch_error(str(exc)):
            raise

    # Architecture not found in transformers registry — try local fallback
    local_dir = find_local_fallback(model_id)
    if local_dir is None:
        # Re-raise the original error with an explanatory note
        raise ValueError(
            f"transformers does not recognise the architecture for '{model_id}' "
            f"and no local fallback is registered in compat._LOCAL_REGISTRY.  "
            f"To add support, place the modeling files in hf_models/<name>/ and "
            f"register the model_type / hub ID in compat._LOCAL_REGISTRY."
        ) from None

    logger.info(
        "Hub loading of '%s' failed (architecture not recognised); "
        "using local fallback: %s",
        model_id, local_dir,
    )
    config = AutoConfig.from_pretrained(str(local_dir), trust_remote_code=True)
    config._hub_model_id = model_id  # preserve original ID for display / Excel
    return config, str(local_dir)


# ── Tiered model instantiation ────────────────────────────────────────────────

def _instantiate_model(config: Any, effective_id: str) -> nn.Module:
    """Instantiate the model class from *config*.

    If the first attempt raises an import-compatibility error (e.g. a removed
    transformers symbol that the version shim did not cover), and the config's
    ``model_type`` has a local fallback registered, reload the config from the
    local directory and retry once.
    """
    from transformers import AutoModelForCausalLM
    from python.zrt.graph.compat import find_local_fallback

    saved_exc: Exception | None = None
    try:
        return AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    except Exception as exc:
        saved_exc = exc
        if not _is_import_compat_error(exc):
            raise

    # Import-compat error — try local fallback keyed by model_type
    model_type = getattr(config, "model_type", None)
    local_dir = (
        find_local_fallback(model_type) if model_type else None
    ) or find_local_fallback(effective_id)

    if local_dir is None or str(local_dir) == effective_id:
        raise  # no new option to try

    logger.info(
        "Model instantiation failed for '%s' (%s: %s); "
        "retrying from local fallback: %s",
        effective_id, type(saved_exc).__name__, saved_exc, local_dir,
    )
    from transformers import AutoConfig
    local_config = AutoConfig.from_pretrained(str(local_dir), trust_remote_code=True)
    # Copy runtime-adjusted fields
    local_config.num_hidden_layers = config.num_hidden_layers
    local_config._full_num_hidden_layers = getattr(config, "_full_num_hidden_layers", None)
    local_config._attn_implementation = getattr(config, "_attn_implementation", "eager")
    _normalize_config(local_config)
    return AutoModelForCausalLM.from_config(local_config, trust_remote_code=True)


# ── Public API ───────────────────────────────────────────────────────────────

def load_model(
    model_id: str,
    num_hidden_layers: int = 4,
    training: bool = False,
    infer_profile: bool = False,
    pre_truncated_config: Any = None,
) -> Tuple[nn.Module, Any, FakeTensorMode]:
    """Load any HF causal LM via FakeTensorMode for op-sequence tracing.

    Parameters
    ----------
    model_id:
        HF Hub ID or local directory. Ignored if pre_truncated_config is provided.
    num_hidden_layers:
        Number of transformer blocks to instantiate. Ignored if pre_truncated_config 
        already has num_hidden_layers set.
    training:
        When True, apply training patches and keep model in train() mode.
    infer_profile:
        When True, infer layer structure and save to config._full_layer_profile.
    pre_truncated_config:
        If provided, use this already-truncated config instead of reloading.
        This is used by pipeline.py to avoid double-loading and preserve truncation.

    Returns
    -------
    (model, config, fake_mode)
    """
    # Step 1: inject version shims + legacy compat attrs
    apply_compat_patches()

    # Step 2: use pre-truncated config if provided, else load fresh
    if pre_truncated_config is not None:
        config = pre_truncated_config
        effective_id = model_id
    else:
        logger.info("Loading config from %s …", model_id)
        config, effective_id = _load_config(model_id)
        # Save full values before any truncation
        config._full_num_hidden_layers = getattr(config, "num_hidden_layers", None)
        config._full_num_nextn_predict_layers = getattr(config, "num_nextn_predict_layers", 0) or 0
        full_compress_ratios = list(getattr(config, "compress_ratios", []))
        if full_compress_ratios:
            config._full_compress_ratios = full_compress_ratios
        config.num_hidden_layers = num_hidden_layers
    
    # Step 2b: infer layer profile if requested (skip if already set by caller)
    # CRITICAL: Infer profile BEFORE truncation, using saved full values
    if infer_profile and not hasattr(config, "_full_layer_profile"):
        # Restore full layer info temporarily for profile inference
        full_compress = getattr(config, "_full_compress_ratios", None)
        full_layers = getattr(config, "_full_num_hidden_layers", None)
        
        # Temporarily restore full values if available
        if full_compress and full_layers:
            saved_compress = getattr(config, "compress_ratios", None)
            saved_layers = config.num_hidden_layers
            config.compress_ratios = full_compress
            config.num_hidden_layers = full_layers
        
        profile = infer_layer_profile(config)
        config._full_layer_profile = profile
        
        # Restore truncated values if we modified them
        if full_compress and full_layers:
            config.compress_ratios = saved_compress if saved_compress else full_compress[:saved_layers]
            config.num_hidden_layers = saved_layers
        
        logger.info(
            "LayerProfile: typical_indices=%s, types: dense=%d, moe=%d, "
            "hca_hash=%d, hca_topk=%d, csa_hash=%d, csa_topk=%d, "
            "swa_hash=%d, swa_topk=%d",
            profile.typical_indices,
            profile.num_dense, profile.num_moe,
            profile.num_hca_hash, profile.num_hca_topk,
            profile.num_csa_hash, profile.num_csa_topk,
            profile.num_swa_hash, profile.num_swa_topk,
        )
    
    _normalize_config(config)

    # Step 3: enter FakeTensorMode — all tensors created inside become FakeTensors
    fake_mode = FakeTensorMode(allow_non_fake_inputs=False)
    fake_mode.__enter__()

    # Step 4: instantiate model (with local-registry fallback on import errors)
    logger.info(
        "Instantiating %s with FakeTensorMode (%d layers, training=%s) …",
        type(config).__name__, num_hidden_layers, training,
    )
    try:
        model = _instantiate_model(config, effective_id)
    except Exception:
        fake_mode.__exit__(None, None, None)
        raise

    if training:
        # train() keeps requires_grad=True on parameters; patch_for_training_capture
        # also calls patch_moe_for_fake / patch_indexer_for_fake / patch_hc_for_capture.
        model.train()
        patch_for_training_capture(model)
    else:
        model.eval()
        patch_moe_for_fake(model)
        patch_indexer_for_fake(model)
        patch_hc_for_capture(model)
        patch_v4_inference_stubs()

    return model, config, fake_mode
