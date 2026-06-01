"""Unit tests for layer_strategy module."""
from types import SimpleNamespace

import pytest

from python.zrt.graph.layer_strategy import (
    LayerProfile,
    LayerType,
    infer_layer_profile,
    scale_layer_costs,
    _select_first_of_each_type,
)


class TestInferV4Profile:
    """Tests for V4 style profile inference (compress_ratios array)."""

    def test_infer_v4_profile_pro(self):
        """DSV4-pro: 3 HCA_HASH + alternating HCA_TOPK/CSA_TOPK (61 transformer layers)."""
        # Full DSV4-pro compress_ratios from config.json (62 elements)
        # num_hash_layers=3 means layers 0,1,2 use hash routing
        # We ignore the last element (MTP), only consider transformer layers
        full_compress_ratios = [128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]
        config = SimpleNamespace(
            num_hidden_layers=61,
            num_hash_layers=3,
            compress_ratios=full_compress_ratios,  # 62 elements
        )
        profile = infer_layer_profile(config)
        
        # After truncating to 61 layers (ignoring MTP):
        # layer 0 (ratio=128) -> HCA_HASH (layer_id < num_hash_layers)
        # layer 1 (ratio=128) -> HCA_HASH
        # layer 2 (ratio=4) -> CSA_HASH
        # layers 3+ -> HCA_TOPK / CSA_TOPK
        assert profile.num_hca_hash == 2  # indices 0,1
        assert profile.num_csa_hash == 1  # index 2
        assert profile.total_layers == 61

    def test_infer_v4_profile_flash(self):
        """DSV4-flash: 2 SWA_HASH + 1 CSA_HASH + alternating CSA_TOPK/HCA_TOPK + 1 SWA_TOPK (43 transformer layers)."""
        # DSV4-flash compress_ratios (43 elements)
        # num_hash_layers=3 means layers 0,1,2 use hash routing
        # Pattern: [0,0,4] + 18*(128,4) + [128,0] = 3+36+3 = 42? Need 43 elements
        # Correct: [0,0,4] + 19*(128,4) + [0] = 3+38+1 = 42? Still wrong
        # Let's make 43 elements: [0,0,4] + 19*(128,4) + [128,0] = 3+38+2 = 43
        full_compress_ratios = [0, 0, 4] + [128, 4] * 19 + [128, 0]  # 43 elements
        config = SimpleNamespace(
            num_hidden_layers=43,
            num_hash_layers=3,
            compress_ratios=full_compress_ratios,
        )
        profile = infer_layer_profile(config)
        
        # 43 layers:
        # layer 0 (ratio=0) -> SWA_HASH (hash routing, layer_id < num_hash_layers)
        # layer 1 (ratio=0) -> SWA_HASH (hash routing)
        # layer 2 (ratio=4) -> CSA_HASH (hash routing)
        # layers 3-41: alternating HCA_TOPK/CSA_TOPK (19 HCA + 19 CSA = 38)
        # layer 42 (ratio=0) -> SWA_TOPK (topk routing)
        assert profile.num_swa_hash == 2  # layers 0,1
        assert profile.num_swa_topk == 1  # layer 42
        assert profile.num_csa_hash == 1  # layer 2
        assert profile.num_hca_hash == 0
        assert profile.num_hca_topk == 20  # indices 3,5,...,39,41 (odd indices 3-41)
        assert profile.num_csa_topk == 19  # indices 4,6,...,40 (even indices 4-40)
        assert profile.total_layers == 43
        # Typical: first of each type
        assert profile.typical_indices == [0, 2, 3, 4, 42]

    def test_infer_v4_profile_no_mtp(self):
        """V4 variant without MTP field, num_hash_layers=2."""
        config = SimpleNamespace(
            num_hidden_layers=6,
            num_hash_layers=2,
            compress_ratios=[128, 128, 4, 128, 4, 128],
        )
        profile = infer_layer_profile(config)
        
        # layer 0,1 -> HCA_HASH (layer_id < num_hash_layers)
        # layer 2 -> CSA_TOPK (layer_id >= num_hash_layers, ratio=4)
        # layer 3 -> HCA_TOPK
        # layer 4 -> CSA_TOPK
        # layer 5 -> HCA_TOPK
        assert profile.num_hca_hash == 2  # indices 0,1
        assert profile.num_csa_hash == 0
        assert profile.num_hca_topk == 2  # indices 3,5
        assert profile.num_csa_topk == 2  # indices 2,4
        assert profile.total_layers == 6

    def test_infer_v4_profile_unknown_ratio(self):
        """Unknown ratio value should be treated as SWA with hash/topk distinction."""
        config = SimpleNamespace(
            num_hidden_layers=3,
            num_hash_layers=1,
            num_nextn_predict_layers=0,
            compress_ratios=[128, 99, 4],
        )
        profile = infer_layer_profile(config)
        
        # layer 0 -> HCA_HASH (layer_id < num_hash_layers)
        # layer 1 -> SWA_TOPK (unknown ratio treated as SWA, layer_id >= num_hash_layers)
        # layer 2 -> CSA_TOPK
        assert profile.layer_types[1] == LayerType.SWA_TOPK
        assert profile.num_hca_hash == 1
        assert profile.num_swa_topk == 1
        assert profile.num_csa_topk == 1

    def test_infer_v4_profile_length_mismatch_warning(self):
        """Mismatch in compress_ratios length should log warning but proceed."""
        config = SimpleNamespace(
            num_hidden_layers=5,
            num_nextn_predict_layers=1,
            compress_ratios=[128, 128, 4, 128],  # length 4, expected 6
        )
        profile = infer_layer_profile(config)
        
        # Should still process available ratios
        assert profile.total_layers == 4


class TestInferDSV3Profile:
    """Tests for DSV3/V3.2 style profile inference."""

    def test_infer_dsv3_profile(self):
        """DSV3: 3 dense + 58 moe."""
        config = SimpleNamespace(
            num_hidden_layers=61,
            first_k_dense_replace=3,
            moe_layer_freq=1,
        )
        profile = infer_layer_profile(config)
        
        assert profile.typical_indices == [0, 3]  # dense[0], moe[3]
        assert profile.num_dense == 3
        assert profile.num_moe == 58
        assert profile.total_layers == 61

    def test_infer_dsv3_profile_with_freq(self):
        """DSV3 variant with moe_layer_freq > 1."""
        config = SimpleNamespace(
            num_hidden_layers=10,
            first_k_dense_replace=2,
            moe_layer_freq=2,
        )
        profile = infer_layer_profile(config)
        
        # layers 0,1 → dense
        # layers 2,4,6,8 → moe (i >= first_k and i % freq == 0)
        # layers 3,5,7,9 → dense
        assert profile.num_dense == 6
        assert profile.num_moe == 4
        assert profile.typical_indices == [0, 2]

    def test_infer_dsv3_profile_no_first_k(self):
        """first_k_dense_replace=0 means all layers follow moe pattern."""
        config = SimpleNamespace(
            num_hidden_layers=10,
            first_k_dense_replace=0,
            moe_layer_freq=1,
        )
        profile = infer_layer_profile(config)
        
        assert profile.num_moe == 10
        assert profile.num_dense == 0


class TestInferPureMoEProfile:
    """Tests for pure MoE models (Mixtral)."""

    def test_infer_pure_moe_local_experts(self):
        """Pure MoE with num_local_experts."""
        config = SimpleNamespace(
            num_hidden_layers=32,
            num_local_experts=8,
        )
        profile = infer_layer_profile(config)
        
        assert profile.typical_indices == [0]
        assert profile.num_moe == 32
        assert all(lt == LayerType.MOE for lt in profile.layer_types)

    def test_infer_pure_moe_routed_experts(self):
        """Pure MoE with n_routed_experts."""
        config = SimpleNamespace(
            num_hidden_layers=32,
            n_routed_experts=256,
        )
        profile = infer_layer_profile(config)
        
        assert profile.num_moe == 32


class TestInferDenseProfile:
    """Tests for standard dense models (Llama, Qwen, Mistral)."""

    def test_infer_dense_profile(self):
        """Standard dense model."""
        config = SimpleNamespace(
            num_hidden_layers=32,
        )
        profile = infer_layer_profile(config)
        
        assert profile.typical_indices == [0]
        assert profile.num_dense == 32
        assert all(lt == LayerType.DENSE for lt in profile.layer_types)

    def test_infer_dense_profile_no_fields(self):
        """Config without any MoE/V4 fields defaults to dense."""
        config = SimpleNamespace(num_hidden_layers=16)
        profile = infer_layer_profile(config)
        
        assert profile.num_dense == 16


class TestSelectFirstOfEachType:
    """Tests for _select_first_of_each_type helper."""

    def test_all_same_type(self):
        """All same type returns only first."""
        result = _select_first_of_each_type([LayerType.DENSE] * 10)
        assert result == [0]

    def test_two_types(self):
        """Two types returns first of each."""
        result = _select_first_of_each_type([
            LayerType.DENSE, LayerType.DENSE, LayerType.MOE, LayerType.MOE
        ])
        assert result == [0, 2]

    def test_three_types(self):
        """Three types returns first of each."""
        result = _select_first_of_each_type([
            LayerType.SWA_HASH, LayerType.SWA_TOPK, LayerType.CSA_HASH, LayerType.HCA_TOPK
        ])
        assert result == [0, 1, 2, 3]

    def test_interleaved(self):
        """Interleaved types still select first occurrence."""
        result = _select_first_of_each_type([
            LayerType.HCA, LayerType.CSA, LayerType.HCA, LayerType.CSA
        ])
        assert result == [0, 1]


class TestScaleLayerCosts:
    """Tests for scale_layer_costs scaling logic."""

    def test_scale_single_type(self):
        """Scale single layer type."""
        profile = LayerProfile(
            layer_types=[LayerType.DENSE] * 32,
            typical_indices=[0],
            num_dense=32,
        )
        layer_costs = {0: 1.5}
        scaled = scale_layer_costs(layer_costs, profile)
        
        assert scaled["dense"] == 48.0
        assert scaled["total"] == 48.0

    def test_scale_two_types(self):
        """Scale two layer types."""
        profile = LayerProfile(
            layer_types=[LayerType.DENSE, LayerType.DENSE, LayerType.MOE],
            typical_indices=[0, 2],
            num_dense=32,
            num_moe=58,
        )
        layer_costs = {0: 1.0, 2: 2.0}
        scaled = scale_layer_costs(layer_costs, profile)
        
        assert scaled["dense"] == 32.0
        assert scaled["moe"] == 116.0
        assert scaled["total"] == 148.0

    def test_scale_missing_cost(self):
        """Missing cost for a type defaults to 0."""
        profile = LayerProfile(
            layer_types=[LayerType.DENSE, LayerType.MOE],
            typical_indices=[0, 1],
            num_dense=10,
            num_moe=20,
        )
        layer_costs = {0: 1.0}  # missing MOE cost
        scaled = scale_layer_costs(layer_costs, profile)
        
        assert scaled["dense"] == 10.0
        assert scaled["moe"] == 0.0
        assert scaled["total"] == 10.0

    def test_scale_v4_profile(self):
        """Scale V4 profile with multiple types."""
        profile = LayerProfile(
            layer_types=[LayerType.HCA_HASH, LayerType.HCA_TOPK, LayerType.CSA_TOPK],
            typical_indices=[0, 1, 2],
            num_hca_hash=10,
            num_hca_topk=21,
            num_csa_topk=30,
            num_hca=31,
            num_csa=30,
        )
        layer_costs = {0: 0.5, 1: 0.5, 2: 0.8}  # HCA_HASH=0.5ms, HCA_TOPK=0.5ms, CSA_TOPK=0.8ms
        scaled = scale_layer_costs(layer_costs, profile)
        
        assert scaled["hca_hash"] == 5.0  # 0.5 * 10
        assert scaled["hca_topk"] == 10.5  # 0.5 * 21
        assert scaled["csa_topk"] == 24.0  # 0.8 * 30
        assert scaled["total"] == 39.5





class TestLayerProfileProperties:
    """Tests for LayerProfile properties."""

    def test_total_layers(self):
        """total_layers property."""
        profile = LayerProfile(
            layer_types=[LayerType.DENSE] * 10,
            typical_indices=[0],
            num_dense=10,
        )
        assert profile.total_layers == 10

    def test_empty_profile(self):
        """Profile with no layers."""
        profile = LayerProfile(
            layer_types=[],
            typical_indices=[],
        )
        assert profile.total_layers == 0
        assert profile.num_dense == 0