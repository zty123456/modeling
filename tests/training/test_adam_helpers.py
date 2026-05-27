"""Unit tests for Adam cost primitives (US-006)."""
import pytest
from zrt.training.models.optimizer import (
    ADAM_UPDATE_BYTES_PER_PARAM,
    adam_state_bytes,
    adam_step_traffic_bytes,
    adam_step_time_s,
)
from zrt.training.models.memory import adam_params_on_rank


class TestAdamCostPrimitives:
    def test_adam_update_bytes_per_param_is_28(self):
        assert ADAM_UPDATE_BYTES_PER_PARAM == 28

    def test_adam_state_bytes(self):
        assert adam_state_bytes(1_000_000) == 1_000_000 * 12
        assert adam_state_bytes(0) == 0

    def test_adam_state_bytes_custom_master(self):
        assert adam_state_bytes(100, master_dtype_bytes=2) == 100 * 6

    def test_adam_step_traffic_bytes(self):
        assert adam_step_traffic_bytes(1_000_000) == 1_000_000 * 28
        assert adam_step_traffic_bytes(0) == 0

    def test_adam_step_time_s_basic(self):
        # 1M params, 2 TB/s peak, no efficiency adjustment
        t = adam_step_time_s(1_000_000, 2e12)
        expected = (1_000_000 * 28) / 2e12
        assert t == pytest.approx(expected, rel=1e-6)

    def test_adam_step_time_s_zero_bw(self):
        assert adam_step_time_s(100, 0.0) == 0.0

    def test_adam_step_time_s_zero_params(self):
        assert adam_step_time_s(0, 2e12) == 0.0

    def test_adam_step_time_s_with_efficiency_override(self):
        t = adam_step_time_s(1_000_000, 2e12, efficiency_override=0.85)
        expected = (1_000_000 * 28) / (2e12 * 0.85)
        assert t == pytest.approx(expected, rel=1e-6)

    def test_adam_step_time_s_with_gpu_name(self):
        # gpu_name triggers achieved_bandwidth_efficiency lookup
        t = adam_step_time_s(1_000_000_000, 2e12, gpu_name="H100")
        assert t > 0  # non-zero for large transfer
        # Large transfer (>100MB) gets 0.85 efficiency
        expected = (1_000_000_000 * 28) / (2e12 * 0.85)
        assert t == pytest.approx(expected, rel=1e-6)


class TestAdamParamsOnRank:
    def test_dense_no_sharding(self):
        P = adam_params_on_rank(
            total_params=1000, n_layers=10, embed_params=100,
            expert_params_full=0, tp=1, pp=1, ep=1, dp=1, zero_stage=0,
        )
        assert P == 1000

    def test_tp_sharding(self):
        P = adam_params_on_rank(
            total_params=1000, n_layers=10, embed_params=100,
            expert_params_full=0, tp=4, pp=1, ep=1, dp=1, zero_stage=0,
        )
        assert P == 250

    def test_pp_sharding(self):
        P = adam_params_on_rank(
            total_params=10000, n_layers=10, embed_params=1000,
            expert_params_full=0, tp=1, pp=2, ep=1, dp=1, zero_stage=0,
        )
        # non_embed = 10000 - 1000 = 9000; non_embed // 2 = 4500; embed // 2 = 500
        assert P == 5000

    def test_dp_zero1_step_view(self):
        """Step-time view: DP applied when zero_stage >= 1."""
        P = adam_params_on_rank(
            total_params=8000, n_layers=10, embed_params=0,
            expert_params_full=0, tp=1, pp=1, ep=1, dp=8, zero_stage=1,
            apply_dp_for_zero=1,
        )
        assert P == 1000

    def test_dp_zero1_storage_view(self):
        """Storage view: DP NOT applied when zero_stage=1 < apply_dp_for_zero=3."""
        P = adam_params_on_rank(
            total_params=8000, n_layers=10, embed_params=0,
            expert_params_full=0, tp=1, pp=1, ep=1, dp=8, zero_stage=1,
            apply_dp_for_zero=3,
        )
        assert P == 8000

    def test_dp_zero3_storage_view(self):
        """Storage view: DP applied when zero_stage=3 >= apply_dp_for_zero=3."""
        P = adam_params_on_rank(
            total_params=8000, n_layers=10, embed_params=0,
            expert_params_full=0, tp=1, pp=1, ep=1, dp=8, zero_stage=3,
            apply_dp_for_zero=3,
        )
        assert P == 1000

    def test_ep_sharding(self):
        """EP splits expert params, non-expert stays intact."""
        # 10000 total, 6000 expert, 4000 non-expert
        P = adam_params_on_rank(
            total_params=10000, n_layers=10, embed_params=0,
            expert_params_full=6000, tp=1, pp=1, ep=3, dp=1, zero_stage=0,
        )
        # non_expert = 10000 - 6000 = 4000; expert_per_rank = 6000 // 3 = 2000
        assert P == 6000

    def test_full_sharding_chain(self):
        """TP + PP + EP + DP (ZeRO-3) all applied."""
        P = adam_params_on_rank(
            total_params=100000, n_layers=10, embed_params=1000,
            expert_params_full=60000, tp=2, pp=2, ep=4, dp=8, zero_stage=3,
            apply_dp_for_zero=1,
        )
        assert P > 0
        assert P < 100000
