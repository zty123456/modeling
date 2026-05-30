import pytest

from server.schemas import EstimateRequest, SearchRequest, TraceRequest


def test_request_schemas_accept_username():
    assert TraceRequest(model_id="m", username="alice").username == "alice"
    assert EstimateRequest(config_content="x", username="bob").username == "bob"
    assert SearchRequest(config_content="x", username="cara").username == "cara"


def test_username_defaults_to_none():
    assert EstimateRequest(config_content="x").username is None


def test_get_stats_reflects_recorded_submissions(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("ZRT_STATS_FILE", str(tmp_path / "stats.json"))
    from server import stats
    from server.main import get_stats

    stats.record_submission("alice", "estimate")
    stats.record_submission("bob", "search")
    result = get_stats()
    assert result["totals"]["estimate"] == 1
    assert result["totals"]["search"] == 1
    assert result["total"] == 2


def test_submit_estimate_records_a_submission(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("ZRT_STATS_FILE", str(tmp_path / "stats.json"))
    from fastapi import BackgroundTasks

    from server import stats
    from server.main import submit_estimate
    from server.schemas import EstimateRequest

    # config_content passes validation; calling the handler directly only
    # *registers* the background job (it does not execute), so no heavy work.
    submit_estimate(
        EstimateRequest(config_content="dummy", username="alice"),
        BackgroundTasks(),
    )
    assert stats.read_totals()["totals"]["estimate"] == 1
