import json
import threading

import pytest

from server import stats


@pytest.fixture
def stats_file(tmp_path, monkeypatch):
    """Point the stats module at an isolated temp file per test."""
    path = tmp_path / "stats.json"
    monkeypatch.setenv("ZRT_STATS_FILE", str(path))
    return path


def test_record_increments_per_user_and_kind(stats_file):
    stats.record_submission("alice", "estimate")
    stats.record_submission("alice", "estimate")
    stats.record_submission("alice", "search")
    data = json.loads(stats_file.read_text(encoding="utf-8"))
    assert data["users"]["alice"] == {"estimate": 2, "search": 1}


def test_read_totals_aggregates_across_users(stats_file):
    stats.record_submission("alice", "estimate")
    stats.record_submission("bob", "estimate")
    stats.record_submission("bob", "search")
    result = stats.read_totals()
    assert result["totals"] == {"trace": 0, "estimate": 2, "search": 1}
    assert result["total"] == 3
    assert result["users"] == 2


def test_blank_or_none_username_falls_back_to_anonymous(stats_file):
    stats.record_submission(None, "trace")
    stats.record_submission("   ", "trace")
    data = json.loads(stats_file.read_text(encoding="utf-8"))
    assert data["users"]["anonymous"]["trace"] == 2


def test_unknown_kind_is_ignored(stats_file):
    stats.record_submission("alice", "bogus")
    assert not stats_file.exists()  # early return → nothing written
    assert stats.read_totals()["total"] == 0


def test_missing_file_reads_as_empty(stats_file):
    assert not stats_file.exists()
    assert stats.read_totals() == {
        "totals": {"trace": 0, "estimate": 0, "search": 0},
        "total": 0,
        "users": 0,
    }


def test_corrupt_file_is_treated_as_empty_then_recovers(stats_file):
    stats_file.write_text("{not valid json", encoding="utf-8")
    assert stats.read_totals()["total"] == 0          # survives corruption
    stats.record_submission("alice", "trace")          # overwrites cleanly
    assert stats.read_totals()["totals"]["trace"] == 1


def test_concurrent_records_do_not_lose_counts(stats_file):
    def worker():
        for _ in range(50):
            stats.record_submission("alice", "estimate")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert stats.read_totals()["totals"]["estimate"] == 8 * 50


def test_read_totals_survives_wrong_shaped_users(stats_file):
    stats_file.write_text('{"users": ["alice"]}', encoding="utf-8")
    assert stats.read_totals() == {
        "totals": {"trace": 0, "estimate": 0, "search": 0},
        "total": 0,
        "users": 0,
    }


def test_read_totals_skips_non_numeric_counts(stats_file):
    stats_file.write_text(
        '{"users": {"alice": {"trace": "abc", "estimate": 4}}}', encoding="utf-8"
    )
    result = stats.read_totals()
    assert result["totals"] == {"trace": 0, "estimate": 4, "search": 0}
    assert result["total"] == 4
    assert result["users"] == 1
