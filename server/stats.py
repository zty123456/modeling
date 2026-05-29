"""Usage statistics for the ZRT-Sim web interface.

Counts task submissions (trace / estimate / search) per username and
persists them to a local JSON file. Best-effort side channel: recording
must never raise into the request path, and a missing or corrupt stats
file is treated as empty.

Storage layout (JSON):
    {"users": {"alice": {"trace": 3, "estimate": 5, "search": 1}}}

Path defaults to server/stats.json; override with the ZRT_STATS_FILE
environment variable (read on every call so tests can isolate it).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger(__name__)

TASK_KINDS = ("trace", "estimate", "search")
_DEFAULT_PATH = Path(__file__).parent / "stats.json"
_lock = threading.Lock()


def _stats_path() -> Path:
    override = os.environ.get("ZRT_STATS_FILE")
    return Path(override) if override else _DEFAULT_PATH


def _load(path: Path) -> dict:
    """Load the stats file, returning {} on any problem."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (ValueError, OSError) as exc:
        log.warning("stats file unreadable (%s); treating as empty", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _atomic_write(path: Path, data: dict) -> None:
    # Assumes the caller holds _lock — the fixed temp filename is only safe
    # under the lock.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _normalize_user(username: str | None) -> str:
    return (username or "").strip() or "anonymous"


def record_submission(username: str | None, kind: str) -> None:
    """Increment the count for (username, kind) by 1.

    Unknown kinds are ignored. Never raises — failures are logged.
    """
    if kind not in TASK_KINDS:
        log.warning("ignoring unknown task kind %r", kind)
        return
    name = _normalize_user(username)
    path = _stats_path()
    try:
        with _lock:
            data = _load(path)
            counts = data.setdefault("users", {}).setdefault(name, {})
            counts[kind] = int(counts.get(kind, 0) or 0) + 1
            _atomic_write(path, data)
    except Exception as exc:  # best-effort: never break the request path
        log.warning("failed to record submission (%s)", exc)


def read_totals() -> dict:
    """Aggregate totals per task kind across all users."""
    with _lock:
        data = _load(_stats_path())
    users = data.get("users", {})
    if not isinstance(users, dict):
        users = {}
    totals = {kind: 0 for kind in TASK_KINDS}
    for counts in users.values():
        if not isinstance(counts, dict):
            continue
        for kind in TASK_KINDS:
            val = counts.get(kind, 0)
            try:
                totals[kind] += int(val)
            except (TypeError, ValueError):
                pass
    return {"totals": totals, "total": sum(totals.values()), "users": len(users)}
