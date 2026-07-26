"""Simple JSON-file-backed store of finished calls - phone number, transcript, collected answers,
and AI-classified tags/summary - so past calls can be reviewed on /telephony after the fact, not
just watched live. No DB needed at this scale; the whole file is read/rewritten each time."""

import json
from pathlib import Path

_HISTORY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "call_history.json"


def _load() -> list[dict]:
    if not _HISTORY_PATH.exists():
        return []
    try:
        return json.loads(_HISTORY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save(records: list[dict]) -> None:
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2))


def add_call(record: dict) -> None:
    records = _load()
    records.append(record)
    _save(records)


def list_calls() -> list[dict]:
    """Most recent first."""
    return list(reversed(_load()))
