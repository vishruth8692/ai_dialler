"""Sequential bulk-calling queue for the attrition bot - takes a CSV of phone numbers (plus
optional rider_name/rider_code/city/store_name/preferred_language columns, same fields as the
single "Place a call" form), places them one at a time, and waits for each call to actually finish
(see app/telephony/call_monitor.py's is_call_active()) plus a fixed gap before starting the next.

This app can only ever have one live call at a time - one Exotel account/number, one
call_monitor live-view feed, one in-process session per call (see call_monitor.py's own
docstring) - so the queue is built around that constraint (wait for real completion) rather than
just firing on a fixed timer, which could easily overlap two live calls if one runs long.
"""

import asyncio
import csv
import io
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.attrition import greeting_prefetch
from app.telephony import call_monitor, exotel_client

logger = logging.getLogger(__name__)

_GAP_SECONDS = 30
_CALL_STATE_POLL_S = 2.0
_MAX_WAIT_FOR_CALL_START_S = 90  # phone never answered / Exotel never connected back to us
_MAX_WAIT_FOR_CALL_END_S = 20 * 60  # failsafe only - a real call should never run this long

_DIAL_RECORD_FIELDS = ("rider_name", "rider_code", "city", "store_name", "preferred_language")


def normalize_phone(raw: str) -> Optional[str]:
    """Returns a dialable +91XXXXXXXXXX string for a valid 10-digit Indian number, tolerating a
    stray +91/91/leading-0 prefix and any spaces/dashes - or None if it doesn't resolve to exactly
    10 digits."""
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) == 10:
        return f"+91{digits}"
    return None


def parse_csv(file_content: bytes) -> tuple[list[dict], list[str]]:
    """Returns (valid_rows, errors). Bad phone numbers are skipped with a message rather than
    failing the whole upload - same "don't let a few bad rows block the good ones" spirit as
    app/attrition/qa_store.py's CSV parser, just with per-row errors surfaced instead of silently
    dropped. valid_rows: [{"to_number": "+91...", "dial_record": {...optional fields...}}, ...]."""
    text = file_content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV appears to be empty or has no header row.")

    header_map = {name.strip().lower(): name for name in reader.fieldnames}
    phone_key = header_map.get("phone_number") or header_map.get("to_number")
    if not phone_key:
        raise ValueError("CSV is missing required column: phone_number")

    rows = []
    errors = []
    for i, row in enumerate(reader, start=2):  # row 1 is the header
        raw_phone = (row.get(phone_key) or "").strip()
        to_number = normalize_phone(raw_phone)
        if not to_number:
            errors.append(f"Row {i}: invalid phone number {raw_phone!r} - must resolve to 10 digits.")
            continue
        dial_record = {}
        for field_name in _DIAL_RECORD_FIELDS:
            key = header_map.get(field_name)
            value = (row.get(key) or "").strip() if key else ""
            if value:
                dial_record[field_name] = value
        rows.append({"to_number": to_number, "dial_record": dial_record})

    if not rows and not errors:
        raise ValueError("No rows found in the CSV.")
    return rows, errors


@dataclass
class _QueueState:
    pending: list[dict] = field(default_factory=list)
    done: list[dict] = field(default_factory=list)  # {"to_number", "status": "called"|"failed", "error"}
    running: bool = False
    total: int = 0


_state = _QueueState()
_worker_task: Optional[asyncio.Task] = None


def status() -> dict:
    return {
        "running": _state.running,
        "pending_count": len(_state.pending),
        "done": list(_state.done),
        "total": _state.total,
    }


def enqueue(rows: list[dict]) -> None:
    global _worker_task
    _state.pending.extend(rows)
    _state.total += len(rows)
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_run())


def clear_pending() -> None:
    """Stops the queue after whatever call is currently in flight finishes - never hangs up a
    live call, just skips placing any more after it."""
    _state.pending.clear()


def reset() -> None:
    """Clears pending AND the done/failed history - for starting a fresh batch. Only meaningful
    once the queue isn't running (the UI hides this while running)."""
    _state.pending.clear()
    _state.done.clear()
    _state.total = 0


async def _wait_until(active: bool, timeout: float) -> bool:
    elapsed = 0.0
    while call_monitor.is_call_active() != active:
        if elapsed >= timeout:
            return False
        await asyncio.sleep(_CALL_STATE_POLL_S)
        elapsed += _CALL_STATE_POLL_S
    return True


async def _run() -> None:
    _state.running = True
    try:
        while _state.pending:
            row = _state.pending.pop(0)
            to_number = row["to_number"]
            dial_record = row["dial_record"]
            try:
                exotel_client.place_call(to_number, dial_record=dial_record, stream_path="/attrition/exotel-stream")
            except Exception as e:
                logger.exception("Bulk call to %s failed to place", to_number)
                _state.done.append({"to_number": to_number, "status": "failed", "error": str(e)})
                continue
            # Same overlap-with-ring-time trick as the single "Place a call" form - see
            # greeting_prefetch.py.
            greeting_prefetch.start(dial_record.get("rider_name", ""))

            started = await _wait_until(active=True, timeout=_MAX_WAIT_FOR_CALL_START_S)
            if not started:
                _state.done.append(
                    {"to_number": to_number, "status": "failed", "error": "Call never connected (no answer?)."}
                )
                continue

            await _wait_until(active=False, timeout=_MAX_WAIT_FOR_CALL_END_S)
            _state.done.append({"to_number": to_number, "status": "called"})

            if _state.pending:
                await asyncio.sleep(_GAP_SECONDS)
    finally:
        _state.running = False
