"""Simple in-memory pub/sub so a browser can watch a real Exotel call happen live - transcript,
bot replies, and debug info, sourced from the actual call in real time. One shared broadcast
channel, not per-call-id routing: this project places one outbound call at a time, so there's
never ambiguity about which call a connected monitor is watching.

Deliberately does not carry audio - the call's audio goes to the phone, not the browser. This is
purely a text/JSON event feed for visibility while a real call is in progress.
"""

import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)

_subscribers: set[WebSocket] = set()

# Tracked alongside the existing "call_lifecycle" started/ended broadcasts (see
# app/streaming/exotel_ws_adapter.py) - a queryable version of the same one-call-at-a-time fact
# this module's docstring already relies on, so app/attrition/call_queue.py can wait for the
# current call to actually finish before placing the next one in a bulk-call CSV run.
_call_active = False


def mark_call_started() -> None:
    global _call_active
    _call_active = True


def mark_call_ended() -> None:
    global _call_active
    _call_active = False


def is_call_active() -> bool:
    return _call_active


async def register(websocket: WebSocket) -> None:
    await websocket.accept()
    _subscribers.add(websocket)


def unregister(websocket: WebSocket) -> None:
    _subscribers.discard(websocket)


async def broadcast(event: dict) -> None:
    if not _subscribers:
        return
    dead = []
    for ws in _subscribers:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _subscribers.discard(ws)
