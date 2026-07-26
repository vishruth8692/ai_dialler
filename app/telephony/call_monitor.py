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
