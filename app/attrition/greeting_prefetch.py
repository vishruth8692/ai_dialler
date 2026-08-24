"""Speculatively generates the attrition bot's greeting the moment a call is placed, instead of
waiting for Exotel's WebSocket to connect (which happens near/at answer time). Both cost the same
Claude round-trip (~3-5s, confirmed via real-call logs) - starting it here overlaps that cost with
time that's already being spent on ringing/call-setup, rather than adding it fully on top after the
rider picks up.

Single pending slot, not keyed by phone number or call SID - this app only ever runs one live call
at a time (see app/telephony/call_monitor.py's docstring, and the bulk queue in call_queue.py is
built around the same fact), so there's never ambiguity about which placed call a given prefetch
belongs to.
"""

import asyncio
import logging
from typing import Optional

from app.attrition import prompts
from app.config import ATTRITION_SAFETY_HELPLINE, CLAUDE_MODEL
from app.llm.claude_client import _shared_async_client

logger = logging.getLogger(__name__)

_pending: Optional[asyncio.Task] = None


async def _generate(rider_name: str) -> str:
    system = prompts.render_system_prompt(prompts.GREETING, rider_name, ATTRITION_SAFETY_HELPLINE)
    response = await _shared_async_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        system=system,
        messages=[{"role": "user", "content": "Begin the call."}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def start(rider_name: str) -> None:
    """Fire-and-forget - call right after exotel_client.place_call() succeeds, before waiting for
    the call to actually connect. Overwrites any still-pending previous prefetch (e.g. an earlier
    call that never connected) rather than accumulating - only ever one call in flight."""
    global _pending
    _pending = asyncio.create_task(_generate(rider_name))


async def take() -> Optional[str]:
    """Consumes the prefetched greeting, awaiting it if still in flight - or None if nothing was
    prefetched (e.g. a call reached this code without going through the normal place_call() path),
    so the caller can fall back to generating it fresh rather than failing the call."""
    global _pending
    task, _pending = _pending, None
    if task is None:
        return None
    try:
        return await task
    except Exception:
        logger.exception("Prefetched greeting failed - falling back to generating it fresh")
        return None
