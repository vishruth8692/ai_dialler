"""Adapter translating Exotel's bidirectional voice-streaming WebSocket protocol into the same
internal event shapes app/streaming/call_ws_handler.py's orchestrator already expects - so the
exact same CallSession/Claude/Sarvam/barge-in state machine built and verified for browser calls
(see app/templates/voice_chat_live.html) handles real phone calls too, with only this transport
layer being Exotel-specific. This is exactly the reuse the streaming architecture was designed for.

VERIFIED against real calls on a live (trial) Exotel account. Confirmed via actual call logs:
  {"event": "start", "stream_sid": "...", "start": {"call_sid", "leg_sid", "account_sid", "from",
                                                       "to", "media_format": {"encoding", "sample_rate", "bit_rate"}}}
  {"event": "media", "stream_sid": "...", "media": {"chunk", "timestamp", "payload": "<b64 PCM>"}}
  {"event": "stop", "stream_sid": "...", "stop": {"call_sid", "leg_sid", "account_sid", "reason"}}
  {"event": "connected"} - sent once, before "start" - harmless, ignored.
Audio confirmed as raw PCM16 little-endian mono at the rate requested via the stream URL's
?sample-rate= query param (see app/telephony/exotel_client.py - requests 16000 to match Sarvam's
STT requirement). Verified via byte-math on a real call: 640 bytes per 20ms chunk = exactly
16000 Hz * 2 bytes/sample * 0.02s.

One real bug this verification caught: outgoing TTS was left at SarvamTTSStream's 24kHz default
instead of matching the negotiated 16kHz, making the bot's replies play back pitch-shifted and
slowed by 1.5x on the first real call - "audio you can't make out" is the exact symptom of a
sample-rate mismatch. Fixed below by passing sample_rate=16000 explicitly.
"""

import asyncio
import base64
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.call_session import CallSession
from app.llm import call_tagger
from app.rag import qa_store
from app.streaming.call_ws_handler import _orchestrate  # package-internal reuse, see module docstring
from app.streaming.sarvam_stt_ws import SarvamSTTStream
from app.streaming.sarvam_tts_ws import SarvamTTSStream
from app.telephony import call_history, call_monitor

logger = logging.getLogger(__name__)


@dataclass
class _ExotelCallInfo:
    stream_sid: Optional[str] = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    call_sid: Optional[str] = None
    phone_number: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dial_record: dict = field(default_factory=dict)
    transcript: list = field(default_factory=list)
    collected_answers: list = field(default_factory=list)
    _pending_bot_text: list = field(default_factory=list)


def _default_configured_check() -> bool:
    return bool(qa_store.get_survey_questions())


def _default_classify_fn(dial_record: dict, collected_answers: list, transcript: list) -> dict:
    return call_tagger.classify_call(collected_answers, transcript)


async def run_exotel_call(
    websocket: WebSocket,
    *,
    call_type: str = "zepto_feedback",
    session_factory: Callable[[dict], object] = lambda dial_record: CallSession(),
    configured_check: Callable[[], bool] = _default_configured_check,
    not_configured_message: str = "Exotel call connected but no survey questions are loaded - closing.",
    classify_fn: Callable[[dict, list, list], dict] = _default_classify_fn,
) -> None:
    """Entry point for Exotel's WebSocket connection - one call per connection, same shape as
    call_ws_handler.run_call() for the browser path. No error messages are sent back on failure
    (unlike the browser path) since there's no UI on the other end to show them to - just log and
    close so the call ends cleanly.

    Shared by both call types (Zepto feedback and rider attrition) - only `session_factory` (which
    session class to drive), `configured_check` (what "ready for a real call" means for that bot),
    and `classify_fn` (how to turn a finished call into a stored record) differ; the ~150 lines of
    Exotel media-handling/history-capture logic below is identical either way.
    """
    await websocket.accept()

    if not configured_check():
        logger.error(not_configured_message)
        await websocket.close()
        return

    # The rider dial record (rider_name, rider_code, city, ...) travels through the same
    # query-string passthrough already proven for ?sample-rate=16000 - Exotel echoes back whatever
    # was in the StreamUrl we gave it when placing the call (see app/telephony/exotel_client.py).
    # Empty {} for the Zepto bot, which doesn't need one.
    dial_record = {k: v for k, v in websocket.query_params.items() if k != "sample-rate"}

    call_info = _ExotelCallInfo(dial_record=dial_record)
    session = session_factory(dial_record)

    call_monitor.mark_call_started()
    await call_monitor.broadcast({"type": "call_lifecycle", "status": "started"})
    try:
        # sample_rate=16000 must match the rate negotiated with Exotel via the ?sample-rate=16000
        # query param in app/telephony/exotel_client.py - confirmed via a real call's 'start' event
        # (media_format) and raw chunk byte-math that Exotel really does send/expect 16kHz, not the
        # 24kHz SarvamTTSStream defaults to. Leaving this at the default was the actual bug behind
        # "audio was garbled/unintelligible" on the first real call - it made the bot's replies play
        # back pitch-shifted and slowed by 24/16 = 1.5x.
        #
        # STT and TTS connections are opened CONCURRENTLY (asyncio.gather), not via a plain
        # `async with (a, b):` - that enters context managers sequentially, and each __aenter__ here
        # is a real websockets.connect() round trip. Confirmed on a real call this was part of a
        # ~4s gap of silence after the rider picked up, before the greeting could even start.
        t0 = time.monotonic()
        async with contextlib.AsyncExitStack() as stack:
            stt, tts = await asyncio.gather(
                stack.enter_async_context(SarvamSTTStream()),
                stack.enter_async_context(
                    SarvamTTSStream(language=session.language_hint, sample_rate=16000)
                ),
            )
            logger.info("[latency] Sarvam STT+TTS connections opened: %.2fs", time.monotonic() - t0)
            await _orchestrate(
                session,
                stt,
                tts,
                make_reader_task=lambda stt_send_q, events_q: _exotel_reader_task(
                    websocket, call_info, stt_send_q, events_q
                ),
                make_writer_task=lambda outbound_q, events_q: _exotel_writer_task(
                    websocket, call_info, outbound_q, events_q
                ),
            )
    except WebSocketDisconnect:
        logger.info("Exotel disconnected mid-call")
    except Exception:
        logger.exception("Exotel call failed")
    finally:
        call_monitor.mark_call_ended()
        await call_monitor.broadcast({"type": "call_lifecycle", "status": "ended"})
        if call_info.transcript:
            # Only log a history entry if the call actually got underway (greeting spoken) - a
            # connection that never sent a real "start" event has nothing worth recording.
            # Classification is a nice-to-have annotation, run off the event loop since it's a
            # blocking API call - never let a classification failure stop the call record from
            # being saved.
            classification = await asyncio.to_thread(
                classify_fn, call_info.dial_record, call_info.collected_answers, call_info.transcript
            )
            record = {
                "call_type": call_type,
                "call_sid": call_info.call_sid,
                "phone_number": call_info.phone_number,
                "dial_record": call_info.dial_record,
                "started_at": call_info.started_at,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "transcript": call_info.transcript,
                "collected_answers": call_info.collected_answers,
            }
            if call_type == "zepto_feedback":
                # Kept as top-level tags/summary fields, unchanged, so existing stored records and
                # telephony.html's history rendering stay compatible.
                record["tags"] = classification["tags"]
                record["summary"] = classification["summary"]
            else:
                record["structured_record"] = classification
            call_history.add_call(record)
        try:
            await websocket.close()
        except Exception:
            pass


_RAW_LOG_MEDIA_SAMPLE_EVERY = 50  # log 1 in N media events verbatim (minus the payload blob) - the
# first call revealed real protocol details can't be trusted from docs alone; this is cheap
# ongoing visibility without flooding logs at ~1 media event per 20-100ms.


async def _exotel_reader_task(
    websocket: WebSocket,
    call_info: _ExotelCallInfo,
    stt_send_q: asyncio.Queue,
    events_q: asyncio.Queue,
) -> None:
    """Sole reader of the Exotel WS. Exotel wraps audio in a JSON envelope (event/media/payload),
    unlike the browser's raw binary frames - unwrap it into the same {"kind":"audio","data":bytes}
    shape _stt_writer_task already expects, so the rest of the pipeline doesn't need to know or
    care which transport it's running over.

    Logs verbosely by design right now (every non-media event in full, a sample of media events,
    and every raw message that doesn't match a known event type) - the first real call revealed
    audio came through garbled, and the previous version of this function would have silently
    ignored anything that didn't match its assumed protocol, exactly the kind of mismatch that
    needs to be visible rather than swallowed. Trim this back down once the real protocol is
    confirmed against a working call."""
    media_count = 0
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            logger.info("Exotel WS disconnected (raw message: %s)", message)
            await events_q.put({"type": "disconnected"})
            return

        text = message.get("text")
        if text is None:
            logger.warning("Exotel sent a non-text WS frame, unexpected: %s", message)
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Exotel sent non-JSON text frame: %r", text[:500])
            continue

        event = payload.get("event")
        if event == "start":
            start_info = payload.get("start", {})
            call_info.stream_sid = payload.get("stream_sid") or start_info.get("stream_sid")
            call_info.call_sid = start_info.get("call_sid")
            call_info.phone_number = start_info.get("from")
            call_info.ready.set()
            logger.info("Exotel 'start' event (full, raw): %s", json.dumps(payload))
            # "from" is Exotel's (confusingly named) field for the number actually being called -
            # confirmed against real call logs, see module docstring. Broadcast separately from the
            # immediate "started" lifecycle event in run_exotel_call() since the phone number isn't
            # known until this "start" event arrives, slightly after the call/WS connection begins.
            await call_monitor.broadcast(
                {"type": "call_info", "phone_number": call_info.phone_number}
            )
        elif event == "media":
            media_count += 1
            if media_count == 1 or media_count % _RAW_LOG_MEDIA_SAMPLE_EVERY == 0:
                media_obj = payload.get("media", {})
                sample = {k: v for k, v in media_obj.items() if k != "payload"}
                b64_payload = media_obj.get("payload", "")
                sample["payload_b64_len"] = len(b64_payload)
                sample["payload_decoded_bytes"] = len(base64.b64decode(b64_payload)) if b64_payload else 0
                logger.info("Exotel 'media' event #%d (payload elided): %s", media_count, sample)
            b64_audio = payload.get("media", {}).get("payload")
            if b64_audio:
                await stt_send_q.put({"kind": "audio", "data": base64.b64decode(b64_audio)})
        elif event == "stop":
            logger.info("Exotel 'stop' event (full, raw): %s", json.dumps(payload))
            await events_q.put({"type": "disconnected"})
            return
        elif event == "connected":
            pass  # benign pre-"start" handshake ack, confirmed harmless on a real call, no-op
        elif event in ("dtmf", "mark", "clear"):
            logger.info("Exotel '%s' event (full, raw, not acted on): %s", event, json.dumps(payload))
        else:
            logger.warning("Exotel sent an UNRECOGNIZED event type %r (full, raw): %s", event, json.dumps(payload))


def _record_for_history(call_info: _ExotelCallInfo, data: dict) -> None:
    """Builds call_info.transcript/collected_answers up from the same UI events already being
    broadcast to the live monitor - no separate instrumentation needed, this just also keeps a
    copy for call_history once the call ends."""
    msg_type = data.get("type")
    if msg_type == "bot_text_chunk":
        call_info._pending_bot_text.append(data["text"])
    elif msg_type == "reply_done":
        if call_info._pending_bot_text:
            call_info.transcript.append(
                {"role": "bot", "text": " ".join(call_info._pending_bot_text)}
            )
            call_info._pending_bot_text = []
    elif msg_type == "transcript":
        call_info.transcript.append(
            {"role": "rider", "text": data["text"], "barge_in": data.get("barge_in", False)}
        )
    elif msg_type == "control":
        call_info.collected_answers = data.get("collected_answers", [])


_START_EVENT_TIMEOUT_S = 10.0  # give up if Exotel's "start" event never arrives


async def _exotel_writer_task(
    websocket: WebSocket,
    call_info: _ExotelCallInfo,
    outbound_q: asyncio.Queue,
    events_q: asyncio.Queue,
) -> None:
    """Sole sender to the Exotel WS. Wraps outgoing audio in Exotel's expected envelope. The JSON
    UI messages the browser path sends (bot_state/transcript/control/...) have no meaning to Exotel
    itself - it only wants audio - but are broadcast to app/telephony/call_monitor.py so a browser
    on /telephony can watch the call happen live, same information /voice/live shows for a browser
    call, just sourced from a real phone call instead.

    Timing confirmed via real calls: the orchestrator starts speaking the greeting immediately on
    entry, which can be (and was, in testing) before Exotel's "start" event/stream_sid arrives.
    Waiting on call_info.ready here delays early audio rather than dropping it - confirmed correct,
    calls connected and played the greeting successfully.

    Bounded by _START_EVENT_TIMEOUT_S so a connection that never sends a real "start" event (e.g.
    a stray/test connection, or a misbehaving client) doesn't sit open indefinitely holding a
    CallSession and open Sarvam STT/TTS connections open forever - confirmed this was a real gap,
    not theoretical, by testing a raw WebSocket connection against this endpoint directly.
    """
    try:
        await asyncio.wait_for(call_info.ready.wait(), timeout=_START_EVENT_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.error(
            "Exotel connection never sent a 'start' event within %.0fs - closing.",
            _START_EVENT_TIMEOUT_S,
        )
        await events_q.put({"type": "disconnected"})
        return

    while True:
        kind, data = await outbound_q.get()
        if kind == "json":
            _record_for_history(call_info, data)
            await call_monitor.broadcast(data)
            continue
        await websocket.send_text(
            json.dumps(
                {
                    "event": "media",
                    "stream_sid": call_info.stream_sid,
                    "media": {"payload": base64.b64encode(data).decode()},
                }
            )
        )
