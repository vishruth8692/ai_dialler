"""WebSocket orchestrator bridging a browser voice call to Sarvam STT/TTS streaming and Claude's
streaming turn logic - with real barge-in support: the mic is always live, and if the rider talks
continuously for 5+ seconds while the bot is speaking, the bot stops and listens. A shorter pause
(a quick "okay"/"haan") is treated as a backchannel acknowledgment and doesn't interrupt anything -
purely duration-based, no per-language word lists or transcript classification needed.

One browser WebSocket connection = one CallSession, scoped to this coroutine's lifetime - no
session-store keying needed, which is also the shape that generalizes directly to a future
real-phone-call handler (one Exotel media-stream WebSocket = one call, same structure).

Concurrency design: every socket (browser WS, Sarvam STT WS) has exactly one dedicated reader task
and one dedicated writer task, fed/drained via queues - never more than one task ever calls
.receive()/.events() on a given socket. This is the fix for a real bug hit earlier in this project
(wrapping a shared WS-receiving generator in asyncio.wait_for()+cancel corrupted it); the same
discipline extends cleanly to a socket now being written to by multiple concurrent producers
(continuous mic audio forwarding vs. orchestrator-issued flushes; TTS audio streaming vs. control
messages) by funneling everything through one queue per socket.

Browser <-> backend protocol (see app/templates/voice_chat_live.html for the client side):
  browser -> backend: binary frames = raw 16-bit PCM audio chunks, streamed continuously for the
                       whole call (no more click-to-talk / turn_end signal - turns are VAD-driven).
  backend -> browser: {"type":"bot_state","state":"speaking"|"listening"} - UI state only, the mic
                         is never gated on this.
                       {"type":"transcript","text":...,"barge_in":bool} once the rider's turn is
                         transcribed (barge_in=true if this continued an interrupted bot reply).
                       {"type":"bot_text_chunk","text":...} per sentence, as the bot composes it.
                       binary frames = raw 16-bit PCM audio chunks of the bot's speech, as
                         synthesized (forward immediately, don't buffer).
                       {"type":"stop_audio"} - a genuine interruption was confirmed; the browser
                         must immediately stop all scheduled playback and clear its queue.
                       {"type":"control","tool_name",...} once the LLM's decision is known.
                       {"type":"reply_done"} once all of this turn's audio has been sent.
                       {"type":"error","message":...} on failure.
"""

import asyncio
import contextlib
import logging
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.call_session import CallSession
from app.rag import qa_store
from app.streaming.sarvam_stt_ws import SarvamSTTStream
from app.streaming.sarvam_tts_ws import SarvamTTSStream

logger = logging.getLogger(__name__)

_RATECARD_RE = re.compile(r"\bratecard\b", re.IGNORECASE)
_RATECARD_DEVANAGARI_RE = re.compile(r"रेटकार्ड")


def _normalize_for_tts(text: str) -> str:
    """Sarvam's TTS mispronounces "ratecard" as one mashed-together word - confirmed on a real
    call ("the voice is still not clear, like ratecard"). The prompt also tells Claude to write
    "rate card" as two words, but this is a deterministic safety net independent of how well the
    model follows that instruction - covers both the Latin-script and Devanagari-transliterated
    spellings Claude has been observed to produce."""
    text = _RATECARD_RE.sub("rate card", text)
    text = _RATECARD_DEVANAGARI_RE.sub("रेट कार्ड", text)
    return text

_TTS_PING_INTERVAL = 30  # seconds; Sarvam's TTS WS closes after 60s of inactivity otherwise
_BARGE_IN_GRACE_S = 0.3  # ignore vad_start right as the bot starts talking (TTS-onset false trigger)
_BARGE_IN_THRESHOLD_S = 5.0  # continuous rider speech past this while the bot talks = real interrupt
_PAUSE_GRACE_S = 1.2  # confirmed on a real call: Sarvam's VAD reports vad_end on ANY pause, even a
# brief mid-sentence one while the rider is still thinking - flushing immediately on every vad_end
# cut their answer off mid-thought and had the bot reply to a fragment. Waiting this long after
# vad_end before actually flushing gives them a beat to resume; if they do, it's folded into the
# same continuous utterance instead of being treated as a separate, completed turn.
_SPEAKING_TIMEOUT_S = 20.0  # safety net: normal turns finish in a few seconds. On a real call, a
# reply whose Claude stream never emitted the ###CONTROL### block left _run_speaking stuck inside
# drive()/forward() with no completion event ever arriving from Sarvam, wedging `speaking` as
# "reply" for the rest of the call - every subsequent transcript was silently dropped as "stray"
# until the rider gave up and hung up. This bounds that failure mode instead of hanging forever.


@dataclass
class SpeakingHandle:
    kind: str = "reply"  # "greeting" | "fallback" | "reply"
    task: Optional[asyncio.Task] = None
    flushes_sent: int = 0
    completions_seen: int = 0
    barge_in_watchdog: Optional[asyncio.Task] = None
    started_at: float = 0.0


async def run_call(websocket: WebSocket) -> None:
    await websocket.accept()

    if not qa_store.get_survey_questions():
        await websocket.send_json(
            {"type": "error", "message": "No survey questions loaded yet - upload a CSV at /qa first."}
        )
        await websocket.close()
        return

    session = CallSession()

    try:
        async with SarvamSTTStream() as stt, SarvamTTSStream(language=session.language_hint) as tts:
            await _orchestrate(
                session,
                stt,
                tts,
                make_reader_task=lambda stt_send_q, events_q: _browser_reader_task(
                    websocket, stt_send_q, events_q
                ),
                make_writer_task=lambda outbound_q, events_q: _browser_writer_task(
                    websocket, outbound_q
                ),
            )
    except WebSocketDisconnect:
        logger.info("Browser disconnected mid-call")
    except Exception as e:
        logger.exception("Voice call failed")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


async def _orchestrate(
    session: CallSession,
    stt: SarvamSTTStream,
    tts: SarvamTTSStream,
    *,
    make_reader_task: Callable[[asyncio.Queue, asyncio.Queue], Awaitable[None]],
    make_writer_task: Callable[[asyncio.Queue, asyncio.Queue], Awaitable[None]],
) -> None:
    """Transport-agnostic: `make_reader_task(stt_send_q, events_q)` and
    `make_writer_task(outbound_q, events_q)` build the transport-specific reader/writer coroutines
    (browser WS today, Exotel WS later - see app/streaming/exotel_ws_adapter.py) - everything below
    is identical regardless of which peer is on the other end, which is exactly the point: the same
    CallSession/Claude/Sarvam/barge-in state
    machine handles both a browser demo call and a real phone call."""
    events_q: asyncio.Queue = asyncio.Queue()
    outbound_q: asyncio.Queue = asyncio.Queue()
    stt_send_q: asyncio.Queue = asyncio.Queue()

    infra = [
        asyncio.create_task(_stt_reader_task(stt, events_q)),
        asyncio.create_task(_stt_writer_task(stt, stt_send_q)),
        asyncio.create_task(make_reader_task(stt_send_q, events_q)),
        asyncio.create_task(make_writer_task(outbound_q, events_q)),
        asyncio.create_task(_tts_keepalive_task(tts)),
    ]

    # `speaking` is the single source of truth for LISTENING (None) vs SPEAKING (a handle) - no
    # separate state enum, so state and handle can never disagree with each other.
    speaking: Optional[SpeakingHandle] = None
    awaiting_transcript = False
    # Pending "confirm the rider is actually done talking" timer while LISTENING - see
    # _PAUSE_GRACE_S. Mirrors the SPEAKING-state barge_in_watchdog pattern but for the opposite
    # direction (deciding whether OUR flush should fire, not whether an interruption should).
    pause_watchdog: Optional[asyncio.Task] = None
    # Set when barge_in_confirmed cancels a SPEAKING handle, consumed by the next transcript so
    # its `barge_in` flag is accurate even when interrupting the greeting/fallback (which have no
    # prior user history entry for pop_pending_user_turn() to detect).
    just_interrupted = False
    # Latency instrumentation - logs STT round-trip (flush -> transcript) and, in _run_speaking,
    # time-to-first-audio-byte, so real call latency can be measured instead of guessed at.
    t_flush_sent: Optional[float] = None

    def start_speaking(*, fixed_text=None, user_transcript=None, kind="reply") -> SpeakingHandle:
        handle = SpeakingHandle(kind=kind, started_at=time.monotonic())
        handle.task = asyncio.create_task(
            _run_speaking(
                session, outbound_q, tts, handle, events_q,
                fixed_text=fixed_text, user_transcript=user_transcript,
            )
        )
        return handle

    async def cancel_speaking(handle: SpeakingHandle) -> None:
        if handle.barge_in_watchdog is not None:
            handle.barge_in_watchdog.cancel()
            handle.barge_in_watchdog = None
        handle.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await handle.task
        # Sarvam may still be pushing audio for flushes already sent before cancellation - drain
        # only what's actually still outstanding (flushes sent minus finals already consumed by
        # forward() before it got cancelled). If forward() already drained everything naturally
        # (e.g. the utterance had already fully finished and we're just cancelling the trailing
        # real-time-playback sleep - see _run_speaking), there's nothing left to drain, and
        # draining a positive count here would hang forever waiting for a 'final' that will never
        # come (confirmed empirically - this was a real bug, not a theoretical one).
        remaining = handle.flushes_sent - handle.completions_seen
        await tts.drain_pending(remaining)

    t_greeting_start = time.monotonic()
    # session.greeting() is async and uses the async Anthropic client directly (previously a
    # blocking sync call wrapped in asyncio.to_thread() - confirmed on real calls that stalled the
    # whole event loop, including the Exotel WS reader task, for the full 3-4.7s round-trip).
    # AttritionCallSession's greeting() also checks app/attrition/greeting_prefetch.py first, for a
    # copy generated speculatively at place-call time, overlapping this cost with ring time instead
    # of paying it after the rider picks up.
    greeting_text = await session.greeting()
    logger.info("[latency] greeting generation (Claude call): %.2fs", time.monotonic() - t_greeting_start)
    speaking = start_speaking(fixed_text=greeting_text, kind="greeting")
    await outbound_q.put(("json", {"type": "bot_state", "state": "speaking"}))

    try:
        while True:
            event = await events_q.get()
            etype = event["type"]

            # Deliberately .info(), not .debug() - the default log level is INFO (see main.py) and
            # .debug() output has silently gone missing during real diagnosis more than once this
            # project already; this line is cheap and exactly what's needed when a call misbehaves.
            logger.info(
                "event=%s speaking=%s watchdog=%s",
                etype, speaking.kind if speaking else None,
                speaking.barge_in_watchdog is not None if speaking else None,
            )

            if etype == "disconnected":
                break

            elif etype == "vad_start":
                # Always create the watchdog on the first vad_start while SPEAKING - continuous
                # speech produces exactly one vad_start for the whole segment (confirmed
                # empirically), so skipping creation here (e.g. because it landed within the
                # grace window) would mean no watchdog ever gets created for that entire
                # utterance, silently disabling interruption for it. The grace period instead
                # just delays how soon the watchdog can fire, not whether it gets created.
                if speaking is not None and speaking.barge_in_watchdog is None:
                    grace_remaining = max(
                        0.0, _BARGE_IN_GRACE_S - (time.monotonic() - speaking.started_at)
                    )
                    speaking.barge_in_watchdog = asyncio.create_task(
                        _barge_in_watchdog(events_q, grace_remaining + _BARGE_IN_THRESHOLD_S)
                    )
                elif speaking is None and pause_watchdog is not None:
                    # Rider resumed talking before the pause-grace window elapsed - that was just
                    # a mid-sentence pause, not the end of their turn. Cancel the pending flush;
                    # whatever they said stays unflushed in STT's buffer and folds into the rest
                    # of what they're now saying, same as the SPEAKING-state backchannel case.
                    pause_watchdog.cancel()
                    pause_watchdog = None

            elif etype == "vad_end":
                if speaking is not None and speaking.barge_in_watchdog is not None:
                    # Rider paused before the threshold - just a brief acknowledgment, not a real
                    # interruption. Discard; the bot keeps talking. No flush needed - whatever was
                    # said stays unflushed in STT's buffer and naturally folds into whatever the
                    # rider says next.
                    speaking.barge_in_watchdog.cancel()
                    speaking.barge_in_watchdog = None
                elif speaking is None and not awaiting_transcript and pause_watchdog is None:
                    # Don't flush immediately - confirmed on a real call that Sarvam's VAD reports
                    # vad_end on ANY pause, including a brief mid-sentence one, and flushing right
                    # away cut the rider off mid-thought. Wait _PAUSE_GRACE_S first; only flush if
                    # they haven't resumed by the time it fires (see "pause_confirmed" below).
                    pause_watchdog = asyncio.create_task(_pause_watchdog(events_q, _PAUSE_GRACE_S))

            elif etype == "pause_confirmed":
                if speaking is None and not awaiting_transcript:
                    pause_watchdog = None
                    awaiting_transcript = True
                    t_flush_sent = time.monotonic()
                    await stt_send_q.put({"kind": "flush"})

            elif etype == "barge_in_confirmed":
                # Rider has been talking continuously past the threshold - stop the bot now.
                # Deliberately does NOT force a flush here: they may still be mid-sentence, so
                # this just falls through to the ordinary speaking-is-None flow above, which
                # waits for their actual pause (vad_end) before flushing - same code path as a
                # normal turn, no separate "forced partial transcript" logic needed.
                if speaking is not None:
                    handle = speaking
                    speaking = None
                    just_interrupted = True
                    await outbound_q.put(("json", {"type": "stop_audio"}))
                    await cancel_speaking(handle)
                    await outbound_q.put(("json", {"type": "bot_state", "state": "listening"}))

            elif etype == "transcript":
                text = event["text"]
                awaiting_transcript = False
                was_interrupted, just_interrupted = just_interrupted, False
                if t_flush_sent is not None:
                    logger.info("[latency] STT flush -> transcript: %.2fs", time.monotonic() - t_flush_sent)
                    t_flush_sent = None

                if speaking is None:
                    if not text.strip():
                        continue  # VAD tripped but nothing was recognized - likely noise, ignore
                    pending = session.pop_pending_user_turn()
                    merged = f"{pending} {text}".strip() if pending else text
                    await outbound_q.put(
                        (
                            "json",
                            {
                                "type": "transcript",
                                "text": merged,
                                "barge_in": was_interrupted,
                            },
                        )
                    )
                    speaking = start_speaking(user_transcript=merged, kind="reply")
                    await outbound_q.put(("json", {"type": "bot_state", "state": "speaking"}))
                # else: stray transcript that arrived after the bot already started speaking
                # again - nothing to do with it, drop.

            elif etype == "speaking_done":
                data = event["data"]
                finished = speaking
                speaking = None
                if finished is not None:
                    if finished.barge_in_watchdog is not None:
                        finished.barge_in_watchdog.cancel()
                    if finished.kind == "reply":
                        await outbound_q.put(
                            (
                                "json",
                                {
                                    "type": "control",
                                    "tool_name": data["tool_name"],
                                    "answer_summary": data["answer_summary"],
                                    "retrieved_context": data["retrieved_context"],
                                    "collected_answers": session.collected_answers,
                                    "ended": session.ended,
                                },
                            )
                        )
                await outbound_q.put(("json", {"type": "reply_done"}))
                if session.ended:
                    break
                await outbound_q.put(("json", {"type": "bot_state", "state": "listening"}))

            elif etype == "stt_error":
                logger.warning("Sarvam STT error: %s", event.get("raw"))
    finally:
        if pause_watchdog is not None:
            pause_watchdog.cancel()
        if speaking is not None and speaking.task is not None and not speaking.task.done():
            if speaking.barge_in_watchdog is not None:
                speaking.barge_in_watchdog.cancel()
            speaking.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await speaking.task
        for t in infra:
            t.cancel()
        await asyncio.gather(*infra, return_exceptions=True)


async def _barge_in_watchdog(events_q: asyncio.Queue, delay: float) -> None:
    await asyncio.sleep(delay)
    await events_q.put({"type": "barge_in_confirmed"})


async def _pause_watchdog(events_q: asyncio.Queue, delay: float) -> None:
    await asyncio.sleep(delay)
    await events_q.put({"type": "pause_confirmed"})


async def _run_speaking(
    session: CallSession,
    outbound_q: asyncio.Queue,
    tts: SarvamTTSStream,
    handle: SpeakingHandle,
    events_q: asyncio.Queue,
    *,
    fixed_text: Optional[str] = None,
    user_transcript: Optional[str] = None,
) -> None:
    """Exactly one of fixed_text/user_transcript is set. Streams sentences to TTS as they're
    ready, forwards resulting audio to outbound_q. Mutates handle.flushes_sent live so a canceller
    can drain the right number of pending TTS 'final' events afterward. On natural completion
    pushes {"type":"speaking_done","data":{...}} to events_q - pushes nothing if cancelled, since
    the canceller already knows what happened and drives whatever comes next itself."""
    llm_done = asyncio.Event()
    # Set as soon as every speech_chunk for this turn has been flushed to TTS - strictly earlier
    # than llm_done, which also waits on the post-reply classification API call (see
    # app/llm/claude_client.py's next_turn_stream() docstring). forward() below waits on THIS, not
    # llm_done, to decide it can stop listening for more TTS 'final' events - using llm_done there
    # was a real bug: confirmed via logs on two live attrition calls, if every expected 'final' had
    # already arrived before classification finished, forward() had no future event left to
    # re-check its exit condition on, and hung until the 20s safety-net timeout forced recovery.
    speech_flushed = asyncio.Event()
    final_data = {"tool_name": None, "answer_summary": None, "reply_text": "", "retrieved_context": []}
    total_audio_bytes = 0

    async def _send_and_flush(text: str) -> None:
        # Timed individually so a future hang (see _SPEAKING_TIMEOUT_S) shows up in logs as a slow
        # send_text vs. a slow flush, rather than just "the reply never finished." INFO, not DEBUG
        # - a real call already hit a 20s stuck-speaking recovery with nothing in the logs showing
        # which flush the matching 'final' never came back for, because this line was DEBUG-level
        # while the app runs at INFO (same class of gap main.py's own logging comment describes).
        t0 = time.monotonic()
        await tts.send_text(_normalize_for_tts(text))
        await tts.flush()
        handle.flushes_sent += 1
        logger.info("send_text+flush #%d took %.2fs", handle.flushes_sent, time.monotonic() - t0)

    async def drive():
        nonlocal final_data
        if fixed_text is not None:
            await outbound_q.put(("json", {"type": "bot_text_chunk", "text": fixed_text}))
            await _send_and_flush(fixed_text)
            final_data["reply_text"] = fixed_text
            speech_flushed.set()
        else:
            async for event in session.handle_user_turn_stream(user_transcript):
                if event["type"] == "speech_chunk":
                    await outbound_q.put(("json", {"type": "bot_text_chunk", "text": event["text"]}))
                    await _send_and_flush(event["text"])
                elif event["type"] == "speech_done":
                    speech_flushed.set()
                else:
                    final_data = event["data"]
            speech_flushed.set()  # belt and suspenders - covers any path that reaches "final" without a "speech_done" (e.g. session.ended's early-return)
        llm_done.set()
        logger.info(
            "drive() done (flushes_sent=%d, completions_seen=%d)",
            handle.flushes_sent, handle.completions_seen,
        )

    async def forward():
        # Deliberately a plain `async for`, not asyncio.wait_for()-wrapped polling - see the
        # module docstring for why (this bit us once already with a shared receive loop).
        nonlocal total_audio_bytes
        first_byte_logged = False
        async for tts_event in tts.events():
            if tts_event["kind"] == "audio":
                if not first_byte_logged:
                    first_byte_logged = True
                    logger.info(
                        "[latency] %s: turn start -> first audio byte: %.2fs",
                        handle.kind, time.monotonic() - handle.started_at,
                    )
                total_audio_bytes += len(tts_event["data"])
                await outbound_q.put(("bytes", tts_event["data"]))
            elif tts_event["kind"] == "final":
                handle.completions_seen += 1
                logger.info(
                    "TTS final #%d (flushes_sent=%d, speech_flushed=%s)",
                    handle.completions_seen, handle.flushes_sent, speech_flushed.is_set(),
                )
                # speech_flushed, not llm_done - see the comment where speech_flushed is declared.
                if speech_flushed.is_set() and handle.completions_seen >= handle.flushes_sent:
                    return
            elif tts_event["kind"] == "error":
                logger.warning("Sarvam TTS error: %s", tts_event["raw"])
                return
            else:
                # Not currently expected on the happy path - logged rather than silently dropped,
                # since an unrecognized message here would otherwise look identical to "forward()
                # is just waiting for the next event" in the logs, the exact ambiguity that made a
                # real 20s stuck-speaking recovery hard to diagnose after the fact.
                logger.warning("Unexpected Sarvam TTS event kind=%r: %s", tts_event["kind"], tts_event.get("raw"))

    try:
        await asyncio.wait_for(asyncio.gather(drive(), forward()), timeout=_SPEAKING_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.error(
            "_run_speaking timed out after %.0fs (kind=%s) - forcing recovery to LISTENING instead "
            "of leaving the call stuck silent for its remainder (see _SPEAKING_TIMEOUT_S comment)",
            _SPEAKING_TIMEOUT_S, handle.kind,
        )
    else:
        # Bytes finish generating/forwarding much faster than the audio actually takes to play out
        # on the rider's end (confirmed empirically - see scratchpad barge-in test notes) - if we
        # declared "done speaking" the moment bytes stop flowing, the barge-in window would close
        # seconds before the rider could actually still be hearing the bot. Hold the SPEAKING state
        # open for the estimated real-time playback duration (raw PCM16 mono: 2 bytes/sample).
        estimated_playback_s = total_audio_bytes / (tts.sample_rate * 2)
        elapsed_s = time.monotonic() - handle.started_at
        remaining_s = estimated_playback_s - elapsed_s
        logger.debug(
            "kind=%s total_audio_bytes=%d estimated_playback_s=%.2f elapsed_s=%.2f remaining_s=%.2f",
            handle.kind, total_audio_bytes, estimated_playback_s, elapsed_s, remaining_s,
        )
        if remaining_s > 0:
            await asyncio.sleep(remaining_s)

    await events_q.put({"type": "speaking_done", "data": final_data})


async def _stt_reader_task(stt: SarvamSTTStream, events_q: asyncio.Queue) -> None:
    """Sole reader of stt.events() for the whole call - translates Sarvam's raw message shapes
    into the orchestrator's simpler event vocabulary."""
    async for raw_event in stt.events():
        etype = raw_event.get("type")
        if etype == "events":
            signal = raw_event.get("data", {}).get("signal_type")
            if signal == "START_SPEECH":
                await events_q.put({"type": "vad_start"})
            elif signal == "END_SPEECH":
                await events_q.put({"type": "vad_end"})
        elif etype == "data":
            await events_q.put({"type": "transcript", "text": raw_event.get("data", {}).get("transcript", "")})
        elif etype == "error":
            await events_q.put({"type": "stt_error", "raw": raw_event})


async def _stt_writer_task(stt: SarvamSTTStream, stt_send_q: asyncio.Queue) -> None:
    """Sole sender to the STT WS for the whole call - drains audio chunks and flush requests in
    the order they were queued, regardless of which part of the orchestrator produced them."""
    while True:
        item = await stt_send_q.get()
        if item["kind"] == "audio":
            await stt.send_audio_chunk(item["data"])
        elif item["kind"] == "flush":
            await stt.flush()


async def _browser_reader_task(
    websocket: WebSocket, stt_send_q: asyncio.Queue, events_q: asyncio.Queue
) -> None:
    """Sole reader of the browser WS for the whole call. The rider's mic streams continuously now
    (no more click-to-talk/turn_end signal), so this just forwards every audio frame to STT."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            await events_q.put({"type": "disconnected"})
            return
        audio_bytes = message.get("bytes")
        if audio_bytes is not None:
            await stt_send_q.put({"kind": "audio", "data": audio_bytes})


async def _browser_writer_task(websocket: WebSocket, outbound_q: asyncio.Queue) -> None:
    """Sole sender to the browser WS for the whole call - both control-message JSON and TTS audio
    bytes now come from multiple concurrent producers (the orchestrator, _run_speaking), so
    everything is funneled through this one queue to avoid concurrent writes to the same socket."""
    while True:
        kind, payload = await outbound_q.get()
        if kind == "json":
            await websocket.send_json(payload)
        elif kind == "bytes":
            await websocket.send_bytes(payload)


async def _tts_keepalive_task(tts: SarvamTTSStream) -> None:
    while True:
        await asyncio.sleep(_TTS_PING_INTERVAL)
        await tts.ping()
