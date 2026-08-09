"""Per-call state machine for the rider attrition calling bot (see ATTRITION_VOICEBOT_KB.md and
app/attrition/prompts.py). Structurally different from app/call_session.py's CallSession: instead
of a fixed list of scripted questions, this is a branching stage machine (identity check -> opening
pitch -> status gate -> open question -> probe -> last straw -> grievance, each possibly ending the
call inline), with a safety hard-stop that can interrupt any stage.

Duck-typed to the exact interface app/streaming/call_ws_handler.py's _orchestrate() already expects
(.greeting(), .handle_user_turn_stream(), .pop_pending_user_turn(), .ended, .collected_answers) - so
it reuses that entire barge-in/pause-grace/speaking-timeout state machine completely unmodified.
"""

import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from app.attrition import prompts
from app.config import ATTRITION_SAFETY_HELPLINE, CLAUDE_MODEL
from app.llm.claude_client import _shared_async_client, _shared_client
from app.llm.sentence_chunker import SentenceChunker

logger = logging.getLogger(__name__)

# Stages that end the call once reached, with the closing remark folded into that same turn's
# reply (see prompts.py's GRIEVANCE instructions) - mirrors the Zepto bot's proven "close in the
# same reply as the last answer" pattern rather than needing one more empty turn.
#
# still_working/temporary_break are deliberately NOT here, unlike an earlier version of this code -
# STATUS_GATE's instructions ask the model to ask a follow-up AND close in the same reply, but
# confirmed on a real call that when it only asks the follow-up, treating the signal as terminal
# hung up on the rider before they could answer their own question. See awaiting_close_ack below.
_TERMINAL_SIGNALS = {"grievance_given"}

_STAGE_TRANSITIONS = {
    prompts.GREETING: {"ready_to_continue": prompts.STATUS_GATE},
    prompts.STATUS_GATE: {
        "stopped": prompts.OPEN_QUESTION,
        "never_started": prompts.OPEN_QUESTION,
    },
    prompts.OPEN_QUESTION: {"reason_given": prompts.PROBE},
    prompts.PROBE: {"probe_answered": prompts.LAST_STRAW, "vague_reason": prompts.LAST_STRAW},
    prompts.LAST_STRAW: {"last_straw_given": prompts.GRIEVANCE},
}


def missing_attrition_settings() -> list[str]:
    """Required before a real attrition call can be placed - the safety hard-stop (§4.7) has
    nothing concrete to give a distressed rider without ATTRITION_SAFETY_HELPLINE. Hard-blocked,
    not a soft warning, since a confirmed test showed the model will NOT reliably substitute
    something else on its own for this specific disclosure (unlike ordinary side-questions, which
    it reliably redirects to the Zepto app's support-ticket flow without needing any config)."""
    return [] if ATTRITION_SAFETY_HELPLINE else ["ATTRITION_SAFETY_HELPLINE"]


def is_attrition_configured() -> bool:
    return not missing_attrition_settings()


@dataclass
class AttritionCallSession:
    dial_record: dict = field(default_factory=dict)  # rider_name, rider_code, city, store_name, preferred_language
    # Read by app/streaming/exotel_ws_adapter.py the same way as CallSession.language_hint, to pick
    # the TTS voice's language.
    language_hint: str = "hindi"
    stage: str = prompts.GREETING
    history: list = field(default_factory=list)
    ended: bool = False
    # Shape-compatible with CallSession.collected_answers so the shared live-monitor UI (which
    # renders {"question", "answer"} pairs) works unmodified for this call type too. The full §8
    # structured record is built separately, offline, by app/llm/attrition_classifier.py.
    collected_answers: list = field(default_factory=list)

    status_gate: Optional[str] = None
    safety_stop: bool = False
    opt_out: bool = False
    callback_requested: bool = False
    wrong_person: bool = False
    # Set when the rider says they're still working or on a break - the NEXT turn (their answer to
    # the "which store"/"when are you back" follow-up) is closed deterministically rather than
    # trusting the model to have already closed in the same reply as the follow-up. See
    # _apply_signal() and handle_user_turn_stream().
    awaiting_close_ack: bool = False

    @property
    def rider_name(self) -> str:
        return self.dial_record.get("rider_name") or ""

    def _system_prompt(self) -> str:
        return prompts.render_system_prompt(self.stage, self.rider_name, ATTRITION_SAFETY_HELPLINE)

    def greeting(self) -> str:
        response = _shared_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=200,
            system=self._system_prompt(),
            messages=[{"role": "user", "content": "Begin the call."}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        self.history.append({"role": "assistant", "content": text})
        return text

    def pop_pending_user_turn(self) -> Optional[str]:
        if self.history and self.history[-1]["role"] == "user":
            return self.history.pop()["content"]
        return None

    def _apply_signal(self, signal: str, detail: Optional[str], user_text: str) -> None:
        if signal == "safety_stop":
            self.safety_stop = True
            self.stage = prompts.SAFETY_STOP
            self.ended = True
            self.collected_answers.append(
                {"question": "Safety disclosure", "answer": detail or user_text}
            )
            return
        if signal == "opt_out":
            self.opt_out = True
            self.ended = True
            return
        if signal == "wrong_person":
            self.wrong_person = True
            self.ended = True
            return
        if signal == "busy_callback":
            self.callback_requested = True
            self.ended = True
            return
        if signal == "end_call":
            self.ended = True
            return
        if signal == "answered_side_question":
            return  # stage unchanged - the reply already answered it and re-asked the question

        if signal in ("still_working", "temporary_break"):
            self.status_gate = signal
            self.awaiting_close_ack = True
        elif signal in ("stopped", "never_started"):
            self.status_gate = signal

        if detail:
            self.collected_answers.append({"question": self.stage, "answer": detail})

        if signal in _TERMINAL_SIGNALS:
            self.ended = True
            return

        self.stage = _STAGE_TRANSITIONS.get(self.stage, {}).get(signal, self.stage)

    async def _classify_turn(self, base_system: str, reply_text: str) -> tuple[str, Optional[str]]:
        """Forced tool-call classification of a turn whose spoken reply has ALREADY been generated
        and spoken - same split "stream speech, then classify separately" pattern proven more
        reliable than a single call self-reporting structured state (see
        app/llm/claude_client.py's next_turn_stream() docstring for why)."""
        system = (
            f"{base_system}\n\nA reply has ALREADY been generated and spoken to the rider, in "
            f"response to their last message:\n\n{reply_text}\n\nClassify this turn using the "
            "advance tool."
        )
        try:
            response = await _shared_async_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=200,
                system=system,
                tools=prompts.ADVANCE_TOOL,
                tool_choice={"type": "tool", "name": "advance"},
                messages=self.history,
            )
            for block in response.content:
                if block.type == "tool_use":
                    return (
                        block.input.get("signal", "answered_side_question"),
                        block.input.get("detail"),
                    )
        except Exception:
            logger.exception("Attrition turn classification failed")
        return "answered_side_question", None

    async def _classify_turn_pre(self, base_system: str) -> tuple[str, Optional[str]]:
        """Classifies the rider's message BEFORE any reply has been generated - used only for the
        GREETING stage's transition (see handle_user_turn_stream), where the acknowledgment + next
        question are scripted verbatim rather than model-generated. Confirmed via two real calls
        that the model would not reliably avoid previewing "why did you stop" here otherwise, even
        with explicit instructions not to - classifying first lets the code, not the model, decide
        what gets said for the common case, while less common signals (wrong person, busy, a side
        question) still fall through to normal generation afterward."""
        system = f"{base_system}\n\nClassify the rider's last message using the advance tool."
        try:
            response = await _shared_async_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=200,
                system=system,
                tools=prompts.ADVANCE_TOOL,
                tool_choice={"type": "tool", "name": "advance"},
                messages=self.history,
            )
            for block in response.content:
                if block.type == "tool_use":
                    return (
                        block.input.get("signal", "answered_side_question"),
                        block.input.get("detail"),
                    )
        except Exception:
            logger.exception("Attrition pre-generation classification failed")
        return "answered_side_question", None

    async def handle_user_turn_stream(self, user_text: str) -> AsyncIterator[dict]:
        """Same shape as CallSession.handle_user_turn_stream(): yields {"type":"speech_chunk",...}
        as sentences complete, then one {"type":"final","data":{...}}."""
        if self.ended:
            yield {
                "type": "final",
                "data": {
                    "user_text": user_text,
                    "tool_name": None,
                    "answer_summary": None,
                    "reply_text": "The call has already ended.",
                    "retrieved_context": [],
                },
            }
            return

        self.history.append({"role": "user", "content": user_text})

        if self.awaiting_close_ack:
            # The rider just answered the still-working/on-a-break follow-up - close
            # deterministically here rather than trusting the model to have already closed in the
            # SAME reply as that follow-up (confirmed on a real call it doesn't reliably do both).
            reply_text = prompts.STATUS_GATE_CLOSE_LINE
            yield {"type": "speech_chunk", "text": reply_text}
            yield {"type": "speech_done"}
            self.collected_answers.append({"question": self.stage, "answer": user_text})
            self.ended = True
            self.history.append({"role": "assistant", "content": reply_text})
            yield {
                "type": "final",
                "data": {
                    "user_text": user_text,
                    "tool_name": self.status_gate,
                    "answer_summary": user_text,
                    "reply_text": reply_text,
                    "retrieved_context": [],
                },
            }
            return

        system = self._system_prompt()

        precomputed: Optional[tuple[str, Optional[str]]] = None
        if self.stage == prompts.GREETING:
            signal, detail = await self._classify_turn_pre(system)
            if signal == "ready_to_continue":
                reply_text = prompts.GREETING_TO_STATUS_GATE_LINE
                yield {"type": "speech_chunk", "text": reply_text}
                yield {"type": "speech_done"}
                self._apply_signal(signal, detail, user_text)
                self.history.append({"role": "assistant", "content": reply_text})
                yield {
                    "type": "final",
                    "data": {
                        "user_text": user_text,
                        "tool_name": signal,
                        "answer_summary": detail,
                        "reply_text": reply_text,
                        "retrieved_context": [],
                    },
                }
                return
            # Any other signal (wrong_person, busy_callback, safety_stop, opt_out,
            # answered_side_question, end_call) still needs a real spoken reply - fall through to
            # normal generation below, reusing this classification instead of running it twice.
            precomputed = (signal, detail)

        chunker = SentenceChunker()
        speech_parts: list[str] = []

        async with _shared_async_client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=300,
            system=system,
            messages=self.history,
        ) as stream:
            async for delta in stream.text_stream:
                for chunk in chunker.feed(delta):
                    speech_parts.append(chunk)
                    yield {"type": "speech_chunk", "text": chunk}

        for chunk in chunker.feed("", final=True):
            speech_parts.append(chunk)
            yield {"type": "speech_chunk", "text": chunk}

        reply_text = " ".join(speech_parts).strip()
        if not reply_text:
            # Confirmed via repeated testing: the model occasionally generates no text at all for a
            # turn (e.g. when it judges - rightly or wrongly - that the previous turn already asked
            # everything relevant). Silence here means dead air on a live call, not just a missing
            # log line - app/streaming/call_ws_handler.py only speaks what it's given, and its
            # speaking-timeout is a last-resort recovery, not something to rely on for every turn.
            # A safe generic acknowledgement is always a valid thing to say mid-call, so fall back
            # to one rather than risk silence.
            logger.warning("Attrition turn produced an empty reply (stage=%s) - using fallback", self.stage)
            reply_text = "Samajh gaya, main note kar raha hoon."
            yield {"type": "speech_chunk", "text": reply_text}

        # See app/llm/claude_client.py's next_turn_stream() for why this is yielded separately
        # from the eventual "final" event - closes a real 20s-stuck-speaking race confirmed on two
        # live attrition calls, caused by classification (a real API call) finishing after
        # forward() in call_ws_handler.py had already seen every TTS 'final' it was waiting for.
        yield {"type": "speech_done"}

        if precomputed is not None:
            signal, detail = precomputed
        else:
            signal, detail = await self._classify_turn(system, reply_text)

        if signal == "safety_stop" and ATTRITION_SAFETY_HELPLINE and ATTRITION_SAFETY_HELPLINE not in reply_text:
            # Confirmed via repeated testing (12 runs across three prompt phrasings) that the model
            # will not reliably say the helpline number in this specific turn - it consistently
            # stops after the concern question alone, even with an explicit "two sentences, both
            # required" instruction. This is the one place in the call where saying the fallback
            # phrase isn't good enough - guarantee it in code instead of continuing to hope the
            # prompt eventually lands.
            logger.warning("Safety-stop reply omitted the helpline - appending it")
            helpline_sentence = f"Aap turant is number pe call kar sakte hain: {ATTRITION_SAFETY_HELPLINE}."
            reply_text = f"{reply_text} {helpline_sentence}".strip()
            yield {"type": "speech_chunk", "text": helpline_sentence}

        self._apply_signal(signal, detail, user_text)

        self.history.append({"role": "assistant", "content": reply_text})

        yield {
            "type": "final",
            "data": {
                "user_text": user_text,
                "tool_name": signal,
                "answer_summary": detail,
                "reply_text": reply_text,
                "retrieved_context": [],
            },
        }
