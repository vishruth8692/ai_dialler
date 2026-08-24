import logging
from typing import AsyncIterator, Optional

import anthropic

from app.config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from app.llm.prompts import (
    CLASSIFY_TOOL,
    SYSTEM_PROMPT_STREAMING_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    TOOLS,
)
from app.llm.sentence_chunker import SentenceChunker

logger = logging.getLogger(__name__)

# Module-level singletons so every call session reuses the same underlying HTTP connection pool
# instead of paying a fresh TCP+TLS handshake on every new CallSession().
_shared_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
_shared_async_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


class ClaudeClient:
    def __init__(self):
        self._client = _shared_client
        self._async_client = _shared_async_client

    def _render_prompt(
        self,
        template: str,
        current_question: Optional[dict],
        retrieved_context: list[dict],
        next_question: Optional[dict],
    ) -> str:
        if current_question:
            current_q_text = (
                f'"{current_question["question"]}" '
                f'(reference/expected answer style: {current_question["answer"]})'
            )
        else:
            current_q_text = "None - all scripted questions have already been asked."

        context_block = (
            "\n".join(f'- Q: {c["question"]}\n  A: {c["answer"]}' for c in retrieved_context)
            or "(no relevant context found for this utterance)"
        )

        if next_question is None:
            remaining_note = (
                "This IS the last scripted question. If they answer it, do NOT ask another "
                "question - close the call warmly instead."
            )
        else:
            # Giving the literal next-question text (not just a remaining-count) is deliberate -
            # confirmed on a real call that without it, the model sometimes invented a plausible
            # but entirely different question instead of the actual scripted one.
            remaining_note = (
                "This is NOT the last scripted question. If they answer it, briefly acknowledge "
                f'and then ask this exact next question (verbatim, or a very close natural '
                f'paraphrase - do not invent a different question): "{next_question["question"]}". '
                "Do NOT close the call yet."
            )

        return template.format(
            current_question=current_q_text,
            remaining_note=remaining_note,
            context_block=context_block,
        )

    def _build_system_prompt(
        self,
        current_question: Optional[dict],
        retrieved_context: list[dict],
        next_question: Optional[dict],
    ) -> str:
        return self._render_prompt(
            SYSTEM_PROMPT_TEMPLATE, current_question, retrieved_context, next_question
        )

    async def generate_greeting(self, first_question: Optional[dict], language_hint: str) -> str:
        first_q_text = first_question["question"] if first_question else "(no questions loaded)"
        system = (
            "You are a friendly voice assistant calling an Indian delivery partner (rider) to "
            f"collect quick feedback. Greet them warmly, preferably in {language_hint} (switch "
            "language if they reply in a different one). If greeting in Hindi, Kannada, Telugu, "
            "Tamil, or Marathi, use the natural casual code-mixed register delivery/gig partners "
            "actually speak (Hinglish, Kanglish, Tenglish, Tanglish, Marathi-English mix), not "
            "formal literary language - keep words like delivery, order, app in English. You are "
            "voiced by a MALE voice - in Hindi/Marathi always use masculine self-referential "
            "grammar (e.g. 'bol raha hoon' NOT 'bol rahi hoon'), never feminine verb forms. Clearly "
            "mention this call is from the Zepto support team, then go straight into the first "
            "question naturally in the same message - skip any preamble about 'having a few quick "
            "questions', get straight to it. Keep the WHOLE greeting to 1 short sentence plus the "
            "question - be fast, this is a phone call and every extra word costs the rider's time.\n\n"
            f"First question to ask: {first_q_text}"
        )
        response = await self._async_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": "Begin the call."}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def next_turn(
        self,
        history: list[dict],
        current_question: Optional[dict],
        retrieved_context: list[dict],
        next_question: Optional[dict] = None,
    ) -> dict:
        system = self._build_system_prompt(current_question, retrieved_context, next_question)

        response = self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            system=system,
            tools=TOOLS,
            tool_choice={"type": "any"},
            messages=history,
        )

        for block in response.content:
            if block.type == "tool_use":
                return {
                    "tool_name": block.name,
                    "reply_text": block.input.get("reply_text", ""),
                    "answer_summary": block.input.get("answer_summary"),
                }

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        return {
            "tool_name": "answer_from_context",
            "reply_text": text or "Sorry, could you say that again?",
            "answer_summary": None,
        }

    async def next_turn_stream(
        self,
        history: list[dict],
        current_question: Optional[dict],
        retrieved_context: list[dict],
        next_question: Optional[dict] = None,
    ) -> AsyncIterator[dict]:
        """Streaming counterpart to next_turn(). Yields {"type": "speech_chunk", "text": ...} as
        sentences complete, followed by exactly one {"type": "final", "tool_name", "answer_summary",
        "reply_text"} once the model finishes speaking and a follow-up classification call (see
        _classify_turn()) completes.

        Speech generation and action classification are deliberately two separate API calls.
        The original design asked one streaming call to both speak AND reliably self-report a
        trailing JSON control block (a delimiter + JSON blob) - confirmed on a real production
        call that Claude sometimes genuinely omits the block entirely (not just a whitespace/
        formatting mismatch), which silently defaulted to answer_from_context and meant the
        survey quietly stopped recording answers - a correctness bug, not just a latency one.
        _classify_turn() reuses the same forced tool_choice mechanism next_turn() already relies
        on without this problem. Because it only runs after generation finishes, and
        app/streaming/call_ws_handler.py holds the "speaking" state open for the estimated
        real-time audio playback duration afterward, this call overlaps with that wait in the
        common case rather than adding fully to perceived latency.
        """
        system = self._render_prompt(
            SYSTEM_PROMPT_STREAMING_TEMPLATE, current_question, retrieved_context, next_question
        )

        chunker = SentenceChunker()
        speech_parts: list[str] = []

        async with self._async_client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=500,
            system=system,
            messages=history,
        ) as stream:
            async for delta in stream.text_stream:
                for chunk in chunker.feed(delta):
                    speech_parts.append(chunk)
                    yield {"type": "speech_chunk", "text": chunk}

        for chunk in chunker.feed("", final=True):
            speech_parts.append(chunk)
            yield {"type": "speech_chunk", "text": chunk}

        # Marks "no more speech_chunks are coming" separately from "the turn is fully done" -
        # app/streaming/call_ws_handler.py's forward() needs the EARLIER signal to decide it can
        # stop waiting on TTS, since _classify_turn() below is a real API call that can easily
        # take longer than Sarvam takes to finish synthesizing what's already been flushed. Without
        # this, forward() only re-checks its exit condition when a NEW TTS event arrives - if every
        # expected 'final' had already arrived before classification finishes, nothing would ever
        # wake it up again, and the turn would hang until the 20s safety-net timeout. Confirmed on
        # two real calls (see call_ws_handler.py's _SPEAKING_TIMEOUT_S comment).
        yield {"type": "speech_done"}

        reply_text = " ".join(speech_parts).strip()
        classification = await self._classify_turn(history, current_question, reply_text)

        yield {
            "type": "final",
            "tool_name": classification["tool_name"],
            "answer_summary": classification["answer_summary"],
            "reply_text": reply_text,
        }

    async def _classify_turn(
        self,
        history: list[dict],
        current_question: Optional[dict],
        reply_text: str,
    ) -> dict:
        """Forced tool-call classification of a turn whose spoken reply has ALREADY been
        generated and spoken - see next_turn_stream()'s docstring for why this replaced the
        delimiter-based approach. Falls back to answer_from_context/None only if the API call
        itself fails (network error etc.) - tool_choice being forced means Claude can't decline
        to call it the way it could omit a trailing delimiter."""
        current_q_text = f'"{current_question["question"]}"' if current_question else "None"
        system = (
            "A voice assistant on a phone call with a delivery partner just spoke the following "
            f"reply, in response to the partner's last message:\n\n{reply_text}\n\n"
            f"Current scripted question at the time: {current_q_text}\n\n"
            "If the reply explains something the partner didn't understand ABOUT the current "
            "scripted question itself (e.g. they said they don't understand the rate card, and "
            "the reply now explains it), that is still record_answer - a \"no\"/\"not clear\" IS "
            "a valid answer to a comprehension question, and the explanation is a courtesy "
            "follow-up, not a new topic. Only use answer_from_context when the partner's message "
            "was about something UNRELATED to the current scripted question.\n\n"
            "Classify this turn using the classify_turn tool."
        )
        try:
            response = await self._async_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=200,
                system=system,
                tools=CLASSIFY_TOOL,
                tool_choice={"type": "tool", "name": "classify_turn"},
                messages=history,
            )
            for block in response.content:
                if block.type == "tool_use":
                    return {
                        "tool_name": block.input.get("action", "answer_from_context"),
                        "answer_summary": block.input.get("answer_summary"),
                    }
        except Exception:
            logger.exception("Turn classification call failed")
        return {"tool_name": "answer_from_context", "answer_summary": None}
