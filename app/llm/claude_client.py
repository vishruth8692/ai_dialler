import json
import logging
from typing import AsyncIterator, Optional

import anthropic

from app.config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from app.llm.prompts import CONTROL_DELIMITER, SYSTEM_PROMPT_STREAMING_TEMPLATE, SYSTEM_PROMPT_TEMPLATE, TOOLS
from app.llm.sentence_chunker import SentenceChunker

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"record_answer", "answer_from_context", "end_call"}

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

    def generate_greeting(self, first_question: Optional[dict], language_hint: str) -> str:
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
        response = self._client.messages.create(
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
        "reply_text"} once the model finishes and the trailing control block has been parsed.

        Claude isn't forced through tool-calling here (tool-use JSON doesn't stream cleanly into
        sentence-chunked TTS) - instead it's instructed to emit plain speech, then a delimiter, then
        a small JSON control block (see SYSTEM_PROMPT_STREAMING_TEMPLATE). If the delimiter is never
        emitted or the JSON fails to parse, this falls back to answer_from_context/None, same
        graceful-degradation philosophy as next_turn()'s no-tool-call fallback above.
        """
        system = self._render_prompt(
            SYSTEM_PROMPT_STREAMING_TEMPLATE, current_question, retrieved_context, next_question
        )

        chunker = SentenceChunker()
        speech_parts: list[str] = []
        buf = ""
        control_raw = ""
        delimiter_found = False
        hold_back = len(CONTROL_DELIMITER) - 1

        async with self._async_client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=500,
            system=system,
            messages=history,
        ) as stream:
            async for delta in stream.text_stream:
                if delimiter_found:
                    control_raw += delta
                    continue

                buf += delta
                idx = buf.find(CONTROL_DELIMITER)
                if idx != -1:
                    for chunk in chunker.feed(buf[:idx], final=True):
                        speech_parts.append(chunk)
                        yield {"type": "speech_chunk", "text": chunk}
                    control_raw = buf[idx + len(CONTROL_DELIMITER) :]
                    buf = ""
                    delimiter_found = True
                else:
                    # Hold back a tail long enough that a delimiter split across two deltas is
                    # never missed - only release text once we're sure it's not part of one.
                    safe_len = max(0, len(buf) - hold_back)
                    if safe_len:
                        for chunk in chunker.feed(buf[:safe_len]):
                            speech_parts.append(chunk)
                            yield {"type": "speech_chunk", "text": chunk}
                        buf = buf[safe_len:]

        if not delimiter_found:
            logger.warning("Claude stream ended without emitting %r - falling back", CONTROL_DELIMITER.strip())
            for chunk in chunker.feed(buf, final=True):
                speech_parts.append(chunk)
                yield {"type": "speech_chunk", "text": chunk}
            tool_name, answer_summary = "answer_from_context", None
        else:
            tool_name, answer_summary = "answer_from_context", None
            try:
                parsed = json.loads(control_raw.strip())
                action = parsed.get("action")
                if action in _VALID_ACTIONS:
                    tool_name = action
                    answer_summary = parsed.get("answer_summary")
                else:
                    logger.warning("Claude control block had invalid action %r - falling back", action)
            except (json.JSONDecodeError, AttributeError):
                logger.warning("Claude control block failed to parse: %r", control_raw)

        yield {
            "type": "final",
            "tool_name": tool_name,
            "answer_summary": answer_summary,
            "reply_text": " ".join(speech_parts).strip(),
        }
