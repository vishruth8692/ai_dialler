"""Per-call conversation state machine.

Claude drives the flow via forced tool calls (see app/llm/prompts.py) so the script position,
answer capture, and call termination stay deterministic in code rather than depending on the
model formatting free text correctly. handle_user_turn_stream() uses a different Claude-side
mechanism (streamed plain text + trailing control block, see claude_client.next_turn_stream) but
lands on the exact same tool_name/answer_summary shape, so both turn methods share one state-
transition helper (_apply_tool_result) and can never drift apart.
"""

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from app.llm.claude_client import ClaudeClient
from app.rag import qa_store


@dataclass
class CallSession:
    language_hint: str = "hindi"
    questions: list = field(default_factory=list)
    current_index: int = 0
    collected_answers: list = field(default_factory=list)
    history: list = field(default_factory=list)
    ended: bool = False
    client: ClaudeClient = field(default_factory=ClaudeClient)

    def __post_init__(self):
        if not self.questions:
            # Only type=="survey" rows are asked aloud, in order - "faq" rows stay searchable via
            # qa_store.retrieve() for grounding side questions but are never asked as questions.
            self.questions = qa_store.get_survey_questions()

    @property
    def current_question(self) -> Optional[dict]:
        if self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    @property
    def next_question(self) -> Optional[dict]:
        """The question that should be asked after the current one is answered, or None if the
        current one is the last. Passed to Claude so it asks the NEXT question's actual wording
        instead of inventing its own - confirmed on a real call that without this, the model
        sometimes made up a plausible-sounding but entirely different question (e.g. asking about
        "overall app experience" instead of the scripted ratecard question) since it only knew a
        remaining-count, never the real text."""
        if self.current_index + 1 < len(self.questions):
            return self.questions[self.current_index + 1]
        return None

    def _retrieve_context(self, user_text: str) -> list[dict]:
        """Merges retrieval on what the rider said with retrieval on the active survey question's
        own text. A short reply like "no, not clear" has no lexical/semantic overlap with e.g. a
        ratecard FAQ entry on its own - but it IS a close match to the survey question itself
        ("Did you understand the ratecard...") - so this surfaces relevant FAQ content even when
        the rider's answer is too terse for retrieve(user_text) alone to find it.

        Each source gets its own reserved slots (2 each) rather than pooling-then-truncating - a
        first-come-first-served merge let a generic reply's own (irrelevant) top matches fill every
        slot before the current-question query ever got a look in, even though the real answer
        ranked #1 there - confirmed as the actual cause of the bot failing to explain the ratecard
        on a real call despite the right FAQ entry existing.

        Filters to type=="faq" only - a survey row's "answer" field is an internal scripting note
        (e.g. "Confirm clarity on ratecard/earnings understanding"), not real informative content,
        so including it here just wastes a context slot (or worse, the survey question matching
        itself crowds out the actual FAQ answer - also confirmed happening before this filter).

        Both queries are embedded in a single batched qa_store.retrieve_multi() call rather than
        two separate retrieve() calls - confirmed on a real production call that each embed call
        cost 2-4s on Railway's CPU-only inference, so two separate calls were adding 4-8s of pure
        embedding latency to every reply."""
        queries = [user_text]
        if self.current_question:
            queries.append(self.current_question["question"])
        results_per_query = qa_store.retrieve_multi(queries, top_k=5)

        seen: set[str] = set()
        combined: list[dict] = []
        for results in results_per_query:
            added = 0
            for item in results:
                if item["type"] != "faq" or item["question"] in seen:
                    continue
                seen.add(item["question"])
                combined.append(item)
                added += 1
                if added >= 2:
                    break
        return combined

    def pop_pending_user_turn(self) -> Optional[str]:
        """If the last history entry is a user message with no matching assistant reply - i.e. a
        handle_user_turn_stream() call was cancelled mid-flight before it could finish (the
        genuine-barge-in case: the bot's reply got cut off) - pop and return its text so the
        caller can merge it with whatever the rider says next into one turn. Returns None if
        there's nothing dangling."""
        if self.history and self.history[-1]["role"] == "user":
            return self.history.pop()["content"]
        return None

    def greeting(self) -> str:
        text = self.client.generate_greeting(
            first_question=self.current_question, language_hint=self.language_hint
        )
        self.history.append({"role": "assistant", "content": text})
        return text

    def _apply_tool_result(self, tool_name: Optional[str], answer_summary: Optional[str], user_text: str) -> None:
        """Mutates collected_answers/current_index/ended based on the LLM's decision. Shared by
        both handle_user_turn and handle_user_turn_stream so their state machines can't drift."""
        if tool_name == "record_answer":
            if self.current_question:
                self.collected_answers.append(
                    {
                        "question": self.current_question["question"],
                        "answer": answer_summary or user_text,
                    }
                )
                self.current_index += 1
            if self.current_index >= len(self.questions):
                self.ended = True
        elif tool_name == "end_call":
            self.ended = True
        # answer_from_context: no state change - just replies and re-asks current question

    def handle_user_turn(self, user_text: str) -> dict:
        """Returns {"user_text", "tool_name", "answer_summary", "reply_text", "retrieved_context"} -
        the full turn, not just the reply, so callers can show what the bot heard and decided."""
        if self.ended:
            return {
                "user_text": user_text,
                "tool_name": None,
                "answer_summary": None,
                "reply_text": "The call has already ended.",
                "retrieved_context": [],
            }

        self.history.append({"role": "user", "content": user_text})

        retrieved = self._retrieve_context(user_text)

        result = self.client.next_turn(
            history=self.history,
            current_question=self.current_question,
            retrieved_context=retrieved,
            next_question=self.next_question,
        )

        tool_name = result["tool_name"]
        reply_text = result["reply_text"]
        answer_summary = result.get("answer_summary")

        self._apply_tool_result(tool_name, answer_summary, user_text)

        self.history.append({"role": "assistant", "content": reply_text})

        return {
            "user_text": user_text,
            "tool_name": tool_name,
            "answer_summary": answer_summary,
            "reply_text": reply_text,
            "retrieved_context": retrieved,
        }

    async def handle_user_turn_stream(self, user_text: str) -> AsyncIterator[dict]:
        """Streaming counterpart to handle_user_turn(). Yields the same
        {"type": "speech_chunk", "text": ...} events as ClaudeClient.next_turn_stream (pass these
        straight to TTS as they arrive), then one closing event:
        {"type": "final", "data": {...}} where data has the exact same shape handle_user_turn()
        returns synchronously."""
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

        # qa_store.retrieve() is a synchronous embedding-model + Chroma call - run off the event
        # loop so it doesn't block other concurrent work (e.g. audio forwarding) during a turn.
        retrieved = await asyncio.to_thread(self._retrieve_context, user_text)

        tool_name = "answer_from_context"
        answer_summary = None
        reply_text = ""

        async for event in self.client.next_turn_stream(
            history=self.history,
            current_question=self.current_question,
            retrieved_context=retrieved,
            next_question=self.next_question,
        ):
            if event["type"] == "speech_chunk":
                yield event
            else:
                tool_name = event["tool_name"]
                answer_summary = event["answer_summary"]
                reply_text = event["reply_text"]

        self._apply_tool_result(tool_name, answer_summary, user_text)

        self.history.append({"role": "assistant", "content": reply_text})

        yield {
            "type": "final",
            "data": {
                "user_text": user_text,
                "tool_name": tool_name,
                "answer_summary": answer_summary,
                "reply_text": reply_text,
                "retrieved_context": retrieved,
            },
        }
