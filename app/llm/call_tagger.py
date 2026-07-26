"""One-shot classification of a finished call into a fixed tag taxonomy plus a short summary, so
past calls can be scanned at a glance on /telephony without re-reading the full transcript. Runs
once per call after it ends - never on the hot path of a live turn."""

import logging

import anthropic

from app.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

TAGS = [
    "Positive experience",
    "Negative experience",
    "Ratecard confusion",
    "Store issue",
    "Payment or payout issue",
    "Account or ID issue",
    "App usability issue",
    "Call ended early",
    "No issues reported",
]

_SYSTEM = (
    "You classify a finished delivery-partner feedback call into zero or more tags from a fixed "
    "list, and write a one-sentence summary. Only use tags from the list, spelled exactly as "
    "given - never invent a new tag. Base this ONLY on the collected answers and transcript "
    "provided, do not guess or add outside information.\n\n"
    f"Tags: {', '.join(TAGS)}"
)


def classify_call(collected_answers: list[dict], transcript: list[dict]) -> dict:
    """Returns {"tags": [...], "summary": "..."}. Falls back to empty tags/summary on any failure -
    this is a nice-to-have annotation and must never block call-history logging."""
    transcript_text = "\n".join(f'{t["role"]}: {t["text"]}' for t in transcript) or "(empty)"
    answers_text = "\n".join(f'{a["question"]} -> {a["answer"]}' for a in collected_answers) or "(none recorded)"
    user_content = f"Collected answers:\n{answers_text}\n\nFull transcript:\n{transcript_text}"

    try:
        response = _client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            tools=[
                {
                    "name": "classify",
                    "description": "Report the tags and summary for this call.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "tags": {"type": "array", "items": {"type": "string", "enum": TAGS}},
                            "summary": {"type": "string"},
                        },
                        "required": ["tags", "summary"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "classify"},
        )
        for block in response.content:
            if block.type == "tool_use":
                return {
                    "tags": [t for t in block.input.get("tags", []) if t in TAGS],
                    "summary": block.input.get("summary", ""),
                }
    except Exception:
        logger.exception("Call tagging failed")
    return {"tags": [], "summary": ""}
