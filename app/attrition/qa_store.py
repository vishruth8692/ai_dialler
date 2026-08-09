"""Flat Q&A knowledge base the attrition bot can draw on when a rider asks a side question it
would otherwise have no information for (see prompts.py's SIDE QUESTIONS guardrail). Deliberately
NOT retrieval/embedding-based like app/rag/qa_store.py - the Zepto bot needed RAG for a large
official FAQ bank; this is a much smaller, curated set for occasional off-topic questions, so every
saved pair is just included directly in the system prompt. Same simple JSON-file pattern as
app/telephony/call_history.py and app/attrition/stage_store.py."""

import csv
import io
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"question", "answer"}

_STORE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "attrition_qa.json"


def list_pairs() -> list[dict]:
    if not _STORE_PATH.exists():
        return []
    try:
        return json.loads(_STORE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to load attrition Q&A pairs - treating as empty")
        return []


def _save(pairs: list[dict]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(pairs, ensure_ascii=False, indent=2))


def add_pair(question: str, answer: str) -> None:
    pairs = list_pairs()
    pairs.append({"question": question.strip(), "answer": answer.strip()})
    _save(pairs)


def add_pairs(new_pairs: list[dict]) -> None:
    """Appends to whatever's already saved, rather than replacing it - unlike the Zepto bot's CSV
    upload (which resets the whole store), this is meant for topping up an existing curated set a
    few rows at a time, e.g. via repeated small CSV uploads alongside one-at-a-time adds."""
    pairs = list_pairs()
    pairs.extend(new_pairs)
    _save(pairs)


def delete_pair(index: int) -> None:
    pairs = list_pairs()
    if 0 <= index < len(pairs):
        pairs.pop(index)
        _save(pairs)


def parse_csv(file_content: bytes) -> list[dict]:
    """Expected columns: question, answer (required). Header matching is case-insensitive and
    tolerates surrounding whitespace - same convention as app/rag/ingest.py's CSV parser, just
    without that one's language/type columns (not meaningful here)."""
    text = file_content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV appears to be empty or has no header row.")

    header_map = {name.strip().lower(): name for name in reader.fieldnames}
    missing = REQUIRED_COLUMNS - set(header_map.keys())
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")

    pairs = []
    for row in reader:
        question = (row.get(header_map["question"]) or "").strip()
        answer = (row.get(header_map["answer"]) or "").strip()
        if not question or not answer:
            continue
        pairs.append({"question": question, "answer": answer})

    if not pairs:
        raise ValueError("No valid question/answer rows found in the CSV.")
    return pairs
