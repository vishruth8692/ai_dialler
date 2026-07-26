"""Parses an uploaded Q&A CSV and loads it into the Chroma store.

Expected columns: question, answer (required), language, type (optional). Header matching is
case-insensitive and tolerates surrounding whitespace.

type is "survey" (asked aloud, in order, as the call script) or "faq" (searchable for grounding
side questions the rider asks, never asked aloud) - defaults to "survey" if the column is omitted
or a row leaves it blank, so existing CSVs without a type column keep working unchanged.
"""

import csv
import io

from app.rag import qa_store

REQUIRED_COLUMNS = {"question", "answer"}
VALID_TYPES = {"survey", "faq"}


def parse_csv(file_content: bytes) -> list[dict]:
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
        language_key = header_map.get("language")
        language = (row.get(language_key) or "").strip().lower() if language_key else ""
        type_key = header_map.get("type")
        row_type = (row.get(type_key) or "").strip().lower() if type_key else ""
        if row_type not in VALID_TYPES:
            row_type = "survey"
        if not question or not answer:
            continue
        pairs.append({"question": question, "answer": answer, "language": language, "type": row_type})

    if not pairs:
        raise ValueError("No valid question/answer rows found in the CSV.")
    return pairs


def ingest_csv(file_content: bytes) -> list[dict]:
    pairs = parse_csv(file_content)
    qa_store.reset_store()
    qa_store.add_qa_pairs(pairs)
    return pairs
