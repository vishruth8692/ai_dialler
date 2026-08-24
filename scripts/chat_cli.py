"""Text-only mock call for testing the conversation flow before wiring real audio/telephony.

Usage: python scripts/chat_cli.py
Type your replies as if you were the delivery partner (in any of the 5 supported languages).
Type 'quit' to exit early.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.call_session import CallSession
from app.rag import qa_store


def main():
    pairs = qa_store.get_survey_questions()
    if not pairs:
        print(
            "No survey questions loaded yet. Start the admin UI (`uvicorn app.main:app --reload`) "
            "and upload a CSV at http://localhost:8000/qa before running this."
        )
        return

    print(f"Loaded {len(pairs)} question(s). Starting mock call...\n")

    session = CallSession()
    print(f"Bot: {asyncio.run(session.greeting())}\n")

    while not session.ended:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            break

        turn = session.handle_user_turn(user_text)
        debug_bits = [f"tool={turn['tool_name']}"]
        if turn["answer_summary"]:
            debug_bits.append(f'captured="{turn["answer_summary"]}"')
        if turn["retrieved_context"]:
            ctx = ", ".join(f'"{c["question"]}"' for c in turn["retrieved_context"])
            debug_bits.append(f"context=[{ctx}]")
        print(f"  [{' | '.join(debug_bits)}]")
        print(f"\nBot: {turn['reply_text']}\n")

    if session.ended:
        print("--- Call ended ---")
    print("\nCollected answers:")
    for item in session.collected_answers:
        print(f"- Q: {item['question']}\n  A: {item['answer']}")


if __name__ == "__main__":
    main()
