"""Spike: is it safe to repeatedly cancel a ClaudeClient.next_turn_stream() call mid-generation?
Watches for ResourceWarnings and growing open-file-descriptor count (a practical, implementation-
agnostic proxy for leaked HTTP connections) across many cancel-mid-stream iterations using the SAME
shared AsyncAnthropic client the real orchestrator will use.

Usage: python scripts/test_claude_stream_cancel.py [n_iterations]
"""

import asyncio
import contextlib
import gc
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.claude_client import ClaudeClient
from app.rag import qa_store


def open_fd_count() -> int:
    try:
        out = subprocess.run(
            ["lsof", "-p", str(os.getpid())], capture_output=True, text=True, timeout=5
        )
        return len(out.stdout.splitlines())
    except Exception:
        return -1


async def run_one_iteration(client: ClaudeClient, history, current_question, retrieved) -> list:
    events_seen = []

    async def consume():
        async for event in client.next_turn_stream(
            history=history,
            current_question=current_question,
            retrieved_context=retrieved,
            next_question=None,
        ):
            events_seen.append(event["type"])

    task = asyncio.create_task(consume())
    # Cancel at a random early point - sometimes almost immediately, sometimes after a chunk or two.
    await asyncio.sleep(random.uniform(0.05, 1.0))
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
    return events_seen


async def main():
    warnings.simplefilter("always", ResourceWarning)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    pairs = qa_store.get_survey_questions()
    if not pairs:
        print("No survey questions loaded - upload a CSV at /qa first.")
        return

    client = ClaudeClient()  # shared module-level AsyncAnthropic singleton, same as production
    history = [{"role": "user", "content": "It was fine, just a bit late because of traffic."}]

    fd_start = open_fd_count()
    print(f"Starting open FD count: {fd_start}\nRunning {n} cancel-mid-stream iterations...\n")

    for i in range(n):
        await run_one_iteration(client, list(history), pairs[0], pairs[:2])
        if (i + 1) % 10 == 0:
            gc.collect()
            print(f"  iteration {i + 1}/{n}: open FDs = {open_fd_count()}")

    gc.collect()
    await asyncio.sleep(0.5)  # let any deferred cleanup/warnings surface
    fd_end = open_fd_count()

    print(f"\nFinal open FD count: {fd_end} (started at {fd_start})")
    if fd_end - fd_start > 10:
        print("WARNING: FD count grew significantly - possible connection leak.")
    else:
        print("PASS: FD count stayed roughly stable across repeated cancellations.")


if __name__ == "__main__":
    asyncio.run(main())
