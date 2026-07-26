"""Measures per-stage latency (STT, Claude, TTS) for a single conversation turn - the actual
functions the live app calls, not a synthetic test.

Usage: python scripts/benchmark_latency.py [n_runs]
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.call_session import CallSession
from app.rag import qa_store
from app.speech import stt_sarvam, tts_sarvam

SAMPLE_REPLIES = [
    "It was good, no issues today.",
    "Bike kharab ho gaya tha, isliye thoda late ho gaya.",
    "Payment thoda late aaya tha last time.",
]


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    if not qa_store.get_survey_questions():
        print("No survey questions loaded - upload a CSV at /qa first.")
        return

    stt_times, claude_times, tts_times, total_times = [], [], [], []

    for i in range(n_runs):
        text = SAMPLE_REPLIES[i % len(SAMPLE_REPLIES)]
        print(f'\nRun {i + 1}/{n_runs}: "{text}"')

        audio_bytes, _ = timed(tts_sarvam.synthesize, text, "english")

        session = CallSession()
        session.greeting()  # one-time call-start cost, excluded from per-turn timing

        t0 = time.perf_counter()

        stt_result, stt_elapsed = timed(
            stt_sarvam.transcribe_bytes, audio_bytes, "sample.wav", "audio/wav"
        )
        transcript = stt_result["transcript"]
        language = stt_result["language_code"] or "english"

        turn, claude_elapsed = timed(session.handle_user_turn, transcript)

        _, tts_elapsed = timed(tts_sarvam.synthesize, turn["reply_text"], language)

        total_elapsed = time.perf_counter() - t0

        print(
            f"  STT: {stt_elapsed:.2f}s | Claude: {claude_elapsed:.2f}s | "
            f"TTS: {tts_elapsed:.2f}s | Total: {total_elapsed:.2f}s"
        )

        stt_times.append(stt_elapsed)
        claude_times.append(claude_elapsed)
        tts_times.append(tts_elapsed)
        total_times.append(total_elapsed)

    def summarize(label, times):
        print(f"{label}: avg={statistics.mean(times):.2f}s  min={min(times):.2f}s  max={max(times):.2f}s")

    print("\n--- Summary ---")
    summarize("STT   ", stt_times)
    summarize("Claude", claude_times)
    summarize("TTS   ", tts_times)
    summarize("TOTAL ", total_times)


if __name__ == "__main__":
    main()
