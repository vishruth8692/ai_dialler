"""Spike: is it safe to cancel a consumer of SarvamTTSStream.events() mid-utterance, drain the
stale in-flight audio, and reuse the SAME connection for the next utterance - or does leftover
audio from the cancelled utterance bleed into the next one?

Verification approach: synthesize utterance 1 (long enough to still be streaming when we cancel),
cancel after only 1-2 chunks, drain_pending(), then synthesize a short, distinctive utterance 2 on
the same connection. Feed the utterance-2 audio through Sarvam's REST STT and confirm the
transcript matches utterance 2 cleanly - if utterance 1's leftover audio bled in, the transcript
would be garbled or contain extra words.

Usage: python scripts/test_tts_cancel_reuse.py
"""

import asyncio
import contextlib
import io
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.speech import stt_sarvam
from app.streaming.sarvam_tts_ws import SarvamTTSStream

UTTERANCE_1 = (
    "This is a fairly long sentence specifically designed to keep streaming for a while so that "
    "we have time to cancel it partway through before it finishes generating all of its audio."
)
UTTERANCE_2 = "Delivery was smooth today."


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


async def main():
    sample_rate = 24000

    async with SarvamTTSStream(language="english", sample_rate=sample_rate) as tts:
        # --- Utterance 1: start, then cancel partway through ---
        await tts.send_text(UTTERANCE_1)
        await tts.flush()

        chunks_before_cancel = []

        async def consume_forever():
            # No self-imposed exit condition - relies entirely on external task.cancel(), exactly
            # how the real orchestrator will stop a SpeakingHandle's forward-audio loop.
            async for event in tts.events():
                if event["kind"] == "audio":
                    chunks_before_cancel.append(event["data"])

        task = asyncio.create_task(consume_forever())
        # Let it read a couple of chunks, then cancel it mid-await - this is the real scenario:
        # CancelledError raised while suspended inside `async for event in tts.events()`.
        while len(chunks_before_cancel) < 2:
            await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        print(f"Read {len(chunks_before_cancel)} audio chunks from utterance 1, then cancelled the "
              f"reading task mid-stream (real task.cancel(), not a voluntary return).")

        # --- Drain whatever's left of utterance 1 (1 flush was sent -> expect 1 final) ---
        await tts.drain_pending(expected_finals=1)
        print("Drained remaining utterance 1 audio.\n")

        # --- Utterance 2 on the SAME connection ---
        await tts.send_text(UTTERANCE_2)
        await tts.flush()

        utterance_2_pcm = b""
        async for event in tts.events():
            if event["kind"] == "audio":
                utterance_2_pcm += event["data"]
            elif event["kind"] == "final":
                break
            elif event["kind"] == "error":
                print("ERROR during utterance 2:", event["raw"])
                return

        print(f"Utterance 2: received {len(utterance_2_pcm)} bytes of PCM audio.")

    # --- Verify utterance 2's audio via STT round-trip ---
    wav_bytes = pcm_to_wav_bytes(utterance_2_pcm, sample_rate)
    result = stt_sarvam.transcribe_bytes(wav_bytes, filename="u2.wav", content_type="audio/wav")
    transcript = result["transcript"]
    print(f"\nTranscript of utterance 2's audio: {transcript!r}")
    print(f"Expected (approximately): {UTTERANCE_2!r}")

    # Loose check: contamination from utterance 1 would show up as extra unrelated words/length
    if len(transcript) > len(UTTERANCE_2) * 2:
        print("\nFAIL: transcript is much longer than expected - likely contaminated with "
              "leftover utterance 1 audio.")
    elif "sentence" in transcript.lower() or "streaming" in transcript.lower():
        print("\nFAIL: transcript contains words from utterance 1 - drain did not fully clear it.")
    else:
        print("\nPASS: utterance 2's audio looks clean, no detectable contamination from utterance 1.")


if __name__ == "__main__":
    asyncio.run(main())
