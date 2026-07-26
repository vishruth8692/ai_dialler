"""Spike test for Sarvam's TTS WebSocket streaming protocol. Answers two things we need to know
before building the real orchestrator:
(a) is each returned audio chunk independently decodable as a standalone MP3 clip?
(b) does one flush reliably correspond to one completion event?

Usage: python scripts/test_tts_ws.py
Writes each received chunk to /tmp/tts_ws_chunk_N.mp3 for standalone-decode inspection.
"""

import asyncio
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets

from app.config import SARVAM_API_KEY

TTS_WS_URL = "wss://api.sarvam.ai/text-to-speech/ws"

TEXT_CHUNKS = [
    "Namaste, main aapki delivery ke baare mein baat karna chahta hoon.",
    "Kya aapko koi dikkat hui thi?",
    "Dhanyavaad, aapka din accha ho.",
]


async def main():
    url = f"{TTS_WS_URL}?model=bulbul:v3&send_completion_event=true"
    print(f"Connecting to {url}")

    async with websockets.connect(
        url, additional_headers={"Api-Subscription-Key": SARVAM_API_KEY}
    ) as ws:
        codec = sys.argv[1] if len(sys.argv) > 1 else "mp3"
        config_msg = {
            "type": "config",
            "data": {
                "target_language_code": "hi-IN",
                "speaker": "shubh",
                "model": "bulbul:v3",
                "pace": 1,
                "speech_sample_rate": "24000",
                "output_audio_codec": codec,
            },
        }
        await ws.send(json.dumps(config_msg))
        print("Sent config.\n")

        flush_count = 0
        completion_count = 0
        chunk_index = 0

        async def sender():
            nonlocal flush_count
            for text in TEXT_CHUNKS:
                await ws.send(json.dumps({"type": "text", "data": {"text": text}}))
                await ws.send(json.dumps({"type": "flush"}))
                flush_count += 1
                print(f'>>> sent text + flush #{flush_count}: "{text}"')
                await asyncio.sleep(0.3)

        async def receiver():
            nonlocal completion_count, chunk_index
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                except asyncio.TimeoutError:
                    print("(timed out waiting for more messages)")
                    break

                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "audio":
                    audio_b64 = msg["data"]["audio"]
                    audio_bytes = base64.b64decode(audio_b64)
                    out_path = Path(f"/tmp/tts_ws_chunk_{chunk_index}.mp3")
                    out_path.write_bytes(audio_bytes)
                    print(f"RECV audio chunk #{chunk_index}: {len(audio_bytes)} bytes -> {out_path}")
                    chunk_index += 1
                elif msg_type == "completion":
                    completion_count += 1
                    print(f"RECV completion event (#{completion_count})")
                    if completion_count >= flush_count:
                        break
                else:
                    print("RECV (other):", raw)

        await asyncio.gather(sender(), receiver())

        print(
            f"\nSummary: sent {flush_count} flushes, received {chunk_index} audio chunks, "
            f"{completion_count} completion events"
        )


if __name__ == "__main__":
    asyncio.run(main())
