"""Spike test for Sarvam's STT WebSocket streaming protocol. Answers two things we need to know
before building the real orchestrator:
(a) does STT emit progressive transcripts before flush, or only after?
(b) is the message immediately after flush reliably the finalized transcript?

Usage: python scripts/test_stt_ws.py path/to/audio.wav
"""

import asyncio
import base64
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets

from app.config import SARVAM_API_KEY

STT_WS_URL = "wss://api.sarvam.ai/speech-to-text/ws"


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_stt_ws.py path/to/audio.wav")
        return
    audio_path = sys.argv[1]

    with wave.open(audio_path, "rb") as wf:
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        print(f"Loaded {audio_path}: {sample_rate}Hz, {wf.getnchannels()}ch, {len(pcm_data)} bytes PCM")

    url = (
        f"{STT_WS_URL}?language-code=unknown&model=saaras:v3&mode=codemix"
        f"&sample_rate={sample_rate}&vad_signals=true"
    )
    print(f"Connecting to {url}")

    async with websockets.connect(
        url, additional_headers={"Api-Subscription-Key": SARVAM_API_KEY}
    ) as ws:
        print("Connected.\n")

        async def sender():
            chunk_size = int(sample_rate * 2 * 0.25)  # ~250ms of 16-bit mono PCM
            for i in range(0, len(pcm_data), chunk_size):
                chunk = pcm_data[i : i + chunk_size]
                msg = {
                    "audio": {
                        "data": base64.b64encode(chunk).decode(),
                        "sample_rate": str(sample_rate),
                        "encoding": "audio/wav",
                    }
                }
                await ws.send(json.dumps(msg))
                await asyncio.sleep(0.1)
            print(">>> done sending audio, sending flush\n")
            await ws.send(json.dumps({"type": "flush"}))

        async def receiver():
            msg_count = 0
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=8)
                    msg_count += 1
                    print(f"RECV #{msg_count}: {raw}")
            except asyncio.TimeoutError:
                print(f"\n(no more messages after 8s idle - received {msg_count} total)")

        await asyncio.gather(sender(), receiver())


if __name__ == "__main__":
    asyncio.run(main())
