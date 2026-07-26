"""Async WebSocket client for Sarvam's streaming speech-to-text API.

Protocol verified against the real API in scripts/test_stt_ws.py:
- wss://api.sarvam.ai/speech-to-text/ws, header Api-Subscription-Key.
- Only sample_rate 8000 or 16000 is accepted here (the REST endpoint accepts a wider range).
- Sending {"type":"flush"} reliably produces exactly one final {"type":"data",...} transcript
  message - no progressive/interim transcripts were observed before flush in testing.
"""

import base64
import json
from typing import AsyncIterator, Optional

import websockets

from app.config import SARVAM_API_KEY

STT_WS_URL = "wss://api.sarvam.ai/speech-to-text/ws"


class SarvamSTTStream:
    def __init__(self, language_code: str = "unknown", sample_rate: int = 16000):
        self._language_code = language_code
        self._sample_rate = sample_rate
        self._ws: Optional[websockets.ClientConnection] = None

    async def __aenter__(self) -> "SarvamSTTStream":
        url = (
            f"{STT_WS_URL}?language-code={self._language_code}&model=saaras:v3&mode=codemix"
            f"&sample_rate={self._sample_rate}&vad_signals=true"
        )
        self._ws = await websockets.connect(
            url, additional_headers={"Api-Subscription-Key": SARVAM_API_KEY}
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def send_audio_chunk(self, pcm16_bytes: bytes) -> None:
        msg = {
            "audio": {
                "data": base64.b64encode(pcm16_bytes).decode(),
                "sample_rate": str(self._sample_rate),
                "encoding": "audio/wav",
            }
        }
        await self._ws.send(json.dumps(msg))

    async def flush(self) -> None:
        await self._ws.send(json.dumps({"type": "flush"}))

    async def events(self) -> AsyncIterator[dict]:
        """Yields parsed server messages verbatim: {"type": "data"|"events"|"error", ...}."""
        async for raw in self._ws:
            yield json.loads(raw)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
