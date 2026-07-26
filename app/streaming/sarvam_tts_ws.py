"""Async WebSocket client for Sarvam's streaming text-to-speech API.

Protocol verified against the real API in scripts/test_tts_ws.py:
- wss://api.sarvam.ai/text-to-speech/ws, header Api-Subscription-Key.
- output_audio_codec="linear16" gives clean, headerless, uniformly-sized raw PCM16LE chunks.
  "mp3" was tried first and rejected: those chunks are fragments of one continuous MP3 bitstream
  (frame sync bytes don't align to chunk boundaries), not independently decodable - linear16 avoids
  that problem entirely since raw PCM can be sliced anywhere.
- One flush reliably produces one {"type":"event","data":{"event_type":"final"}} message.
"""

import base64
import json
from typing import AsyncIterator, Optional

import websockets

from app.config import SARVAM_API_KEY
from app.speech.tts_sarvam import LANGUAGE_CODES

TTS_WS_URL = "wss://api.sarvam.ai/text-to-speech/ws"


def _resolve_language_code(language: str) -> str:
    if language in LANGUAGE_CODES.values():
        return language
    return LANGUAGE_CODES.get(language.lower(), "en-IN")


class SarvamTTSStream:
    def __init__(self, language: str = "english", speaker: str = "shubh", sample_rate: int = 24000):
        self._language = language
        self._speaker = speaker
        self._sample_rate = sample_rate
        self._ws: Optional[websockets.ClientConnection] = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def __aenter__(self) -> "SarvamTTSStream":
        url = f"{TTS_WS_URL}?model=bulbul:v3&send_completion_event=true"
        self._ws = await websockets.connect(
            url, additional_headers={"Api-Subscription-Key": SARVAM_API_KEY}
        )

        config_msg = {
            "type": "config",
            "data": {
                "target_language_code": _resolve_language_code(self._language),
                "speaker": self._speaker,
                "model": "bulbul:v3",
                "pace": 1,
                "speech_sample_rate": str(self._sample_rate),
                "output_audio_codec": "linear16",
            },
        }
        await self._ws.send(json.dumps(config_msg))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def send_text(self, text: str) -> None:
        await self._ws.send(json.dumps({"type": "text", "data": {"text": text}}))

    async def flush(self) -> None:
        await self._ws.send(json.dumps({"type": "flush"}))

    async def ping(self) -> None:
        await self._ws.send(json.dumps({"type": "ping"}))

    async def events(self) -> AsyncIterator[dict]:
        """Yields a tagged union so callers don't need to know the raw Sarvam message shape:
        {"kind": "audio", "data": bytes}  - raw PCM16LE chunk, ready to forward/play
        {"kind": "final"}                 - this flush's audio has fully arrived
        {"kind": "error", "raw": dict}
        {"kind": "other", "raw": dict}     - anything else (e.g. non-final events)
        """
        async for raw in self._ws:
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "audio":
                yield {"kind": "audio", "data": base64.b64decode(msg["data"]["audio"])}
            elif msg_type == "event" and msg.get("data", {}).get("event_type") == "final":
                yield {"kind": "final"}
            elif msg_type == "error":
                yield {"kind": "error", "raw": msg}
            else:
                yield {"kind": "other", "raw": msg}

    async def drain_pending(self, expected_finals: int) -> None:
        """Call after a consumer of events() was cancelled mid-utterance, before reusing this
        connection for the next one. Reads and discards until `expected_finals` more 'final'
        events have arrived, relying on Sarvam still emitting exactly one final per already-sent
        flush even when the client stopped consuming mid-stream - otherwise those stale audio
        bytes would sit in the socket and bleed into the start of the next utterance's audio.
        """
        if expected_finals <= 0:
            return
        seen = 0
        async for event in self.events():
            if event["kind"] == "final":
                seen += 1
                if seen >= expected_finals:
                    return
            elif event["kind"] == "error":
                return

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
