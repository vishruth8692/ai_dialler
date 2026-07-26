"""Sarvam AI speech-to-text (Saaras/Saarika) wrapper.

Docs: https://docs.sarvam.ai/api-reference/speech-to-text/transcribe
"""

import io
import mimetypes
from pathlib import Path
from typing import BinaryIO

import requests

from app.config import SARVAM_API_KEY

STT_URL = "https://api.sarvam.ai/speech-to-text"

# Reused across calls so repeated requests get HTTP keep-alive instead of a fresh TCP+TLS
# handshake every time.
_session = requests.Session()


def _post(file_obj: BinaryIO, filename: str, content_type: str, language_code: str, mode: str) -> dict:
    response = _session.post(
        STT_URL,
        headers={"api-subscription-key": SARVAM_API_KEY},
        data={"model": "saaras:v3", "language_code": language_code, "mode": mode},
        files={"file": (filename, file_obj, content_type)},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "transcript": data.get("transcript", ""),
        "language_code": data.get("language_code", ""),
    }


def transcribe(audio_path: str, language_code: str = "unknown", mode: str = "codemix") -> dict:
    """Returns {"transcript": str, "language_code": str}. language_code='unknown' auto-detects.

    mode='codemix' preserves the partner's natural code-switching (e.g. Hinglish) in the transcript
    instead of normalizing everything into one language's pure script.
    """
    filename = Path(audio_path).name
    content_type = mimetypes.guess_type(filename)[0] or "audio/wav"
    with open(audio_path, "rb") as f:
        return _post(f, filename, content_type, language_code, mode)


def transcribe_bytes(
    audio_bytes: bytes,
    filename: str = "audio.webm",
    content_type: str = "audio/webm",
    language_code: str = "unknown",
    mode: str = "codemix",
) -> dict:
    """Same as transcribe() but takes raw bytes directly - avoids a disk round-trip for uploads
    that are already in memory (e.g. from a FastAPI request)."""
    return _post(io.BytesIO(audio_bytes), filename, content_type, language_code, mode)
