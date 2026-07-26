"""Sarvam AI text-to-speech (Bulbul) wrapper.

Docs: https://docs.sarvam.ai/api-reference/text-to-speech/convert
"""

import base64

import requests

from app.config import SARVAM_API_KEY

TTS_URL = "https://api.sarvam.ai/text-to-speech"

# A bare requests.post() opens a fresh TCP+TLS connection every call. Reusing a Session lets
# repeated calls to the same host keep the connection alive (HTTP keep-alive).
_session = requests.Session()

LANGUAGE_CODES = {
    "hindi": "hi-IN",
    "kannada": "kn-IN",
    "telugu": "te-IN",
    "tamil": "ta-IN",
    "marathi": "mr-IN",
    "english": "en-IN",
}


def synthesize(text: str, language: str = "english", speaker: str = "shubh") -> bytes:
    """Returns raw WAV audio bytes for the given text.

    `language` accepts either a friendly name ("hindi") or a BCP-47 code straight from
    Sarvam STT's detected language_code ("hi-IN") - both resolve to the same TTS language.
    """
    if language in LANGUAGE_CODES.values():
        target_language_code = language
    else:
        target_language_code = LANGUAGE_CODES.get(language.lower(), "en-IN")

    response = _session.post(
        TTS_URL,
        headers={"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"},
        json={
            "text": text,
            "target_language_code": target_language_code,
            "model": "bulbul:v3",
            "speaker": speaker,
            "speech_sample_rate": "24000",
            "output_audio_codec": "wav",
        },
        timeout=30,
    )
    response.raise_for_status()
    audio_b64 = response.json()["audios"][0]
    return base64.b64decode(audio_b64)
