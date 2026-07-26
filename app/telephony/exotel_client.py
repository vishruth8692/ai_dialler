"""REST client for Exotel's Calls API - places an outbound call and connects it to our WebSocket
for bidirectional voice streaming once answered.

NOT YET LIVE-TESTED. Unlike every other external integration in this project (Sarvam, Claude), this
was built from Exotel's public developer docs (https://developer.exotel.com/docs/agentstream/
developer-guide) rather than verified against a real account - no Exotel credentials exist yet as
of writing. Re-verify the exact endpoint path, auth scheme, and parameter names here against a real
account before trusting it, the same way scripts/test_stt_ws.py etc. verified Sarvam's protocol
before it was relied on.
"""

import logging

import requests

from app.config import (
    EXOTEL_API_KEY,
    EXOTEL_API_TOKEN,
    EXOTEL_CALLER_ID,
    EXOTEL_SID,
    EXOTEL_SUBDOMAIN,
    PUBLIC_BASE_URL,
)

logger = logging.getLogger(__name__)

_session = requests.Session()


class ExotelNotConfigured(Exception):
    """Raised when required Exotel settings are missing from .env."""


def is_configured() -> bool:
    return bool(EXOTEL_SID and EXOTEL_API_KEY and EXOTEL_API_TOKEN and EXOTEL_CALLER_ID)


def missing_settings() -> list[str]:
    """Returns the names of required settings that are still empty, for a setup-status UI."""
    settings = {
        "EXOTEL_SID": EXOTEL_SID,
        "EXOTEL_API_KEY": EXOTEL_API_KEY,
        "EXOTEL_API_TOKEN": EXOTEL_API_TOKEN,
        "EXOTEL_CALLER_ID": EXOTEL_CALLER_ID,
        "PUBLIC_BASE_URL": PUBLIC_BASE_URL,
    }
    return [name for name, value in settings.items() if not value]


def place_call(to_number: str) -> dict:
    """Places an outbound call to `to_number`, streaming its audio bidirectionally to our
    /telephony/exotel-stream WebSocket once answered. Returns Exotel's response JSON.

    Raises ExotelNotConfigured if required settings are missing, or requests.HTTPError on failure.
    """
    if not is_configured():
        raise ExotelNotConfigured(f"Missing settings: {', '.join(missing_settings())}")

    # sample-rate=16000 matches what Sarvam's STT WebSocket requires (see
    # app/streaming/sarvam_stt_ws.py) so no resampling is needed on audio coming from the caller.
    base = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    stream_url = f"{base}/telephony/exotel-stream?sample-rate=16000"

    url = f"https://{EXOTEL_SUBDOMAIN}/v1/Accounts/{EXOTEL_SID}/Calls/connect.json"
    response = _session.post(
        url,
        auth=(EXOTEL_API_KEY, EXOTEL_API_TOKEN),
        data={
            "From": to_number,
            "CallerId": EXOTEL_CALLER_ID,
            "StreamUrl": stream_url,
            "StreamType": "bidirectional",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()
