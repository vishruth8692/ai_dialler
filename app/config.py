import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

EXOTEL_SID = os.getenv("EXOTEL_SID", "")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY", "")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_API_TOKEN", "")
EXOTEL_CALLER_ID = os.getenv("EXOTEL_CALLER_ID", "")
EXOTEL_SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")

# Public HTTPS/WSS base URL for this server (e.g. an ngrok URL during testing, or a real deployed
# host later) - Exotel needs to reach us from the internet, "localhost" won't work. Used to build
# the wss:// stream URL passed to Exotel's Calls API.
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))
QA_UPLOADS_DIR = BASE_DIR / "data" / "qa_uploads"

SUPPORTED_LANGUAGES = ["hindi", "kannada", "telugu", "tamil", "marathi", "english"]
