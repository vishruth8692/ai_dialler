import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
# Used only for the offline post-call attrition classifier (app/llm/attrition_classifier.py) - the
# KB doc specifies Haiku for that pass (cheap, adequate for structured extraction from a transcript
# that already exists), not the live conversational model.
HAIKU_MODEL = os.getenv("HAIKU_MODEL", "claude-haiku-4-5-20251001")

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

# Rider attrition calling bot (see ATTRITION_VOICEBOT_KB.md) - only the safety hard-stop's helpline
# is a real, separately-configured requirement; ordinary side-questions (money, ID, "connect me to
# someone") are handled by pointing riders to the Zepto app's own support-ticket flow instead, which
# needs no config. Tested that the model reliably says the ticket-raise line for those ordinary
# cases, but consistently would NOT say it in the safety-disclosure case specifically (it read as
# tone-deaf right after asking an injured rider if they're okay) - a real phone number given plainly
# is what actually gets said reliably there, so this one value stays.
ATTRITION_SAFETY_HELPLINE = os.getenv("ATTRITION_SAFETY_HELPLINE", "")
