"""Editable overrides for the attrition bot's per-stage instructions (app/attrition/prompts.py) -
lets someone tune what each stage actually asks from the /attrition tab, without touching code.
Same simple JSON-file pattern as app/telephony/call_history.py.

Only the per-stage instruction text is stored here - guardrails, persona, and the anchor reminder
stay in code (app/attrition/prompts.py) so an edit here can never weaken them, only change what
each stage asks."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_STORE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "attrition_stage_overrides.json"


def load_overrides() -> dict[str, str]:
    if not _STORE_PATH.exists():
        return {}
    try:
        return json.loads(_STORE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to load attrition stage overrides - falling back to defaults")
        return {}


def save_overrides(overrides: dict[str, str]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(overrides, ensure_ascii=False, indent=2))
