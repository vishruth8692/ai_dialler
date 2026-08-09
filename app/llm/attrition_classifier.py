"""Offline, post-call classification of a finished attrition call into the full structured record
(§8 of ATTRITION_VOICEBOT_KB.md) - mirrors app/llm/call_tagger.py's pattern (one forced tool-call
from the transcript, run once after the call ends) but with the much richer schema this bot's KB
specifies. Per the KB: "Emitted once per attempted call by the offline classifier (Haiku) from the
transcript - never by the voice model mid-call" - this module is exactly that classifier.

Mechanical fields the model has no way of knowing (call_id, timestamps, duration, transcript_uri,
rider_code, attempt_no) are deliberately NOT part of the tool schema - the caller fills those in
directly from what it already knows and merges them with this module's output.
"""

import logging

import anthropic

from app.config import ANTHROPIC_API_KEY, HAIKU_MODEL

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_L1_REASONS = [
    "EARNINGS",
    "PAYOUT_TRUST",
    "ACCESS_BLOCKED",
    "STORE_EXPERIENCE",
    "WORK_CONDITIONS",
    "PERSONAL",
    "ALT_WORK",
    "ASSET_VEHICLE",
    "NEVER_REALLY_STARTED",
    "UNKNOWN_REFUSED",
]

_L2_REASONS = [
    "EARN_PER_ORDER_LOW", "EARN_LOW_ORDER_VOLUME", "EARN_INCENTIVE_UNREACHABLE",
    "EARN_SLOT_UNAVAILABLE", "EARN_COST_SQUEEZE", "EARN_BETTER_ELSEWHERE", "EARN_RATECARD_OPAQUE",
    "PAY_NOT_RECEIVED", "PAY_SURGE_NOT_CREDITED", "PAY_BONUS_NOT_CREDITED", "PAY_DEDUCT_MDND",
    "PAY_DEDUCT_BAG", "PAY_DEDUCT_KIT_ASSET", "PAY_DEDUCT_SECURITY_DEPOSIT", "PAY_DEDUCT_TDS",
    "PAY_DEDUCT_UNEXPLAINED", "PAY_COD_QR_MISPOST", "PAY_COD_DEPOSIT_NOT_REFLECTING", "PAY_BANK_DETAILS",
    "ACC_ID_BLOCKED", "ACC_CHECKIN_FAIL", "ACC_NO_VACANCY", "ACC_STORE_CHANGE_STUCK", "ACC_DOC_KYC",
    "ACC_APP_UNUSABLE",
    "STORE_STAFF_BEHAVIOUR", "STORE_GATEKEEPING", "STORE_WAIT_TIME", "STORE_HYGIENE", "STORE_MATERIAL",
    "STORE_DISTANCE",
    "WORK_HOURS_FATIGUE", "WORK_WEATHER", "WORK_LONG_DISTANCE", "WORK_CUSTOMER_BEHAVIOUR",
    "WORK_SAFETY_INCIDENT", "WORK_PRESSURE_METRICS",
    "PERS_HEALTH_INJURY", "PERS_FAMILY", "PERS_RELOCATED", "PERS_EDUCATION", "PERS_OTHER",
    "ALT_COMPETITOR_QCOMM", "ALT_RIDE_HAIL", "ALT_SALARIED_JOB", "ALT_OWN_BUSINESS",
    "ALT_RETURNED_TO_PREV_JOB",
    "VEH_BREAKDOWN_SOLD", "VEH_RENTAL_COST", "VEH_LICENSE_CHALLAN", "VEH_FUEL_COST", "VEH_NO_ACCESS",
    "NEW_NO_KIT", "NEW_NO_TRAINING", "NEW_BLOCKED_AT_STORE", "NEW_ONBOARDING_STUCK",
    "NEW_EXPECTATION_GAP", "NEW_ID_NEVER_ACTIVATED",
    "UNK_REFUSED", "UNK_CALL_CUT", "UNK_LANGUAGE", "UNK_VAGUE",
]

_INTERNAL_ROUTES = [
    "welfare", "id_block", "joining_bonus", "kit_deduction", "mdnd", "tds", "payout", "deductions",
    "store_change", "store_staff", "store_hygiene", "store_ops", "vendor", "cod_qr", "onboarding",
]

_INFO_GAPS = [
    "ratecard", "incentive_structure", "vehicle_source", "kit_process", "slot_system", "insurance",
    "employment_status", "fuel_allowance", "login_requirement", "referral_terms", "app_install",
    "work_flexibility", "other",
]

_SYSTEM = (
    "You classify a finished rider-attrition feedback call into a structured record, from the "
    "transcript alone. Base every field ONLY on what's in the transcript - never guess or infer "
    "beyond what was actually said. Leave a field null/empty rather than fabricate a value.\n\n"
    f"L1 reason categories: {', '.join(_L1_REASONS)}\n"
    f"L2 reason codes (pick the one matching the correct L1 family): {', '.join(_L2_REASONS)}\n\n"
    "Disambiguation rules, in order: (1) a specific uncredited amount/bonus/surge/deduction makes "
    "the primary reason PAYOUT_TRUST even if they opened with earnings language; (2) being blocked "
    "or unable to log in beats not wanting to work - ACCESS_BLOCKED; (3) if they left because of a "
    "problem, the problem is primary and ALT_WORK (if any) is secondary; if they left purely for a "
    "better rate with no complaint, ALT_WORK is primary; (4) 'low earnings' alone with nothing "
    "specific after probing is UNK_VAGUE, never a guessed L2; (5) any injury/accident/assault/threat "
    "sets severity=high and welfare_flag=true regardless of the stated reason; (6) the last-straw "
    "answer decides primary_reason when several reasons were mentioned."
)


def classify_call(dial_record: dict, collected_answers: list[dict], transcript: list[dict]) -> dict:
    """Returns a dict matching the content-derived subset of §8's schema. Falls back to an
    all-null/empty record on any failure - this must never block call-history logging."""
    transcript_text = "\n".join(f'{t["role"]}: {t["text"]}' for t in transcript) or "(empty)"
    answers_text = (
        "\n".join(f'{a["question"]} -> {a["answer"]}' for a in collected_answers) or "(none recorded)"
    )
    user_content = (
        f"Dial record: {dial_record}\n\n"
        f"Stage-by-stage captured answers:\n{answers_text}\n\n"
        f"Full transcript:\n{transcript_text}"
    )

    try:
        response = _client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=800,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            tools=[_CLASSIFY_TOOL],
            tool_choice={"type": "tool", "name": "classify_attrition_call"},
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input
    except Exception:
        logger.exception("Attrition call classification failed")
    return _EMPTY_RECORD


_EMPTY_RECORD = {
    "status_gate": None,
    "primary_reason_l1": None,
    "primary_reason_l2": None,
    "secondary_reasons_l2": [],
    "last_straw": None,
    "other_reason_text": None,
    "reason_confidence": "low",
    "grievance": {"raised_with": None, "resolution_experience": None, "ticket_reference": None},
    "money_claim": None,
    "store_context": {"store_name": None, "city": None, "person_role_complained_about": None},
    "alt_work": None,
    "welfare_flag": False,
    "fraud_report": False,
    "do_not_call": False,
    "wants_to_return": None,
    "severity": "low",
    "internal_route": None,
    "internal_note": None,
    "info_gaps": [],
    "unanswered_question": None,
}

_CLASSIFY_TOOL = {
    "name": "classify_attrition_call",
    "description": "Report the structured attrition record for this call.",
    "input_schema": {
        "type": "object",
        "properties": {
            "status_gate": {
                "type": ["string", "null"],
                "enum": ["STILL_WORKING", "TEMPORARY_BREAK", "STOPPED", "NEVER_STARTED", "WRONG_PERSON", None],
            },
            "primary_reason_l1": {"type": ["string", "null"], "enum": _L1_REASONS + [None]},
            "primary_reason_l2": {"type": ["string", "null"], "enum": _L2_REASONS + [None]},
            "secondary_reasons_l2": {"type": "array", "items": {"type": "string", "enum": _L2_REASONS}},
            "last_straw": {"type": ["string", "null"], "description": "Verbatim last-straw answer."},
            "other_reason_text": {"type": ["string", "null"], "description": "Verbatim reason if no L2 fits."},
            "reason_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "grievance": {
                "type": "object",
                "properties": {
                    "raised_with": {
                        "type": ["string", "null"],
                        "enum": ["none", "chat", "phone_ticket", "store_captain", "rclm_srclm", "other", None],
                    },
                    "resolution_experience": {
                        "type": ["string", "null"],
                        "enum": ["resolved", "partly_resolved", "no_response", "made_worse", "still_open", None],
                    },
                    "ticket_reference": {"type": ["string", "null"]},
                },
                "required": ["raised_with", "resolution_experience", "ticket_reference"],
            },
            "money_claim": {
                "type": ["object", "null"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["not_received", "surge", "bonus", "deduction", "cod", "security_deposit", "tds", "other"],
                    },
                    "amount_inr": {"type": ["number", "null"]},
                    "period": {"type": ["string", "null"], "description": "As the rider stated it."},
                    "order_code": {"type": ["string", "null"]},
                },
            },
            "store_context": {
                "type": "object",
                "properties": {
                    "store_name": {"type": ["string", "null"]},
                    "city": {"type": ["string", "null"]},
                    "person_role_complained_about": {
                        "type": ["string", "null"],
                        "enum": ["captain", "store_staff", "senior_rider", "rsi", "vendor", "other", None],
                    },
                },
                "required": ["store_name", "city", "person_role_complained_about"],
            },
            "alt_work": {
                "type": ["object", "null"],
                "properties": {
                    "platform_named": {"type": ["string", "null"]},
                    "claimed_delta": {"type": ["string", "null"]},
                },
            },
            "welfare_flag": {"type": "boolean"},
            "fraud_report": {"type": "boolean"},
            "do_not_call": {"type": "boolean"},
            "wants_to_return": {"type": ["boolean", "null"]},
            "severity": {"type": "string", "enum": ["high", "medium", "low"]},
            "internal_route": {"type": ["string", "null"], "enum": _INTERNAL_ROUTES + [None]},
            "internal_note": {
                "type": ["string", "null"],
                "description": "One line, must be a specific who/what/where/amount, or null.",
            },
            "info_gaps": {"type": "array", "items": {"type": "string", "enum": _INFO_GAPS}},
            "unanswered_question": {"type": ["string", "null"], "description": "Verbatim, if any."},
        },
        "required": [
            "status_gate", "primary_reason_l1", "primary_reason_l2", "secondary_reasons_l2",
            "last_straw", "other_reason_text", "reason_confidence", "grievance", "money_claim",
            "store_context", "alt_work", "welfare_flag", "fraud_report", "do_not_call",
            "wants_to_return", "severity", "internal_route", "internal_note", "info_gaps",
            "unanswered_question",
        ],
    },
}
