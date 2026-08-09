import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

# INFO-level logs (e.g. real-time call diagnostics in app/streaming/exotel_ws_adapter.py) were
# being silently swallowed by the default WARNING root logger level - confirmed this cost real
# debugging time earlier in this project. Set explicitly so `logger.info(...)` calls are visible.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.attrition import call_queue as attrition_call_queue
from app.attrition import prompts as attrition_prompts
from app.attrition import qa_store as attrition_qa_store
from app.attrition import stage_store as attrition_stage_store
from app.attrition.call_session import AttritionCallSession, is_attrition_configured, missing_attrition_settings
from app.config import (
    ATTRITION_SAFETY_HELPLINE,
    EXOTEL_API_KEY,
    EXOTEL_API_TOKEN,
    EXOTEL_CALLER_ID,
    EXOTEL_SID,
    PUBLIC_BASE_URL,
    QA_UPLOADS_DIR,
)
from app.llm import attrition_classifier
from app.rag import qa_store
from app.rag.ingest import ingest_csv
from app.streaming.exotel_ws_adapter import run_exotel_call
from app.telephony import call_history, call_monitor, exotel_client

app = FastAPI(title="AI Dialler")


@app.on_event("startup")
async def _warm_up_embedder() -> None:
    # Loads the RAG embedding model (and its one-time HF Hub metadata fetch) at startup instead of
    # on a live call's first retrieve() - see qa_store.warm_up() docstring for the real-call impact
    # this had (11+ seconds added to the first reply). Off the event loop since it's a blocking load.
    import asyncio

    await asyncio.to_thread(qa_store.warm_up)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/health")
async def health():
    """Platform health check target (e.g. Railway) - deliberately does nothing but confirm the
    process is up and serving, not a dependency check (DB/embedder readiness is handled by the
    startup warm-up, not here)."""
    return {"status": "ok"}


@app.get("/")
async def root():
    return RedirectResponse(url="/qa")


@app.get("/qa")
async def qa_page(request: Request, message: Optional[str] = None, error: Optional[str] = None):
    pairs = qa_store.get_all_pairs()
    return templates.TemplateResponse(
        request,
        "qa_upload.html",
        {"pairs": pairs, "message": message, "error": error},
    )


@app.post("/qa/upload")
async def qa_upload(file: UploadFile = File(...)):
    content = await file.read()

    QA_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe_name = Path(file.filename or "upload.csv").name
    (QA_UPLOADS_DIR / f"{stamp}-{safe_name}").write_bytes(content)

    try:
        pairs = ingest_csv(content)
    except ValueError as e:
        return RedirectResponse(url=f"/qa?error={quote(str(e))}", status_code=303)
    except Exception:
        return RedirectResponse(
            url=f"/qa?error={quote('Could not process that file — is it a valid CSV?')}",
            status_code=303,
        )

    message = f"Loaded {len(pairs)} Q&A pairs from {safe_name}"
    return RedirectResponse(url=f"/qa?message={quote(message)}", status_code=303)


def _exotel_settings_status() -> list[tuple[str, bool]]:
    return [
        ("EXOTEL_SID", bool(EXOTEL_SID)),
        ("EXOTEL_API_KEY", bool(EXOTEL_API_KEY)),
        ("EXOTEL_API_TOKEN", bool(EXOTEL_API_TOKEN)),
        ("EXOTEL_CALLER_ID", bool(EXOTEL_CALLER_ID)),
        ("PUBLIC_BASE_URL", bool(PUBLIC_BASE_URL)),
    ]


def _attrition_settings_status() -> list[tuple[str, bool]]:
    # Same underlying Exotel requirement as /telephony, plus ATTRITION_SAFETY_HELPLINE - the one
    # remaining attrition-specific value, needed for the safety hard-stop (see
    # app/attrition/call_session.py's missing_attrition_settings() docstring for why it's a hard
    # block, unlike the Zepto bot's soft warning). Ordinary side-questions need no config - they're
    # handled by pointing riders to the Zepto app's own support-ticket flow.
    return _exotel_settings_status() + [("ATTRITION_SAFETY_HELPLINE", bool(ATTRITION_SAFETY_HELPLINE))]


def _attrition_stage_flow() -> dict:
    """The call's flow chart, as rendered on /attrition: each stage's editable instruction text
    plus the (fixed, non-editable) arrow labels that show which classified signal moves the call
    to the next stage - that graph itself lives in app/attrition/call_session.py's
    _STAGE_TRANSITIONS and isn't user-editable, only restated here for display."""
    effective = attrition_prompts.effective_stage_text()
    defaults = attrition_prompts.default_stage_text()

    def node(key, label, arrow_in, arrow_out, side_note=None, placeholder_hint=None):
        return {
            "key": key,
            "label": label,
            "arrow_in": arrow_in,
            "arrow_out": arrow_out,
            "side_note": side_note,
            "placeholder_hint": placeholder_hint,
            "text": effective[key],
            "is_default": effective[key] == defaults[key],
        }

    main = [
        node(
            attrition_prompts.GREETING,
            "1 · Greeting & identity check",
            None,
            "ready_to_continue",
            placeholder_hint='Uses {identity_line} as a placeholder, filled in automatically with '
            "the rider's name (or a generic phrase if none was given on the dial-in form). Don't "
            "remove it unless you want to skip the identity check.",
        ),
        node(
            attrition_prompts.STATUS_GATE,
            "2 · Status gate",
            "ready_to_continue",
            "stopped / never_started",
            side_note="If the rider is still working or on a temporary break, the call closes "
            "warmly right here - stages 3-6 below never run.",
        ),
        node(
            attrition_prompts.OPEN_QUESTION,
            "3 · Open question — why did you stop",
            "stopped / never_started",
            "reason_given",
        ),
        node(attrition_prompts.PROBE, "4 · Probe — get specific", "reason_given", "probe_answered / vague_reason"),
        node(
            attrition_prompts.LAST_STRAW,
            "5 · Last straw",
            "probe_answered / vague_reason",
            "last_straw_given",
        ),
        node(attrition_prompts.GRIEVANCE, "6 · Grievance & close", "last_straw_given", "call ends"),
    ]
    safety = node(
        attrition_prompts.SAFETY_STOP,
        "⚠ Safety stop — interrupts any stage above",
        "injury / accident / assault / threat / distress, from any stage",
        "call ends",
        placeholder_hint="Uses {safety_helpline} as a placeholder, filled in automatically from the "
        "ATTRITION_SAFETY_HELPLINE setting. The code guarantees this number is spoken even if this "
        "text omits it, so don't rely on removing it to suppress the number.",
    )
    return {"main": main, "safety": safety}


def _attrition_calls() -> list[dict]:
    return [c for c in call_history.list_calls() if c.get("call_type") == "attrition"]


@app.get("/attrition")
async def attrition_page(request: Request, message: Optional[str] = None, error: Optional[str] = None):
    settings_status = _attrition_settings_status()
    return templates.TemplateResponse(
        request,
        "attrition.html",
        {
            "settings_status": settings_status,
            "configured": all(is_set for _, is_set in settings_status),
            "message": message,
            "error": error,
            "history": _attrition_calls(),
            "stage_flow": _attrition_stage_flow(),
            "qa_pairs": attrition_qa_store.list_pairs(),
            "queue_status": attrition_call_queue.status(),
        },
    )


_CSV_COLUMNS = [
    "call_sid", "phone_number", "rider_name", "rider_code", "city", "store_name",
    "started_at", "ended_at",
    "status_gate", "primary_reason_l1", "primary_reason_l2", "reason_confidence",
    "last_straw", "other_reason_text",
    "severity", "welfare_flag", "fraud_report", "do_not_call", "wants_to_return",
    "grievance_raised_with", "grievance_resolution_experience", "grievance_ticket_reference",
    "money_claim_type", "money_claim_amount_inr", "money_claim_period", "money_claim_order_code",
    "store_context_store_name", "store_context_city", "store_context_person_role_complained_about",
    "alt_work_platform_named", "alt_work_claimed_delta",
    "internal_route", "internal_note", "info_gaps", "unanswered_question",
    "transcript",
]


def _flatten_attrition_call(call: dict) -> dict:
    """One spreadsheet row per call - flattens the nested §8 structured_record (see
    app/llm/attrition_classifier.py) into the CSV columns above. Missing/not-yet-classified
    records (call.structured_record is None) just leave those columns blank rather than error."""
    rec = call.get("structured_record") or {}
    dial = call.get("dial_record") or {}
    grievance = rec.get("grievance") or {}
    money_claim = rec.get("money_claim") or {}
    store_context = rec.get("store_context") or {}
    alt_work = rec.get("alt_work") or {}
    transcript = " | ".join(f'{t.get("role")}: {t.get("text")}' for t in call.get("transcript") or [])
    return {
        "call_sid": call.get("call_sid"),
        "phone_number": call.get("phone_number"),
        "rider_name": dial.get("rider_name"),
        "rider_code": dial.get("rider_code"),
        "city": dial.get("city"),
        "store_name": dial.get("store_name"),
        "started_at": call.get("started_at"),
        "ended_at": call.get("ended_at"),
        "status_gate": rec.get("status_gate"),
        "primary_reason_l1": rec.get("primary_reason_l1"),
        "primary_reason_l2": rec.get("primary_reason_l2"),
        "reason_confidence": rec.get("reason_confidence"),
        "last_straw": rec.get("last_straw"),
        "other_reason_text": rec.get("other_reason_text"),
        "severity": rec.get("severity"),
        "welfare_flag": rec.get("welfare_flag"),
        "fraud_report": rec.get("fraud_report"),
        "do_not_call": rec.get("do_not_call"),
        "wants_to_return": rec.get("wants_to_return"),
        "grievance_raised_with": grievance.get("raised_with"),
        "grievance_resolution_experience": grievance.get("resolution_experience"),
        "grievance_ticket_reference": grievance.get("ticket_reference"),
        "money_claim_type": money_claim.get("type"),
        "money_claim_amount_inr": money_claim.get("amount_inr"),
        "money_claim_period": money_claim.get("period"),
        "money_claim_order_code": money_claim.get("order_code"),
        "store_context_store_name": store_context.get("store_name"),
        "store_context_city": store_context.get("city"),
        "store_context_person_role_complained_about": store_context.get("person_role_complained_about"),
        "alt_work_platform_named": alt_work.get("platform_named"),
        "alt_work_claimed_delta": alt_work.get("claimed_delta"),
        "internal_route": rec.get("internal_route"),
        "internal_note": rec.get("internal_note"),
        "info_gaps": ";".join(rec.get("info_gaps") or []),
        "unanswered_question": rec.get("unanswered_question"),
        "transcript": transcript,
    }


@app.get("/attrition/calls/export.csv")
async def attrition_export_csv():
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS)
    writer.writeheader()
    for call in _attrition_calls():
        writer.writerow(_flatten_attrition_call(call))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="attrition-calls-{stamp}.csv"'},
    )


@app.get("/attrition/calls/export.json")
async def attrition_export_json():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Response(
        content=json.dumps(_attrition_calls(), ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="attrition-calls-{stamp}.json"'},
    )


class AttritionStagesPayload(BaseModel):
    stages: dict[str, str]


class AttritionResetPayload(BaseModel):
    stage: Optional[str] = None


@app.get("/attrition/stages")
async def attrition_get_stages():
    """JSON view of the current flow - used by the /attrition page's JS after a reset, so it can
    refresh without a full page reload."""
    return JSONResponse(
        {
            "stages": attrition_prompts.effective_stage_text(),
            "defaults": attrition_prompts.default_stage_text(),
        }
    )


@app.post("/attrition/stages")
async def attrition_save_stages(payload: AttritionStagesPayload):
    """Saves edited stage wording. Every call re-renders its system prompt fresh each turn (see
    AttritionCallSession._system_prompt()), so a save here takes effect on the very next call turn
    placed after it - no restart needed."""
    stripped = {stage: text.strip() for stage, text in payload.stages.items()}
    errors = {}
    for stage, text in stripped.items():
        err = attrition_prompts.validate_stage_text(stage, text)
        if err:
            errors[stage] = err
    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=400)

    defaults = attrition_prompts.default_stage_text()
    overrides = attrition_stage_store.load_overrides()
    for stage, text in stripped.items():
        # Only persist an override when the text actually differs from the default - an
        # unmodified "Save changes" click, or typing the default text back in, should behave as a
        # true no-op rather than silently marking every stage "customized".
        if stage in defaults and text == defaults[stage]:
            overrides.pop(stage, None)
        else:
            overrides[stage] = text
    attrition_stage_store.save_overrides(overrides)
    return JSONResponse({"ok": True, "defaults": defaults})


@app.post("/attrition/stages/reset")
async def attrition_reset_stages(payload: AttritionResetPayload):
    overrides = attrition_stage_store.load_overrides()
    if payload.stage:
        overrides.pop(payload.stage, None)
    else:
        overrides = {}
    attrition_stage_store.save_overrides(overrides)
    return JSONResponse(
        {
            "ok": True,
            "stages": attrition_prompts.effective_stage_text(),
            "defaults": attrition_prompts.default_stage_text(),
        }
    )


class AttritionQaPayload(BaseModel):
    question: str
    answer: str


class AttritionQaDeletePayload(BaseModel):
    index: int


@app.get("/attrition/qa")
async def attrition_get_qa():
    return JSONResponse({"pairs": attrition_qa_store.list_pairs()})


@app.post("/attrition/qa")
async def attrition_add_qa(payload: AttritionQaPayload):
    question = payload.question.strip()
    answer = payload.answer.strip()
    if not question or not answer:
        return JSONResponse({"ok": False, "error": "Both question and answer are required."}, status_code=400)
    attrition_qa_store.add_pair(question, answer)
    return JSONResponse({"ok": True, "pairs": attrition_qa_store.list_pairs()})


@app.post("/attrition/qa/delete")
async def attrition_delete_qa(payload: AttritionQaDeletePayload):
    attrition_qa_store.delete_pair(payload.index)
    return JSONResponse({"ok": True, "pairs": attrition_qa_store.list_pairs()})


@app.post("/attrition/qa/upload")
async def attrition_upload_qa(file: UploadFile = File(...)):
    """Bulk-adds from a CSV (question, answer columns) - appends to whatever's already saved
    rather than replacing it, so this can be used alongside one-at-a-time adds."""
    content = await file.read()
    try:
        pairs = attrition_qa_store.parse_csv(content)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    except Exception:
        return JSONResponse(
            {"ok": False, "error": "Could not process that file - is it a valid CSV?"}, status_code=400
        )
    attrition_qa_store.add_pairs(pairs)
    return JSONResponse({"ok": True, "added": len(pairs), "pairs": attrition_qa_store.list_pairs()})


@app.post("/attrition/call")
async def attrition_place_call(
    to_number: str = Form(...),
    rider_name: str = Form(""),
    rider_code: str = Form(""),
    city: str = Form(""),
    store_name: str = Form(""),
    preferred_language: str = Form(""),
):
    if not is_attrition_configured():
        return RedirectResponse(
            url=f"/attrition?error={quote('Missing settings: ' + ', '.join(missing_attrition_settings()))}",
            status_code=303,
        )
    if attrition_call_queue.status()["running"]:
        # This app can only run one live call at a time (see call_monitor.py's docstring) - a
        # manual call placed while the bulk queue is also trying to place one would race it.
        return RedirectResponse(
            url=f"/attrition?error={quote('A bulk call queue is currently running - wait for it to finish or clear it first.')}",
            status_code=303,
        )

    dial_record = {
        k: v
        for k, v in {
            "rider_name": rider_name,
            "rider_code": rider_code,
            "city": city,
            "store_name": store_name,
            "preferred_language": preferred_language,
        }.items()
        if v
    }
    try:
        result = exotel_client.place_call(to_number, dial_record=dial_record, stream_path="/attrition/exotel-stream")
    except exotel_client.ExotelNotConfigured as e:
        return RedirectResponse(url=f"/attrition?error={quote(str(e))}", status_code=303)
    except Exception as e:
        return RedirectResponse(
            url=f"/attrition?error={quote(f'Call failed: {e}')}", status_code=303
        )

    call_sid = result.get("Call", {}).get("Sid", "unknown")
    return RedirectResponse(
        url=f"/attrition?message={quote(f'Call placed to {to_number} (SID: {call_sid})')}",
        status_code=303,
    )


@app.post("/attrition/calls/upload")
async def attrition_upload_calls(file: UploadFile = File(...)):
    """Bulk-calling: parses a CSV of phone numbers (+ optional dial-record columns) and enqueues
    the valid rows - see app/attrition/call_queue.py for how they actually get placed (one at a
    time, 30s gap after each finishes, never overlapping a live call)."""
    if not is_attrition_configured():
        return JSONResponse(
            {"ok": False, "error": "Missing settings: " + ", ".join(missing_attrition_settings())},
            status_code=400,
        )
    content = await file.read()
    try:
        rows, errors = attrition_call_queue.parse_csv(content)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    except Exception:
        return JSONResponse(
            {"ok": False, "error": "Could not process that file - is it a valid CSV?"}, status_code=400
        )
    attrition_call_queue.enqueue(rows)
    return JSONResponse({"ok": True, "enqueued": len(rows), "errors": errors, "status": attrition_call_queue.status()})


@app.get("/attrition/calls/queue-status")
async def attrition_queue_status():
    return JSONResponse(attrition_call_queue.status())


@app.post("/attrition/calls/queue/clear")
async def attrition_queue_clear():
    attrition_call_queue.clear_pending()
    return JSONResponse(attrition_call_queue.status())


@app.post("/attrition/calls/queue/reset")
async def attrition_queue_reset():
    attrition_call_queue.reset()
    return JSONResponse(attrition_call_queue.status())


@app.websocket("/attrition/exotel-stream")
async def attrition_exotel_stream(websocket: WebSocket):
    await run_exotel_call(
        websocket,
        call_type="attrition",
        session_factory=lambda dial_record: AttritionCallSession(dial_record=dial_record),
        configured_check=is_attrition_configured,
        not_configured_message=(
            "Exotel attrition call connected but required settings are missing "
            f"({', '.join(missing_attrition_settings())}) - closing."
        ),
        classify_fn=attrition_classifier.classify_call,
    )


@app.websocket("/attrition/monitor-ws")
async def attrition_monitor_ws(websocket: WebSocket):
    """Same shared call_monitor feed /telephony/monitor-ws uses - see call_monitor.py's docstring:
    one outbound call at a time app-wide, so there's never ambiguity about which call is live."""
    await call_monitor.register(websocket)
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        call_monitor.unregister(websocket)
