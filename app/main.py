import base64
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

# INFO-level logs (e.g. real-time call diagnostics in app/streaming/exotel_ws_adapter.py) were
# being silently swallowed by the default WARNING root logger level - confirmed this cost real
# debugging time earlier in this project. Set explicitly so `logger.info(...)` calls are visible.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from fastapi import (
    Cookie,
    FastAPI,
    File,
    Form,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.call_session import CallSession
from app.config import (
    EXOTEL_API_KEY,
    EXOTEL_API_TOKEN,
    EXOTEL_CALLER_ID,
    EXOTEL_SID,
    PUBLIC_BASE_URL,
    QA_UPLOADS_DIR,
)
from app.rag import qa_store
from app.rag.ingest import ingest_csv
from app.speech import stt_sarvam, tts_sarvam
from app.streaming.call_ws_handler import run_call
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

# In-memory store of mock-call sessions, keyed by a per-browser cookie. Fine for local testing;
# real calls in Phase 4 will be keyed by the Exotel call SID instead.
_chat_sessions: dict[str, CallSession] = {}


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


class ChatMessage(BaseModel):
    text: str


@app.get("/chat")
async def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {})


@app.post("/chat/start")
async def chat_start(response: Response):
    if not qa_store.get_survey_questions():
        return JSONResponse(
            {"error": "No survey questions loaded yet — upload a CSV at /qa first."}, status_code=400
        )

    session_id = str(uuid.uuid4())
    session = CallSession()

    try:
        greeting = session.greeting()
    except Exception as e:
        return JSONResponse({"error": f"Claude call failed: {e}"}, status_code=502)

    _chat_sessions[session_id] = session
    response.set_cookie("session_id", session_id, httponly=True, samesite="lax")
    return {"reply": greeting, "ended": session.ended}


@app.post("/chat/send")
async def chat_send(payload: ChatMessage, session_id: Optional[str] = Cookie(None)):
    session = _chat_sessions.get(session_id) if session_id else None
    if session is None:
        return JSONResponse({"error": "No active call — click 'Start new call' first."}, status_code=400)

    try:
        turn = session.handle_user_turn(payload.text)
    except Exception as e:
        return JSONResponse({"error": f"Claude call failed: {e}"}, status_code=502)

    return {
        "reply": turn["reply_text"],
        "tool_name": turn["tool_name"],
        "answer_summary": turn["answer_summary"],
        "retrieved_context": turn["retrieved_context"],
        "ended": session.ended,
        "collected_answers": session.collected_answers,
    }


@app.get("/voice")
async def voice_page(request: Request):
    return templates.TemplateResponse(request, "voice_chat.html", {})


@app.post("/voice/start")
async def voice_start(response: Response):
    if not qa_store.get_survey_questions():
        return JSONResponse(
            {"error": "No survey questions loaded yet — upload a CSV at /qa first."}, status_code=400
        )

    session_id = str(uuid.uuid4())
    session = CallSession()

    try:
        greeting_text = session.greeting()
        audio_bytes = tts_sarvam.synthesize(greeting_text, language=session.language_hint)
    except Exception as e:
        return JSONResponse({"error": f"Could not start call: {e}"}, status_code=502)

    _chat_sessions[session_id] = session
    response.set_cookie("session_id", session_id, httponly=True, samesite="lax")
    return {
        "reply_text": greeting_text,
        "audio_base64": base64.b64encode(audio_bytes).decode(),
        "ended": session.ended,
    }


@app.post("/voice/transcribe")
async def voice_transcribe(audio: UploadFile = File(...), session_id: Optional[str] = Cookie(None)):
    """Step 1 of 2: transcribe only, so the UI can show what the rider said right away instead of
    waiting for the LLM + TTS round-trip too."""
    session = _chat_sessions.get(session_id) if session_id else None
    if session is None:
        return JSONResponse(
            {"error": "No active call — click 'Start new call' first."}, status_code=400
        )

    content = await audio.read()
    content_type = audio.content_type or "audio/webm"
    filename = audio.filename or "recording.webm"

    try:
        stt_result = stt_sarvam.transcribe_bytes(content, filename=filename, content_type=content_type)
    except Exception as e:
        return JSONResponse({"error": f"Transcription failed: {e}"}, status_code=502)

    return {
        "transcript": stt_result["transcript"].strip(),
        "language_code": stt_result["language_code"] or session.language_hint,
    }


class VoiceReplyRequest(BaseModel):
    text: str
    language_code: str = "english"


@app.post("/voice/reply")
async def voice_reply(payload: VoiceReplyRequest, session_id: Optional[str] = Cookie(None)):
    """Step 2 of 2: run the transcript through Claude + TTS."""
    session = _chat_sessions.get(session_id) if session_id else None
    if session is None:
        return JSONResponse(
            {"error": "No active call — click 'Start new call' first."}, status_code=400
        )

    transcript = payload.text.strip()
    language = payload.language_code or session.language_hint

    if not transcript:
        reply_text = "Sorry, I didn't catch that — could you say it again?"
        try:
            audio_bytes = tts_sarvam.synthesize(reply_text, language=language)
        except Exception as e:
            return JSONResponse({"error": f"Speech synthesis failed: {e}"}, status_code=502)
        return {
            "reply_text": reply_text,
            "tool_name": None,
            "answer_summary": None,
            "retrieved_context": [],
            "audio_base64": base64.b64encode(audio_bytes).decode(),
            "ended": session.ended,
            "collected_answers": session.collected_answers,
        }

    try:
        turn = session.handle_user_turn(transcript)
        reply_text = turn["reply_text"]
        audio_bytes = tts_sarvam.synthesize(reply_text, language=language)
    except Exception as e:
        return JSONResponse({"error": f"Voice call failed: {e}"}, status_code=502)

    return {
        "reply_text": reply_text,
        "tool_name": turn["tool_name"],
        "answer_summary": turn["answer_summary"],
        "retrieved_context": turn["retrieved_context"],
        "audio_base64": base64.b64encode(audio_bytes).decode(),
        "ended": session.ended,
        "collected_answers": session.collected_answers,
    }


@app.get("/voice/live")
async def voice_live_page(request: Request):
    return templates.TemplateResponse(request, "voice_chat_live.html", {})


@app.websocket("/voice/ws")
async def voice_ws(websocket: WebSocket):
    await run_call(websocket)


def _exotel_settings_status() -> list[tuple[str, bool]]:
    return [
        ("EXOTEL_SID", bool(EXOTEL_SID)),
        ("EXOTEL_API_KEY", bool(EXOTEL_API_KEY)),
        ("EXOTEL_API_TOKEN", bool(EXOTEL_API_TOKEN)),
        ("EXOTEL_CALLER_ID", bool(EXOTEL_CALLER_ID)),
        ("PUBLIC_BASE_URL", bool(PUBLIC_BASE_URL)),
    ]


@app.get("/telephony")
async def telephony_page(request: Request, message: Optional[str] = None, error: Optional[str] = None):
    settings_status = _exotel_settings_status()
    return templates.TemplateResponse(
        request,
        "telephony.html",
        {
            "settings_status": settings_status,
            "configured": all(is_set for _, is_set in settings_status),
            "message": message,
            "error": error,
            "history": call_history.list_calls(),
        },
    )


@app.post("/telephony/call")
async def telephony_place_call(to_number: str = Form(...)):
    try:
        result = exotel_client.place_call(to_number)
    except exotel_client.ExotelNotConfigured as e:
        return RedirectResponse(url=f"/telephony?error={quote(str(e))}", status_code=303)
    except Exception as e:
        return RedirectResponse(
            url=f"/telephony?error={quote(f'Call failed: {e}')}", status_code=303
        )

    call_sid = result.get("Call", {}).get("Sid", "unknown")
    return RedirectResponse(
        url=f"/telephony?message={quote(f'Call placed to {to_number} (SID: {call_sid})')}",
        status_code=303,
    )


@app.websocket("/telephony/exotel-stream")
async def telephony_exotel_stream(websocket: WebSocket):
    await run_exotel_call(websocket)


@app.websocket("/telephony/monitor-ws")
async def telephony_monitor_ws(websocket: WebSocket):
    """Browser-facing feed for /telephony's live call view - broadcasts the same transcript/reply/
    debug events a real Exotel call produces, sourced from app/telephony/call_monitor.py. Carries
    no audio (the call's audio goes to the phone, not here)."""
    await call_monitor.register(websocket)
    try:
        while True:
            # Bare .receive() returns a {"type":"websocket.disconnect"} message on disconnect -
            # it does NOT raise WebSocketDisconnect the way .receive_text()/.receive_json() do.
            # Calling .receive() again after that message raises RuntimeError (confirmed on a
            # real call when a monitor tab was closed) - check the message type explicitly.
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        call_monitor.unregister(websocket)
