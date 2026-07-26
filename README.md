# AI Dialler — Delivery Partner Feedback Voice Bot

Outbound voice bot for Indian delivery partners: dials a number, greets the partner, asks a fixed
feedback Q&A script, understands free-form replies via Claude, and answers off-script questions
using only the uploaded Q&A content (RAG — the bot is not fine-tuned on this data). Supports real
barge-in: talk over the bot for 5+ seconds and it stops to listen; a quick "okay" doesn't interrupt.

## Stack

| Layer            | Choice                          | Status |
|------------------|----------------------------------|--------|
| Telephony        | Exotel                          | Working — verified against real calls |
| STT / TTS        | Sarvam AI (Saarika/Saaras / Bulbul) | Working — REST + streaming WS |
| LLM              | Claude API                       | Working |
| Languages        | Hindi, Kannada, Telugu, Tamil, Marathi, English |
| RAG store        | Chroma (local) + multilingual sentence-transformers | Working |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set ANTHROPIC_API_KEY and SARVAM_API_KEY at minimum
```

```bash
uvicorn app.main:app --reload --port 8008
```

## Pages

- **`/qa`** — upload a Q&A CSV (`question,answer`, optional `language`, `type`). `type` is
  `survey` (asked aloud, in order — the call script; default if omitted) or `faq` (only used to
  answer side questions, never asked aloud).
- **`/chat`** — text-only mock call. Fastest way to test conversation logic/Q&A content changes.
- **`/voice`** — batch voice call (record → upload → wait → play). Simple REST-based fallback.
- **`/voice/live`** — real-time streaming voice call with barge-in. Mic is always on once you click
  "Start call"; talk over the bot for 5+ seconds to interrupt it, a brief "okay" won't.
- **`/telephony`** — real Exotel phone calls: place a call, watch a live transcript as it happens,
  and review past calls below (AI-tagged summary + full transcript per call).

## Real phone calls (Exotel)

- **Exotel account** (https://exotel.com/) — sign up, complete **TRAI DLT registration** (legally
  required in India for commercial calling), buy an Exophone number, get API SID/key/token. Set
  `EXOTEL_SID`, `EXOTEL_API_KEY`, `EXOTEL_API_TOKEN`, `EXOTEL_CALLER_ID` in `.env`.
- **A public HTTPS/WSS URL** — Exotel connects *to* this server; `localhost` won't work. For local
  dev, use `ngrok http 8008` and set `PUBLIC_BASE_URL` to the ngrok URL it gives you. For the
  shared/deployed instance, see **Deployment** below instead — ngrok is dev-only.
- **Trial accounts can only call numbers verified in the Exotel dashboard** (Settings → Users, or a
  missed call to Exotel's verification number). Calling anyone else needs KYC approval on the
  account first.

## Deployment (Railway)

The app needs long-lived WebSockets (Sarvam STT/TTS, the Exotel media stream, the `/telephony`
live monitor) and local persistent disk (Chroma vector store, uploaded CSVs, call history) — not
classic serverless. It also keeps call state in one process's memory
(`app/telephony/call_monitor.py`, per-call `CallSession`), so it must run as **exactly one
instance**, never autoscaled/replicated.

1. Push this repo to a **private** GitHub repo.
2. Create a Railway project from that repo (it auto-detects the `Dockerfile`).
3. Attach a **Volume** mounted at `/app/data` — this is what makes the Chroma store, uploaded
   CSVs, and call history durable across deploys/restarts.
4. Set **replicas = 1** explicitly.
5. Set env vars in the Railway dashboard: `ANTHROPIC_API_KEY`, `SARVAM_API_KEY`, `EXOTEL_SID`,
   `EXOTEL_API_KEY`, `EXOTEL_API_TOKEN`, `EXOTEL_CALLER_ID`, `EXOTEL_SUBDOMAIN` (only if not the
   default `api.exotel.com`), `CLAUDE_MODEL` (optional).
6. Deploy once to get Railway's assigned `https://<name>.up.railway.app` domain, then set
   `PUBLIC_BASE_URL` to that exact URL and redeploy — this is what builds the `wss://` stream URL
   Exotel calls back into.
7. Visit `/qa` on the deployed URL and upload your Q&A CSV — the volume starts empty on first
   deploy, so this is a required one-time step.

**No authentication is enabled** — anyone with the URL can place real, costed calls and read real
rider transcripts. Keep the URL unlisted/shared only with trusted people until auth is added.

## Scripts

- `scripts/chat_cli.py` — text-only mock call in the terminal.
- `scripts/test_stt.py` / `test_tts.py` — Sarvam REST STT/TTS sanity checks.
- `scripts/test_stt_ws.py` / `test_tts_ws.py` — Sarvam streaming WebSocket protocol checks.
- `scripts/test_tts_cancel_reuse.py` — verifies TTS connection reuse after mid-utterance cancellation
  (barge-in support).
- `scripts/test_claude_stream_cancel.py` — verifies Claude streaming cancellation doesn't leak
  connections.
- `scripts/benchmark_latency.py` — per-stage latency benchmark for the batch (`/voice`) path.

## Project layout

- `app/rag/` — Q&A store (Chroma + multilingual embeddings), CSV ingestion.
- `app/llm/` — Claude prompts, conversation control-block parsing, call tagging.
- `app/call_session.py` — per-call state machine (script position, answer capture, RAG retrieval).
- `app/streaming/` — Sarvam STT/TTS WebSocket clients and the transport-agnostic call orchestrator
  (shared by both the browser demo and real Exotel calls).
- `app/telephony/` — Exotel REST client, live call monitor pub/sub, call history store.
