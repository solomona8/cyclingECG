
# Cloud ECG Analyzer (features + optional GPT narrative)

This backend computes deterministic ECG features from Apple Watch single‑lead data and (optionally) calls OpenAI to generate a patient‑friendly narrative.

## iOS App

A native **iPhone app** is now available in the `iOS/` directory! The app provides:
- **HealthKit Integration**: Extract ECG recordings directly from Apple Watch
- **Real-time Analysis**: Send recordings to this backend for detailed analysis
- **Multiple Export Formats**: Export as JSON, CSV, PDF, or TXT
- **Beautiful UI**: Modern SwiftUI interface with classification badges and detailed metrics

See [iOS/README.md](iOS/README.md) for complete setup instructions and documentation.

## Endpoints
- `POST /v1/ecg/analyze` — returns features + summary; includes `narrative` if `include_narrative=true` and `OPENAI_API_KEY` is set.
- `GET /v1/ecg/recordings/{recording_id}` — retrieve by id.

## Env vars
- `API_KEY` — enables `Authorization: Bearer <API_KEY>`
- `OPENAI_API_KEY` — optional, enables narrative
- `OPENAI_MODEL` — optional, defaults to `gpt-5`

## Local run
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export API_KEY="dev_secret_123"
uvicorn app.main:app --host 0.0.0.0 --port 8000

Docs: http://127.0.0.1:8000/docs

## Deploy on Render
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Env: set `API_KEY`, optionally `OPENAI_API_KEY`, `OPENAI_MODEL`
- Your URL: `https://<service>.onrender.com/v1/ecg/analyze`

## Lovable setup
Method: POST
URL: https://<service>.onrender.com/v1/ecg/analyze
Headers: { "Authorization": "Bearer <API_KEY>", "Content-Type": "application/json" }
Body: your ECG JSON; add `"include_narrative": true` to get GPT output.
