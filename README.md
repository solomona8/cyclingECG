
# Cloud ECG Analyzer (features + optional GPT narrative)

This backend computes deterministic ECG features from Apple Watch single‑lead data and (optionally) calls OpenAI to generate a patient‑friendly narrative.

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
