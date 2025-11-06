
# ECG Analyzer Minimal Backend

A tiny FastAPI service exposing two endpoints to integrate your Lovable UI with a real API.

## Endpoints
- POST /v1/ecg/analyze — submit ECG payload (see payload.json from earlier)
- GET /v1/ecg/recordings/{recording_id} — retrieve prior analysis

## Run locally
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export API_KEY="dev_secret_123"  # optional; leave unset to disable auth
uvicorn main:app --host 0.0.0.0 --port 8000

## Test
curl -X POST "http://localhost:8000/v1/ecg/analyze"   -H "Content-Type: application/json"   -H "Authorization: Bearer dev_secret_123"   -d @payload.json

## Deploy options
- Vercel: serverless function or ASGI adapter
- Render: Web Service → Build: pip install -r requirements.txt → Start: uvicorn main:app --host 0.0.0.0 --port $PORT
- Railway/Fly/Azure App Service: similar

Public base URL examples:
- Vercel: https://<project>.vercel.app
- Render: https://<service>.onrender.com
- Railway: https://<app>.up.railway.app

Lovable endpoint: https://<your-deployment-domain>/v1/ecg/analyze
