# ECG Backend Deployment Guide

This guide explains how to deploy your ECG analysis backend to the cloud and connect it to your Lovable web app.

## Table of Contents

1. [Quick Start - Deploy to Render](#quick-start---deploy-to-render)
2. [Alternative Deployment Options](#alternative-deployment-options)
3. [Connecting Your Lovable App](#connecting-your-lovable-app)
4. [Environment Variables](#environment-variables)
5. [Testing Your Deployment](#testing-your-deployment)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start - Deploy to Render

Render is recommended for its simplicity and free tier.

### Prerequisites

- GitHub account
- Render account (free tier available at https://render.com)
- This code pushed to a GitHub repository

### Step 1: Push Code to GitHub

```bash
# If not already in a git repo
git init
git add .
git commit -m "Prepare ECG backend for deployment"

# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy to Render

**Option A: Using render.yaml (Recommended)**

1. Go to https://render.com/dashboard
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml` and configure everything
5. Click **"Apply"**

**Option B: Manual Setup**

1. Go to https://render.com/dashboard
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `cycling-ecg-api` (or your choice)
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
5. Click **"Create Web Service"**

### Step 3: Configure Environment Variables (Optional)

In your Render dashboard:

1. Go to your service → **"Environment"**
2. Add variables (all optional):
   - `API_KEY`: Set a secure token if you want API authentication
   - `OPENAI_API_KEY`: Your OpenAI API key for AI-generated narratives
   - `OPENAI_MODEL`: `gpt-4-turbo` or `gpt-4`
3. Save changes (will trigger a redeploy)

### Step 4: Get Your API URL

Once deployed, Render gives you a URL like:
```
https://cycling-ecg-api.onrender.com
```

**Important**: Free tier services spin down after 15 minutes of inactivity. First request after sleep takes ~30 seconds.

---

## Alternative Deployment Options

### Railway.app

1. Visit https://railway.app
2. Click **"Start a New Project"** → **"Deploy from GitHub repo"**
3. Select your repository
4. Railway auto-detects Python and uses `requirements.txt`
5. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables in Settings → Variables

### Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Create app
flyctl launch

# Deploy
flyctl deploy
```

### Docker (Any Platform)

```bash
# Build image
docker build -t ecg-backend .

# Run locally
docker run -p 8000:8000 ecg-backend

# Push to Docker Hub or deploy to any container platform
```

### Vercel (Serverless)

Requires additional setup for ASGI adapter. Not recommended due to cold start times for ML workloads.

---

## Connecting Your Lovable App

Once deployed, you'll connect your Lovable app to your backend API.

### Step 1: Update API Base URL in Lovable

In your Lovable project, find where you make API calls and update the base URL:

**Before:**
```javascript
const API_BASE_URL = 'http://localhost:8000';
```

**After:**
```javascript
const API_BASE_URL = 'https://cycling-ecg-api.onrender.com';
// Or your Railway/Fly.io URL
```

### Step 2: Handle CORS (If Needed)

The backend already has CORS enabled for all origins in `app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # All origins allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For production, you may want to restrict origins to only your Lovable app domain:

```python
allow_origins=[
    "https://your-app.lovable.app",
    "https://your-custom-domain.com"
]
```

### Step 3: Example API Call from Lovable

**Analyze ECG Data:**

```javascript
async function analyzeECG(samples, samplingRate = 512) {
  try {
    const response = await fetch(`${API_BASE_URL}/v1/ecg/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // If you set API_KEY in environment:
        // 'Authorization': 'Bearer YOUR_API_KEY'
      },
      body: JSON.stringify({
        recording_id: `web_${Date.now()}`,
        samples: samples,  // Array of numbers
        sampling_rate_hz: samplingRate,
        units: "mV",
        lead: "I",
        start_timestamp_utc: new Date().toISOString()
      })
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const result = await response.json();

    // Result contains:
    // - features.heart_rate_bpm
    // - features.rhythm_classification
    // - features.intervals (QRS, QT, QTc)
    // - narrative.patient_summary
    // - narrative.clinician_notes

    return result;
  } catch (error) {
    console.error('ECG analysis failed:', error);
    throw error;
  }
}

// Usage
const ecgSamples = [/* your ECG data array */];
const results = await analyzeECG(ecgSamples, 512);
console.log(`Heart Rate: ${results.features.heart_rate_bpm.mean} BPM`);
console.log(`Rhythm: ${results.features.rhythm_classification}`);
console.log(`Summary: ${results.narrative.patient_summary}`);
```

### Step 4: Handle Cold Starts (Render Free Tier)

Free tier services sleep after 15 minutes. Handle this in your UI:

```javascript
async function analyzeECGWithRetry(samples, samplingRate = 512, retries = 2) {
  for (let i = 0; i < retries; i++) {
    try {
      // Show loading message on first attempt
      if (i === 0) {
        showMessage("Analyzing ECG...");
      } else {
        showMessage("Waking up server, please wait...");
      }

      const result = await analyzeECG(samples, samplingRate);
      return result;
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5s
    }
  }
}
```

---

## Environment Variables

Copy `.env.example` to `.env` for local development:

```bash
cp .env.example .env
```

### Available Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `API_KEY` | No | Bearer token for API auth | `your-secret-key-123` |
| `OPENAI_API_KEY` | No | OpenAI API key for narratives | `sk-...` |
| `OPENAI_MODEL` | No | OpenAI model to use | `gpt-4-turbo` |
| `PORT` | No* | Server port | `8000` |

*Cloud platforms set `PORT` automatically

---

## Testing Your Deployment

### 1. Health Check

```bash
curl https://cycling-ecg-api.onrender.com/health
```

Expected response:
```json
{"status": "healthy"}
```

### 2. Test ECG Analysis

```bash
curl -X POST https://cycling-ecg-api.onrender.com/v1/ecg/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "recording_id": "test_001",
    "samples": [0.1, 0.15, 0.12, 0.08, 0.05, ...],
    "sampling_rate_hz": 512,
    "units": "mV",
    "lead": "I",
    "start_timestamp_utc": "2025-12-09T12:00:00Z"
  }'
```

### 3. Interactive API Documentation

Visit in your browser:
- Swagger UI: `https://cycling-ecg-api.onrender.com/docs`
- ReDoc: `https://cycling-ecg-api.onrender.com/redoc`

You can test all endpoints directly from the Swagger UI.

---

## Troubleshooting

### Service Won't Start

**Check Build Logs**: In Render/Railway dashboard, view deployment logs

Common issues:
- Missing dependencies → Check `requirements.txt`
- Wrong Python version → Render uses Python 3.7 by default, specify 3.11 in environment
- Port binding → Ensure using `--host 0.0.0.0 --port $PORT`

### CORS Errors in Browser

If you see CORS errors in your Lovable app console:

1. Check `app/main.py` has CORS middleware enabled
2. Add your Lovable domain to `allow_origins` list
3. Redeploy

### Slow First Request (Render Free Tier)

This is normal - service sleeps after 15 minutes of inactivity.

Solutions:
1. Upgrade to paid tier ($7/month for always-on)
2. Keep-alive ping from your Lovable app every 10 minutes
3. Show "Waking up server" message to users

### API Key Authentication Fails

If you set `API_KEY` environment variable:

```javascript
headers: {
  'Authorization': `Bearer ${YOUR_API_KEY}`
}
```

Must match exactly.

### OpenAI Narratives Not Working

Currently, `app/openai_narrative.py` is not integrated into the main endpoint. This is a work-in-progress feature.

---

## Next Steps

1. ✅ Deploy your backend to Render (or alternative)
2. ✅ Get your deployment URL
3. ✅ Update your Lovable app's API base URL
4. ✅ Test ECG analysis from your web app
5. Optional: Set up custom domain
6. Optional: Add monitoring (UptimeRobot, etc.)
7. Optional: Implement API key authentication

---

## Support

- **Backend Issues**: Check logs in Render dashboard
- **API Documentation**: Visit `/docs` endpoint on your deployed URL
- **Lovable Integration**: Check browser console for errors

## Upgrading from Free Tier

When ready for production:

- **Render**: $7/month for always-on, faster cold starts
- **Railway**: Pay-as-you-go, ~$5-10/month typical usage
- **Fly.io**: Free tier more generous, then pay-as-you-go

---

## Security Best Practices

1. **Set API_KEY** for production to prevent unauthorized access
2. **Restrict CORS** to only your domains
3. **Use HTTPS** (automatic on Render/Railway/Fly)
4. **Validate input** (already implemented with Pydantic)
5. **Monitor usage** to prevent abuse
6. **Keep dependencies updated**: `pip install --upgrade -r requirements.txt`

---

**You're all set!** Your ECG backend is ready to power your Lovable web app with real-time cardiac analysis.
