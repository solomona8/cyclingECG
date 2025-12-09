# Quick Start: Connect Lovable to Your ECG API

Your API URL: **https://cyclingecg.onrender.com**

## ⚡ 3-Minute Setup

### 1. Verify Render is Running

Open: https://cyclingecg.onrender.com/docs

If it loads → You're ready! ✅
If it doesn't → Check Render dashboard and deploy.

### 2. Required Render Settings

In your Render dashboard (https://dashboard.render.com):

- **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variable**: `PYTHON_VERSION` = `3.11.0`

### 3. Add to Your Lovable App

```javascript
// Add at top of your file
const API_URL = 'https://cyclingecg.onrender.com';

// Analyze ECG function
async function analyzeECG(samples, samplingRate = 512) {
  const response = await fetch(`${API_URL}/v1/ecg/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      recording_id: `web_${Date.now()}`,
      samples: samples,          // Your ECG data array (min 100 samples)
      sampling_rate_hz: samplingRate,  // 128, 250, 256, or 512
      units: "mV",
      lead: "I",
      start_timestamp_utc: new Date().toISOString()
    })
  });

  return await response.json();
}

// Use it
const result = await analyzeECG(yourECGData, 512);
console.log(`Heart Rate: ${result.features.heart_rate_bpm.mean} BPM`);
console.log(`Rhythm: ${result.features.rhythm_classification}`);
console.log(`Summary: ${result.narrative.patient_summary}`);
```

### 4. Handle Cold Starts (Free Tier)

Free tier sleeps after 15 min. First request takes 30-50 seconds:

```javascript
async function analyzeWithWait(samples, samplingRate, setStatusMessage) {
  setStatusMessage('Analyzing ECG... (may take 30-50s if server was sleeping)');

  try {
    const result = await analyzeECG(samples, samplingRate);
    setStatusMessage('Complete!');
    return result;
  } catch (error) {
    setStatusMessage(`Error: ${error.message}`);
    throw error;
  }
}
```

## 📊 What You Get Back

```javascript
{
  features: {
    heart_rate_bpm: { mean: 72, min: 65, max: 80 },
    rhythm_classification: "sinus",
    intervals: {
      qrs_duration_ms: 90,
      qt_interval_ms: 380,
      qtc_ms: 420
    },
    morphology: {
      pvc_count: 0,
      pac_count: 0,
      ectopy_burden_percent: 0
    },
    hrv: { sdnn_ms: 50, rmssd_ms: 42 },
    signal_quality: { overall_quality: "good" }
  },
  narrative: {
    patient_summary: "ECG shows normal sinus rhythm...",
    clinician_notes: "QTc: 420ms, QRS: 90ms...",
    safety_flags: []
  }
}
```

## 🧪 Test It

1. **API Test**: `curl https://cyclingecg.onrender.com/health`
2. **Swagger UI**: https://cyclingecg.onrender.com/docs (test all endpoints)
3. **Sample Data**: Use 2560 samples (5 seconds @ 512 Hz) minimum

## ⚠️ Important Notes

- **Free Tier**: First request after 15 min takes 30-50 seconds (cold start)
- **Minimum Data**: At least 100 samples required, 2560+ recommended
- **Sampling Rates**: Only 128, 250, 256, or 512 Hz supported
- **CORS**: Already enabled for all origins

## 🚀 Upgrade to Production

When ready ($7/month):
- Always-on (no cold starts)
- Faster performance
- Better reliability

**Render Dashboard** → Your Service → Upgrade Plan

---

## 📚 Full Documentation

- **Complete Lovable Integration**: See `LOVABLE-INTEGRATION.md`
- **General Deployment Guide**: See `DEPLOYMENT.md`
- **Backend API Details**: See `README-ECG-BACKEND.md`

---

**That's it!** Copy the code above into your Lovable app and you're analyzing ECGs in the cloud.
