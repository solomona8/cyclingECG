# Lovable App Integration Guide

Your ECG backend is configured to deploy at: **https://cyclingecg.onrender.com**

## Step 1: Verify Render Configuration

Go to your Render dashboard (https://dashboard.render.com) and check your `cyclingecg` service:

### Required Settings

**Build & Deploy:**
- **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Branch**: `main` or `claude/apple-watch-data-extraction-01GXvTS5e9r2psgrXhe73bva`

**Environment:**
- **Python Version**: Add environment variable `PYTHON_VERSION = 3.11.0`
- **Optional Variables**:
  - `API_KEY` - Set if you want authentication
  - `OPENAI_API_KEY` - Set if you want AI-generated narratives

### If Service Isn't Running

1. Check **Logs** tab for errors
2. Click **Manual Deploy** → **Deploy latest commit**
3. Wait 2-3 minutes for build to complete
4. Free tier services sleep after 15 minutes - first request takes 30-50 seconds

---

## Step 2: Test Your API

Once deployed, test these endpoints:

### Health Check
```bash
curl https://cyclingecg.onrender.com/health
```
Expected: `{"status":"healthy"}`

### Interactive Documentation
Open in browser: **https://cyclingecg.onrender.com/docs**

You can test all endpoints directly from this Swagger UI interface.

---

## Step 3: Integrate with Your Lovable App

### Configuration

Add your API base URL (at the top of your main file or in a config):

```javascript
// config.js or at top of your main component
const API_CONFIG = {
  baseURL: 'https://cyclingecg.onrender.com',
  timeout: 60000, // 60 seconds for free tier cold starts
};
```

### Core ECG Analysis Function

```javascript
/**
 * Analyze ECG data via cloud API
 * @param {number[]} samples - ECG voltage samples
 * @param {number} samplingRateHz - Sample rate (128, 250, 256, or 512 Hz)
 * @param {string} units - Units: "mV", "uV", or "LSB"
 * @returns {Promise<Object>} Analysis results
 */
async function analyzeECG(samples, samplingRateHz = 512, units = "mV") {
  if (!samples || samples.length < 100) {
    throw new Error('At least 100 samples required for analysis');
  }

  const requestBody = {
    recording_id: `lovable_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    samples: samples,
    sampling_rate_hz: samplingRateHz,
    units: units,
    lead: "I", // Single-lead ECG
    start_timestamp_utc: new Date().toISOString(),
    device_info: {
      manufacturer: "Browser",
      model: "Web ECG",
      os_version: navigator.userAgent,
    },
  };

  try {
    const response = await fetch(`${API_CONFIG.baseURL}/v1/ecg/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // If you set API_KEY in Render environment:
        // 'Authorization': `Bearer ${YOUR_API_KEY}`,
      },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(API_CONFIG.timeout),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API error ${response.status}: ${errorText}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('ECG analysis failed:', error);
    throw error;
  }
}
```

### Handle Free Tier Cold Starts

Render free tier services sleep after 15 minutes. Handle this gracefully:

```javascript
/**
 * Analyze ECG with retry logic for cold starts
 * @param {Function} setStatusMessage - Function to update UI status
 */
async function analyzeECGWithRetry(samples, samplingRateHz, units, setStatusMessage) {
  const maxRetries = 2;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      if (attempt === 0) {
        setStatusMessage('Analyzing ECG...');
      } else {
        setStatusMessage('Server was sleeping, waking up... (30-50 seconds)');
      }

      const result = await analyzeECG(samples, samplingRateHz, units);
      setStatusMessage('Analysis complete!');
      return result;

    } catch (error) {
      if (attempt === maxRetries - 1) {
        setStatusMessage(`Analysis failed: ${error.message}`);
        throw error;
      }

      // Wait before retry
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }
}
```

### Display Results

```javascript
/**
 * Format and display ECG analysis results
 */
function displayResults(result) {
  const { features, narrative } = result;

  // Basic metrics
  const heartRate = features.heart_rate_bpm.mean.toFixed(0);
  const rhythm = features.rhythm_classification.replace(/_/g, ' ').toUpperCase();

  // Intervals
  const qrsDuration = features.intervals?.qrs_duration_ms?.toFixed(0) || 'N/A';
  const qtInterval = features.intervals?.qt_interval_ms?.toFixed(0) || 'N/A';
  const qtc = features.intervals?.qtc_ms?.toFixed(0) || 'N/A';

  // Arrhythmias
  const pvcCount = features.morphology?.pvc_count || 0;
  const pacCount = features.morphology?.pac_count || 0;
  const ectopyBurden = features.morphology?.ectopy_burden_percent?.toFixed(1) || 0;

  // HRV
  const sdnn = features.hrv?.sdnn_ms?.toFixed(0) || 'N/A';
  const rmssd = features.hrv?.rmssd_ms?.toFixed(0) || 'N/A';

  // Quality
  const quality = features.signal_quality?.overall_quality || 'unknown';

  // Narratives
  const patientSummary = narrative?.patient_summary || 'Analysis complete';
  const clinicianNotes = narrative?.clinician_notes || '';
  const safetyFlags = narrative?.safety_flags || [];

  return {
    // Primary display
    heartRate,
    rhythm,
    patientSummary,
    quality,

    // Detailed metrics
    intervals: {
      qrs: `${qrsDuration} ms`,
      qt: `${qtInterval} ms`,
      qtc: `${qtc} ms`,
    },

    arrhythmias: {
      pvcs: pvcCount,
      pacs: pacCount,
      burden: `${ectopyBurden}%`,
    },

    hrv: {
      sdnn: `${sdnn} ms`,
      rmssd: `${rmssd} ms`,
    },

    // Clinical info
    clinicianNotes,
    safetyFlags,

    // Full raw data
    raw: result,
  };
}
```

### Complete Example (React)

```jsx
import React, { useState } from 'react';

function ECGAnalyzer({ ecgData, samplingRate = 512 }) {
  const [status, setStatus] = useState('');
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const result = await analyzeECGWithRetry(
        ecgData,
        samplingRate,
        "mV",
        setStatus
      );

      const formatted = displayResults(result);
      setResults(formatted);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ecg-analyzer">
      <button
        onClick={handleAnalyze}
        disabled={loading || !ecgData || ecgData.length < 100}
      >
        {loading ? 'Analyzing...' : 'Analyze ECG'}
      </button>

      {status && (
        <div className="status-message">
          {status}
        </div>
      )}

      {error && (
        <div className="error-message">
          Error: {error}
        </div>
      )}

      {results && (
        <div className="results">
          <h3>ECG Analysis Results</h3>

          {/* Primary Results */}
          <div className="primary-results">
            <div className="metric">
              <span className="label">Heart Rate:</span>
              <span className="value">{results.heartRate} BPM</span>
            </div>
            <div className="metric">
              <span className="label">Rhythm:</span>
              <span className="value">{results.rhythm}</span>
            </div>
            <div className="metric">
              <span className="label">Quality:</span>
              <span className="value">{results.quality}</span>
            </div>
          </div>

          {/* Patient Summary */}
          <div className="summary">
            <p>{results.patientSummary}</p>
          </div>

          {/* Detailed Metrics */}
          <details>
            <summary>Detailed Measurements</summary>

            <h4>Intervals</h4>
            <ul>
              <li>QRS Duration: {results.intervals.qrs}</li>
              <li>QT Interval: {results.intervals.qt}</li>
              <li>QTc (Corrected): {results.intervals.qtc}</li>
            </ul>

            <h4>Arrhythmias</h4>
            <ul>
              <li>PVCs: {results.arrhythmias.pvcs}</li>
              <li>PACs: {results.arrhythmias.pacs}</li>
              <li>Ectopy Burden: {results.arrhythmias.burden}</li>
            </ul>

            <h4>Heart Rate Variability</h4>
            <ul>
              <li>SDNN: {results.hrv.sdnn}</li>
              <li>RMSSD: {results.hrv.rmssd}</li>
            </ul>
          </details>

          {/* Clinician Notes */}
          {results.clinicianNotes && (
            <details>
              <summary>Clinician Notes</summary>
              <p>{results.clinicianNotes}</p>
            </details>
          )}

          {/* Safety Flags */}
          {results.safetyFlags.length > 0 && (
            <div className="safety-flags warning">
              <h4>⚠️ Safety Flags</h4>
              <ul>
                {results.safetyFlags.map((flag, i) => (
                  <li key={i}>{flag}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ECGAnalyzer;
```

---

## Step 4: Testing Checklist

### Before Testing
- [ ] Render service shows "Live" status
- [ ] Health endpoint responds: `curl https://cyclingecg.onrender.com/health`
- [ ] Swagger UI loads: https://cyclingecg.onrender.com/docs

### During Testing
- [ ] First request may take 30-50 seconds (cold start)
- [ ] Check browser console for any CORS errors
- [ ] Verify response contains all expected fields

### Sample Test Data

Use this sample ECG data for testing (simulated normal sinus rhythm):

```javascript
// Generate 5 seconds of simulated ECG at 512 Hz
const testECGData = generateTestECG(512, 5);

function generateTestECG(samplingRate, durationSeconds) {
  const samples = samplingRate * durationSeconds;
  const ecg = [];
  const heartRate = 72; // bpm
  const beatInterval = (60 / heartRate) * samplingRate;

  for (let i = 0; i < samples; i++) {
    const phase = (i % beatInterval) / beatInterval;

    // Simplified QRS complex
    let value = 0;
    if (phase > 0.1 && phase < 0.2) {
      // P wave
      value = 0.1 * Math.sin((phase - 0.1) * 10 * Math.PI);
    } else if (phase > 0.25 && phase < 0.35) {
      // QRS complex
      value = 1.0 * Math.sin((phase - 0.25) * 10 * Math.PI);
    } else if (phase > 0.4 && phase < 0.6) {
      // T wave
      value = 0.3 * Math.sin((phase - 0.4) * 5 * Math.PI);
    }

    // Add small noise
    value += (Math.random() - 0.5) * 0.02;

    ecg.push(value);
  }

  return ecg;
}
```

---

## Step 5: Common Issues & Solutions

### "Connection failed" or "Network error"

**Cause**: Service is sleeping (free tier) or not deployed

**Solution**:
1. Open https://cyclingecg.onrender.com/docs in browser first to wake it up
2. Wait 30-50 seconds
3. Try again

### CORS Error in Browser Console

**Cause**: CORS not configured properly

**Solution**: Already configured in `app/main.py`. If issues persist:
1. Check Render logs for startup errors
2. Verify build succeeded
3. Redeploy if needed

### "Recording must have at least 100 samples"

**Cause**: Not enough data sent

**Solution**: Ensure your ECG data array has ≥100 samples (about 0.2 seconds at 512 Hz)

### Timeout Errors

**Cause**: Free tier cold start taking too long

**Solution**: Increase timeout in fetch (already set to 60s in examples above)

### "Unsupported sampling rate"

**Cause**: Invalid sampling rate

**Solution**: Use one of: 128, 250, 256, or 512 Hz

---

## Step 6: Going to Production

When ready to deploy your Lovable app to production:

### Recommended Upgrades

1. **Upgrade Render to Paid** ($7/month)
   - Always-on (no cold starts)
   - Faster CPU
   - Better reliability

2. **Add API Key Authentication**
   - Set `API_KEY` in Render environment
   - Add to your Lovable app config
   - Include in Authorization header

3. **Restrict CORS** (in `app/main.py`)
   ```python
   allow_origins=[
       "https://your-app.lovable.app",
       "https://your-custom-domain.com"
   ]
   ```

4. **Add Monitoring**
   - Render provides built-in metrics
   - Consider UptimeRobot for uptime monitoring
   - Set up Render email alerts

5. **Keep Alive (Optional for Free Tier)**
   ```javascript
   // Ping every 10 minutes to prevent sleep
   setInterval(() => {
     fetch('https://cyclingecg.onrender.com/health');
   }, 10 * 60 * 1000);
   ```

---

## API Reference

### POST /v1/ecg/analyze

Analyze ECG samples and return comprehensive cardiac metrics.

**Request:**
```json
{
  "recording_id": "string",
  "samples": [0.1, 0.15, ...],
  "sampling_rate_hz": 512,
  "units": "mV",
  "lead": "I",
  "start_timestamp_utc": "2025-12-09T12:00:00Z"
}
```

**Response:**
```json
{
  "recording_id": "string",
  "timestamp_utc": "2025-12-09T12:00:01Z",
  "features": {
    "rhythm_classification": "sinus",
    "heart_rate_bpm": {"mean": 72.0, "min": 65.0, "max": 80.0},
    "intervals": {
      "qrs_duration_ms": 90.0,
      "qt_interval_ms": 380.0,
      "qtc_ms": 420.0
    },
    "morphology": {
      "pvc_count": 0,
      "pac_count": 0,
      "ectopy_burden_percent": 0.0
    },
    "hrv": {
      "sdnn_ms": 50.0,
      "rmssd_ms": 42.0
    },
    "signal_quality": {
      "overall_quality": "good"
    }
  },
  "narrative": {
    "patient_summary": "Normal sinus rhythm detected...",
    "clinician_notes": "QTc: 420ms, QRS: 90ms...",
    "safety_flags": []
  }
}
```

---

## Support

- **API Documentation**: https://cyclingecg.onrender.com/docs
- **Render Dashboard**: https://dashboard.render.com
- **Check Service Logs**: Render Dashboard → Your Service → Logs

---

**You're all set!** Copy the code examples above into your Lovable app and start analyzing ECGs in the cloud.
