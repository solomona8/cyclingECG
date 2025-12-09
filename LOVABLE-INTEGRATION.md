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

**⚠️ CRITICAL: DO NOT USE LINEAR RAMPS FOR ECG DATA!**

If your ECG data looks like `[0, 1, 2, 3, ...]` or any constant increment pattern, the analysis will fail because the bandpass filter removes DC components. You need **realistic waveforms** with P waves, QRS complexes, and T waves.

#### Option 1: Use the Backend Test Data Generator (Recommended)

The backend now provides a `/v1/ecg/generate_test_data` endpoint that generates realistic ECG:

```javascript
/**
 * Fetch realistic test ECG data from the backend
 */
async function getTestECGData(durationSec = 10, samplingRateHz = 512, heartRateBpm = 72) {
  const response = await fetch(
    `${API_CONFIG.baseURL}/v1/ecg/generate_test_data?` +
    `duration_sec=${durationSec}&` +
    `sampling_rate_hz=${samplingRateHz}&` +
    `heart_rate_bpm=${heartRateBpm}`
  );

  if (!response.ok) {
    throw new Error(`Failed to generate test data: ${response.status}`);
  }

  const data = await response.json();
  console.log(data.note); // Confirms it's realistic data, not a linear ramp
  return data.samples; // Array of realistic ECG values in mV
}

// Usage:
const testECGData = await getTestECGData(10, 512, 72);
const result = await analyzeECG(testECGData, 512, "mV");
```

#### Option 2: Generate Test Data Locally (Client-Side)

If you prefer to generate test data in the browser:

```javascript
/**
 * Generate realistic synthetic ECG with P waves, QRS complexes, and T waves
 * @param {number} samplingRateHz - Sample rate (128, 256, or 512 Hz)
 * @param {number} durationSeconds - Duration (1-30 seconds)
 * @param {number} heartRateBpm - Heart rate (40-180 bpm)
 * @returns {number[]} Array of ECG voltage samples in mV
 */
function generateTestECG(samplingRateHz = 512, durationSeconds = 10, heartRateBpm = 72) {
  const numSamples = samplingRateHz * durationSeconds;
  const rrSec = 60.0 / heartRateBpm;

  // Generate beat times with heart rate variability
  const beatTimes = [];
  let currentTime = 0.3; // Start after 0.3 seconds
  while (currentTime < durationSeconds) {
    beatTimes.push(currentTime);
    // Add HRV: small random variation in RR interval
    currentTime += rrSec + (Math.random() - 0.5) * 0.04;
  }

  // Initialize with baseline noise
  const ecg = Array(numSamples).fill(0).map(() => (Math.random() - 0.5) * 0.1);

  // Add cardiac waveforms for each beat
  beatTimes.forEach(beatTime => {
    const beatIdx = Math.floor(beatTime * samplingRateHz);

    // QRS complex (scaled for sampling rate)
    const qrsWidth = Math.floor(20 * samplingRateHz / 512);

    if (beatIdx + qrsWidth < numSamples) {
      // Q wave (small negative deflection)
      const qWidth = Math.max(1, Math.floor(qrsWidth / 4));
      for (let i = 0; i < qWidth; i++) {
        ecg[beatIdx + i] -= 0.2;
      }

      // R wave (large positive deflection - this is what R-peak detection looks for!)
      const rWidth = Math.max(1, Math.floor(qrsWidth / 2));
      for (let i = 0; i < rWidth; i++) {
        ecg[beatIdx + qWidth + i] += 1.0;
      }

      // S wave (negative deflection)
      for (let i = qWidth + rWidth; i < qrsWidth; i++) {
        ecg[beatIdx + i] -= 0.3;
      }
    }

    // P wave (before QRS)
    const pIdx = beatIdx - Math.floor(40 * samplingRateHz / 512);
    const pWidth = Math.floor(15 * samplingRateHz / 512);
    if (pIdx > 0 && pIdx + pWidth < numSamples) {
      for (let i = 0; i < pWidth; i++) {
        ecg[pIdx + i] += 0.15;
      }
    }

    // T wave (after QRS)
    const tIdx = beatIdx + qrsWidth + Math.floor(60 * samplingRateHz / 512);
    const tWidth = Math.floor(30 * samplingRateHz / 512);
    if (tIdx + tWidth < numSamples) {
      for (let i = 0; i < tWidth; i++) {
        ecg[tIdx + i] += 0.25;
      }
    }
  });

  return ecg;
}

// Usage:
const testECGData = generateTestECG(512, 10, 72);
const result = await analyzeECG(testECGData, 512, "mV");
```

#### How to Verify Your Test Data is Correct

Before sending to the backend, verify your ECG data:

```javascript
function verifyECGData(samples) {
  if (samples.length < 100) {
    console.error("❌ Too few samples:", samples.length);
    return false;
  }

  // Check for linear ramp (BAD!)
  const diffs = [];
  for (let i = 1; i < Math.min(100, samples.length); i++) {
    diffs.push(samples[i] - samples[i-1]);
  }
  const avgDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
  const diffStd = Math.sqrt(
    diffs.map(d => (d - avgDiff) ** 2).reduce((a, b) => a + b, 0) / diffs.length
  );

  if (diffStd < Math.abs(avgDiff) * 0.01 && avgDiff !== 0) {
    console.error("❌ LINEAR RAMP DETECTED! This will fail R-peak detection.");
    console.error(`   Average increment: ${avgDiff}, Std: ${diffStd}`);
    console.error("   Use generateTestECG() or /v1/ecg/generate_test_data instead!");
    return false;
  }

  console.log("✅ ECG data looks realistic");
  console.log(`   Samples: ${samples.length}`);
  console.log(`   Range: ${Math.min(...samples).toFixed(3)} to ${Math.max(...samples).toFixed(3)} mV`);
  console.log(`   Increment std: ${diffStd.toFixed(6)} (good variability)`);
  return true;
}

// Before analyzing:
const testData = generateTestECG(512, 10, 72);
if (verifyECGData(testData)) {
  const result = await analyzeECG(testData, 512, "mV");
}
```

---

## Step 5: Common Issues & Solutions

### "No R-peaks detected" or "0 beats detected"

**Cause**: You're sending **linear ramp test data** (e.g., `[0, 1, 2, 3, ...]`) instead of realistic ECG waveforms

**How to identify**:
- Check backend logs on Render for: `⚠️ WARNING: Data appears to be a LINEAR RAMP`
- Your data increments are constant (e.g., always +0.0019 between samples)
- After bandpass filtering, the signal becomes near-zero

**Solution**:
1. Replace your test data generator with one of the options in "Sample Test Data" above
2. Use `/v1/ecg/generate_test_data` endpoint to get realistic data
3. Run `verifyECGData()` before sending to ensure proper waveforms
4. Real ECG needs P waves, QRS complexes, and T waves - not a ramp!

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
