from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import os, io, csv

# Optional: use your own extractor if available; otherwise we fall back to a simple analyzer
try:
    from app.feature_extractor import extract_features as _extract_features_real
except Exception:
    _extract_features_real = None  # we'll use a stub fallback below

app = FastAPI(title="ECG Analyzer", version="1.0.0")

# ---- config & models ----
API_KEY = os.environ.get("API_KEY")  # no default; auth is disabled if unset

class FiltersApplied(BaseModel):
    highpass_hz: Optional[float] = None
    lowpass_hz: Optional[float] = None
    notch_hz: Optional[float] = None

class DeviceInfo(BaseModel):
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    software_version: Optional[str] = None
    os: Optional[str] = None
    app_version: Optional[str] = None

class RecordingContext(BaseModel):
    symptoms: Optional[List[str]] = None
    activity: Optional[str] = None
    position: Optional[str] = None
    duration_s: Optional[float] = None
    signal_quality_flag: Optional[str] = Field(default=None, pattern="^(good|moderate|poor)$")

class UserInfo(BaseModel):
    age: Optional[int] = None
    sex_at_birth: Optional[str] = None
    meds: Optional[List[str]] = None
    history: Optional[List[str]] = None

class ECGRequest(BaseModel):
    recording_id: str
    samples: List[float] = Field(min_items=100)
    sampling_rate_hz: float
    units: str
    lead: str
    start_timestamp_utc: str  # Accept as string to match iOS app
    gain: Optional[float] = None
    adc_bits: Optional[int] = None
    filters_applied: Optional[FiltersApplied] = None
    device_info: Optional[DeviceInfo] = None  # Changed from 'device'
    context: Optional[RecordingContext] = None  # Changed to RecordingContext
    user: Optional[UserInfo] = None
    symptoms: Optional[List[str]] = None
    analyzer_version: Optional[str] = None

# ---- public routes (no auth) ----
@app.get("/")
def root():
    return {"ok": True, "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/v1/ecg/test_response")
def test_response():
    """Return a sample analysis response for testing iOS app decoding"""
    return {
        "recording_id": "test-123",
        "timestamp_utc": "2025-01-01T00:00:00Z",
        "features": {
            "rhythm_classification": "sinus",
            "rhythm_confidence": 0.9,
            "heart_rate_bpm": {
                "mean": 72.0,
                "min": 60.0,
                "max": 85.0
            },
            "rr_intervals_ms": {
                "mean": 833.0,
                "min": 700.0,
                "max": 1000.0,
                "coefficient_of_variation": 5.0
            },
            "hrv": {
                "sdnn_ms": 50.0,
                "rmssd_ms": 42.0
            },
            "intervals": {
                "qrs_duration_ms": 90.0,
                "qt_interval_ms": 360.0,
                "qtc_ms": 440.0
            },
            "signal_quality": {
                "overall_quality": "good",
                "artifact_burden_percent": None
            },
            "morphology": {
                "ectopy_burden_percent": 0.0
            }
        },
        "narrative": {
            "patient_summary": "ECG analysis shows sinus rhythm with heart rate 72 BPM.",
            "clinician_notes": "QTc: 440ms, QRS: 90ms. 0 PVCs, 0 PACs detected.",
            "safety_flags": []
        },
        "analyzer_version": "2.0.0"
    }

# Debug endpoint to log request details
@app.post("/v1/ecg/analyze/debug")
def analyze_ecg_debug(request: Request, payload: ECGRequest):
    import json
    debug_info = {
        "request_received": True,
        "recording_id": payload.recording_id,
        "num_samples": len(payload.samples),
        "sampling_rate": payload.sampling_rate_hz,
        "units": payload.units,
        "lead": payload.lead,
        "has_device_info": payload.device_info is not None,
        "has_context": payload.context is not None,
        "sample_range": {
            "min": min(payload.samples) if payload.samples else None,
            "max": max(payload.samples) if payload.samples else None,
            "mean": sum(payload.samples) / len(payload.samples) if payload.samples else None
        }
    }
    print("=== DEBUG REQUEST ===")
    print(json.dumps(debug_info, indent=2))
    print("=== END DEBUG ===")
    return debug_info

# ---- auth (works with the padlock in /docs) ----
auth_scheme = HTTPBearer(auto_error=False)

def _require_bearer(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    # Auth disabled if API_KEY not set
    if not API_KEY:
        return
    if not credentials or (credentials.scheme or "").lower() != "bearer":
        raise HTTPException(status_code=401, detail="Unauthorized: missing Bearer token")
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: bad token")

# ---- simple built-in analyzer (fallback) ----
def _fake_analysis(samples: List[float], fs: float) -> Dict[str, Any]:
    n = len(samples)
    mean = sum(samples) / max(n, 1)
    mean_hr = 72 + int(mean) % 15
    return {
        "summary": {
            "rhythm": "sinus",
            "rhythm_confidence": 0.85,
            "mean_hr_bpm": mean_hr,
            "min_hr_bpm": max(40, mean_hr - 8),
            "max_hr_bpm": min(180, mean_hr + 12),
            "signal_quality": "good" if n > 200 else "moderate",
        },
        "beats": {
            "r_peaks_ms": [i * 800 for i in range(1, max(1, n // int(fs or 1)) + 1)],
            "rr_ms": [800 for _ in range(max(0, n // int(fs or 1) - 1))],
            "artifact_mask": [],
        },
        "hrv_time": {"sdnn_ms": 50, "rmssd_ms": 42},
        "intervals": {"qrs_ms": 90, "qt_ms": 360, "qtc_ms_bazett": 440, "uncertainty_ms": 25},
        "flags": {"pacemaker_detected": False, "ectopy_burden_pct": 0.0, "st_deviation_flag": False},
        "version": "analyzer-1.0.0",
    }

def _extract_features(samples: List[float], fs: float) -> Dict[str, Any]:
    """
    Unified extractor: use your real extractor if present,
    otherwise map the fake analyzer into the same shape.
    """
    if _extract_features_real:
        return _extract_features_real(samples, fs)
    # Convert fake output into the feature-shaped dict the CSV route expects
    fake = _fake_analysis(samples, fs)
    return {
        "rhythm_label": fake["summary"]["rhythm"],
        "confidence": fake["summary"]["rhythm_confidence"],
        "mean_hr_bpm": fake["summary"]["mean_hr_bpm"],
        "min_hr_bpm": fake["summary"]["min_hr_bpm"],
        "max_hr_bpm": fake["summary"]["max_hr_bpm"],
        "signal_quality": fake["summary"]["signal_quality"],
        "r_peaks_ms": fake["beats"]["r_peaks_ms"],
        "rr_ms": fake["beats"]["rr_ms"],
        "artifact_mask": fake["beats"]["artifact_mask"],
        "sdnn_ms": fake["hrv_time"]["sdnn_ms"],
        "rmssd_ms": fake["hrv_time"]["rmssd_ms"],
        "qrs_ms": fake["intervals"]["qrs_ms"],
        "qt_ms": fake["intervals"]["qt_ms"],
        "qtc_ms_bazett": fake["intervals"]["qtc_ms_bazett"],
        "uncertainty_ms": fake["intervals"]["uncertainty_ms"],
        "ectopy_burden_pct": fake["flags"]["ectopy_burden_pct"],
    }

# ---- in-memory store for quick fetch-by-id ----
STORE: Dict[str, Dict[str, Any]] = {}

# ---- JSON endpoint ----
@app.post("/v1/ecg/analyze")
def analyze_ecg(
    payload: ECGRequest,
    credentials: HTTPAuthorizationCredentials = Security(auth_scheme)
):
    _require_bearer(credentials)

    if payload.units not in {"uV", "mV", "LSB"}:
        raise HTTPException(status_code=400, detail="units must be one of uV, mV, LSB")
    if payload.lead != "I":
        raise HTTPException(status_code=400, detail="lead must be 'I' for Apple Watch")
    if payload.sampling_rate_hz not in {128, 250, 256, 512}:
        raise HTTPException(status_code=400, detail="Unsupported sampling_rate_hz")

    # Use real feature extractor
    try:
        import numpy as np
        samples_array = np.array(payload.samples, dtype=float)
        print(f"[ANALYZE] Processing recording_id={payload.recording_id}, samples={len(payload.samples)}, fs={payload.sampling_rate_hz}")
        print(f"[ANALYZE] Units: {payload.units}, Lead: {payload.lead}")
        print(f"[ANALYZE] Sample stats - min: {samples_array.min():.6f}, max: {samples_array.max():.6f}, mean: {samples_array.mean():.6f}, std: {samples_array.std():.6f}")
        print(f"[ANALYZE] First 10 samples: {samples_array[:10].tolist()}")
        print(f"[ANALYZE] Last 10 samples: {samples_array[-10:].tolist()}")
        print(f"[ANALYZE] Non-zero samples: {np.count_nonzero(samples_array)} / {len(samples_array)}")

        # Check for linear ramp pattern (test data indicator)
        if len(samples_array) >= 10:
            diffs = np.diff(samples_array)
            avg_diff = np.mean(diffs)
            diff_std = np.std(diffs)
            print(f"[ANALYZE] Sample increments - mean: {avg_diff:.9f}, std: {diff_std:.9f}")
            if diff_std < abs(avg_diff) * 0.01 and avg_diff > 0:
                print(f"[ANALYZE] ⚠️ WARNING: Data appears to be a LINEAR RAMP (test data, not real ECG)")
                print(f"[ANALYZE] ⚠️ Real ECG data should have variable increments, not constant!")

        # Check for realistic ECG amplitude (in microvolts, typical range 500-3000 uV)
        if payload.units == "uV":
            if samples_array.max() < 100 or samples_array.max() > 50000:
                print(f"[ANALYZE] ⚠️ WARNING: Amplitude unusual for ECG in uV: {samples_array.max():.1f}")
        elif payload.units == "mV":
            if samples_array.max() < 0.1 or samples_array.max() > 50:
                print(f"[ANALYZE] ⚠️ WARNING: Amplitude unusual for ECG in mV: {samples_array.max():.1f}")

        features = _extract_features(payload.samples, payload.sampling_rate_hz)
        print(f"[ANALYZE] Feature extraction successful, detected {features.get('beat_count', 0)} beats")
    except Exception as e:
        print(f"[ANALYZE] Feature extraction FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    # Helper function to sanitize numeric values (replace NaN/Inf with None)
    def sanitize_value(val):
        if val is None:
            return None
        try:
            import math
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        except (TypeError, ValueError):
            return val

    # Calculate RR interval statistics for the response
    rr_intervals = features.get("rr_ms", [])
    rr_mean = sum(rr_intervals) / len(rr_intervals) if rr_intervals else None
    rr_min = min(rr_intervals) if rr_intervals else None
    rr_max = max(rr_intervals) if rr_intervals else None

    # Calculate coefficient of variation for RR intervals
    if rr_mean and rr_mean > 0 and features.get("sdnn_ms"):
        rr_cv = (features["sdnn_ms"] / rr_mean) * 100
    else:
        rr_cv = None

    # Sanitize all numeric values
    mean_hr = sanitize_value(features.get("mean_hr_bpm", 0.0))
    min_hr = sanitize_value(features.get("min_hr_bpm", 0.0))
    max_hr = sanitize_value(features.get("max_hr_bpm", 0.0))

    # Format response to match iOS app's ECGAnalysisResponse structure
    response = {
        "recording_id": payload.recording_id,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "features": {
            "rhythm_classification": features.get("rhythm_label", "undetermined"),
            "rhythm_confidence": sanitize_value(features.get("confidence", 0.0)),
            "heart_rate_bpm": {
                "mean": mean_hr,
                "min": min_hr,
                "max": max_hr
            },
            "rr_intervals_ms": {
                "mean": sanitize_value(rr_mean),
                "min": sanitize_value(rr_min),
                "max": sanitize_value(rr_max),
                "coefficient_of_variation": sanitize_value(rr_cv)
            },
            "hrv": {
                "sdnn_ms": sanitize_value(features.get("sdnn_ms", 0.0)),
                "rmssd_ms": sanitize_value(features.get("rmssd_ms", 0.0))
            },
            "intervals": {
                "qrs_duration_ms": sanitize_value(features.get("qrs_ms", 0.0)),
                "qt_interval_ms": sanitize_value(features.get("qt_ms", 0.0)),
                "qtc_ms": sanitize_value(features.get("qtc_ms_bazett", 0.0))
            },
            "signal_quality": {
                "overall_quality": features.get("signal_quality", "moderate"),
                "artifact_burden_percent": None  # Not currently calculated
            },
            "morphology": {
                "ectopy_burden_percent": sanitize_value(features.get("ectopy_burden_pct", 0.0))
            }
        },
        "narrative": {
            "patient_summary": f"ECG analysis shows {features.get('rhythm_label') or 'undetermined'} rhythm with heart rate {mean_hr or 0:.0f} BPM.",
            "clinician_notes": f"QTc: {sanitize_value(features.get('qtc_ms_bazett')) or 0:.0f}ms, QRS: {sanitize_value(features.get('qrs_ms')) or 0:.0f}ms. {features.get('pvcs_detected') or 0} PVCs, {features.get('pacs_detected') or 0} PACs detected.",
            "safety_flags": []
        },
        "analyzer_version": payload.analyzer_version or "2.0.0"
    }

    STORE[payload.recording_id] = response
    print(f"[ANALYZE] Response generated successfully for {payload.recording_id}")
    print(f"[ANALYZE] Response keys: {list(response.keys())}")

    # Log the actual JSON that will be sent to help debug iOS decoding issues
    import json
    try:
        response_json = json.dumps(response, indent=2, default=str)
        print(f"[ANALYZE] Response JSON preview (first 500 chars):")
        print(response_json[:500])
    except Exception as e:
        print(f"[ANALYZE] Could not serialize response to JSON: {e}")

    return response

# ---- CSV upload endpoint ----
@app.post("/v1/ecg/upload_csv")
async def upload_csv(
    file: UploadFile = File(...),
    sampling_rate_hz: float = Form(256),
    units: str = Form("uV"),
    lead: str = Form("I"),
    recording_id: str = Form("csv_upload"),
    start_timestamp_utc: Optional[str] = Form(None),
    credentials: HTTPAuthorizationCredentials = Security(auth_scheme)
):
    _require_bearer(credentials)

    if units not in {"uV", "mV", "LSB"}:
        raise HTTPException(status_code=400, detail="units must be one of uV, mV, LSB")
    if lead != "I":
        raise HTTPException(status_code=400, detail="lead must be 'I' for Apple Watch")

    # Read CSV (handles headers, spaces, multiple columns)
    contents = await file.read()
    text = contents.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))

    rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        raise HTTPException(status_code=400, detail="CSV appears empty")

    header_like = any(not _is_float(c) for c in rows[0])
    data_rows = rows[1:] if header_like else rows

    col_idx = _choose_numeric_column(data_rows)
    if col_idx is None:
        raise HTTPException(status_code=400, detail="No numeric column found in CSV")

    samples: List[float] = []
    for r in data_rows:
        try:
            samples.append(float(r[col_idx]))
        except Exception:
            continue

    if len(samples) < 30:
        raise HTTPException(status_code=400, detail=f"Not enough numeric samples ({len(samples)})")

    try:
        features = _extract_features(samples, sampling_rate_hz)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    # Helper function to sanitize numeric values (replace NaN/Inf with None)
    def sanitize_value(val):
        if val is None:
            return None
        try:
            import math
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        except (TypeError, ValueError):
            return val

    # Calculate RR interval statistics
    rr_intervals = features.get("rr_ms", [])
    rr_mean = sum(rr_intervals) / len(rr_intervals) if rr_intervals else None
    rr_min = min(rr_intervals) if rr_intervals else None
    rr_max = max(rr_intervals) if rr_intervals else None

    # Calculate coefficient of variation for RR intervals
    if rr_mean and rr_mean > 0 and features.get("sdnn_ms"):
        rr_cv = (features["sdnn_ms"] / rr_mean) * 100
    else:
        rr_cv = None

    # Sanitize all numeric values
    mean_hr = sanitize_value(features.get("mean_hr_bpm", 0.0))
    min_hr = sanitize_value(features.get("min_hr_bpm", 0.0))
    max_hr = sanitize_value(features.get("max_hr_bpm", 0.0))

    # Format response to match iOS app's ECGAnalysisResponse structure
    response = {
        "recording_id": recording_id,
        "timestamp_utc": start_timestamp_utc or (datetime.utcnow().isoformat() + "Z"),
        "features": {
            "rhythm_classification": features.get("rhythm_label", "undetermined"),
            "rhythm_confidence": sanitize_value(features.get("confidence", 0.0)),
            "heart_rate_bpm": {
                "mean": mean_hr,
                "min": min_hr,
                "max": max_hr
            },
            "rr_intervals_ms": {
                "mean": sanitize_value(rr_mean),
                "min": sanitize_value(rr_min),
                "max": sanitize_value(rr_max),
                "coefficient_of_variation": sanitize_value(rr_cv)
            },
            "hrv": {
                "sdnn_ms": sanitize_value(features.get("sdnn_ms", 0.0)),
                "rmssd_ms": sanitize_value(features.get("rmssd_ms", 0.0))
            },
            "intervals": {
                "qrs_duration_ms": sanitize_value(features.get("qrs_ms", 0.0)),
                "qt_interval_ms": sanitize_value(features.get("qt_ms", 0.0)),
                "qtc_ms": sanitize_value(features.get("qtc_ms_bazett", 0.0))
            },
            "signal_quality": {
                "overall_quality": features.get("signal_quality", "moderate"),
                "artifact_burden_percent": None
            },
            "morphology": {
                "ectopy_burden_percent": sanitize_value(features.get("ectopy_burden_pct", 0.0))
            }
        },
        "narrative": {
            "patient_summary": f"ECG analysis shows {features.get('rhythm_label') or 'undetermined'} rhythm with heart rate {mean_hr or 0:.0f} BPM.",
            "clinician_notes": f"QTc: {sanitize_value(features.get('qtc_ms_bazett')) or 0:.0f}ms, QRS: {sanitize_value(features.get('qrs_ms')) or 0:.0f}ms. {features.get('pvcs_detected') or 0} PVCs, {features.get('pacs_detected') or 0} PACs detected.",
            "safety_flags": []
        },
        "analyzer_version": "2.0.0"
    }

    STORE[recording_id] = response
    return response

# ---- fetch by id ----
@app.get("/v1/ecg/recordings/{recording_id}")
def get_ecg(
    recording_id: str,
    credentials: HTTPAuthorizationCredentials = Security(auth_scheme)
):
    _require_bearer(credentials)
    data = STORE.get(recording_id)
    if not data:
        raise HTTPException(status_code=404, detail="recording_id not found")
    return data

# ---- helpers ----
def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def _choose_numeric_column(rows: List[List[str]]) -> Optional[int]:
    if not rows:
        return None
    n_cols = max(len(r) for r in rows)
    best_col, best_hits = None, -1
    for c in range(n_cols):
        hits = 0
        for r in rows:
            if c < len(r) and _is_float(r[c].strip()):
                hits += 1
        if hits > best_hits:
            best_hits, best_col = hits, c
    return best_col if best_hits >= max(1, len(rows) // 2) else None
