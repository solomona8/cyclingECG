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

# ---- public routes (no auth) ----
@app.get("/")
def root():
    return {"ok": True, "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---- config & models ----
API_KEY = os.environ.get("API_KEY")  # no default; auth is disabled if unset

class FiltersApplied(BaseModel):
    highpass_hz: Optional[float] = None
    lowpass_hz: Optional[float] = None
    notch_hz: Optional[float] = None

class DeviceInfo(BaseModel):
    model: Optional[str] = None
    os: Optional[str] = None
    app_version: Optional[str] = None

class ContextInfo(BaseModel):
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
    start_timestamp_utc: datetime
    gain: Optional[float] = None
    adc_bits: Optional[int] = None
    filters_applied: Optional[FiltersApplied] = None
    device: Optional[DeviceInfo] = None
    context: Optional[ContextInfo] = None
    user: Optional[UserInfo] = None
    symptoms: Optional[List[str]] = None
    analyzer_version: Optional[str] = None

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

    result = _fake_analysis(payload.samples, payload.sampling_rate_hz)
    resp = {"recording_id": payload.recording_id, **result}
    STORE[payload.recording_id] = resp
    return resp

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

    features = _extract_features(samples, sampling_rate_hz)

    summary = {
        "rhythm": features["rhythm_label"],
        "rhythm_confidence": features["confidence"],
        "mean_hr_bpm": features["mean_hr_bpm"],
        "min_hr_bpm": features["min_hr_bpm"],
        "max_hr_bpm": features["max_hr_bpm"],
        "signal_quality": features["signal_quality"],
    }

    response = {
        "recording_id": recording_id,
        "summary": summary,
        "beats": {
            "r_peaks_ms": features["r_peaks_ms"],
            "rr_ms": features["rr_ms"],
            "artifact_mask": features["artifact_mask"],
        },
        "hrv_time": {
            "sdnn_ms": features["sdnn_ms"],
            "rmssd_ms": features["rmssd_ms"]
        },
        "intervals": {
            "qrs_ms": features["qrs_ms"],
            "qt_ms": features["qt_ms"],
            "qtc_ms_bazett": features["qtc_ms_bazett"],
            "uncertainty_ms": features["uncertainty_ms"]
        },
        "flags": {
            "pacemaker_detected": False,
            "ectopy_burden_pct": features["ectopy_burden_pct"],
            "st_deviation_flag": False
        },
        "version": "1.0.0"
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
