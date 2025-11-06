from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import os
from fastapi import UploadFile, File, Form
import io, csv

app = FastAPI(title="ECG Analyzer", version="1.0.0")

# in app/main.py near the top, after app = FastAPI(...)
@app.get("/")
def root():
    return {"ok": True, "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

# in app/main.py near the top, after app = FastAPI(...)
@app.get("/")
def root():
    return {"ok": True, "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

API_KEY = os.environ.get("API_KEY", "changeme")

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

def fake_analysis(samples: List[float], fs: float) -> Dict[str, Any]:
    n = len(samples)
    mean = sum(samples)/max(n,1)
    mean_hr = 72 + int(mean) % 15
    return {
        "summary": {
            "rhythm": "sinus",
            "rhythm_confidence": 0.85,
            "mean_hr_bpm": mean_hr,
            "min_hr_bpm": max(40, mean_hr - 8),
            "max_hr_bpm": min(180, mean_hr + 12),
            "signal_quality": "good" if n > 200 else "moderate"
        },
        "beats": {
            "r_peaks_ms": [i*800 for i in range(1, min(30, n//int(fs)) + 1)],
            "rr_ms": [800 for _ in range(min(29, n//int(fs) - 1))],
            "artifact_mask": []
        },
        "hrv_time": {"sdnn_ms": 50, "rmssd_ms": 42},
        "intervals": {"qrs_ms": 90, "qt_ms": 360, "qtc_ms_bazett": 440, "uncertainty_ms": 25},
        "flags": {"pacemaker_detected": False, "ectopy_burden_pct": 0.0, "st_deviation_flag": False},
        "version": "analyzer-1.0.0"
    }

STORE: Dict[str, Dict[str, Any]] = {}

def _check_auth(auth_header: Optional[str]):
    if not API_KEY:
        return
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: missing Bearer token")
    token = auth_header.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: bad token")

@app.post("/v1/ecg/analyze")
def analyze_ecg(payload: ECGRequest, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    if payload.units not in {"uV", "mV", "LSB"}:
        raise HTTPException(status_code=400, detail="units must be one of uV, mV, LSB")
    if payload.lead != "I":
        raise HTTPException(status_code=400, detail="lead must be 'I' for Apple Watch")
    if payload.sampling_rate_hz not in {128, 250, 256, 512}:
        raise HTTPException(status_code=400, detail="Unsupported sampling_rate_hz")
    result = fake_analysis(payload.samples, payload.sampling_rate_hz)
    resp = {"recording_id": payload.recording_id, **result}
    STORE[payload.recording_id] = resp
    return resp

@app.post("/v1/ecg/upload_csv")
async def upload_csv(
    file: UploadFile = File(...),
    sampling_rate_hz: float = Form(256),
    units: str = Form("uV"),
    lead: str = Form("I"),
    recording_id: str = Form("csv_upload"),
    start_timestamp_utc: str = Form(None),  # optional
    authorization: str | None = Header(None),
):
    _check_auth(authorization)

    if units not in {"uV", "mV", "LSB"}:
        raise HTTPException(status_code=400, detail="units must be one of uV, mV, LSB")
    if lead != "I":
        raise HTTPException(status_code=400, detail="lead must be 'I' for Apple Watch")

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

    samples = []
    for r in data_rows:
        try:
            samples.append(float(r[col_idx]))
        except Exception:
            continue

    if len(samples) < 30:
        raise HTTPException(status_code=400, detail=f"Not enough numeric samples ({len(samples)})")

    features = extract_features(samples, sampling_rate_hz)

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
        "version": APP_VERSION
    }

    STORE[recording_id] = response
    return response


@app.get("/v1/ecg/recordings/{recording_id}")
def get_ecg(recording_id: str, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    data = STORE.get(recording_id)
    if not data:
        raise HTTPException(status_code=404, detail="recording_id not found")
    return data
# add to your imports at the top of app/main.py
from fastapi import UploadFile, File, Form
import io, csv

@app.post("/v1/ecg/upload_csv")
async def upload_csv(
    file: UploadFile = File(...),
    sampling_rate_hz: float = Form(256),
    units: str = Form("uV"),
    lead: str = Form("I"),
    recording_id: str = Form("csv_upload"),
    start_timestamp_utc: str = Form(None),  # optional
    authorization: str | None = Header(None),
):
    """
    Accept a CSV file and analyze it.
    - CSV may have a header.
    - If multiple columns, the first numeric-looking column is used.
    - Form fields let you pass metadata alongside the file.
    """
    _check_auth(authorization)

    if units not in {"uV", "mV", "LSB"}:
        raise HTTPException(status_code=400, detail="units must be one of uV, mV, LSB")
    if lead != "I":
        raise HTTPException(status_code=400, detail="lead must be 'I' for Apple Watch")

    # Read CSV in memory
    contents = await file.read()
    text = contents.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))

    # Parse rows -> choose the first numeric column
    rows = list(reader)
    # remove empty rows
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        raise HTTPException(status_code=400, detail="CSV appears empty")

    # If header suspected, keep both; detection is simple
    header_like = any(not _is_float(c) for c in rows[0])
    data_rows = rows[1:] if header_like else rows

    # Pick the first column that looks numeric across most rows
    col_idx = _choose_numeric_column(data_rows)
    if col_idx is None:
        raise HTTPException(status_code=400, detail="No numeric column found in CSV")

    # Extract numeric samples
    samples = []
    for r in data_rows:
        try:
            samples.append(float(r[col_idx]))
        except Exception:
            # skip non-numeric rows
            continue

    if len(samples) < 30:
        raise HTTPException(status_code=400, detail=f"Not enough numeric samples ({len(samples)})")

    # Run the same feature extraction
    features = extract_features(samples, sampling_rate_hz)

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
        "version": APP_VERSION
    }

    STORE[recording_id] = response
    return response


# ---- helper functions (add near the bottom or above) ----
def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def _choose_numeric_column(rows: list[list[str]]) -> int | None:
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
    # require that at least half the rows are numeric in that column
    return best_col if best_hits >= max(1, len(rows) // 2) else None
def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def _choose_numeric_column(rows: list[list[str]]) -> int | None:
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
