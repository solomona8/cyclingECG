
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import os

app = FastAPI(title="ECG Analyzer", version="1.0.0")

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

@app.get("/v1/ecg/recordings/{recording_id}")
def get_ecg(recording_id: str, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    data = STORE.get(recording_id)
    if not data:
        raise HTTPException(status_code=404, detail="recording_id not found")
    return data
