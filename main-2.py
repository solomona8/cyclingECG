
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from app.feature_extractor import extract_features
from app.openai_narrative import generate_narrative

APP_VERSION = "1.1.0"

app = FastAPI(title="ECG Cloud Analyzer", version=APP_VERSION)

API_KEY = os.environ.get("API_KEY")  # If unset, auth is disabled
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Optional for narrative

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
    include_narrative: Optional[bool] = False

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

    features = extract_features(payload.samples, payload.sampling_rate_hz)

    summary = {
        "rhythm": features["rhythm_label"],
        "rhythm_confidence": features["confidence"],
        "mean_hr_bpm": features["mean_hr_bpm"],
        "min_hr_bpm": features["min_hr_bpm"],
        "max_hr_bpm": features["max_hr_bpm"],
        "signal_quality": features["signal_quality"],
    }

    response = {
        "recording_id": payload.recording_id,
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

    # Optional: narrative via OpenAI
    if payload.include_narrative and OPENAI_API_KEY:
        try:
            narrative = generate_narrative(
                features=features,
                patient={
                    "age": payload.user.age if payload.user and payload.user.age is not None else None,
                    "sex_at_birth": payload.user.sex_at_birth if payload.user else None,
                    "symptoms": payload.symptoms or []
                },
                openai_api_key=OPENAI_API_KEY
            )
            response["narrative"] = narrative
        except Exception as e:
            response["narrative_error"] = str(e)

    STORE[payload.recording_id] = response
    return response

@app.get("/v1/ecg/recordings/{recording_id}")
def get_ecg(recording_id: str, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    data = STORE.get(recording_id)
    if not data:
        raise HTTPException(status_code=404, detail="recording_id not found")
    return data
