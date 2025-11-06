
import json
from typing import Dict, Any
import os
import requests

OPENAI_URL = "https://api.openai.com/v1/responses"
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-5")

def generate_narrative(features: Dict[str, Any], patient: Dict[str, Any], openai_api_key: str) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    prompt = {
        "model": MODEL_NAME,
        "input": [
            {"role": "system", "content": "You are a conservative, safety-first cardiology assistant. Use provided features from a single-lead Apple Watch ECG. Do NOT provide a diagnosis; emphasize limitations."},
            {"role": "user", "content": json.dumps({
                "patient": patient,
                "features": {
                    "mean_hr_bpm": features["mean_hr_bpm"],
                    "min_hr_bpm": features["min_hr_bpm"],
                    "max_hr_bpm": features["max_hr_bpm"],
                    "sdnn_ms": features["sdnn_ms"],
                    "rmssd_ms": features["rmssd_ms"],
                    "signal_quality": features["signal_quality"],
                    "rhythm_label": features["rhythm_label"],
                    "confidence": features["confidence"]
                }
            })}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ecg_report",
                "schema": {
                    "type": "object",
                    "required": ["patient_summary", "clinician_note", "safety_flags"],
                    "properties": {
                        "patient_summary": {"type": "string"},
                        "clinician_note": {"type": "string"},
                        "safety_flags": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        }
    }
    resp = requests.post(OPENAI_URL, headers=headers, data=json.dumps(prompt), timeout=30)
    resp.raise_for_status()
    return resp.json()
