
from typing import List, Dict, Any
import numpy as np

def _zscore(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x)
    sigma = np.std(x) + 1e-8
    return (x - mu) / sigma

def _simple_peak_detect(sig: np.ndarray, fs: float) -> List[int]:
    x = _zscore(sig.astype(float))
    dx = np.diff(x, prepend=x[0])
    energy = dx**2
    win = max(3, int(0.08 * fs))
    kernel = np.ones(win) / win
    smooth = np.convolve(energy, kernel, mode="same")
    thr = np.percentile(smooth, 90) * 0.5
    cand = np.where(smooth > thr)[0]

    r_peaks = []
    refractory = int(0.2 * fs)
    last = -refractory
    for i in cand:
        if i - last >= refractory:
            w = int(0.04 * fs)
            lo = max(0, i - w)
            hi = min(len(x) - 1, i + w)
            loc = lo + int(np.argmax(x[lo:hi+1]))
            r_peaks.append(loc)
            last = loc
    return r_peaks

def _rr_intervals_ms(r_idx: List[int], fs: float) -> List[float]:
    if len(r_idx) < 2:
        return []
    rr = np.diff(np.array(r_idx)) / fs * 1000.0
    return rr.tolist()

def _hr_stats(rr_ms: List[float]) -> Dict[str, float]:
    if not rr_ms:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    hr = 60000.0 / np.array(rr_ms)
    return {"mean": float(np.mean(hr)), "min": float(np.min(hr)), "max": float(np.max(hr))}

def _hrv(rr_ms: List[float]) -> Dict[str, float]:
    if len(rr_ms) < 2:
        return {"sdnn": 0.0, "rmssd": 0.0}
    rr = np.array(rr_ms)
    sdnn = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
    diff = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diff**2))) if len(diff) > 0 else 0.0
    return {"sdnn": sdnn, "rmssd": rmssd}

def _basic_quality(sig: np.ndarray) -> str:
    amp = np.percentile(sig, 95) - np.percentile(sig, 5)
    if amp < 5: 
        return "poor"
    if amp < 20:
        return "moderate"
    return "good"

def extract_features(samples: List[float], fs: float) -> Dict[str, Any]:
    x = np.asarray(samples, dtype=float)
    r_idx = _simple_peak_detect(x, fs)
    rr_ms = _rr_intervals_ms(r_idx, fs)
    h = _hr_stats(rr_ms)
    hrv = _hrv(rr_ms)

    qrs_ms = 90.0
    qt_ms = 360.0
    # crude Bazett using mean HR
    mean_hr = max(h['mean'], 1e-3)
    rr_s = 60.0 / mean_hr
    qtc_bazett = qt_ms / (rr_s**0.5) if rr_s > 0 else qt_ms

    rhythm_label = "sinus"
    confidence = 0.8
    if len(rr_ms) >= 15:
        cv = np.std(rr_ms) / (np.mean(rr_ms) + 1e-6)
        if cv > 0.12:
            rhythm_label = "possible_af"
            confidence = min(0.95, 0.6 + cv)

    signal_quality = _basic_quality(x)

    return {
        "r_peaks_ms": [int(i / fs * 1000) for i in r_idx],
        "rr_ms": [float(v) for v in rr_ms],
        "mean_hr_bpm": float(h["mean"]),
        "min_hr_bpm": float(h["min"]),
        "max_hr_bpm": float(h["max"]),
        "sdnn_ms": float(hrv["sdnn"]),
        "rmssd_ms": float(hrv["rmssd"]),
        "qrs_ms": qrs_ms,
        "qt_ms": qt_ms,
        "qtc_ms_bazett": float(qtc_bazett),
        "uncertainty_ms": 25.0,
        "ectopy_burden_pct": 0.0,
        "artifact_mask": [],
        "signal_quality": signal_quality,
        "rhythm_label": rhythm_label,
        "confidence": float(confidence),
    }
