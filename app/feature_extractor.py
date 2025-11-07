from typing import List, Dict, Any, Tuple
import numpy as np

def extract_features(samples: List[float], fs: float) -> Dict[str, Any]:
    """
    Lightweight single-lead ECG feature extractor.
    - Detects R-peaks using a simple Pan-Tompkins style pipeline:
      diff -> square -> moving window integration -> adaptive threshold + refractory.
    - Computes RR intervals, HR stats, SDNN, RMSSD.
    Assumes Apple Watch-like sampling (e.g., 256 Hz). Works for 128–512 Hz.
    """

    x = np.asarray(samples, dtype=float)
    n = len(x)
    if n < max(30, int(2 * fs)):
        # Not enough data for reliable detection
        return _fallback_no_beats(n, fs)

    # 1) detrend / normalize (simple)
    x = x - np.nanmean(x)
    if np.nanstd(x) > 0:
        x = x / np.nanstd(x)

    # 2) derivative + square (emphasize QRS)
    dx = np.diff(x, prepend=x[0])
    energy = dx ** 2

    # 3) moving window integration (~150 ms)
    win = max(3, int(0.15 * fs))  # e.g., 38 samples @ 256 Hz
    mwi = _moving_average(energy, win)

    # 4) adaptive threshold: median/percentile-based
    #    Use a slightly conservative threshold to reduce false positives
    baseline = np.median(mwi) + 0.5 * (np.percentile(mwi, 95) - np.median(mwi))
    thr = max(baseline, np.mean(mwi) + 0.5 * np.std(mwi))

    # 5) find candidate crossings, then pick local maxima in original x
    refractory = int(0.30 * fs)  # 300 ms
    cand_idx = np.where(mwi > thr)[0]
    r_peaks = _select_peaks(cand_idx, x, refractory)

    # If none found, relax threshold once
    if len(r_peaks) < 1:
        thr2 = np.median(mwi) + 0.25 * (np.percentile(mwi, 95) - np.median(mwi))
        cand_idx = np.where(mwi > thr2)[0]
        r_peaks = _select_peaks(cand_idx, x, refractory)

    if len(r_peaks) < 2:
        # Still nothing useful: return minimal feature set
        return _fallback_no_beats(n, fs)

    r_peaks = np.asarray(r_peaks, dtype=int)
    r_peaks_ms = (r_peaks / fs) * 1000.0
    rr_ms = np.diff(r_peaks_ms)

    # Basic HR stats
    mean_rr = float(np.mean(rr_ms))
    mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0
    min_hr = 60000.0 / float(np.max(rr_ms)) if np.max(rr_ms) > 0 else 0.0
    max_hr = 60000.0 / float(np.min(rr_ms)) if np.min(rr_ms) > 0 else 0.0

    # HRV time-domain
    sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 2 else 0.0

    # Simple heuristics for rhythm and quality
    rhythm_label, confidence = _rhythm_and_confidence(rr_ms)
    signal_quality = "good" if n >= 200 and len(r_peaks) >= 2 else "moderate"

    # Placeholders for intervals (single-lead without T/P delineation)
    qrs_ms = 90.0
    qt_ms = 360.0
    qtc_ms_bazett = 440.0
    uncertainty_ms = 25.0

    return {
        "rhythm_label": rhythm_label,
        "confidence": confidence,
        "mean_hr_bpm": float(round(mean_hr, 1)),
        "min_hr_bpm": float(round(min_hr, 1)),
        "max_hr_bpm": float(round(max_hr, 1)),
        "signal_quality": signal_quality,
        "r_peaks_ms": r_peaks_ms.tolist(),
        "rr_ms": rr_ms.tolist(),
        "artifact_mask": [],
        "sdnn_ms": float(round(sdnn, 1)),
        "rmssd_ms": float(round(rmssd, 1)),
        "qrs_ms": qrs_ms,
        "qt_ms": qt_ms,
        "qtc_ms_bazett": qtc_ms_bazett,
        "uncertainty_ms": uncertainty_ms,
        "ectopy_burden_pct": 0.0,
    }

# ---------------- helpers ----------------

def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[w:] - c[:-w]) / float(w)
    # pad to original length
    pad_left = w // 2
    pad_right = len(x) - len(ma) - pad_left
    return np.pad(ma, (pad_left, max(0, pad_right)), mode="edge")

def _select_peaks(candidates: np.ndarray, raw: np.ndarray, refractory: int) -> List[int]:
    """
    Collapse contiguous above-threshold runs; for each run, pick the local max
    of the raw signal. Enforce a refractory period between selected peaks.
    """
    if candidates.size == 0:
        return []
    runs = []
    start = candidates[0]
    prev = candidates[0]
    for idx in candidates[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev))
            start = idx
            prev = idx
    runs.append((start, prev))

    # pick local maxima in raw for each run
    run_peaks = []
    for a, b in runs:
        seg = raw[a:b+1]
        if seg.size == 0:
            continue
        local = int(a + np.argmax(seg))
        run_peaks.append(local)

    # enforce refractory
    selected = []
    last = -10**9
    for p in run_peaks:
        if p - last >= refractory:
            selected.append(p)
            last = p
        else:
            # keep the stronger of the two if too close
            if raw[p] > raw[last]:
                selected[-1] = p
                last = p
    return selected

def _rhythm_and_confidence(rr_ms: np.ndarray) -> Tuple[str, float]:
    if len(rr_ms) < 2:
        return "undetermined", 0.3
    # Coefficient of variation as a crude regularity measure
    cv = float(np.std(rr_ms) / (np.mean(rr_ms) + 1e-6))
    if cv < 0.05:
        return "sinus", 0.9
    elif cv < 0.12:
        return "sinus_irregular", 0.75
    else:
        return "irregular", 0.6

def _fallback_no_beats(n: int, fs: float) -> Dict[str, Any]:
    # Minimal output if we cannot detect beats
    return {
        "rhythm_label": "undetermined",
        "confidence": 0.3,
        "mean_hr_bpm": 0.0,
        "min_hr_bpm": 0.0,
        "max_hr_bpm": 0.0,
        "signal_quality": "poor" if n < int(1.0 * fs) else "moderate",
        "r_peaks_ms": [],
        "rr_ms": [],
        "artifact_mask": [],
        "sdnn_ms": 0.0,
        "rmssd_ms": 0.0,
        "qrs_ms": 0.0,
        "qt_ms": 0.0,
        "qtc_ms_bazett": 0.0,
        "uncertainty_ms": 50.0,
        "ectopy_burden_pct": 0.0,
    }
