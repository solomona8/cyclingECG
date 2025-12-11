from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy import signal

def extract_features(samples: List[float], fs: float) -> Dict[str, Any]:
    """
    Enhanced single-lead ECG feature extractor with advanced analysis.

    Features:
    - R-peak detection using Pan-Tompkins style pipeline
    - QRS onset/offset detection for true QRS duration
    - T-wave detection using differential and zero-crossing methods
    - QT interval measurement with Bazett correction
    - P-wave detection for PR interval
    - PVC/PAC detection based on morphology and timing
    - Ectopy burden calculation
    - Enhanced signal quality assessment

    Optimized for Apple Watch ECG (single-lead, 128-512 Hz).
    """

    x = np.asarray(samples, dtype=float)
    n = len(x)
    print(f"[FEATURE_EXTRACT] Input signal: n={n}, min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}, std={x.std():.6f}")

    if n < max(30, int(2 * fs)):
        print(f"[FEATURE_EXTRACT] Signal too short: {n} < {max(30, int(2 * fs))}")
        return _fallback_no_beats(n, fs)

    # 1) Preprocessing: detrend, normalize, filter
    x = _preprocess_signal(x, fs)
    print(f"[FEATURE_EXTRACT] After preprocessing: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}, std={x.std():.6f}")

    # 2) R-peak detection
    r_peaks = _detect_r_peaks(x, fs)
    print(f"[FEATURE_EXTRACT] R-peaks detected: {len(r_peaks)} peaks at indices {r_peaks[:10] if len(r_peaks) > 0 else 'none'}")

    if len(r_peaks) < 2:
        print(f"[FEATURE_EXTRACT] Too few R-peaks ({len(r_peaks)} < 2), returning fallback")
        return _fallback_no_beats(n, fs)

    r_peaks = np.asarray(r_peaks, dtype=int)
    r_peaks_ms = (r_peaks / fs) * 1000.0
    rr_ms = np.diff(r_peaks_ms)

    # 3) QRS complex detection and duration
    qrs_features = _detect_qrs_complexes(x, r_peaks, fs)

    # 4) T-wave detection
    t_waves = _detect_t_waves(x, r_peaks, fs)

    # 5) P-wave detection
    p_waves = _detect_p_waves(x, r_peaks, fs)

    # 6) QT interval measurement
    qt_intervals = _measure_qt_intervals(r_peaks, t_waves, fs)

    # 7) Arrhythmia detection (PVCs, PACs)
    ectopy_results = _detect_ectopy(x, r_peaks, rr_ms, qrs_features, fs)

    # 8) Heart rate statistics
    mean_rr = float(np.mean(rr_ms))
    mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0
    min_hr = 60000.0 / float(np.max(rr_ms)) if np.max(rr_ms) > 0 else 0.0
    max_hr = 60000.0 / float(np.min(rr_ms)) if np.min(rr_ms) > 0 else 0.0

    # 9) HRV time-domain
    sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 2 else 0.0

    # 10) Rhythm classification
    rhythm_label, confidence = _rhythm_and_confidence(rr_ms, ectopy_results)

    # 11) Signal quality assessment
    signal_quality = _assess_signal_quality(x, r_peaks, qrs_features, fs)

    # 12) Compute average intervals
    avg_qrs = np.mean([f['duration_ms'] for f in qrs_features]) if qrs_features else 90.0

    if qt_intervals:
        avg_qt = np.mean([q['qt_ms'] for q in qt_intervals])
        avg_qtc = np.mean([q['qtc_bazett'] for q in qt_intervals])
        qt_uncertainty = np.std([q['qt_ms'] for q in qt_intervals]) if len(qt_intervals) > 1 else 20.0
    else:
        # Estimation based on heart rate (Bazett's formula in reverse)
        avg_qt = 360.0 if mean_hr == 0 else 440.0 * np.sqrt(mean_rr / 1000.0)
        avg_qtc = 440.0
        qt_uncertainty = 35.0

    # 13) Calculate detailed interval metrics
    p_wave_durations = []
    pr_intervals = []
    pr_segments = []
    st_segments = []
    t_wave_durations = []

    # Match P waves, QRS complexes, and T waves for each beat
    for i in range(min(len(p_waves), len(qrs_features), len(t_waves))):
        p_wave = p_waves[i]
        qrs = qrs_features[i]
        t_wave = t_waves[i]

        # P wave duration
        p_wave_durations.append(p_wave['duration_ms'])

        # PR interval: from P wave onset to QRS onset
        pr_ms = ((qrs['onset'] - p_wave['onset']) / fs) * 1000.0
        if 120 <= pr_ms <= 220:  # Physiologically valid PR interval
            pr_intervals.append(pr_ms)

        # PR segment: from P wave offset to QRS onset
        pr_seg_ms = ((qrs['onset'] - p_wave['offset']) / fs) * 1000.0
        if 50 <= pr_seg_ms <= 120:  # Physiologically valid PR segment
            pr_segments.append(pr_seg_ms)

        # ST segment: from QRS offset to T wave onset
        st_seg_ms = ((t_wave['onset'] - qrs['offset']) / fs) * 1000.0
        if 50 <= st_seg_ms <= 150:  # Physiologically valid ST segment
            st_segments.append(st_seg_ms)

        # T wave duration
        t_wave_durations.append(t_wave['duration_ms'])

    # Calculate averages
    avg_p_wave = np.mean(p_wave_durations) if p_wave_durations else None
    avg_pr = np.mean(pr_intervals) if pr_intervals else None
    avg_pr_seg = np.mean(pr_segments) if pr_segments else None
    avg_st_seg = np.mean(st_segments) if st_segments else None
    avg_t_wave = np.mean(t_wave_durations) if t_wave_durations else None

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
        "p_wave_ms": float(round(avg_p_wave, 1)) if avg_p_wave else None,
        "pr_ms": float(round(avg_pr, 1)) if avg_pr else None,
        "pr_seg_ms": float(round(avg_pr_seg, 1)) if avg_pr_seg else None,
        "qrs_ms": float(round(avg_qrs, 1)),
        "st_seg_ms": float(round(avg_st_seg, 1)) if avg_st_seg else None,
        "t_wave_ms": float(round(avg_t_wave, 1)) if avg_t_wave else None,
        "qt_ms": float(round(avg_qt, 1)),
        "qtc_ms_bazett": float(round(avg_qtc, 1)),
        "uncertainty_ms": float(round(qt_uncertainty, 1)),
        "ectopy_burden_pct": ectopy_results['burden_pct'],
        "pvcs_detected": ectopy_results['pvc_count'],
        "pacs_detected": ectopy_results['pac_count'],
        "morphology_variance": float(round(ectopy_results['morphology_variance'], 3)),
        "beat_count": len(r_peaks),
    }

# ============================================================================
# PREPROCESSING
# ============================================================================

def _preprocess_signal(x: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass filter and normalize the signal."""
    # Remove baseline wander and high-frequency noise
    # Bandpass: 0.5-40 Hz (typical for ECG)
    nyq = fs / 2.0
    low = 0.5 / nyq
    high = min(40.0 / nyq, 0.99)

    if low < high:
        try:
            b, a = signal.butter(2, [low, high], btype='band')
            x_filt = signal.filtfilt(b, a, x)
        except:
            x_filt = x
    else:
        x_filt = x

    # Normalize
    x_filt = x_filt - np.mean(x_filt)
    std = np.std(x_filt)
    if std > 0:
        x_filt = x_filt / std

    return x_filt

# ============================================================================
# R-PEAK DETECTION
# ============================================================================

def _detect_r_peaks(x: np.ndarray, fs: float) -> List[int]:
    """Enhanced R-peak detection with adaptive thresholding."""
    # Derivative + square (emphasize QRS)
    dx = np.diff(x, prepend=x[0])
    energy = dx ** 2

    # Moving window integration (~150 ms)
    win = max(3, int(0.15 * fs))
    mwi = _moving_average(energy, win)

    print(f"[R_PEAK_DETECT] Energy: max={energy.max():.6f}, mean={energy.mean():.6f}")
    print(f"[R_PEAK_DETECT] MWI: max={mwi.max():.6f}, median={np.median(mwi):.6f}, p95={np.percentile(mwi, 95):.6f}")

    # Adaptive threshold
    baseline = np.median(mwi) + 0.5 * (np.percentile(mwi, 95) - np.median(mwi))
    thr = max(baseline, np.mean(mwi) + 0.5 * np.std(mwi))

    print(f"[R_PEAK_DETECT] Primary threshold: {thr:.6f}")

    # Find candidate crossings
    refractory = int(0.30 * fs)  # 300 ms
    cand_idx = np.where(mwi > thr)[0]
    print(f"[R_PEAK_DETECT] Candidates above primary threshold: {len(cand_idx)}")

    r_peaks = _select_peaks(cand_idx, x, refractory)
    print(f"[R_PEAK_DETECT] Peaks after refractory filtering: {len(r_peaks)}")

    # If none found, relax threshold
    if len(r_peaks) < 1:
        thr2 = np.median(mwi) + 0.25 * (np.percentile(mwi, 95) - np.median(mwi))
        print(f"[R_PEAK_DETECT] Trying fallback threshold: {thr2:.6f}")
        cand_idx = np.where(mwi > thr2)[0]
        print(f"[R_PEAK_DETECT] Candidates above fallback threshold: {len(cand_idx)}")
        r_peaks = _select_peaks(cand_idx, x, refractory)
        print(f"[R_PEAK_DETECT] Peaks after refractory filtering (fallback): {len(r_peaks)}")

    return r_peaks

# ============================================================================
# QRS COMPLEX DETECTION
# ============================================================================

def _detect_qrs_complexes(x: np.ndarray, r_peaks: np.ndarray, fs: float) -> List[Dict[str, Any]]:
    """Detect QRS onset and offset for each R-peak to measure duration."""
    qrs_features = []
    search_win = int(0.08 * fs)  # 80ms search window each side

    for r_idx in r_peaks:
        # Search for QRS onset (before R-peak)
        onset_start = max(0, r_idx - search_win)
        onset_seg = x[onset_start:r_idx]

        # QRS onset: point where derivative first exceeds threshold
        if len(onset_seg) > 3:
            d_onset = np.abs(np.diff(onset_seg))
            threshold = np.mean(d_onset) + 0.5 * np.std(d_onset)
            onset_candidates = np.where(d_onset > threshold)[0]
            if len(onset_candidates) > 0:
                qrs_onset = onset_start + onset_candidates[0]
            else:
                qrs_onset = onset_start
        else:
            qrs_onset = onset_start

        # Search for QRS offset (after R-peak)
        offset_end = min(len(x), r_idx + search_win)
        offset_seg = x[r_idx:offset_end]

        # QRS offset: point where signal returns near baseline
        if len(offset_seg) > 3:
            d_offset = np.abs(np.diff(offset_seg))
            threshold = np.mean(d_offset) + 0.5 * np.std(d_offset)
            # Find last significant deflection
            offset_candidates = np.where(d_offset > threshold)[0]
            if len(offset_candidates) > 0:
                qrs_offset = r_idx + offset_candidates[-1]
            else:
                qrs_offset = offset_end - 1
        else:
            qrs_offset = offset_end - 1

        duration_ms = ((qrs_offset - qrs_onset) / fs) * 1000.0

        # Sanity check: QRS should be 60-120ms typically
        if duration_ms < 40:
            duration_ms = 80.0  # Default if too narrow
        elif duration_ms > 180:
            duration_ms = 120.0  # Default if too wide

        qrs_features.append({
            'r_peak': int(r_idx),
            'onset': int(qrs_onset),
            'offset': int(qrs_offset),
            'duration_ms': float(duration_ms),
            'amplitude': float(x[r_idx])
        })

    return qrs_features

# ============================================================================
# T-WAVE DETECTION
# ============================================================================

def _detect_t_waves(x: np.ndarray, r_peaks: np.ndarray, fs: float) -> List[Dict[str, Any]]:
    """Detect T-wave features including onset, peak, offset, and duration."""
    t_waves = []

    for i, r_idx in enumerate(r_peaks):
        # T-wave search window: 150-450ms after R-peak
        t_start_search = r_idx + int(0.15 * fs)
        t_end_search = min(len(x), r_idx + int(0.45 * fs))

        # Stop search before next R-peak if it exists
        if i + 1 < len(r_peaks):
            t_end_search = min(t_end_search, r_peaks[i + 1])

        if t_end_search <= t_start_search:
            continue

        t_seg = x[t_start_search:t_end_search]

        if len(t_seg) < 3:
            continue

        # Find T-wave peak (usually largest deflection in window)
        # T-wave can be positive or negative
        abs_seg = np.abs(t_seg)
        t_peak_local = np.argmax(abs_seg)
        t_peak = t_start_search + t_peak_local

        # Validate: T-wave should have reasonable amplitude (at least 10% of R-peak)
        if abs(x[t_peak]) > 0.1 * abs(x[r_idx]):
            # Detect T wave onset and offset
            # Search 80ms before and after peak
            onset_window = int(0.08 * fs)

            # T wave onset - find where signal starts rising after ST segment
            t_onset_start = max(t_start_search, t_peak - onset_window)
            onset_seg = x[t_onset_start:t_peak]
            if len(onset_seg) > 2:
                d_onset = np.abs(np.diff(onset_seg))
                threshold = np.mean(d_onset) + 0.3 * np.std(d_onset)
                onset_candidates = np.where(d_onset > threshold)[0]
                if len(onset_candidates) > 0:
                    t_onset = t_onset_start + onset_candidates[0]
                else:
                    t_onset = t_onset_start
            else:
                t_onset = t_onset_start

            # T wave offset - find where signal returns to baseline
            t_offset_end = min(len(x), t_peak + onset_window)
            if i + 1 < len(r_peaks):
                t_offset_end = min(t_offset_end, r_peaks[i + 1])

            offset_seg = x[t_peak:t_offset_end]
            if len(offset_seg) > 2:
                d_offset = np.abs(np.diff(offset_seg))
                threshold = np.mean(d_offset) + 0.3 * np.std(d_offset)
                offset_candidates = np.where(d_offset > threshold)[0]
                if len(offset_candidates) > 0:
                    t_offset = t_peak + offset_candidates[-1]
                else:
                    t_offset = t_offset_end - 1
            else:
                t_offset = t_offset_end - 1

            # Calculate T wave duration (typical: 160ms)
            t_duration_ms = ((t_offset - t_onset) / fs) * 1000.0

            # Sanity check
            if t_duration_ms < 100:
                t_duration_ms = 160.0  # Default
            elif t_duration_ms > 250:
                t_duration_ms = 180.0  # Default

            t_waves.append({
                'peak': int(t_peak),
                'onset': int(t_onset),
                'offset': int(t_offset),
                'duration_ms': float(t_duration_ms)
            })
        else:
            # If no significant T-wave, estimate
            estimated_peak = int(r_idx + int(0.30 * fs))
            t_waves.append({
                'peak': estimated_peak,
                'onset': estimated_peak - int(0.08 * fs),
                'offset': estimated_peak + int(0.08 * fs),
                'duration_ms': 160.0
            })

    return t_waves

# ============================================================================
# P-WAVE DETECTION
# ============================================================================

def _detect_p_waves(x: np.ndarray, r_peaks: np.ndarray, fs: float) -> List[Dict[str, Any]]:
    """Detect P-wave features including onset, peak, offset, and duration."""
    p_waves = []

    for r_idx in r_peaks:
        # P-wave search window: 80-250ms before R-peak
        p_end_search = r_idx - int(0.08 * fs)
        p_start_search = max(0, r_idx - int(0.25 * fs))

        if p_end_search <= p_start_search:
            continue

        p_seg = x[p_start_search:p_end_search]

        if len(p_seg) < 3:
            continue

        # P-wave is typically smaller, look for local maxima
        # Smooth to reduce noise
        if len(p_seg) > 5:
            p_smooth = _moving_average(np.abs(p_seg), min(5, len(p_seg) // 3))
        else:
            p_smooth = np.abs(p_seg)

        p_peak_local = np.argmax(p_smooth)
        p_peak = p_start_search + p_peak_local

        # Validate: P-wave should be smaller than R-peak
        if abs(x[p_peak]) < 0.5 * abs(x[r_idx]) and abs(x[p_peak]) > 0.05:
            # Detect P wave onset and offset
            # Search 40ms before and after peak for onset/offset
            onset_window = int(0.04 * fs)

            # P wave onset - find where signal starts rising before peak
            p_onset_start = max(0, p_peak - onset_window)
            onset_seg = x[p_onset_start:p_peak]
            if len(onset_seg) > 2:
                d_onset = np.abs(np.diff(onset_seg))
                threshold = np.mean(d_onset) + 0.3 * np.std(d_onset)
                onset_candidates = np.where(d_onset > threshold)[0]
                if len(onset_candidates) > 0:
                    p_onset = p_onset_start + onset_candidates[0]
                else:
                    p_onset = p_onset_start
            else:
                p_onset = p_onset_start

            # P wave offset - find where signal returns to baseline after peak
            p_offset_end = min(len(x), p_peak + onset_window)
            offset_seg = x[p_peak:p_offset_end]
            if len(offset_seg) > 2:
                d_offset = np.abs(np.diff(offset_seg))
                threshold = np.mean(d_offset) + 0.3 * np.std(d_offset)
                offset_candidates = np.where(d_offset > threshold)[0]
                if len(offset_candidates) > 0:
                    p_offset = p_peak + offset_candidates[-1]
                else:
                    p_offset = p_offset_end - 1
            else:
                p_offset = p_offset_end - 1

            # Calculate P wave duration (typical: 80-120ms)
            p_duration_ms = ((p_offset - p_onset) / fs) * 1000.0

            # Sanity check
            if p_duration_ms < 60:
                p_duration_ms = 80.0  # Default
            elif p_duration_ms > 120:
                p_duration_ms = 100.0  # Default

            p_waves.append({
                'peak': int(p_peak),
                'onset': int(p_onset),
                'offset': int(p_offset),
                'duration_ms': float(p_duration_ms)
            })

    return p_waves

# ============================================================================
# QT INTERVAL MEASUREMENT
# ============================================================================

def _measure_qt_intervals(r_peaks: np.ndarray, t_waves: List[Dict[str, Any]], fs: float) -> List[Dict[str, float]]:
    """Measure QT intervals and apply Bazett correction."""
    qt_intervals = []

    for i in range(min(len(r_peaks), len(t_waves))):
        # Use T-wave offset for QT interval (Q to end of T)
        qt_samples = t_waves[i]['offset'] - r_peaks[i]
        qt_ms = (qt_samples / fs) * 1000.0

        # Physiologically valid QT: 200-600ms
        if 200 <= qt_ms <= 600:
            # Bazett correction: QTc = QT / sqrt(RR)
            # Get corresponding RR interval
            if i > 0:
                rr_ms = ((r_peaks[i] - r_peaks[i-1]) / fs) * 1000.0
            elif i + 1 < len(r_peaks):
                rr_ms = ((r_peaks[i+1] - r_peaks[i]) / fs) * 1000.0
            else:
                rr_ms = 1000.0  # Assume 60 bpm

            qtc_bazett = qt_ms / np.sqrt(rr_ms / 1000.0) if rr_ms > 0 else qt_ms

            qt_intervals.append({
                'qt_ms': float(qt_ms),
                'qtc_bazett': float(qtc_bazett),
                'rr_ms': float(rr_ms)
            })

    return qt_intervals

# ============================================================================
# ECTOPY DETECTION (PVCs, PACs)
# ============================================================================

def _detect_ectopy(x: np.ndarray, r_peaks: np.ndarray, rr_ms: np.ndarray,
                   qrs_features: List[Dict], fs: float) -> Dict[str, Any]:
    """Detect premature ventricular contractions (PVCs) and premature atrial contractions (PACs)."""

    if len(r_peaks) < 3 or len(qrs_features) < 3:
        return {
            'pvc_count': 0,
            'pac_count': 0,
            'burden_pct': 0.0,
            'morphology_variance': 0.0
        }

    pvc_count = 0
    pac_count = 0

    # Compute morphology features for each beat
    beat_morphologies = []
    for qrs in qrs_features:
        # Extract QRS complex waveform
        onset, offset = qrs['onset'], qrs['offset']
        if offset > onset and offset < len(x):
            waveform = x[onset:offset]
            # Normalize and compute features
            if len(waveform) > 0:
                waveform_norm = waveform - np.mean(waveform)
                if np.std(waveform_norm) > 0:
                    waveform_norm = waveform_norm / np.std(waveform_norm)
                beat_morphologies.append(waveform_norm)

    # Compute median RR interval for prematurity detection
    median_rr = np.median(rr_ms) if len(rr_ms) > 0 else 800.0

    for i in range(1, len(r_peaks) - 1):
        is_ectopic = False
        is_wide_qrs = False

        # Check 1: Premature beat (RR interval < 80% of median)
        if i < len(rr_ms):
            is_premature = rr_ms[i] < 0.80 * median_rr
        else:
            is_premature = False

        # Check 2: Wide QRS (>120ms suggests ventricular origin)
        if i < len(qrs_features):
            qrs_duration = qrs_features[i]['duration_ms']
            is_wide_qrs = qrs_duration > 120.0

        # Check 3: Morphology different from surrounding beats
        is_morphology_different = False
        if i < len(beat_morphologies) and i > 0 and i < len(beat_morphologies) - 1:
            # Compare with previous and next beats
            curr_morph = beat_morphologies[i]
            prev_morph = beat_morphologies[i-1]
            next_morph = beat_morphologies[i+1]

            # Resize to match lengths for correlation
            min_len = min(len(curr_morph), len(prev_morph), len(next_morph))
            if min_len > 5:
                curr = curr_morph[:min_len]
                prev = prev_morph[:min_len]
                next = next_morph[:min_len]

                # Correlation with neighbors
                corr_prev = np.corrcoef(curr, prev)[0, 1] if np.std(curr) > 0 and np.std(prev) > 0 else 1.0
                corr_next = np.corrcoef(curr, next)[0, 1] if np.std(curr) > 0 and np.std(next) > 0 else 1.0

                # Low correlation suggests different morphology
                is_morphology_different = (corr_prev < 0.7 or corr_next < 0.7)

        # Classify ectopic beats
        if is_premature:
            if is_wide_qrs or is_morphology_different:
                pvc_count += 1  # Premature + wide/different = PVC
                is_ectopic = True
            else:
                pac_count += 1  # Premature + narrow/similar = PAC
                is_ectopic = True

    total_beats = len(r_peaks)
    ectopic_beats = pvc_count + pac_count
    burden_pct = (ectopic_beats / total_beats * 100.0) if total_beats > 0 else 0.0

    # Morphology variance across all beats
    if len(beat_morphologies) > 2:
        # Compute pairwise correlations
        correlations = []
        for i in range(len(beat_morphologies) - 1):
            curr = beat_morphologies[i]
            next_beat = beat_morphologies[i + 1]
            min_len = min(len(curr), len(next_beat))
            if min_len > 5:
                c1 = curr[:min_len]
                c2 = next_beat[:min_len]
                if np.std(c1) > 0 and np.std(c2) > 0:
                    corr = np.corrcoef(c1, c2)[0, 1]
                    correlations.append(corr)

        morphology_variance = 1.0 - np.mean(correlations) if correlations else 0.0
    else:
        morphology_variance = 0.0

    return {
        'pvc_count': int(pvc_count),
        'pac_count': int(pac_count),
        'burden_pct': float(round(burden_pct, 1)),
        'morphology_variance': float(morphology_variance)
    }

# ============================================================================
# RHYTHM CLASSIFICATION
# ============================================================================

def _rhythm_and_confidence(rr_ms: np.ndarray, ectopy_results: Dict) -> Tuple[str, float]:
    """Enhanced rhythm classification considering ectopy."""
    if len(rr_ms) < 2:
        return "undetermined", 0.3

    # Coefficient of variation
    cv = float(np.std(rr_ms) / (np.mean(rr_ms) + 1e-6))

    # Check for high ectopy burden
    ectopy_burden = ectopy_results['burden_pct']

    if ectopy_burden > 10:
        return "frequent_ectopy", 0.7
    elif cv < 0.05:
        return "sinus", 0.9
    elif cv < 0.12:
        return "sinus_irregular", 0.75
    elif cv > 0.3:
        return "irregular_possible_afib", 0.6
    else:
        return "irregular", 0.65

# ============================================================================
# SIGNAL QUALITY ASSESSMENT
# ============================================================================

def _assess_signal_quality(x: np.ndarray, r_peaks: np.ndarray,
                          qrs_features: List[Dict], fs: float) -> str:
    """Assess overall signal quality based on multiple factors."""
    n = len(x)

    # Factor 1: Signal length
    duration_sec = n / fs
    if duration_sec < 5:
        return "poor"

    # Factor 2: Number of detected beats
    expected_beats = duration_sec * 1.0  # At least 60 bpm
    if len(r_peaks) < expected_beats:
        quality_score = 0
    else:
        quality_score = 1

    # Factor 3: QRS consistency
    if qrs_features:
        qrs_durations = [f['duration_ms'] for f in qrs_features]
        qrs_cv = np.std(qrs_durations) / (np.mean(qrs_durations) + 1e-6)
        if qrs_cv < 0.15:
            quality_score += 1

    # Factor 4: Signal-to-noise ratio estimate
    signal_std = np.std(x)
    if signal_std > 0.5:  # Good amplitude variation
        quality_score += 1

    if quality_score >= 3:
        return "good"
    elif quality_score >= 2:
        return "moderate"
    else:
        return "poor"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Compute moving average with edge padding."""
    if w <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[w:] - c[:-w]) / float(w)
    pad_left = w // 2
    pad_right = len(x) - len(ma) - pad_left
    return np.pad(ma, (pad_left, max(0, pad_right)), mode="edge")

def _select_peaks(candidates: np.ndarray, raw: np.ndarray, refractory: int) -> List[int]:
    """Select R-peaks from candidates, enforcing refractory period."""
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

    run_peaks = []
    for a, b in runs:
        seg = raw[a:b+1]
        if seg.size == 0:
            continue
        local = int(a + np.argmax(seg))
        run_peaks.append(local)

    selected = []
    last = -10**9
    for p in run_peaks:
        if p - last >= refractory:
            selected.append(p)
            last = p
        else:
            if raw[p] > raw[last]:
                selected[-1] = p
                last = p
    return selected

def _fallback_no_beats(n: int, fs: float) -> Dict[str, Any]:
    """Return minimal features when no beats detected."""
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
        "p_wave_ms": None,
        "pr_ms": None,
        "pr_seg_ms": None,
        "qrs_ms": 0.0,
        "st_seg_ms": None,
        "t_wave_ms": None,
        "qt_ms": 0.0,
        "qtc_ms_bazett": 0.0,
        "uncertainty_ms": 50.0,
        "ectopy_burden_pct": 0.0,
        "pvcs_detected": 0,
        "pacs_detected": 0,
        "morphology_variance": 0.0,
        "beat_count": 0,
    }
