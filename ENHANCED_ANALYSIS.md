# Enhanced ECG Analysis Features

## Overview

The ECG analysis backend has been significantly enhanced with advanced signal processing and arrhythmia detection capabilities. The enhanced feature extractor now provides:

- ✅ **Real QRS detection and duration measurement** (not placeholders)
- ✅ **QT interval detection with Bazett correction**
- ✅ **P-wave detection for PR interval measurement**
- ✅ **T-wave detection for repolarization analysis**
- ✅ **PVC/PAC detection** (Premature Ventricular/Atrial Contractions)
- ✅ **Ectopy burden calculation**
- ✅ **Morphology variance analysis**
- ✅ **Enhanced signal quality assessment**

## New Features

### 1. Advanced QRS Analysis

**QRS Onset & Offset Detection:**
- Automatically detects QRS complex boundaries
- Measures true QRS duration (normal: 60-120ms, wide: >120ms)
- Uses derivative-based threshold detection
- Physiological validation with sanity checks

**Output Fields:**
```json
{
  "intervals": {
    "qrs_ms": 85.0,  // Measured QRS duration
    ...
  }
}
```

### 2. QT Interval Measurement

**T-Wave Detection:**
- Searches 150-450ms window after each R-peak
- Detects T-wave peak (positive or negative)
- Validates amplitude (must be >10% of R-peak)
- Falls back to 300ms estimate if T-wave unclear

**QT Interval Calculation:**
- Measures from R-peak to T-wave end
- Applies physiological validation (200-600ms)
- Computes Bazett-corrected QTc: `QTc = QT / sqrt(RR)`

**Output Fields:**
```json
{
  "intervals": {
    "qt_ms": 345.6,           // Measured QT interval
    "qtc_ms_bazett": 488.0,   // Bazett-corrected QTc
    "uncertainty_ms": 20.0     // Measurement uncertainty
  }
}
```

**Clinical Significance:**
- Normal QTc: <440ms (men), <460ms (women)
- Prolonged QTc: Risk marker for arrhythmias
- QTc >500ms: Significant risk

### 3. P-Wave Detection

**PR Interval Measurement:**
- Searches 80-250ms before each R-peak
- Detects P-wave peak (atrial depolarization)
- Validates amplitude (<50% of R-peak)
- Physiological range: 120-200ms

**Output Fields:**
```json
{
  "intervals": {
    "pr_ms": 153.5  // PR interval (null if not detected)
  }
}
```

**Clinical Significance:**
- Short PR (<120ms): Possible pre-excitation
- Long PR (>200ms): First-degree AV block

### 4. Arrhythmia Detection (PVCs & PACs)

**Detection Criteria:**

**Premature Ventricular Contractions (PVCs):**
- Premature beat (RR < 80% of median)
- Wide QRS (>120ms) OR different morphology
- High correlation with morphology variance

**Premature Atrial Contractions (PACs):**
- Premature beat (RR < 80% of median)
- Normal QRS width (<120ms)
- Similar morphology to sinus beats

**Morphology Analysis:**
- Extracts QRS waveform for each beat
- Computes cross-correlation with neighbors
- Low correlation (<0.7) indicates different morphology

**Output Fields:**
```json
{
  "arrhythmia": {
    "ectopy_burden_pct": 13.5,      // % of ectopic beats
    "pvcs_detected": 5,              // Count of PVCs
    "pacs_detected": 2,              // Count of PACs
    "morphology_variance": 0.050    // Beat-to-beat variance (0-1)
  }
}
```

**Clinical Significance:**
- Occasional PVCs/PACs: Usually benign
- Burden >10%: Consider further evaluation
- Morphology variance >0.15: Suggests multiple ectopic foci

### 5. Enhanced Rhythm Classification

**New Rhythm Categories:**
- `sinus`: Regular, CV <5%
- `sinus_irregular`: Mild irregularity, CV 5-12%
- `irregular`: Moderate irregularity, CV 12-30%
- `irregular_possible_afib`: High irregularity, CV >30%
- `frequent_ectopy`: Ectopy burden >10%
- `undetermined`: Insufficient data

**Output Fields:**
```json
{
  "summary": {
    "rhythm": "sinus",
    "rhythm_confidence": 0.90
  }
}
```

### 6. Signal Quality Assessment

**Multi-Factor Quality Score:**
1. **Duration**: <5s = poor
2. **Beat Detection**: Compares expected vs detected beats
3. **QRS Consistency**: CV of QRS durations <15%
4. **SNR Estimate**: Signal standard deviation >0.5

**Quality Levels:**
- **Good**: Score ≥3 factors
- **Moderate**: Score 2 factors
- **Poor**: Score <2 factors

**Output Fields:**
```json
{
  "summary": {
    "signal_quality": "good"
  }
}
```

## API Response Format

### Complete Example Response

```json
{
  "recording_id": "ABC123",
  "summary": {
    "rhythm": "sinus",
    "rhythm_confidence": 0.90,
    "mean_hr_bpm": 72.5,
    "min_hr_bpm": 68.0,
    "max_hr_bpm": 78.0,
    "signal_quality": "good"
  },
  "beats": {
    "r_peaks_ms": [800, 1620, 2410, ...],
    "rr_ms": [820, 790, 810, ...],
    "artifact_mask": [],
    "beat_count": 36
  },
  "hrv_time": {
    "sdnn_ms": 45.2,
    "rmssd_ms": 38.5
  },
  "intervals": {
    "qrs_ms": 85.0,
    "qt_ms": 345.6,
    "qtc_ms_bazett": 420.0,
    "pr_ms": 153.5,
    "uncertainty_ms": 20.0
  },
  "arrhythmia": {
    "ectopy_burden_pct": 2.8,
    "pvcs_detected": 1,
    "pacs_detected": 0,
    "morphology_variance": 0.045
  },
  "flags": {
    "pacemaker_detected": false,
    "st_deviation_flag": false
  },
  "version": "2.0.0"
}
```

## Technical Details

### Signal Preprocessing

**Bandpass Filter:**
- Frequency range: 0.5-40 Hz
- Filter type: Butterworth (2nd order)
- Removes baseline wander and high-frequency noise

**Normalization:**
- Zero mean
- Unit standard deviation

### R-Peak Detection

**Pan-Tompkins Style Pipeline:**
1. Derivative (emphasizes QRS slopes)
2. Squaring (emphasizes high frequencies)
3. Moving window integration (150ms)
4. Adaptive threshold (median + percentile-based)
5. Refractory period enforcement (300ms)

### Computational Performance

- **Typical processing time**: 50-100ms for 30-second ECG
- **Memory usage**: <10MB per recording
- **Optimized for**: Single-lead, 128-512 Hz sampling

### Limitations

**Single-Lead Constraints:**
- Cannot perform 12-lead analysis
- Limited ST-segment analysis
- P-wave detection less reliable than multi-lead
- Cannot determine electrical axis

**Morphology Detection:**
- Based on correlation, not absolute templates
- May miss subtle morphology changes
- Best with good signal quality

**QT Measurement:**
- T-wave end detection challenging in single-lead
- Uncertainty ±20-35ms typical
- Requires visual confirmation for clinical decisions

## Clinical Use Cases

### 1. Rhythm Screening
- Detect atrial fibrillation
- Identify frequent ectopy
- Monitor heart rate trends

### 2. QT Monitoring
- Drug safety monitoring (QT-prolonging medications)
- Congenital long QT syndrome screening
- Post-cardiac event monitoring

### 3. Ectopy Quantification
- PVC burden assessment
- PAC frequency tracking
- Morphology variance (suggests multiple foci)

### 4. HRV Analysis
- Autonomic function assessment
- SDNN: Overall HRV
- RMSSD: Parasympathetic activity

## Testing

Run the test suite:

```bash
python test_enhanced_analysis.py
```

**Test Coverage:**
- Normal sinus rhythm
- ECG with PVCs
- Short recordings (edge cases)
- Elevated heart rate
- Signal quality assessment

## Migration from v1.0

### Breaking Changes
None - response format extended, not changed.

### New Fields
All new fields use `.get()` with defaults for backward compatibility:
- `pr_ms`: Defaults to `null`
- `pvcs_detected`: Defaults to `0`
- `pacs_detected`: Defaults to `0`
- `morphology_variance`: Defaults to `0.0`
- `beat_count`: Defaults to `len(r_peaks_ms)`

### Version Update
- v1.0: Placeholder intervals, basic rhythm
- v2.0: Real measurements, advanced arrhythmia detection

## References

### Algorithms
- **Pan-Tompkins**: QRS detection (1985)
- **Bazett Formula**: QTc correction (1920)
- **Morphology Correlation**: Ectopy detection

### Clinical Guidelines
- AHA/ACC/HRS: QT interval measurement
- ESC: Arrhythmia classification
- Apple Watch ECG: Single-lead interpretation guidelines

## Support

For questions or issues:
- Check test suite: `test_enhanced_analysis.py`
- Review algorithm details in: `app/feature_extractor.py`
- API documentation: `/docs` endpoint

## Disclaimer

**This analysis tool is for research and informational purposes only. It is not FDA-cleared or intended for clinical diagnosis. All results should be reviewed by qualified healthcare providers.**
