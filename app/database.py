"""
Database models and operations for ECG analysis history
"""
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import statistics
import os

Base = declarative_base()

class ECGAnalysis(Base):
    """Store historical ECG analysis results"""
    __tablename__ = 'ecg_analyses'

    id = Column(Integer, primary_key=True)
    recording_id = Column(String, unique=True, index=True)
    timestamp_utc = Column(DateTime, index=True)

    # Rhythm metrics
    rhythm_classification = Column(String)
    rhythm_confidence = Column(Float)

    # Heart rate metrics
    hr_mean = Column(Float)
    hr_min = Column(Float)
    hr_max = Column(Float)

    # RR interval metrics
    rr_mean = Column(Float)
    rr_min = Column(Float)
    rr_max = Column(Float)
    rr_cv = Column(Float)

    # HRV metrics
    hrv_sdnn = Column(Float)
    hrv_rmssd = Column(Float)

    # Interval metrics
    p_wave_duration = Column(Float, nullable=True)
    pr_interval = Column(Float, nullable=True)
    pr_segment = Column(Float, nullable=True)
    qrs_duration = Column(Float)
    st_segment = Column(Float, nullable=True)
    t_wave_duration = Column(Float, nullable=True)
    qt_interval = Column(Float)
    qtc = Column(Float)

    # Signal quality
    signal_quality = Column(String)
    artifact_burden = Column(Float, nullable=True)

    # Morphology
    ectopy_burden = Column(Float, nullable=True)

    # Store complete JSON for reference
    full_analysis_json = Column(JSON)


# Database setup
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./ecg_data.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def save_analysis(recording_id: str, timestamp_utc: datetime, analysis_data: Dict) -> None:
    """Save an ECG analysis result to the database"""
    db = SessionLocal()
    try:
        features = analysis_data.get('features', {})

        # Extract nested values safely
        hr = features.get('heart_rate_bpm', {})
        rr = features.get('rr_intervals_ms', {})
        hrv = features.get('hrv', {})
        intervals = features.get('intervals', {})
        quality = features.get('signal_quality', {})
        morphology = features.get('morphology', {})

        analysis = ECGAnalysis(
            recording_id=recording_id,
            timestamp_utc=timestamp_utc,
            rhythm_classification=features.get('rhythm_classification'),
            rhythm_confidence=features.get('rhythm_confidence'),
            hr_mean=hr.get('mean'),
            hr_min=hr.get('min'),
            hr_max=hr.get('max'),
            rr_mean=rr.get('mean'),
            rr_min=rr.get('min'),
            rr_max=rr.get('max'),
            rr_cv=rr.get('coefficient_of_variation'),
            hrv_sdnn=hrv.get('sdnn_ms'),
            hrv_rmssd=hrv.get('rmssd_ms'),
            p_wave_duration=intervals.get('p_wave_duration_ms'),
            pr_interval=intervals.get('pr_interval_ms'),
            pr_segment=intervals.get('pr_segment_ms'),
            qrs_duration=intervals.get('qrs_duration_ms'),
            st_segment=intervals.get('st_segment_ms'),
            t_wave_duration=intervals.get('t_wave_duration_ms'),
            qt_interval=intervals.get('qt_interval_ms'),
            qtc=intervals.get('qtc_ms'),
            signal_quality=quality.get('overall_quality'),
            artifact_burden=quality.get('artifact_burden_percent'),
            ectopy_burden=morphology.get('ectopy_burden_percent'),
            full_analysis_json=analysis_data
        )

        # Check if recording already exists
        existing = db.query(ECGAnalysis).filter(ECGAnalysis.recording_id == recording_id).first()
        if existing:
            # Update existing record
            for key, value in analysis.__dict__.items():
                if not key.startswith('_'):
                    setattr(existing, key, value)
        else:
            # Add new record
            db.add(analysis)

        db.commit()
    finally:
        db.close()


def calculate_30day_stats(metric_name: str, current_value: Optional[float]) -> Dict[str, Optional[float]]:
    """
    Calculate 30-day average and standard deviation for a metric

    Returns:
        {
            "avg_30d": float or None,
            "std_30d": float or None,
            "is_outlier": bool (True if >2 SD from mean)
        }
    """
    if current_value is None:
        return {"avg_30d": None, "std_30d": None, "is_outlier": False}

    db = SessionLocal()
    try:
        # Get data from last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)

        # Query based on metric name
        query = db.query(ECGAnalysis).filter(ECGAnalysis.timestamp_utc >= thirty_days_ago)
        records = query.all()

        if not records or len(records) < 2:
            return {"avg_30d": None, "std_30d": None, "is_outlier": False}

        # Extract values for the specific metric
        values = []
        for record in records:
            val = getattr(record, metric_name, None)
            if val is not None:
                values.append(val)

        if len(values) < 2:
            return {"avg_30d": None, "std_30d": None, "is_outlier": False}

        # Calculate statistics
        avg = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0

        # Check if current value is outlier (>2 SD from mean)
        is_outlier = abs(current_value - avg) > (2 * std) if std > 0 else False

        return {
            "avg_30d": round(avg, 2),
            "std_30d": round(std, 2),
            "is_outlier": is_outlier
        }
    finally:
        db.close()


def get_30day_stats_for_all_metrics(features: Dict) -> Dict[str, Dict]:
    """
    Calculate 30-day stats for all metrics in the analysis

    Returns a dictionary with the same structure as features, but with added stats
    """
    hr = features.get('heart_rate_bpm', {})
    rr = features.get('rr_intervals_ms', {})
    hrv = features.get('hrv', {})
    intervals = features.get('intervals', {})
    morphology = features.get('morphology', {})

    stats = {
        'heart_rate_bpm': {
            'mean': calculate_30day_stats('hr_mean', hr.get('mean')),
            'min': calculate_30day_stats('hr_min', hr.get('min')),
            'max': calculate_30day_stats('hr_max', hr.get('max'))
        },
        'rr_intervals_ms': {
            'mean': calculate_30day_stats('rr_mean', rr.get('mean')),
            'min': calculate_30day_stats('rr_min', rr.get('min')),
            'max': calculate_30day_stats('rr_max', rr.get('max')),
            'coefficient_of_variation': calculate_30day_stats('rr_cv', rr.get('coefficient_of_variation'))
        },
        'hrv': {
            'sdnn_ms': calculate_30day_stats('hrv_sdnn', hrv.get('sdnn_ms')),
            'rmssd_ms': calculate_30day_stats('hrv_rmssd', hrv.get('rmssd_ms'))
        },
        'intervals': {
            'p_wave_duration_ms': calculate_30day_stats('p_wave_duration', intervals.get('p_wave_duration_ms')),
            'pr_interval_ms': calculate_30day_stats('pr_interval', intervals.get('pr_interval_ms')),
            'pr_segment_ms': calculate_30day_stats('pr_segment', intervals.get('pr_segment_ms')),
            'qrs_duration_ms': calculate_30day_stats('qrs_duration', intervals.get('qrs_duration_ms')),
            'st_segment_ms': calculate_30day_stats('st_segment', intervals.get('st_segment_ms')),
            't_wave_duration_ms': calculate_30day_stats('t_wave_duration', intervals.get('t_wave_duration_ms')),
            'qt_interval_ms': calculate_30day_stats('qt_interval', intervals.get('qt_interval_ms')),
            'qtc_ms': calculate_30day_stats('qtc', intervals.get('qtc_ms'))
        },
        'morphology': {
            'ectopy_burden_percent': calculate_30day_stats('ectopy_burden', morphology.get('ectopy_burden_percent'))
        }
    }

    return stats
