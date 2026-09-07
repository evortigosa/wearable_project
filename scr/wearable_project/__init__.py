"""
Wearable Data Processing and Modeling project
"""

__version__ = "2.1.0"

from .policies import (
    FeaturePolicy,
    feature_policy_origin,
    load_feature_policies,
    resolve_feature_policy,
)
from .processing import ProcessingConfig, process_dataset, process_participant
from .statistics import StatisticsConfig, summarize_dataset
from .timeseries import SegmentConfig, build_time_series, segment_continuous_intervals
from .windowing import event_window_overlaps, fixed_window_timedelta, window_feature

__all__ = [
    "FeaturePolicy",
    "ProcessingConfig",
    "SegmentConfig",
    "StatisticsConfig",
    "build_time_series",
    "event_window_overlaps",
    "fixed_window_timedelta",
    "feature_policy_origin",
    "load_feature_policies",
    "process_dataset",
    "process_participant",
    "resolve_feature_policy",
    "segment_continuous_intervals",
    "summarize_dataset",
    "window_feature",
]
