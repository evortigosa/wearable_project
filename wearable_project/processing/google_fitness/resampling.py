"""
Wearable Data Processing and Modeling project
Google Fitness native processing intentionally excludes fixed-window resampling.
"""


from wearable_project.exceptions import ResamplingOutOfScopeError


def resample_dataset(*args, **kwargs):
    raise ResamplingOutOfScopeError(
        "Google Fitness native processing writes only source-resolution events. "
        "Fixed-window resampling belongs to a separate derived-data layer."
    )
