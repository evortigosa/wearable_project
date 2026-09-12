"""
Wearable Data Processing and Modeling project
Reserved for the later optional resampling milestone.
"""


from wearable_project.exceptions import ResamplingOutOfScopeError


def resample_dataset(*args, **kwargs):
    raise ResamplingOutOfScopeError(
        "Milestone one intentionally writes only cleaned native events. "
        "Fixed-window resampling will be a separate derived-data command."
    )
