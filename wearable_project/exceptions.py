"""
Wearable Data Processing and Modeling project
Shared exceptions.
"""


class WearableProjectError(Exception):
    """Base package error."""


class InputLayoutError(WearableProjectError):
    """Input does not follow the participant/month directory contract."""


class DuplicateMonthError(InputLayoutError):
    """Two files map to the same canonical participant month."""


class PayloadDecodeError(WearableProjectError):
    """A nested HealthKit payload cannot be decoded safely."""


class ParticipantProcessingError(WearableProjectError):
    """A participant cannot be processed transactionally."""


class SnapshotConflictError(WearableProjectError):
    """A cumulative snapshot violates the selected incremental policy."""


class OutputValidationError(WearableProjectError):
    """A staged participant output is invalid."""


class ResamplingOutOfScopeError(WearableProjectError):
    """Resampling is intentionally excluded from milestone one."""
