"""
Wearable Data Processing and Modeling project
Project-specific exceptions.
"""


class WearableProjectError(Exception):
    """ Base class for expected pipeline failures. """


class SchemaError(WearableProjectError):
    """ Raised when an input table does not satisfy the required schema. """


class PayloadParseError(WearableProjectError):
    """ Raised when a serialized payload cannot be decoded safely. """


class ConfigurationError(WearableProjectError):
    """ Raised when pipeline configuration is inconsistent or unsafe. """
