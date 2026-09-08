"""
Wearable Data Processing and Modeling project
Runtime and path-safety helpers.
"""

from __future__ import annotations
from datetime import datetime, timezone
import os
from pathlib import Path
from ..exceptions import ConfigurationError


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def cap_workers(requested:int | None) -> int:
    """ Return a valid worker count that is robust when CPU detection fails. """

    cpu_count= os.cpu_count() or 1
    default= max(1, cpu_count - 1)
    if requested is None:
        return default
    if requested <= 0:
        raise ConfigurationError("max_workers must be a positive integer.")
    return min(requested, cpu_count)


def ensure_disjoint_roots(input_root:Path, output_root:Path) -> None:
    """ Reject layouts that could overwrite input data or recursively process output. """

    input_resolved= input_root.resolve()
    output_resolved= output_root.resolve()
    if input_resolved == output_resolved:
        raise ConfigurationError("Input and output roots must be different directories.")
    if output_resolved.is_relative_to(input_resolved):
        raise ConfigurationError("Output root must not be located inside the input root.")
    if input_resolved.is_relative_to(output_resolved):
        raise ConfigurationError("Input root must not be located inside the output root.")