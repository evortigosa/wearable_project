"""
Wearable Data Processing and Modeling project
Deterministic serialization helpers.
"""


from __future__ import annotations
from datetime import datetime
import hashlib
import json
from typing import Any
import numpy as np
import pandas as pd



def stable_json(value:Any) -> str:
    """ Serialize nested/scalar values deterministically for hashing and CSV output. """

    def default(obj:Any) -> Any:
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=default)


def canonical_cell(value:Any) -> Any:
    """ Convert unhashable nested cells to deterministic strings for deduplication. """

    if isinstance(value, (dict, list, tuple, set, np.ndarray)):
        return stable_json(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def serialize_nested_columns(df:pd.DataFrame) -> pd.DataFrame:
    """ Return a copy whose nested object cells are canonical JSON strings. """

    result= df.copy()
    for column in result.select_dtypes(include=["object"]).columns:
        result[column]= result[column].map(
            lambda value: stable_json(value)
            if isinstance(value, (dict, list, tuple, set, np.ndarray))
            else value
        )
    return result


def config_digest(payload:dict[str, Any]) -> str:
    serialized= json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
