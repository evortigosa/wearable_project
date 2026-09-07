"""
Wearable Data Processing and Modeling project
Shared path, serialization and logging helpers.
"""

from __future__ import annotations
from collections.abc import Iterable
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from typing import Any
import numpy as np
import pandas as pd
from .exceptions import ConfigurationError

LOGGER= logging.getLogger(__name__)
MONTHLY_FILE_RE= re.compile(r"^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])\.csv$")
SAFE_FEATURE_RE= re.compile(r"^[A-Za-z0-9_-]+$")
MAX_FEATURE_STEM_LENGTH= 120


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


def safe_feature_stem(feature_name:str) -> str:
    """ Create a collision-resistant filename stem from an arbitrary feature name. """

    feature_name= str(feature_name).strip()
    if (
        feature_name and len(feature_name) <= MAX_FEATURE_STEM_LENGTH and SAFE_FEATURE_RE.fullmatch(feature_name)
    ):
        return feature_name
    sanitized= "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in feature_name).strip("_")
    sanitized= sanitized or "feature"
    digest= hashlib.sha1(feature_name.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    prefix_length= MAX_FEATURE_STEM_LENGTH - len(digest) - 2
    return f"{sanitized[:prefix_length]}__{digest}"


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


def file_fingerprint(path:Path, mode:str= "metadata") -> dict[str, Any]:
    """ Fingerprint a file for idempotent resume checks. """

    stat= path.stat()
    result:dict[str, Any]= {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if mode == "sha256":
        digest= hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"]= digest.hexdigest()
    elif mode != "metadata":
        raise ConfigurationError("fingerprint_mode must be 'metadata' or 'sha256'.")
    return result


def config_digest(payload:dict[str, Any]) -> str:
    serialized= json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def atomic_write_text(path:Path, text:str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary= path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path:Path, payload:Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_write_dataframe(df:pd.DataFrame, path:Path, output_format:str) -> None:
    """ Write a DataFrame completely before exposing the destination path. """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary= path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if output_format == "csv":
        serialize_nested_columns(df).to_csv(temporary, index=False)
    elif output_format == "parquet":
        try:
            df.to_parquet(temporary, index=False)
        except ImportError as exc:
            raise ConfigurationError(
                "Parquet output requires the optional 'pyarrow' or 'fastparquet' dependency."
            ) from exc
    else:
        raise ConfigurationError("output_format must be 'csv' or 'parquet'.")
    os.replace(temporary, path)


def read_table(path:Path, *, chunksize:int | None= None) -> pd.DataFrame | Iterable[pd.DataFrame]:
    if path.suffix.casefold() == ".parquet":
        if chunksize is not None:
            raise ConfigurationError("Chunked reads are not supported for Parquet in this helper.")
        return pd.read_parquet(path)
    return pd.read_csv(path, chunksize=chunksize)
