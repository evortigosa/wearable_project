"""
Wearable Data Processing and Modeling project
Facts about a representative sample root, counted directly from its CSV files with the csv module.
The sample-based tests compare what the DataLoaders report with these counts, so they hold for any representative
sample rather than for one snapshot of it. Nothing here goes through the DataLoaders, so the comparison checks the
loaders against the files themselves.
"""


from __future__ import annotations
import csv
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def _feature(root: str, feature: str) -> tuple[tuple[dict[str, str], ...], int]:
    """Every data row of the feature in the root, and the number of participant files holding it."""

    rows: list[dict[str, str]] = []
    files = 0
    for path in sorted(Path(root).glob(f"[!.]*/{feature}.csv")):
        files += 1
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return tuple(rows), files


def rows(root: Path | str, feature: str) -> int:
    """Data rows of the feature, across every participant file in the root."""

    return len(_feature(str(root), feature)[0])


def files(root: Path | str, feature: str) -> int:
    """Participant files of the feature in the root: one per participant who has it."""

    return _feature(str(root), feature)[1]


def nonblank(root: Path | str, feature: str, column: str) -> int:
    """Rows whose cell in ``column`` is not blank; an absent column counts as blank."""

    return sum(1 for row in _feature(str(root), feature)[0] if (row.get(column) or "").strip())


def equal(root: Path | str, feature: str, column: str, value: str) -> int:
    """Rows whose cell in ``column`` holds exactly ``value``."""

    return sum(1 for row in _feature(str(root), feature)[0] if (row.get(column) or "") == value)


def included(root: Path | str, feature: str) -> int:
    """Rows included by default: a blank cell, or an absent include_by_default column, means included."""

    return sum(1 for row in _feature(str(root), feature)[0] if (row.get("include_by_default") or "").strip() != "0")
