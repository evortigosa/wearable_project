"""
Wearable Data Processing and Modeling project
Read participant/month files from an extracted raw root or directly from a ZIP.
"""


from __future__ import annotations
import csv
import io
import re
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, TextIO


MONTH_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{1,2})\.csv$")
EXPORT_MARKER = "export-ended.txt"


def raise_csv_field_limit() -> int:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit //= 10


def canonical_month(name: str) -> str | None:
    match = MONTH_RE.match(name)
    if match is None:
        return None
    year = int(match.group("year"))
    month = int(match.group("month"))
    if not 1 <= month <= 12:
        return None
    return f"{year:04d}-{month:02d}"


@dataclass(frozen=True, slots=True)
class RawFile:
    participant_id: str
    canonical_month: str
    basename: str
    member_path: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ParticipantFiles:
    participant_id: str
    files: tuple[RawFile, ...]
    export_ended_present: bool
    invalid_csv_names: tuple[str, ...]
    duplicate_months: tuple[tuple[str, tuple[str, ...]], ...]


class RawInput:
    kind: str

    def participants(self) -> tuple[ParticipantFiles, ...]:
        raise NotImplementedError

    @contextmanager
    def open_text(self, file: RawFile) -> Iterator[TextIO]:
        raise NotImplementedError

    def descriptor(self) -> dict[str, object]:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self) -> "RawInput":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class DirectoryInput(RawInput):
    kind = "directory"

    def __init__(self, path: Path):
        self.original_path = path.resolve()
        self.root = self._resolve_root(self.original_path)
        self._participants = self._discover()

    @staticmethod
    def _looks_like_root(path: Path) -> bool:
        if not path.is_dir():
            return False
        for child in path.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            try:
                if any(canonical_month(item.name) for item in child.iterdir() if item.is_file()):
                    return True
            except OSError:
                continue
        return False

    @classmethod
    def _resolve_root(cls, path: Path) -> Path:
        if cls._looks_like_root(path):
            return path
        children = [child for child in path.iterdir() if child.is_dir() and not child.name.startswith(".")]
        if len(children) == 1 and cls._looks_like_root(children[0]):
            return children[0]
        raise ValueError(
            f"Could not identify participant directories below {path}. Expected participant/YYYY-M.csv files."
        )

    def _discover(self) -> tuple[ParticipantFiles, ...]:
        result: list[ParticipantFiles] = []
        for participant_dir in sorted(
            (p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.name,
        ):
            by_month: dict[str, list[RawFile]] = {}
            invalid: list[str] = []
            marker = False
            for path in sorted(participant_dir.iterdir(), key=lambda p: p.name):
                if not path.is_file():
                    continue
                if path.name == EXPORT_MARKER:
                    marker = True
                    continue
                if path.suffix.lower() != ".csv":
                    continue
                month = canonical_month(path.name)
                if month is None:
                    invalid.append(path.name)
                    continue
                raw = RawFile(
                    participant_id=participant_dir.name,
                    canonical_month=month,
                    basename=path.name,
                    member_path=str(path),
                    size_bytes=path.stat().st_size,
                )
                by_month.setdefault(month, []).append(raw)
            files = tuple(sorted((item for items in by_month.values() for item in items), key=lambda x: (x.canonical_month, x.basename)))
            duplicates = tuple(
                (month, tuple(item.basename for item in items))
                for month, items in sorted(by_month.items())
                if len(items) > 1
            )
            if files or marker or invalid:
                result.append(ParticipantFiles(participant_dir.name, files, marker, tuple(invalid), duplicates))
        return tuple(result)

    def participants(self) -> tuple[ParticipantFiles, ...]:
        return self._participants

    @contextmanager
    def open_text(self, file: RawFile) -> Iterator[TextIO]:
        with Path(file.member_path).open("r", encoding="utf-8-sig", newline="") as handle:
            yield handle

    def descriptor(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "input_path": str(self.original_path),
            "resolved_root": str(self.root),
            "participant_count": len(self._participants),
            "monthly_file_count": sum(len(p.files) for p in self._participants),
            "source_bytes": sum(f.size_bytes for p in self._participants for f in p.files),
        }


class ZipInput(RawInput):
    kind = "zip"

    def __init__(self, path: Path):
        self.path = path.resolve()
        self._zip = zipfile.ZipFile(self.path)
        self._participants = self._discover()

    def _discover(self) -> tuple[ParticipantFiles, ...]:
        monthly: dict[str, dict[str, list[RawFile]]] = {}
        invalid: dict[str, list[str]] = {}
        markers: set[str] = set()
        for info in self._zip.infolist():
            if info.is_dir():
                continue
            posix = PurePosixPath(info.filename)
            if len(posix.parts) < 2 or any(part.startswith("__MACOSX") for part in posix.parts):
                continue
            participant_id = posix.parent.name
            if not participant_id or participant_id.startswith("."):
                continue
            if posix.name == EXPORT_MARKER:
                markers.add(participant_id)
                continue
            if posix.suffix.lower() != ".csv":
                continue
            month = canonical_month(posix.name)
            if month is None:
                invalid.setdefault(participant_id, []).append(posix.name)
                continue
            raw = RawFile(
                participant_id=participant_id,
                canonical_month=month,
                basename=posix.name,
                member_path=info.filename,
                size_bytes=info.file_size,
            )
            monthly.setdefault(participant_id, {}).setdefault(month, []).append(raw)
        participant_ids = sorted(set(monthly) | set(invalid) | markers)
        result: list[ParticipantFiles] = []
        for participant_id in participant_ids:
            by_month = monthly.get(participant_id, {})
            files = tuple(sorted((item for items in by_month.values() for item in items), key=lambda x: (x.canonical_month, x.basename)))
            duplicates = tuple(
                (month, tuple(item.basename for item in items))
                for month, items in sorted(by_month.items()) if len(items) > 1
            )
            result.append(
                ParticipantFiles(
                    participant_id, files, participant_id in markers, tuple(sorted(invalid.get(participant_id, []))),
                    duplicates,
                )
            )
        return tuple(result)

    def participants(self) -> tuple[ParticipantFiles, ...]:
        return self._participants

    @contextmanager
    def open_text(self, file: RawFile) -> Iterator[TextIO]:
        binary = self._zip.open(file.member_path, "r")
        text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
        try:
            yield text
        finally:
            text.close()

    def descriptor(self) -> dict[str, object]:
        stat = self.path.stat()
        return {
            "kind": self.kind,
            "input_path": str(self.path),
            "archive_size_bytes": stat.st_size,
            "archive_mtime_ns": stat.st_mtime_ns,
            "participant_count": len(self._participants),
            "monthly_file_count": sum(len(p.files) for p in self._participants),
            "uncompressed_monthly_bytes": sum(f.size_bytes for p in self._participants for f in p.files),
        }

    def close(self) -> None:
        self._zip.close()


def open_raw_input(path: Path) -> RawInput:
    path = path.expanduser().resolve()
    if path.is_dir():
        return DirectoryInput(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        return ZipInput(path)
    raise ValueError(f"Input must be an extracted raw root or ZIP archive: {path}")
