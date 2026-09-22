"""
Wearable Data Processing and Modeling project
Shared implementation for HPP-style Apple wearable feature loaders. Native processing and curated roots keep
the same participant / feature directory layout, so a single loader implementation can serve both phases
without altering the stored files.
"""


from __future__ import annotations
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal
import hashlib
import json
import math
import sqlite3
import warnings
import numpy as np
import pandas as pd
from wearable_project.exceptions import (
    DataLoaderConfigurationError, DataLoaderError, DataLoaderPathError, DataLoaderReadError,
)
from wearable_project.DataLoaders._profile import ProfileReport, format_bytes, render_profile_text
from wearable_project.processing.registry import get_feature_spec


ProcessingPhase = Literal["native", "curated"]

DEFAULT_NATIVE_ROOT = Path(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/third-party/EV_cleaned_apple_healthkit"
)
DEFAULT_CURATED_ROOT = Path(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/third-party/EV_curated_apple_healthkit"
)

_TEMPORAL_COLUMNS = (
    "start_date",
    "end_date",
    "datetime",
    "created_at",
    "updated_at",
    "modified_date",
)

# Each processing phase writes its own state database at the root of its output tree. The presence of
# exactly one of these is a reliable marker of what a directory actually holds, which lets a loader
# detect a phase/root mismatch instead of trusting the declared label. Trimmed sample trees often carry
# neither marker; detection then returns None and no claim is made.
_PHASE_STATE_FILES: dict[ProcessingPhase, str] = {
    "native": ".wearable_state.sqlite",
    "curated": ".wearable_curation_state.sqlite",
}


class DataLoaderStateError(DataLoaderReadError):
    """
    A curated file disagrees with the curation state database that recorded it, or that database cannot
    be read safely. Subclasses ``DataLoaderReadError`` so existing handlers for read failures still apply.
    Typical causes are a damaged CSV, a stale state database, a loader incompatible with the writer version,
    or an undocumented serialization change.
    """


class DataLoaderSizeError(DataLoaderError):
    """
    A ``get_data`` call would return, or has begun returning, more rows than its ``max_rows`` limit. Raised
    before any feature file is parsed when the phase's state database gives an exact row count, and otherwise
    as soon as the rows retained so far exceed the limit.
    """


# Default ceiling on the rows one get_data call may return. Peak memory while loading is roughly 350-1,200
# bytes per retained row depending on the feature's width (measured on the representative samples), so this
# keeps a single call to the order of 10-30 GB at peak. It is a class attribute of AppleHealthFeatureLoader so
# it can be raised, lowered, or disabled for a whole session in one assignment.
DEFAULT_MAX_ROWS = 50_000_000
_USE_DEFAULT_MAX_ROWS: Any = object()


StateValidation = Literal["auto", "required", "off"]
_STATE_VALIDATION_MODES: tuple[str, ...] = ("auto", "required", "off")

# Dense logical schema for curated files. The curation writer stores these columns sparsely: a column is
# physically written only when at least one row in that participant-feature file needs a non-default value
# (for include_by_default, only when some row is excluded). Absence therefore means "every row holds the
# default", and the loader restores the default so every curated frame exposes the same logical columns.
# Insertion order follows the curation-appended order documented in README.md. The engine itself counts a
# missing acquisition method as "unclassified", which is why that label is used here.
_DENSE_CURATED_DEFAULTS: dict[str, Any] = {
    "acquisition_method": "unclassified",
    "curation_status": "pass",
    "curation_flags": "",
    "include_by_default": True,
}
# Columns whose blank cells, not only an absent column, mean "the default". curation_status and
# include_by_default are never blank inside a written column, so a blank there is left for reconciliation
# to report rather than silently filled.
_DENSE_FILL_BLANKS: frozenset[str] = frozenset({"acquisition_method", "curation_flags"})
_CURATION_FLAG_SEPARATOR = ";"

# Reading the state database must never write to the data tree. SQLite's ``mode=ro`` still creates
# ``-shm``/``-wal`` side files beside a WAL database, and fails outright from a read-only directory, so the
# loader opens it with ``immutable=1`` instead. That flag ignores the write-ahead log, which is only safe
# when the log holds no frames; a non-empty ``-wal`` file means a writer is active or did not shut down
# cleanly, and the loader refuses rather than read stale state.
_STATE_WAL_SUFFIX = "-wal"
# Rollback-journal databases (the native state database uses this mode) signal an in-progress or interrupted
# write with a non-empty ``-journal`` file, the counterpart of a non-empty ``-wal``. ``immutable=1`` would read
# either kind of database inconsistently in that state.
_STATE_ACTIVE_SUFFIXES = (_STATE_WAL_SUFFIX, "-journal")

# Text forms accepted when normalizing a true/false column. Anything else leaves the column unconverted, so
# an unexpected token is preserved rather than silently turned into missing data.
_BOOLEAN_TEXT: dict[str, bool] = {
    "true": True, "t": True, "yes": True, "y": True, "1": True, "1.0": True,
    "false": False, "f": False, "no": False, "n": False, "0": False, "0.0": False,
}
_UNRECOGNIZED = object()

_PROJECTION_SCHEMA: Any = None


def _projection_schema() -> Any:
    """
    Import ``curation.schema`` on first use and cache it.
    Deferred rather than imported at module scope so that importing a feature loader does not pull the
    curation package, whose ``__init__`` eagerly loads the policy, evidence and engine modules. The cost
    is therefore paid once per process, on the first ``get_data`` call, instead of on every import.
    """

    global _PROJECTION_SCHEMA
    if _PROJECTION_SCHEMA is None:
        from wearable_project.curation import schema
        _PROJECTION_SCHEMA = schema
    return _PROJECTION_SCHEMA


def _empty_participant_metadata() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "participant_id": pd.Series(dtype="object"),
            "rows": pd.Series(dtype="int64"),
            "first_date": pd.Series(pd.DatetimeIndex([], tz="UTC")),
            "last_date": pd.Series(pd.DatetimeIndex([], tz="UTC")),
            "stored_rows": pd.Series(dtype="int64"),
            "state_verified": pd.Series(dtype="boolean"),
            "policy_current": pd.Series(dtype="boolean"),
            "acquisition_classified_fraction": pd.Series(dtype="Float64"),
        }
    )
    frame.index = pd.Index([], dtype="object", name="RegistrationCode")
    return frame


def _empty_columns_metadata() -> pd.DataFrame:
    frame = pd.DataFrame(
        columns=["role", "description", "dtype", "non_null", "registry_unit", "dense_default"]
    )
    frame.index = pd.Index([], dtype="object", name="column")
    return frame


@dataclass(slots=True)
class LoaderData:
    """
    Container returned by :meth:`AppleHealthFeatureLoader.get_data`, shaped like the HPP ``LoaderData``.
    ``df``
        The feature rows, indexed by ``RegistrationCode`` and ``Date``.
    ``df_metadata``
        One row per participant present in ``df``: returned and stored row counts, the returned date span,
        and, in the curated phase, whether the participant's file was verified against the curation state
        database and curated under the installed policy.
    ``df_columns_metadata``
        One row per column of ``df``: its declared role and description, dtype, non-null count, the
        registry unit for measurement columns, and the dense default for reconstructed curation columns.
    ``load_report``
        Provenance of this call: root, phase, files read, filters, projection, and state verification.
        Formerly ``metadata``; the old name remains as a deprecated alias.
    """

    df: pd.DataFrame
    df_metadata: pd.DataFrame = field(default_factory=_empty_participant_metadata)
    df_columns_metadata: pd.DataFrame = field(default_factory=_empty_columns_metadata)
    load_report: dict[str, Any] = field(default_factory=dict)

    @property
    def metadata(self) -> dict[str, Any]:
        """Deprecated alias of :attr:`load_report`, kept so pre-0.2.0 callers continue to work."""

        warnings.warn(
            "LoaderData.metadata is deprecated and will be removed in a future release; "
            "use LoaderData.load_report. LoaderData.df_metadata now holds the per-participant table.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.load_report


class AppleHealthFeatureLoader:
    """
    Base class for one Apple HealthKit feature. Subclasses only declare ``feature_name`` and, when necessary,
    a different date anchor. Stored data are never modified by a loader.
    """

    feature_name: ClassVar[str]
    date_column: ClassVar[str] = "start_date"
    date_semantics: ClassVar[str] = "event_start"
    registration_prefix: ClassVar[str] = "10K_"
    # Session-wide ceiling for get_data(max_rows=...). Set None to disable, e.g.
    # ``AppleHealthFeatureLoader.default_max_rows = None``; a per-call max_rows always takes precedence.
    default_max_rows: ClassVar[int | None] = DEFAULT_MAX_ROWS

    def __init__(
        self, phase: ProcessingPhase | str = "curated", *,
        native_root: str | Path = DEFAULT_NATIVE_ROOT,
        curated_root: str | Path = DEFAULT_CURATED_ROOT,
        root: str | Path | None = None,
        state_validation: StateValidation | str = "auto",
    ) -> None:
        """
        ``state_validation`` governs how curated files are checked against the curation state database:
        ``"auto"`` verifies every file when the database exists and reports the load as unverified when it
        does not; ``"required"`` fails when the database is absent; ``"off"`` skips verification. Dense
        reconstruction of the curated logical schema happens in every mode. The native phase has no
        curation state, so ``"required"`` is rejected there. No filesystem access happens here.
        """

        normalized_phase = str(phase).strip().lower()
        if normalized_phase not in {"native", "curated"}:
            raise DataLoaderConfigurationError(
                "phase must be either 'native' or 'curated'; "
                f"received {phase!r}"
            )
        normalized_validation = str(state_validation).strip().lower()
        if normalized_validation not in _STATE_VALIDATION_MODES:
            raise DataLoaderConfigurationError(
                "state_validation must be one of "
                + ", ".join(repr(mode) for mode in _STATE_VALIDATION_MODES)
                + f"; received {state_validation!r}"
            )
        if normalized_validation == "required" and normalized_phase == "native":
            raise DataLoaderConfigurationError(
                "state_validation='required' applies only to phase='curated'; "
                "the native phase has no curation state to verify against"
            )
        self.phase: ProcessingPhase = normalized_phase  # type: ignore[assignment]
        self.state_validation: StateValidation = normalized_validation  # type: ignore[assignment]
        self.native_root = Path(native_root).expanduser()
        self.curated_root = Path(curated_root).expanduser()
        self._root_override = Path(root).expanduser() if root is not None else None

    @property
    def _data_index_names(self) -> list[str]:
        """HPP-style index names exposed by every wearable feature loader."""

        return ["RegistrationCode", "Date"]

    @property
    def data_root(self) -> Path:
        """Resolved root used by this loader instance."""

        if self._root_override is not None:
            return self._root_override
        return self.curated_root if self.phase == "curated" else self.native_root

    @property
    def filename(self) -> str:
        return f"{self.feature_name}.csv"

    @property
    def uses_root_override(self) -> bool:
        """True when an explicit ``root`` was supplied instead of the phase's default location."""

        return self._root_override is not None

    @staticmethod
    def detect_phase(root: Path) -> ProcessingPhase | None:
        """
        Report which processing phase a directory holds, judged by its state database.
        Returns ``None`` when neither nor both markers are present, i.e. whenever the directory cannot
        speak for itself. Trimmed sample trees are the common ``None`` case and are entirely valid.
        """

        found = [phase for phase, marker in _PHASE_STATE_FILES.items() if (root / marker).is_file()]
        if len(found) != 1:
            return None
        return found[0]

    def _checked_root(self) -> tuple[Path, ProcessingPhase | None]:
        """
        Resolve the data root, require it to exist, and check the declared phase against the tree's own
        state database. ``root`` overrides the phase default, so the declared phase is an assertion about the
        tree rather than a description of it; a mismatched root fails here instead of producing results whose
        reported phase is wrong.
        """

        root = self.data_root
        if not root.is_dir():
            raise DataLoaderPathError(f"{self.phase} wearable root does not exist or is not a directory: {root}")
        detected_phase = self.detect_phase(root)
        if detected_phase is not None and detected_phase != self.phase:
            raise DataLoaderConfigurationError(
                f"phase={self.phase!r} was requested but {root} contains "
                f"{detected_phase} data (found {_PHASE_STATE_FILES[detected_phase]!r}); "
                f"pass phase={detected_phase!r} or point root at a {self.phase} tree"
            )
        return root, detected_phase

    def info(self, *, include_evidence: bool = False):
        """
        Return usage, processing, and curation information for this loader. The report reflects this instance's
        phase and resolved root but performs no filesystem access. ``print(loader.info())`` gives a readable
        report; ``loader.info().as_dict()`` returns the structured form.
        """

        # Local import avoids a module cycle: DataLoaders.info imports this base class to construct
        # feature-specific reports.
        from wearable_project.DataLoaders.info import _feature_report

        return _feature_report(self, include_evidence=include_evidence)

    def get_data(
        self, registration_codes: str | int | Iterable[str | int] | None = None, *,
        start_date: str | pd.Timestamp | None = None, end_date: str | pd.Timestamp | None = None,
        columns: Sequence[str] | None = None, default_inclusion_only: bool = False, sort_index: bool = True,
        projection: str = "default", max_rows: int | None = _USE_DEFAULT_MAX_ROWS,
    ) -> LoaderData:
        """
        Load this feature across the requested participants.
        Parameters
        ----------
        registration_codes:
            Optional HPP registration code(s), e.g. ``"10K_1235738253"``. Bare participant folder IDs and integers
            are accepted as well. ``None`` discovers every participant containing this feature.
        start_date, end_date:
            Optional inclusive UTC bounds applied to the feature's HPP ``Date`` anchor. For almost all features
            this is ``start_date``. For ``ActivitySummary`` it is the retained outer ``datetime`` because
            curation intentionally does not invent a canonical summary day.
        columns:
            Optional subset of data columns returned after indexing. The source date column is still read as
            needed to construct ``Date``. An explicit list overrides ``projection`` entirely, so any stored
            column remains reachable by name.
        projection:
            Named column set, declared in ``curation.schema``. ``"default"`` returns timing, measurements,
            units, curation verdicts, acquisition, UTC offset, quality flags and feature-specific context.
            ``"analysis"`` adds the unit-resolution audit trail, device descriptors, deduplication bookkeeping
            and the IANA time zone. ``"full"`` returns every stored column, including identifiers, free-text
            device labels, the raw metadata payload and the ECG waveform; it reproduces the pre-0.2.0 loader
            behavior. Withheld column names are listed in ``LoaderData.metadata``.
        default_inclusion_only:
            Curated-only opt-in filter. When true, retain rows whose ``include_by_default`` is true.
            The default is false so ``get_data()`` never silently hides review/excluded rows.
            In the curated phase ``include_by_default`` is part of the dense logical schema: files that do not
            store the column hold the default ``True`` for every row, and that reconstruction is verified
            against the curation state database when one is present. When it cannot be verified, the
            filter still applies but a ``UserWarning`` reports how many kept rows rest on the unverified
            sparse-storage convention.
        sort_index:
            Sort by RegistrationCode and Date before returning.
        max_rows:
            Ceiling on the rows this call may return; ``None`` disables it. Defaults to the class attribute
            ``default_max_rows`` (50,000,000 unless changed for the session). When the phase's state database
            gives an exact count, an oversized request raises ``DataLoaderSizeError`` before any feature file
            is parsed; otherwise, for example with date bounds, the limit is enforced as rows are retained.
            Use ``profile()`` to see a request's size without loading it.
        """

        root, detected_phase = self._checked_root()

        if default_inclusion_only and self.phase != "curated":
            raise DataLoaderConfigurationError("default_inclusion_only is only defined for phase='curated'")

        # A bare string is one column name, not a sequence of single-character names.
        if isinstance(columns, str):
            columns = [columns]
        projection = str(projection).strip().lower()
        schema = _projection_schema()
        if not schema.is_known_projection(projection):
            raise DataLoaderConfigurationError(
                f"unknown projection {projection!r}; choose one of "
                + ", ".join(repr(name) for name in schema.available_projections())
            )
        # An explicit column list is an exact request and takes precedence over any named set.
        projecting = columns is None and projection != "full"

        participant_ids, missing_requested = self._resolve_participants(root, registration_codes)

        # Curated files expose a dense logical schema, verified against the curation state database when
        # it is present. The native phase has no curation, so nothing is reconstructed there.
        dense = self.phase == "curated"
        manifest: dict[str, dict[str, Any]] | None = None
        state_note: str | None = "not applicable to the native phase"
        installed_fingerprint: str | None = None
        if dense:
            manifest, state_note = self._open_manifest(root)
            if manifest is not None:
                installed_fingerprint = self._installed_policy_fingerprint()

        frames: list[pd.DataFrame] = []
        files_read = 0
        loaded_participants: list[str] = []
        read_participants: list[str] = []
        stored_rows: dict[str, int] = {}
        verified_participants: set[str] = set()
        diverged_participants: list[str] = []
        inclusion_counts_unverifiable = 0
        rows_reconstructed: Counter[str] = Counter()
        requested_columns = list(dict.fromkeys(columns)) if columns is not None else None
        requested_columns_seen: set[str] = set()
        # Column names as they appear on disk, in order of first appearance, so a call that selects no
        # rows can still return the schema it would have returned had rows survived.
        observed_columns: dict[str, None] = {}
        # The same, restricted to what the projection admits; this is the schema an empty projected
        # result must carry so it concatenates with a populated one.
        projected_columns: dict[str, None] = {}
        # Each column's dtype after timestamp parsing, from the first file that carries it. Used only to type
        # an empty result like a populated one, so concatenation does not degrade datetimes or numbers to text.
        column_dtypes: dict[str, list[Any]] = {}
        files_parsed = 0
        # Columns present on disk with no declared role in curation.schema: evidence that the stored
        # schema has moved ahead of the role table.
        undeclared_columns: dict[str, None] = {}
        inclusion_rows_dropped = 0
        inclusion_rows_reconstructed = 0
        inclusion_rows_unverified = 0

        lower = self._normalize_bound(start_date, "start_date")
        upper = self._normalize_bound(end_date, "end_date")
        if lower is not None and upper is not None and lower > upper:
            raise DataLoaderConfigurationError(
                f"start_date {lower.isoformat()} is after end_date {upper.isoformat()}"
            )

        limit = self._resolve_max_rows(max_rows)
        # Row counts come from the phase's state table, so an oversized request is refused before any
        # CSV is parsed. The native table is advisory; with state_validation='off' no state is read.
        estimate_state: dict[str, dict[str, Any]] | None = manifest
        estimate_note: str | None = state_note if manifest is None else None
        if not dense:
            estimate_state, estimate_note = self._open_native_outputs(root)
        size_estimate = self._size_estimate(
            estimate_state, participant_ids, root, dense=dense,
            default_inclusion_only=default_inclusion_only,
            date_filtered=start_date is not None or end_date is not None,
            presence_known=registration_codes is None,
        )
        if size_estimate is not None:
            size_estimate["source"] = "curation_state" if dense else "native_state"
            expected = size_estimate["rows_expected"]
            if limit is not None and expected is not None and expected > limit:
                raise DataLoaderSizeError(
                    f"{self.feature_name} get_data() would return {expected:,} rows from "
                    f"{size_estimate['participants']:,} participant file(s) "
                    f"({format_bytes(size_estimate['bytes_on_disk'])} on disk), above max_rows={limit:,}. "
                    + self._size_advice()
                )
        acquisition_total: Counter[str] = Counter()
        acquisition_by_participant: dict[str, Counter[str]] = {}
        retained_rows = 0

        for participant_id in participant_ids:
            path = root / participant_id / self.filename
            if not path.is_file():
                continue
            frame = self._read_feature_file(path)
            files_read += 1
            read_participants.append(participant_id)
            stored_rows[participant_id] = int(len(frame))

            if self.date_column not in frame.columns:
                raise DataLoaderReadError(
                    f"{path} does not contain required date anchor column "
                    f"{self.date_column!r} for {self.feature_name}"
                )

            # Dense reconstruction and reconciliation both act on the complete stored file, before any date
            # or inclusion filter, because the state database describes whole files. Checking a filtered
            # frame against whole-file counts would fail on every legitimately filtered call.
            inclusion_reconstructed = False
            inclusion_verified = False
            if dense:
                frame, filled, inclusion_reconstructed = self._densify(frame)
                rows_reconstructed.update(filled)
                if manifest is not None:
                    entry = manifest.get(participant_id)
                    if entry is None:
                        raise DataLoaderStateError(
                            f"{path} is present but the curation state database has no curation_outputs "
                            f"row for participant {participant_id!r} and feature {self.feature_name!r}; the "
                            "file cannot be verified and may be foreign or partially written. For a "
                            "deliberately assembled sample tree, pass state_validation='off'."
                        )
                    inclusion_verified = self._reconcile(frame, path, entry)
                    if not inclusion_verified:
                        inclusion_counts_unverifiable += 1
                    verified_participants.add(participant_id)
                    if (
                        installed_fingerprint is not None
                        and str(entry["policy_fingerprint"]) != installed_fingerprint
                    ):
                        diverged_participants.append(participant_id)

            # Whether a column exists is a property of the file, not of the rows that survive the
            # filters below. Record it here so an empty date window cannot be mistaken for an absent
            # column, and so the schema is known even when nothing survives.
            observed_columns.update(dict.fromkeys(frame.columns))
            undeclared_columns.update(dict.fromkeys(schema.undeclared_columns(frame.columns)))
            if projecting:
                projected_columns.update(
                    dict.fromkeys(schema.columns_for_projection(projection, frame.columns))
                )
            if requested_columns is not None:
                requested_columns_seen.update(
                    name for name in requested_columns if name in frame.columns
                )

            frame = self._parse_temporal_columns(frame, path)
            files_parsed += 1
            for column, dtype in frame.dtypes.items():
                column_dtypes.setdefault(column, []).append(dtype)
            dates = frame[self.date_column]
            if lower is not None:
                frame = frame.loc[dates >= lower].copy()
                dates = frame[self.date_column]
            if upper is not None:
                frame = frame.loc[dates <= upper].copy()

            if default_inclusion_only:
                # Always present here: default_inclusion_only is curated-only, and curated frames are dense.
                include = self._as_boolean(frame["include_by_default"])
                inclusion_rows_dropped += int((~include).sum())
                frame = frame.loc[include].copy()
                if inclusion_reconstructed:
                    inclusion_rows_reconstructed += int(len(frame))
                    if not inclusion_verified:
                        inclusion_rows_unverified += int(len(frame))

            if frame.empty:
                continue

            # Coverage describes the rows returned, whatever columns were requested, so it is counted after
            # every row filter and before the column projection.
            acquisition = self._acquisition_counts(frame)
            acquisition_total.update(acquisition)
            acquisition_by_participant[participant_id] = acquisition
            retained_rows += int(len(frame))
            if limit is not None and retained_rows > limit:
                raise DataLoaderSizeError(
                    f"{self.feature_name} get_data() exceeded max_rows={limit:,}: {retained_rows:,} rows were "
                    f"retained after reading {files_read:,} of {len(participant_ids):,} participant file(s). "
                    + self._size_advice()
                )

            # Keep the index anchor independently of the returned column set. This means ``columns=[...]`` does
            # not unexpectedly force the source date column back into the analytical columns.
            date_values = frame[self.date_column].copy()

            if requested_columns is not None:
                # Curated output is intentionally sparse: a derived column may be absent from a participant-feature
                # file when it was never needed there. Only the columns this file has are kept; concatenation fills
                # the rest, exactly as it does for projections. Filling per file with pd.NA instead would create
                # object columns, and pandas 3 lets those decide the concatenated dtype.
                frame = frame.loc[:, [name for name in requested_columns if name in frame.columns]].copy()
            elif projecting:
                frame = frame.loc[:, schema.columns_for_projection(projection, frame.columns)].copy()

            registration_code = self._to_registration_code(participant_id)
            frame.insert(0, "RegistrationCode", registration_code)
            frame.insert(1, "Date", date_values.loc[frame.index])
            frames.append(frame)
            loaded_participants.append(participant_id)

        if files_read and requested_columns is not None:
            unknown = [
                name for name in requested_columns if name not in requested_columns_seen
            ]
            if unknown:
                raise DataLoaderReadError(
                    f"requested column(s) are not present in any {self.feature_name} file: "
                    + ", ".join(unknown)
                )

        # The columns this call returns are fixed by the files it read, not by which rows survived the filters:
        # a date window or the inclusion filter must not change the schema, and an empty result must carry the
        # same columns as a populated one. Participant subsets can still differ, because each file stores only
        # the columns it uses; an explicit columns= list fixes the schema across any subset.
        merged_dtypes = {
            name: self._merged_dtype(found, missing=len(found) < files_parsed)
            for name, found in column_dtypes.items()
        }
        if requested_columns is not None:
            schema_columns = list(requested_columns)
        else:
            schema_columns = [
                name for name in (projected_columns if projecting else observed_columns)
                if name not in self._data_index_names
            ]

        if frames:
            # Pandas currently emits a FutureWarning when one participant has an all-NA placeholder for a sparse
            # curated column while another participant carries real values. That layout is intentional in the
            # curated contract, and the explicit column union below is stable across the supported pandas versions.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=("The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*"),
                    category=FutureWarning,
                )
                combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
            combined = combined.set_index(self._data_index_names, drop=True)
            combined.index = combined.index.set_names(self._data_index_names)
            for name in schema_columns:
                if name not in combined.columns:
                    combined[name] = pd.Series(index=combined.index, dtype=merged_dtypes.get(name, np.dtype(object)))
            combined = combined.loc[:, schema_columns]
            if sort_index:
                combined = combined.sort_index()
        else:
            combined = self._empty_frame(
                schema_columns if files_parsed else requested_columns, dtypes=merged_dtypes,
            )
        # Declared dtypes are applied once, after concatenation: categoricals built per participant would carry
        # different category sets and pandas would fall back to object on concatenation.
        combined = self._normalize_dtypes(combined, schema)

        # A manifest row whose file is absent means something curation wrote is no longer on disk. The rows
        # returned are still correct and verified, so this is reported rather than raised: withdrawing a
        # participant's data must not make the whole feature unloadable until the manifest is updated.
        manifest_files_missing: list[str] = []
        if manifest is not None:
            in_scope = set(manifest) if registration_codes is None else set(manifest) & set(participant_ids)
            manifest_files_missing = sorted(in_scope - set(read_participants))

        declared_withheld = (
            sorted(set(observed_columns) - set(projected_columns) - set(undeclared_columns))
            if projecting else []
        )
        report: dict[str, Any] = {
            "feature": self.feature_name,
            "phase": self.phase,
            "root": str(root),
            "date_column": self.date_column,
            "date_semantics": self.date_semantics,
            "files_read": files_read,
            "participants_loaded": len(set(loaded_participants)),
            "participant_ids_loaded": sorted(set(loaded_participants)),
            "requested_participants_missing_from_root": missing_requested,
            "rows": int(len(combined)),
            "default_inclusion_only": bool(default_inclusion_only),
            "root_override": self.uses_root_override,
            "detected_phase": detected_phase,
            "projection": projection if requested_columns is None else "columns",
            "columns_withheld": declared_withheld,
            "columns_undeclared": sorted(undeclared_columns),
            "dense_reconstruction": dense,
            "rows_reconstructed": dict(sorted(rows_reconstructed.items())),
            "state_validation": self.state_validation,
            "state_verified": (manifest is not None) if dense else None,
            "state_verification_note": state_note,
            "files_verified": len(verified_participants),
            "inclusion_counts_unverifiable": inclusion_counts_unverifiable,
            "policy_fingerprint_installed": installed_fingerprint,
            "participants_with_diverged_policy": sorted(set(diverged_participants)),
            "manifest_files_missing": manifest_files_missing,
            "max_rows": limit,
            "size_estimate": size_estimate,
            "size_estimate_note": estimate_note if size_estimate is None else None,
            "acquisition_method_counts": dict(sorted(acquisition_total.items())),
            "acquisition_classified_fraction": self._classified_fraction(acquisition_total),
        }

        if undeclared_columns:
            warnings.warn(
                f"{len(undeclared_columns)} column(s) in {self.feature_name} files have no declared role in "
                f"curation.schema: {', '.join(sorted(undeclared_columns))}. They are returned only by "
                "projection='full' or an explicit columns= request. Add them to COLUMN_ROLES.",
                UserWarning,
                stacklevel=2,
            )
        if diverged_participants:
            warnings.warn(
                f"{len(set(diverged_participants))} {self.feature_name} file(s) were curated under a policy "
                "fingerprint that differs from the installed curation registry. Their status and inclusion "
                "values reflect the policy in force when they were written, until the curated root is rebuilt. "
                "See load_report['participants_with_diverged_policy'].",
                UserWarning,
                stacklevel=2,
            )
        if manifest_files_missing:
            warnings.warn(
                f"The curation state database records {len(manifest_files_missing)} {self.feature_name} "
                "file(s) that are no longer on disk; the returned rows are verified but incomplete relative "
                "to the manifest. See load_report['manifest_files_missing'].",
                UserWarning,
                stacklevel=2,
            )

        if default_inclusion_only:
            report["default_inclusion_rows_dropped"] = inclusion_rows_dropped
            report["default_inclusion_rows_reconstructed"] = inclusion_rows_reconstructed
            report["default_inclusion_rows_unverified"] = inclusion_rows_unverified
            if inclusion_rows_unverified:
                reason = state_note if manifest is None else (
                    "the state database predates inclusion counts for some files"
                )
                warnings.warn(
                    f"default_inclusion_only=True kept {inclusion_rows_unverified} {self.feature_name} "
                    "row(s) whose include_by_default was reconstructed from the sparse-storage convention "
                    f"without count verification ({reason}). The convention holds for managed curated roots; "
                    "verify against the curation state database before relying on it.",
                    UserWarning,
                    stacklevel=2,
                )

        df_metadata = self._build_df_metadata(
            combined, stored_rows, verified_participants, set(diverged_participants),
            dense=dense, verified=manifest is not None, fingerprint_checked=installed_fingerprint is not None,
            acquisition=acquisition_by_participant,
        )
        df_columns_metadata = self._build_df_columns_metadata(combined, schema, dense=dense)
        return LoaderData(
            df=combined, df_metadata=df_metadata,
            df_columns_metadata=df_columns_metadata, load_report=report,
        )

    def profile(
        self, registration_codes: str | int | Iterable[str | int] | None = None,
    ) -> ProfileReport:
        """
        Describe what this root holds for the feature, without parsing any feature CSV.
        Figures come from the phase's state database (``curation_outputs`` for curated roots,
        ``feature_outputs`` for native roots) and one ``stat`` per file. They cover the files ``get_data``
        would read for the same participants, so ``profile().summary["rows"]`` equals the row count of an
        unfiltered ``get_data`` call. Curated profiles add curation status, default inclusion,
        acquisition-method coverage, unit-resolution coverage, and policy currency. Integrity checks compare
        each file's presence and size with its state record. Without a readable state database the profile
        falls back to the filesystem and reports bytes but not rows. Nothing is written.
        """

        root, _ = self._checked_root()
        dense = self.phase == "curated"
        requested = registration_codes is not None
        participant_ids, _ = self._resolve_participants(root, registration_codes)

        state: dict[str, dict[str, Any]] | None = None
        note: str | None = None
        installed: str | None = None
        if dense:
            if self.state_validation == "off":
                note = "state reads disabled by state_validation='off'"
            else:
                state, note = self._open_manifest(root)
                if state is not None:
                    installed = self._installed_policy_fingerprint()
        else:
            state, note = self._open_native_outputs(root)
        source = "filesystem" if state is None else ("curation_state" if dense else "native_state")
        rows_key = "curated_rows" if dense else "row_count"

        # Without a requested subset, also include every participant the state records, so a file that
        # has disappeared from disk is reported rather than silently absent from the profile.
        population = set(participant_ids)
        if state is not None and not requested:
            population |= set(state)

        records: list[dict[str, Any]] = []
        acquisition_total: Counter[str] = Counter()
        status_total: Counter[str] = Counter()
        unit_status_total: Counter[str] = Counter()
        included = excluded = unverifiable = canonical_rows = ambiguous_rows = diverged = 0
        for participant_id in sorted(population):
            path = root / participant_id / self.filename
            on_disk = path.is_file()
            entry = state.get(participant_id) if state is not None else None
            if not on_disk and entry is None:
                continue
            record: dict[str, Any] = {
                "RegistrationCode": self._to_registration_code(participant_id),
                "participant_id": participant_id,
                "on_disk": on_disk,
                "bytes_on_disk": int(path.stat().st_size) if on_disk else None,
            }
            if state is not None:
                record["has_state"] = entry is not None
                record["rows"] = int(entry[rows_key]) if entry is not None else None
                record["size_matches"] = bool(
                    entry is not None and on_disk and record["bytes_on_disk"] == int(entry["size_bytes"])
                )
            loadable = on_disk and entry is not None
            if dense and entry is not None:
                stored = int(entry["curated_rows"])
                inclusion_valid = (
                    int(entry["included_by_default_rows"]) + int(entry["excluded_by_default_rows"]) == stored
                )
                acquisition = Counter(json.loads(entry["acquisition_counts_json"] or "{}"))
                record.update({
                    "pass_rows": int(entry["pass_rows"]),
                    "review_rows": int(entry["review_rows"]),
                    "exclude_default_rows": int(entry["exclude_default_rows"]),
                    "included_by_default_rows": int(entry["included_by_default_rows"]) if inclusion_valid else None,
                    "excluded_by_default_rows": int(entry["excluded_by_default_rows"]) if inclusion_valid else None,
                    "acquisition_classified_fraction": self._classified_fraction(acquisition),
                    "canonical_value_rows": int(entry["canonical_value_rows"]),
                    "ambiguous_unit_rows": int(entry["ambiguous_unit_rows"]),
                    "policy_current": (
                        None if installed is None else str(entry["policy_fingerprint"]) == installed
                    ),
                })
                if loadable:
                    status_total.update({
                        "pass": int(entry["pass_rows"]), "review": int(entry["review_rows"]),
                        "exclude_default": int(entry["exclude_default_rows"]),
                    })
                    if inclusion_valid:
                        included += int(entry["included_by_default_rows"])
                        excluded += int(entry["excluded_by_default_rows"])
                    else:
                        unverifiable += 1
                    acquisition_total.update(acquisition)
                    unit_status_total.update(json.loads(entry["unit_status_counts_json"] or "{}"))
                    canonical_rows += int(entry["canonical_value_rows"])
                    ambiguous_rows += int(entry["ambiguous_unit_rows"])
                    if record["policy_current"] is False:
                        diverged += 1
            records.append(record)

        participants = self._profile_frame(records, dense=dense, with_state=state is not None)
        loadable_mask = participants["on_disk"].astype(bool)
        if state is not None:
            loadable_mask &= participants["has_state"].astype(bool)
        rows_total = int(participants.loc[loadable_mask, "rows"].sum()) if state is not None else None
        files_without_state = (
            int((participants["on_disk"] & ~participants["has_state"]).sum()) if state is not None else None
        )
        limit = self.default_max_rows
        get_data_rows = rows_total if state is not None and not files_without_state else None
        summary: dict[str, Any] = {
            "feature": self.feature_name,
            "phase": self.phase,
            "root": str(root),
            "source": source,
            "source_note": note if state is None else None,
            "participants": int(len(participants)),
            "files_on_disk": int(participants["on_disk"].sum()),
            "rows": rows_total,
            "bytes_on_disk": int(participants["bytes_on_disk"].fillna(0).sum()),
            "status_counts": dict(sorted(status_total.items())) if dense and state is not None else None,
            "default_inclusion": (
                {"included": included, "excluded": excluded, "unverifiable_files": unverifiable}
                if dense and state is not None else None
            ),
            "acquisition": (
                {"counts": dict(sorted(acquisition_total.items())),
                 "classified_fraction": self._classified_fraction(acquisition_total)}
                if dense and state is not None else None
            ),
            "unit_resolution": (
                {"status_counts": dict(sorted(unit_status_total.items())),
                 "canonical_value_rows": canonical_rows, "ambiguous_unit_rows": ambiguous_rows,
                 "resolved_fraction": (
                     canonical_rows / (canonical_rows + ambiguous_rows) if canonical_rows + ambiguous_rows else None
                 )}
                if dense and state is not None else None
            ),
            "policy": (
                {"installed_fingerprint": installed, "files_diverged": diverged}
                if dense and state is not None else None
            ),
            "integrity": {
                "files_missing_on_disk": (
                    int((~participants["on_disk"]).sum()) if state is not None else None
                ),
                "files_without_state": files_without_state,
                "files_size_mismatch": (
                    int((participants["on_disk"] & participants["has_state"] & ~participants["size_matches"]).sum())
                    if state is not None else None
                ),
            },
            "load": {
                "max_rows": limit,
                "default_get_data_rows": get_data_rows,
                "exceeds_max_rows": bool(limit is not None and get_data_rows is not None and get_data_rows > limit),
            },
        }
        return ProfileReport(summary=summary, participants=participants, text=render_profile_text(summary))

    @staticmethod
    def _profile_frame(records: list[dict[str, Any]], *, dense: bool, with_state: bool) -> pd.DataFrame:
        """Build the per-participant profile table with stable, nullable dtypes."""

        columns: dict[str, str] = {"participant_id": "object", "on_disk": "boolean", "bytes_on_disk": "Int64"}
        if with_state:
            columns.update({"has_state": "boolean", "rows": "Int64", "size_matches": "boolean"})
            if dense:
                columns.update({
                    "pass_rows": "Int64", "review_rows": "Int64", "exclude_default_rows": "Int64",
                    "included_by_default_rows": "Int64", "excluded_by_default_rows": "Int64",
                    "acquisition_classified_fraction": "Float64",
                    "canonical_value_rows": "Int64", "ambiguous_unit_rows": "Int64", "policy_current": "boolean",
                })
        frame = pd.DataFrame.from_records(records, columns=["RegistrationCode", *columns])
        for column, dtype in columns.items():
            frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
            frame[column] = pd.array(frame[column].tolist(), dtype=dtype) if dtype != "object" else frame[column]
        return frame.set_index("RegistrationCode").sort_index()

    def _resolve_max_rows(self, max_rows: Any) -> int | None:
        limit = self.default_max_rows if max_rows is _USE_DEFAULT_MAX_ROWS else max_rows
        if limit is None:
            return None
        if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or int(limit) < 1:
            raise DataLoaderConfigurationError(f"max_rows must be a positive integer or None; received {limit!r}")
        return int(limit)

    def _size_estimate(
        self, state: Mapping[str, Mapping[str, Any]] | None, participant_ids: Sequence[str], root: Path, *,
        dense: bool, default_inclusion_only: bool, date_filtered: bool, presence_known: bool,
    ) -> dict[str, Any] | None:
        """
        Size of a get_data request from the phase's state table. ``rows_expected`` is exact, or None when
        date bounds, a file the state does not describe, or missing inclusion counts leave it unknown.
        """

        if state is None:
            return None
        rows_key = "curated_rows" if dense else "row_count"
        exact = not date_filtered
        participants = rows_stored = rows_retained = bytes_on_disk = 0
        for participant_id in participant_ids:
            entry = state.get(participant_id)
            on_disk = presence_known or (root / participant_id / self.filename).is_file()
            if entry is None:
                if on_disk:
                    exact = False
                continue
            if not on_disk:
                continue
            stored = int(entry[rows_key])
            participants += 1
            rows_stored += stored
            bytes_on_disk += int(entry["size_bytes"])
            if dense and default_inclusion_only:
                included = int(entry["included_by_default_rows"])
                if included + int(entry["excluded_by_default_rows"]) == stored:
                    rows_retained += included
                else:
                    exact = False
                    rows_retained += stored
            else:
                rows_retained += stored
        return {
            "participants": participants, "rows_stored": rows_stored,
            "rows_expected": rows_retained if exact else None, "bytes_on_disk": bytes_on_disk,
        }

    def _size_advice(self) -> str:
        return (
            "Narrow the request with registration_codes, start_date/end_date, or default_inclusion_only; "
            f"inspect it first with {type(self).__name__}().profile(); or pass max_rows=None to load it anyway."
        )

    @staticmethod
    def _acquisition_counts(frame: pd.DataFrame) -> Counter[str]:
        """Acquisition methods of the rows in ``frame``; absent or blank values count as unclassified."""

        if "acquisition_method" not in frame.columns:
            return Counter({"unclassified": int(len(frame))})
        text = frame["acquisition_method"].astype("string").str.strip()
        known = (text.notna() & text.ne("")).fillna(False).to_numpy(dtype=bool)
        labels = text.astype(object).where(known, "unclassified")
        return Counter({str(name): int(count) for name, count in labels.value_counts().items()})

    @staticmethod
    def _classified_fraction(counts: Mapping[str, int]) -> float | None:
        total = sum(counts.values())
        if not total:
            return None
        return (total - counts.get("unclassified", 0)) / total

    def _resolve_participants(
        self, root: Path, registration_codes: str | int | Iterable[str | int] | None,
    ) -> tuple[list[str], list[str]]:
        if registration_codes is None:
            ids = sorted(
                path.name for path in root.iterdir()
                if path.is_dir() and not path.name.startswith(".") and (path / self.filename).is_file()
            )
            return ids, []

        if isinstance(registration_codes, (str, bytes)) or not isinstance(registration_codes, Iterable):
            values: Iterable[Any] = (registration_codes,)
        else:
            values = registration_codes

        normalized: list[str] = []
        for value in values:
            participant_id = self._to_participant_id(value)
            if participant_id not in normalized:
                normalized.append(participant_id)

        missing = [pid for pid in normalized if not (root / pid).is_dir()]
        return normalized, missing

    @classmethod
    def _to_registration_code(cls, participant_id: str | int) -> str:
        return f"{cls.registration_prefix}{cls._to_participant_id(participant_id)}"

    @classmethod
    def _to_participant_id(cls, registration_code: Any) -> str:
        """
        Normalize one registration code or participant folder ID to the folder ID.
        Accepts strings and integers, including numpy integers and whole-number floats, since codes often
        arrive from a DataFrame column. A missing, fractional, or non-numeric-scalar code is rejected rather
        than silently reported as a participant absent from the root.
        """

        if registration_code is None or registration_code is pd.NA or registration_code is pd.NaT:
            raise DataLoaderConfigurationError(
                "registration code is missing (None/NA); drop it, or pass registration_codes=None for everyone"
            )
        if isinstance(registration_code, (bool, np.bool_)):
            raise DataLoaderConfigurationError(f"registration code must be a string or integer; received {registration_code!r}")
        if isinstance(registration_code, (int, np.integer)):
            value = str(int(registration_code))
        elif isinstance(registration_code, (float, np.floating)):
            if math.isnan(registration_code):
                raise DataLoaderConfigurationError(
                    "registration code is missing (NaN); drop it, or pass registration_codes=None for everyone"
                )
            if not math.isfinite(registration_code) or not float(registration_code).is_integer():
                raise DataLoaderConfigurationError(
                    f"registration code {registration_code!r} is not a whole number; a missing or fractional code "
                    "usually means the column it came from held missing values"
                )
            value = str(int(registration_code))
        elif isinstance(registration_code, str):
            value = registration_code.strip()
        else:
            raise DataLoaderConfigurationError(
                f"registration code must be a string or integer; received {type(registration_code).__name__}"
            )
        if value[: len(cls.registration_prefix)].upper() == cls.registration_prefix.upper():
            value = value[len(cls.registration_prefix) :]
        if not value:
            raise DataLoaderConfigurationError("empty participant/registration code")
        return value

    @staticmethod
    def _normalize_bound(value: str | pd.Timestamp | None, label: str,) -> pd.Timestamp | None:
        if value is None:
            return None
        if pd.api.types.is_scalar(value) and pd.isna(value):
            raise DataLoaderConfigurationError(
                f"{label}={value!r} is missing; pass None for no bound"
            )
        if isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)):
            raise DataLoaderConfigurationError(
                f"{label}={value!r} is a number; pass a date, a datetime, or a string such as '2024-01-01' "
                "(pandas would read a number as nanoseconds since 1970)"
            )
        try:
            timestamp = pd.Timestamp(value)
        except Exception as exc:  # pragma: no cover - pandas exception details vary
            raise DataLoaderConfigurationError(
                f"could not parse {label}={value!r} as a timestamp"
            ) from exc
        if pd.isna(timestamp):
            raise DataLoaderConfigurationError(f"{label}={value!r} does not describe a timestamp")
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    def _read_feature_file(self, path: Path) -> pd.DataFrame:
        """
        Read one stored feature file so that every value comes back exactly as written.
        ``float_precision="round_trip"`` parses each float to the nearest double, as Python's ``float()`` does;
        pandas' default parser misses by one unit in the last place on roughly 3% of stored float values.
        Only empty cells are missing: pandas' default NA tokens would otherwise turn legitimate text such as
        the Dexcom trend arrow ``"None"`` into missing data.
        """

        try:
            return pd.read_csv(
                path, low_memory=False, keep_default_na=False, na_values=[""], float_precision="round_trip",
            )
        except Exception as exc:
            raise DataLoaderReadError(f"failed reading {path}: {exc}") from exc

    def _read_date_anchor(self, path: Path) -> pd.Series:
        try:
            anchor = pd.read_csv(path, usecols=[self.date_column])[self.date_column]
            return pd.to_datetime(anchor, utc=True, errors="raise", format="mixed")
        except Exception as exc:
            raise DataLoaderReadError(
                f"failed reading date anchor {self.date_column!r} from {path}: {exc}"
            ) from exc

    @staticmethod
    def _parse_temporal_columns(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
        frame = frame.copy()
        for column in _TEMPORAL_COLUMNS:
            if column not in frame.columns:
                continue
            try:
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise", format="mixed")
            except Exception as exc:
                raise DataLoaderReadError(
                    f"failed parsing timestamp column {column!r} in {path}: {exc}"
                ) from exc
        return frame

    @staticmethod
    def _as_boolean(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            return numeric.fillna(0).astype(int).astype(bool)
        lowered = series.astype("string").str.strip().str.lower()
        return lowered.isin({"1", "true", "t", "yes", "y"})

    def _open_manifest(self, root: Path) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
        """
        Read this feature's ``curation_outputs`` rows from the curation state database, keyed by participant.
        Returns ``(None, reason)`` when verification is disabled or, in ``auto`` mode, when the database is
        absent. The database is opened with ``immutable=1`` so the read never writes to the data tree.
        """

        if self.state_validation == "off":
            return None, "verification disabled by state_validation='off'"
        database = root / _PHASE_STATE_FILES["curated"]
        if not database.is_file():
            if self.state_validation == "required":
                raise DataLoaderStateError(
                    f"state_validation='required' but the curation state database is absent: {database}"
                )
            return None, "curation state database absent"
        active = self._active_write_file(database)
        if active is not None:
            raise DataLoaderStateError(
                f"{active} holds uncommitted writes: a curation run may be in progress or did not shut down "
                "cleanly, so the state database cannot be read consistently without writing to the data "
                "tree. Retry once curation has finished, or pass state_validation='off' to load unverified."
            )
        uri = f"{database.resolve().as_uri()}?immutable=1"
        try:
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    "SELECT * FROM curation_outputs WHERE feature = ?", (self.feature_name,)
                ).fetchall()
        except sqlite3.Error as exc:
            raise DataLoaderStateError(f"could not read curation_outputs from {database}: {exc}") from exc
        return {str(row["participant_id"]): dict(row) for row in rows}, None

    @staticmethod
    def _active_write_file(database: Path) -> Path | None:
        """Return a non-empty WAL or rollback journal beside ``database``: evidence of a writer at work."""

        for suffix in _STATE_ACTIVE_SUFFIXES:
            side = database.with_name(database.name + suffix)
            if side.is_file() and side.stat().st_size > 0:
                return side
        return None

    def _open_native_outputs(self, root: Path) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
        """
        Read this feature's ``feature_outputs`` rows from the native processing state database, keyed by
        participant. Unlike curated verification this is advisory: an absent, busy, or unreadable database
        yields ``(None, reason)`` and the caller carries on without it. Opened with ``immutable=1``.
        """

        database = root / _PHASE_STATE_FILES["native"]
        if not database.is_file():
            return None, "native state database absent"
        active = self._active_write_file(database)
        if active is not None:
            return None, f"native state database has uncommitted writes ({active.name})"
        uri = f"{database.resolve().as_uri()}?immutable=1"
        try:
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    "SELECT participant_id, row_count, size_bytes FROM feature_outputs WHERE feature = ?",
                    (self.feature_name,),
                ).fetchall()
        except sqlite3.Error as exc:
            return None, f"native state database unreadable: {exc}"
        return {str(row["participant_id"]): dict(row) for row in rows}, None

    def _installed_policy_fingerprint(self) -> str | None:
        """The installed curation policy fingerprint for this feature, or None when it has no policy."""

        from wearable_project.curation.registry import get_policy

        try:
            return str(get_policy(self.feature_name, allow_fallback=False).fingerprint())
        except KeyError:
            return None

    @classmethod
    def _densify(cls, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], bool]:
        """
        Restore the curated logical schema from sparse storage.
        Absent columns take their documented default for every row; for ``acquisition_method`` and
        ``curation_flags`` blank cells do too. ``include_by_default`` is returned as a plain boolean. Row
        count and every other column are unchanged. Returns the frame, per-column counts of reconstructed
        rows, and whether ``include_by_default`` was absent from the stored file.
        """

        filled: dict[str, int] = {}
        rows = int(len(frame))
        inclusion_reconstructed = "include_by_default" not in frame.columns
        for column, default in _DENSE_CURATED_DEFAULTS.items():
            if column not in frame.columns:
                frame[column] = default
                filled[column] = rows
            elif column in _DENSE_FILL_BLANKS:
                values = frame[column]
                blank = (
                    values.isna().to_numpy(dtype=bool)
                    | values.astype("string").str.strip().eq("").fillna(False).to_numpy(dtype=bool)
                )
                count = int(blank.sum())
                if count:
                    frame[column] = values.astype(object).where(~blank, default)
                    filled[column] = count
        frame["include_by_default"] = cls._as_boolean(frame["include_by_default"]).astype(bool)
        return frame, filled, inclusion_reconstructed

    @staticmethod
    def _reconcile(frame: pd.DataFrame, path: Path, entry: Mapping[str, Any]) -> bool:
        """
        Check a densely reconstructed curated file against its ``curation_outputs`` row and raise
        ``DataLoaderStateError`` listing every disagreement. Returns False when the row's inclusion counts
        cannot be used: a state database migrated from an engine that predates them stores 0/0 for files
        with rows, and those two checks are then skipped rather than reported as a mismatch.
        """

        problems: list[str] = []

        def compare(label: str, observed: Any, recorded: Any) -> None:
            if observed != recorded:
                problems.append(f"{label}: file {observed!r} vs state {recorded!r}")

        recorded_rows = int(entry["curated_rows"])
        # File integrity first: size is free, and the hash (about 7% of parsing time) proves the bytes are
        # exactly what curation wrote. It is only computed when sizes agree, since a size mismatch already
        # implies a hash mismatch. The count checks below are a separate guarantee: they prove this loader's
        # dense reconstruction agrees with the logical state the writer recorded.
        size = int(path.stat().st_size)
        compare("size_bytes", size, int(entry["size_bytes"]))
        if size == int(entry["size_bytes"]):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            compare("sha256", digest.hexdigest(), str(entry["sha256"]))
        compare("rows", int(len(frame)), recorded_rows)
        status = frame["curation_status"]
        compare("pass rows", int((status == "pass").sum()), int(entry["pass_rows"]))
        compare("review rows", int((status == "review").sum()), int(entry["review_rows"]))
        compare(
            "exclude_default rows",
            int((status == "exclude_default").sum()), int(entry["exclude_default_rows"]),
        )

        included_recorded = int(entry["included_by_default_rows"])
        excluded_recorded = int(entry["excluded_by_default_rows"])
        inclusion_checked = included_recorded + excluded_recorded == recorded_rows
        if inclusion_checked:
            include = frame["include_by_default"]
            compare("included_by_default rows", int(include.sum()), included_recorded)
            compare("excluded_by_default rows", int((~include).sum()), excluded_recorded)

        acquisition = Counter(str(value) for value in frame["acquisition_method"])
        compare(
            "acquisition counts",
            dict(sorted(acquisition.items())),
            dict(sorted(json.loads(entry["acquisition_counts_json"] or "{}").items())),
        )
        flags: Counter[str] = Counter()
        for value in frame["curation_flags"]:
            flags.update(part for part in str(value).split(_CURATION_FLAG_SEPARATOR) if part)
        compare(
            "flag counts",
            dict(sorted(flags.items())),
            dict(sorted(json.loads(entry["flag_counts_json"] or "{}").items())),
        )

        if problems:
            raise DataLoaderStateError(
                f"{path} disagrees with its curation_outputs row in the curation state database: "
                + "; ".join(problems)
                + ". The file may be damaged, the state database stale, or the writer version "
                "incompatible with this loader; its rows should not be used until this is resolved."
            )
        return inclusion_checked

    def _normalize_dtypes(self, frame: pd.DataFrame, schema: Any) -> pd.DataFrame:
        """Apply the dtypes declared in ``curation.schema``; values are unchanged."""

        categorical = schema.categorical_columns(self.feature_name)
        for column in frame.columns:
            values = frame[column]
            if column in categorical and (
                pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values)
            ):
                frame[column] = values.astype("category")
            elif column in schema.NULLABLE_BOOLEAN_COLUMNS:
                converted = self._as_nullable_boolean(values)
                if converted is not None:
                    frame[column] = converted
        return frame

    @staticmethod
    def _as_nullable_boolean(values: pd.Series) -> pd.Series | None:
        """
        Convert a true/false column to the pandas nullable boolean dtype, keeping missing values as NA.
        Returns None, leaving the column untouched, if any value is not a recognizable boolean.
        """

        if pd.api.types.is_bool_dtype(values):
            return values.astype("boolean")
        missing = values.isna().to_numpy(dtype=bool)
        result = np.zeros(len(values), dtype=bool)
        if (~missing).any():
            present = values.to_numpy(dtype=object)[~missing]
            mapping: dict[Any, bool] = {}
            for value in pd.unique(present):
                if isinstance(value, (bool, np.bool_)):
                    token: Any = bool(value)
                elif isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
                    token = bool(value)
                else:
                    token = _BOOLEAN_TEXT.get(str(value).strip().lower(), _UNRECOGNIZED)
                if token is _UNRECOGNIZED:
                    return None
                mapping[value] = token
            result[~missing] = [mapping[value] for value in present]
        return pd.Series(pd.arrays.BooleanArray(result, missing), index=values.index, name=values.name)

    def _build_df_metadata(
        self, frame: pd.DataFrame, stored_rows: Mapping[str, int], verified_participants: set[str],
        diverged: set[str], *, dense: bool, verified: bool, fingerprint_checked: bool,
        acquisition: Mapping[str, Counter[str]],
    ) -> pd.DataFrame:
        """One row per participant present in ``frame``, indexed by RegistrationCode."""

        if frame.empty:
            return _empty_participant_metadata()
        codes = frame.index.get_level_values("RegistrationCode")
        dates = pd.Series(frame.index.get_level_values("Date"), index=codes)
        grouped = dates.groupby(level=0, sort=True)
        metadata = pd.DataFrame(
            {"rows": grouped.size(), "first_date": grouped.min(), "last_date": grouped.max()}
        )
        metadata.index.name = "RegistrationCode"
        participant_ids = [self._to_participant_id(code) for code in metadata.index]
        metadata.insert(0, "participant_id", participant_ids)
        metadata["rows"] = metadata["rows"].astype("int64")
        metadata["stored_rows"] = pd.array(
            [stored_rows.get(pid, 0) for pid in participant_ids], dtype="int64"
        )
        if dense and verified:
            state = [pid in verified_participants for pid in participant_ids]
            policy = (
                [pid not in diverged for pid in participant_ids] if fingerprint_checked
                else [pd.NA] * len(participant_ids)
            )
        else:
            # Unverified curated loads are known not to be verified; the native phase has nothing to verify.
            state = [False if dense else pd.NA] * len(participant_ids)
            policy = [pd.NA] * len(participant_ids)
        metadata["state_verified"] = pd.array(state, dtype="boolean")
        metadata["policy_current"] = pd.array(policy, dtype="boolean")
        metadata["acquisition_classified_fraction"] = pd.array(
            [self._classified_fraction(acquisition.get(pid, Counter())) for pid in participant_ids], dtype="Float64",
        )
        return metadata

    def _build_df_columns_metadata(self, frame: pd.DataFrame, schema: Any, *, dense: bool) -> pd.DataFrame:
        """One row per column of ``frame``: role, description, dtype, non-null count, units, dense default."""

        if not len(frame.columns):
            return _empty_columns_metadata()
        spec = get_feature_spec(self.feature_name)
        unit_policy = spec.unit_policy
        records = []
        for column in frame.columns:
            role = schema.role_of(column)
            registry_unit = None
            if unit_policy is not None:
                if role is schema.ColumnRole.MEASUREMENT and column in spec.measurement_columns:
                    registry_unit = unit_policy.raw_unit
                elif column == "canonical_value":
                    registry_unit = unit_policy.canonical_unit
            records.append({
                "column": column,
                "role": role.value if role is not None else "undeclared",
                "description": (
                    schema.ROLE_DESCRIPTIONS[role] if role is not None
                    else "No declared role in curation.schema."
                ),
                "dtype": str(frame[column].dtype),
                "non_null": int(frame[column].notna().sum()),
                "registry_unit": registry_unit,
                "dense_default": _DENSE_CURATED_DEFAULTS.get(column) if dense else None,
            })
        return pd.DataFrame.from_records(records).set_index("column")

    @staticmethod
    def _merged_dtype(dtypes: Sequence[Any], *, missing: bool) -> Any:
        """
        The dtype concatenation gives a column seen with ``dtypes`` across files, where ``missing`` means some
        file lacked it and so contributes missing values. Integers are promoted to float to hold them, booleans
        become object, and datetimes and text keep their dtype. Disagreeing non-numeric dtypes become object.
        """

        if not dtypes:
            return np.dtype(object)
        numeric = [d for d in dtypes if pd.api.types.is_numeric_dtype(d) and not pd.api.types.is_bool_dtype(d)]
        if len(numeric) == len(dtypes):
            if missing or any(pd.api.types.is_float_dtype(d) for d in numeric):
                return np.dtype("float64")
            return np.result_type(*numeric)
        if len({str(d) for d in dtypes}) == 1:
            if missing and pd.api.types.is_bool_dtype(dtypes[0]):
                return np.dtype(object)
            return dtypes[0]
        return np.dtype(object)

    def _empty_frame(
        self, columns: Sequence[str] | None, *, observed_columns: Sequence[str] | None = None,
        dtypes: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Build the zero-row frame returned when nothing survives the filters.
        An explicit ``columns`` projection defines the schema exactly. Otherwise, the columns observed
        while reading are used. Each column takes the dtype it had in the files read, and the index levels
        take the dtypes a populated result would have, so an empty result concatenates with a populated one
        without turning datetimes or numbers into text. Declared categoricals still carry no categories, so
        concatenating them follows pandas' usual rule for differing categories. Only when no file was read
        at all is the schema genuinely unknown, and the frame is then returned with no columns.
        """

        if columns is not None:
            data_columns = list(dict.fromkeys(columns))
        else:
            data_columns = [
                name
                for name in dict.fromkeys(observed_columns or ())
                if name not in self._data_index_names
            ]
        known = dtypes or {}
        empty = pd.DataFrame({name: pd.Series(dtype=known.get(name, "object")) for name in data_columns})
        anchor = known.get(self.date_column)
        date_level = pd.Index([], dtype=anchor) if anchor is not None else pd.DatetimeIndex([], tz="UTC")
        # The default text dtype of the running pandas version, so the code level matches a populated result.
        code_level = pd.Index([], dtype=pd.Series(["10K_"]).dtype)
        empty.index = pd.MultiIndex.from_arrays([code_level, date_level], names=self._data_index_names)
        return empty
