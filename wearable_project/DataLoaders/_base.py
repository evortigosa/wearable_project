"""
Wearable Data Processing and Modeling project
Shared implementation for HPP-style Apple wearable feature loaders. Native processing and curated roots keep
the same participant / feature directory layout, so a single loader implementation can serve both phases
without altering the stored files.
"""


from __future__ import annotations
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal
import warnings
import pandas as pd
from wearable_project.exceptions import DataLoaderConfigurationError, DataLoaderPathError, DataLoaderReadError


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


@dataclass(slots=True)
class LoaderData:
    """
    Container returned by :meth:`AppleHealthFeatureLoader.get_data`. HPP loaders expose their principal
    table through ``.df``. ``metadata`` is intentionally additive: existing code can use only ``.df``
    while callers that need provenance can inspect which root/files/participants were read.
    """

    df: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


class AppleHealthFeatureLoader:
    """
    Base class for one Apple HealthKit feature. Subclasses only declare ``feature_name`` and, when necessary,
    a different date anchor. Stored data are never modified by a loader.
    """

    feature_name: ClassVar[str]
    date_column: ClassVar[str] = "start_date"
    date_semantics: ClassVar[str] = "event_start"
    registration_prefix: ClassVar[str] = "10K_"

    def __init__(
        self, phase: ProcessingPhase | str = "curated", *,
        native_root: str | Path = DEFAULT_NATIVE_ROOT,
        curated_root: str | Path = DEFAULT_CURATED_ROOT,
        root: str | Path | None = None,
    ) -> None:
        normalized_phase = str(phase).strip().lower()
        if normalized_phase not in {"native", "curated"}:
            raise DataLoaderConfigurationError(
                "phase must be either 'native' or 'curated'; "
                f"received {phase!r}"
            )
        self.phase: ProcessingPhase = normalized_phase  # type: ignore[assignment]
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
        projection: str = "default",
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
            Curated-only opt-in filter. When true, retain rows whose ``include_by_default`` evaluates to true.
            The default is false so ``get_data()`` never silently hides review/excluded rows.
            Rows in participant files that carry no ``include_by_default`` column are kept, because the
            curated contract records that column only where an inclusion decision was made. Those rows
            are counted in ``LoaderData.metadata`` and raise a ``UserWarning`` so the partial filter is
            visible rather than silent.
        sort_index:
            Sort by RegistrationCode and Date before returning.
        """

        root = self.data_root
        if not root.is_dir():
            raise DataLoaderPathError(f"{self.phase} wearable root does not exist or is not a directory: {root}")

        # ``root`` overrides the phase default, so the declared phase is an assertion about the tree
        # rather than a description of it. Check the assertion against the tree's own state database
        # before reading, so a mismatched smoke-test root fails loudly instead of producing a frame
        # whose reported phase is wrong.
        detected_phase = self.detect_phase(root)
        if detected_phase is not None and detected_phase != self.phase:
            raise DataLoaderConfigurationError(
                f"phase={self.phase!r} was requested but {root} contains "
                f"{detected_phase} data (found {_PHASE_STATE_FILES[detected_phase]!r}); "
                f"pass phase={detected_phase!r} or point root at a {self.phase} tree"
            )

        if default_inclusion_only and self.phase != "curated":
            raise DataLoaderConfigurationError(
                "default_inclusion_only is only defined for phase='curated'"
            )

        schema = _projection_schema()
        if not schema.is_known_projection(projection):
            raise DataLoaderConfigurationError(
                f"unknown projection {projection!r}; choose one of "
                + ", ".join(repr(name) for name in schema.available_projections())
            )
        # An explicit column list is an exact request and takes precedence over any named set.
        projecting = columns is None and projection != "full"

        participant_ids, missing_requested = self._resolve_participants(root, registration_codes)

        frames: list[pd.DataFrame] = []
        files_read = 0
        loaded_participants: list[str] = []
        requested_columns = list(dict.fromkeys(columns)) if columns is not None else None
        requested_columns_seen: set[str] = set()
        # Column names as they appear on disk, in order of first appearance, so a call that selects no
        # rows can still return the schema it would have returned had rows survived.
        observed_columns: dict[str, None] = {}
        # The same, restricted to what the projection admits; this is the schema an empty projected
        # result must carry so it concatenates with a populated one.
        projected_columns: dict[str, None] = {}
        inclusion_rows_dropped = 0
        inclusion_rows_unflagged = 0
        participants_without_inclusion_flag: list[str] = []

        lower = self._normalize_bound(start_date, "start_date")
        upper = self._normalize_bound(end_date, "end_date")
        if lower is not None and upper is not None and lower > upper:
            raise DataLoaderConfigurationError(
                f"start_date {lower.isoformat()} is after end_date {upper.isoformat()}"
            )

        for participant_id in participant_ids:
            path = root / participant_id / self.filename
            if not path.is_file():
                continue
            frame = self._read_feature_file(path)
            files_read += 1

            if self.date_column not in frame.columns:
                raise DataLoaderReadError(
                    f"{path} does not contain required date anchor column "
                    f"{self.date_column!r} for {self.feature_name}"
                )

            # Whether a column exists is a property of the file, not of the rows that survive the
            # filters below. Record it here so an empty date window cannot be mistaken for an absent
            # column, and so the schema is known even when nothing survives.
            observed_columns.update(dict.fromkeys(frame.columns))
            if projecting:
                projected_columns.update(
                    dict.fromkeys(schema.columns_for_projection(projection, frame.columns))
                )
            if requested_columns is not None:
                requested_columns_seen.update(
                    name for name in requested_columns if name in frame.columns
                )

            frame = self._parse_temporal_columns(frame, path)
            dates = frame[self.date_column]
            if lower is not None:
                frame = frame.loc[dates >= lower].copy()
                dates = frame[self.date_column]
            if upper is not None:
                frame = frame.loc[dates <= upper].copy()

            if default_inclusion_only:
                if "include_by_default" in frame.columns:
                    include = self._as_boolean(frame["include_by_default"])
                    inclusion_rows_dropped += int((~include).sum())
                    frame = frame.loc[include].copy()
                else:
                    # The curated contract records include_by_default only where a decision was made,
                    # so absence means "no decision" and the rows are kept. That is correct but easy to
                    # mistake for a filter that ran, hence the accounting and the warning below.
                    inclusion_rows_unflagged += int(len(frame))
                    participants_without_inclusion_flag.append(participant_id)

            if frame.empty:
                continue

            # Keep the index anchor independently of the returned column set. This means ``columns=[...]`` does
            # not unexpectedly force the source date column back into the analytical columns.
            date_values = frame[self.date_column].copy()

            if requested_columns is not None:
                present = [name for name in requested_columns if name in frame.columns]
                frame = frame.loc[:, present].copy()
                # Curated output is intentionally sparse: a derived column may be absent from a participant-feature
                # file when it was never needed there. Cohort loading therefore fills such columns with NA rather
                # than treating the file as malformed.
                for name in requested_columns:
                    if name not in frame.columns:
                        frame[name] = pd.NA
                frame = frame.loc[:, requested_columns]
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
            if sort_index:
                combined = combined.sort_index()
        else:
            combined = self._empty_frame(
                requested_columns, observed_columns=list(projected_columns if projecting else observed_columns),
            )

        metadata = {
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
            "columns_withheld": (
                sorted(set(observed_columns) - set(projected_columns)) if projecting else []
            ),
        }
        if default_inclusion_only:
            metadata["default_inclusion_rows_dropped"] = inclusion_rows_dropped
            metadata["default_inclusion_rows_unflagged"] = inclusion_rows_unflagged
            metadata["participants_without_inclusion_flag"] = sorted(
                set(participants_without_inclusion_flag)
            )
            if inclusion_rows_unflagged:
                warnings.warn(
                    f"default_inclusion_only=True kept {inclusion_rows_unflagged} "
                    f"{self.feature_name} row(s) that carry no include_by_default column, across "
                    f"{len(set(participants_without_inclusion_flag))} participant(s); the curated "
                    "contract records that column only where an inclusion decision was made, so these "
                    "rows were not filtered. See LoaderData.metadata for the per-participant breakdown.",
                    UserWarning,
                    stacklevel=2,
                )
        return LoaderData(df=combined, metadata=metadata)

    def _resolve_participants(
        self, root: Path, registration_codes: str | int | Iterable[str | int] | None,
    ) -> tuple[list[str], list[str]]:
        if registration_codes is None:
            ids = sorted(
                path.name for path in root.iterdir() if path.is_dir() and (path / self.filename).is_file()
            )
            return ids, []

        if isinstance(registration_codes, (str, int)):
            values: Iterable[str | int] = (registration_codes,)
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
        value = str(participant_id).strip()
        if value.startswith(cls.registration_prefix):
            return value
        return f"{cls.registration_prefix}{value}"

    @classmethod
    def _to_participant_id(cls, registration_code: str | int) -> str:
        value = str(registration_code).strip()
        if value.startswith(cls.registration_prefix):
            value = value[len(cls.registration_prefix) :]
        if not value:
            raise DataLoaderConfigurationError("empty participant/registration code")
        return value

    @staticmethod
    def _normalize_bound(value: str | pd.Timestamp | None, label: str,) -> pd.Timestamp | None:
        if value is None:
            return None
        try:
            timestamp = pd.Timestamp(value)
        except Exception as exc:  # pragma: no cover - pandas exception details vary
            raise DataLoaderConfigurationError(
                f"could not parse {label}={value!r} as a timestamp"
            ) from exc
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    def _read_feature_file(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, low_memory=False)
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

    def _empty_frame(
        self, columns: Sequence[str] | None, *, observed_columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Build the zero-row frame returned when nothing survives the filters.
        An explicit ``columns`` projection defines the schema exactly. Otherwise, the columns observed
        while reading are used, so an empty result concatenates cleanly with non-empty results from the
        same call. Only when no file was read at all is the schema genuinely unknown, and the frame is
        then returned with no columns.
        """

        if columns is not None:
            data_columns = list(dict.fromkeys(columns))
        else:
            data_columns = [
                name
                for name in dict.fromkeys(observed_columns or ())
                if name not in self._data_index_names
            ]
        empty = pd.DataFrame(columns=data_columns)
        empty.index = pd.MultiIndex.from_arrays(
            [pd.Index([], dtype="object"), pd.DatetimeIndex([], tz="UTC")], names=self._data_index_names,
        )
        return empty
