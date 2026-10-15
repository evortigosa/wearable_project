"""
Wearable Data Processing and Modeling project
Version-aware Google Fitness candidate decoders used by the WP2 parser. These functions consume the immutable WP1
registry. They normalize source values into native-shaped candidates but do not reconcile, deduplicate, write,
or infer units.
"""


from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
import re
from typing import Any, Iterable, Mapping
from wearable_project.processing.google_fitness.registry import (
    ACTIVITY_MEASUREMENT_ORDER, PayloadEnvelope, STEP_COUNT_DATA_SOURCE_ID, STEP_COUNT_DATA_TYPE_NAME,
    UNREGISTERED_ACTIVITY_POLICY, get_activity_route, get_feature_spec, get_source_feature_version_spec,
)


_STEP_SUMMARY_FIELDS = frozenset({"date", "value"})
_STEP_RAW_FIELDS = frozenset(
    {
        "steps",
        "start_date",
        "end_date",
        "data_source_id",
        "data_type_name",
        "origin_data_source_id",
        "app_name",
        "app_package_name",
        "device_manufacturer",
        "device_model",
        "device_type",
        "device_uid",
    }
)


@dataclass(frozen=True, slots=True)
class _CandidateContext:
    outer_id: str
    source_feature: str
    version: str
    outer_datetime: str
    payload_index: int


@dataclass(frozen=True, slots=True)
class _CandidateOutcome:
    native_feature: str | None
    record: dict[str, Any] | None
    reason_codes: tuple[str, ...] = ()
    warning: tuple[str, str] | None = None


class _CandidateValueError(ValueError):
    pass


def _normalize_datetime(value: Any, *, field_name: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value.strip():
        raise _CandidateValueError(f"{field_name}:missing_or_nonstring_timestamp")
    text = value.strip()
    candidates = [text, text.replace("Z", "+00:00")]
    if re.search(r"[+-]\d{4}$", text):
        candidates.append(text[:-5] + text[-5:-2] + ":" + text[-2:])
    parsed: datetime | None = None
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            break
        except ValueError:
            continue
    if parsed is None:
        raise _CandidateValueError(f"{field_name}:invalid_timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), parsed


def _normalize_epoch_milliseconds(value: Any, *, field_name: str) -> tuple[str, datetime]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _CandidateValueError(f"{field_name}:missing_or_nonnumeric_timestamp")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise _CandidateValueError(f"{field_name}:nonintegral_epoch_milliseconds")
    milliseconds = int(number)
    seconds, remainder = divmod(milliseconds, 1000)
    try:
        parsed = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=remainder * 1000)
    except (OverflowError, OSError, ValueError) as exc:
        raise _CandidateValueError(f"{field_name}:invalid_epoch_milliseconds") from exc
    return parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), parsed


def _normalize_date(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _CandidateValueError(f"{field_name}:missing_or_nonstring_date")
    text = value.strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise _CandidateValueError(f"{field_name}:invalid_date") from exc


def _finite_number(value: Any, *, field_name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _CandidateValueError(f"{field_name}:missing_or_nonnumeric_value")
    if not math.isfinite(float(value)):
        raise _CandidateValueError(f"{field_name}:nonfinite_value")
    return value


def _integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _CandidateValueError(f"{field_name}:missing_or_noninteger_value")
    return value


def _required_string(value: Any, *, field_name: str, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise _CandidateValueError(f"{field_name}:missing_or_nonstring_value")
    if nonempty and not value.strip():
        raise _CandidateValueError(f"{field_name}:empty_string")
    return value


def _optional_string(item: Mapping[str, Any], field_name: str) -> str | None:
    if field_name not in item or item[field_name] is None:
        return None
    value = item[field_name]
    if not isinstance(value, str):
        raise _CandidateValueError(f"{field_name}:nonstring_value")
    return value


def _unexpected_fields(item: Mapping[str, Any], allowed: Iterable[str], *, prefix: str = "") -> tuple[str, ...]:
    extras = sorted(str(key) for key in set(item).difference(allowed))
    if not extras:
        return ()
    label = ",".join(extras)
    return (f"{prefix}unexpected_source_fields:{label}",)


def _extract_error_code(exc: _CandidateValueError) -> str:
    return str(exc)


def _unique_sorted(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _project_registered_record(feature: str, values: Mapping[str, Any]) -> dict[str, Any]:
    spec = get_feature_spec(feature)
    record: dict[str, Any] = {}
    for column in spec.columns:
        if column.name not in values:
            if not column.nullable:
                raise _CandidateValueError(f"{column.name}:missing_required_native_value")
            record[column.name] = None
        else:
            record[column.name] = values[column.name]
    extra = set(values).difference(spec.column_names)
    if extra:
        raise _CandidateValueError("native_projection:unexpected_values:" + ",".join(sorted(extra)))
    return record


def _project_unregistered_activity_record(values: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for column in UNREGISTERED_ACTIVITY_POLICY.columns:
        if column.name not in values:
            if not column.nullable:
                raise _CandidateValueError(f"{column.name}:missing_required_native_value")
            record[column.name] = None
        else:
            record[column.name] = values[column.name]
    extra = set(values).difference(UNREGISTERED_ACTIVITY_POLICY.column_names)
    if extra:
        raise _CandidateValueError("native_projection:unexpected_values:" + ",".join(sorted(extra)))
    return record


def _decode_activity_candidate(item: Any, context: _CandidateContext) -> _CandidateOutcome:
    if not isinstance(item, Mapping):
        return _CandidateOutcome(native_feature=None, record=None, reason_codes=("non_dict_list_item",),)

    reasons = list(_unexpected_fields(item, get_source_feature_version_spec(
        "Activities", context.version
    ).expected_source_fields))

    try:
        raw_label = _required_string(item.get("activity_name"), field_name="activity_name", nonempty=True)
    except _CandidateValueError as exc:
        return _CandidateOutcome(
            native_feature=None, record=None, reason_codes=_unique_sorted([*reasons, _extract_error_code(exc)]),
        )

    try:
        route = get_activity_route(raw_label)
    except KeyError:
        route = None

    try:
        start_date, start = _normalize_epoch_milliseconds(item.get("start"), field_name="start")
        end_date, end = _normalize_epoch_milliseconds(item.get("end"), field_name="end")
        if end <= start:
            reasons.append("end:not_strictly_after_start")
    except _CandidateValueError as exc:
        reasons.append(_extract_error_code(exc))
        start_date = None
        end_date = None

    measurement_values: dict[str, int | float | None] = {}
    for name in ACTIVITY_MEASUREMENT_ORDER:
        if name not in item or item[name] is None:
            measurement_values[name] = None
            continue
        try:
            measurement_values[name] = _finite_number(item[name], field_name=name)
        except _CandidateValueError as exc:
            reasons.append(_extract_error_code(exc))
            measurement_values[name] = None

    tracked = item.get("tracked")
    if not isinstance(tracked, bool):
        reasons.append("tracked:missing_or_nonboolean_value")
    try:
        device = _required_string(item.get("device"), field_name="device")
    except _CandidateValueError as exc:
        reasons.append(_extract_error_code(exc))
        device = None
    try:
        source_name = _required_string(item.get("source_name"), field_name="source_name")
    except _CandidateValueError as exc:
        reasons.append(_extract_error_code(exc))
        source_name = None
    try:
        source_id = _optional_string(item, "source_id")
    except _CandidateValueError as exc:
        reasons.append(_extract_error_code(exc))
        source_id = None

    if route is not None:
        for name in ACTIVITY_MEASUREMENT_ORDER:
            if measurement_values[name] is not None and name not in route.measurements:
                reasons.append(f"{name}:unregistered_measurement_for_activity")
        if source_id is not None and not route.include_source_id:
            reasons.append("source_id:unregistered_for_activity")
        native_feature = route.native_feature
    else:
        native_feature = UNREGISTERED_ACTIVITY_POLICY.native_feature

    if reasons:
        return _CandidateOutcome(native_feature=None, record=None, reason_codes=_unique_sorted(reasons),)

    assert start_date is not None
    assert end_date is not None
    assert isinstance(tracked, bool)
    assert device is not None
    assert source_name is not None

    values: dict[str, Any] = {
        "start_date": start_date,
        "end_date": end_date,
        "tracked": tracked,
        "device": device,
        "source_name": source_name,
        "collecting_method_version": context.version,
        "outer_id": context.outer_id,
        "payload_index": context.payload_index,
    }

    if route is None:
        values["raw_activity_name"] = raw_label
        for name in ACTIVITY_MEASUREMENT_ORDER:
            values[name] = measurement_values[name]
        values["source_id"] = source_id
        record = _project_unregistered_activity_record(values)
        return _CandidateOutcome(
            native_feature=native_feature,
            record=record,
            warning=(
                "unregistered-activity",
                f"Unregistered activity label {raw_label!r} routed to "
                f"{UNREGISTERED_ACTIVITY_POLICY.native_filename}",
            ),
        )

    if route.native_state is not None:
        values["value"] = route.native_state
    for name in route.measurements:
        values[name] = measurement_values[name]
    if route.include_source_id:
        values["source_id"] = source_id
    record = _project_registered_record(native_feature, values)
    return _CandidateOutcome(native_feature=native_feature, record=record)


def _decode_interval_total_candidate(item: Any, context: _CandidateContext) -> _CandidateOutcome:
    if not isinstance(item, Mapping):
        return _CandidateOutcome(None, None, ("non_dict_list_item",))
    contract = get_source_feature_version_spec(context.source_feature, context.version)
    reasons = list(_unexpected_fields(item, contract.expected_source_fields))
    value_field = "calorie" if context.source_feature == "Calories" else "distance"

    try:
        start_date, start = _normalize_datetime(item.get("start_date"), field_name="start_date")
        end_date, end = _normalize_datetime(item.get("end_date"), field_name="end_date")
        if end < start:
            reasons.append("end_date:before_start_date")
    except _CandidateValueError as exc:
        reasons.append(_extract_error_code(exc))
        start_date = None
        end_date = None

    try:
        value = _finite_number(item.get(value_field), field_name=value_field)
    except _CandidateValueError as exc:
        reasons.append(_extract_error_code(exc))
        value = None
    try:
        source_day_label = _required_string(item.get("day"), field_name="day")
    except _CandidateValueError as exc:
        reasons.append(_extract_error_code(exc))
        source_day_label = None

    if reasons:
        return _CandidateOutcome(None, None, _unique_sorted(reasons))
    assert start_date is not None and end_date is not None
    assert value is not None and source_day_label is not None
    record = _project_registered_record(
        context.source_feature,
        {
            "start_date": start_date,
            "end_date": end_date,
            "value": value,
            "source_day_label": source_day_label,
            "collecting_method_version": context.version,
            "outer_id": context.outer_id,
        },
    )
    return _CandidateOutcome(context.source_feature, record)


def _decode_anthropometric_candidate(item: Any, context: _CandidateContext) -> _CandidateOutcome:
    if not isinstance(item, Mapping):
        return _CandidateOutcome(None, None, ("non_dict_list_item", "missing_or_nonnumeric_value"),)
    contract = get_source_feature_version_spec(context.source_feature, context.version)
    reasons = list(_unexpected_fields(item, contract.expected_source_fields))

    try:
        start_date, start = _normalize_datetime(item.get("start_date"), field_name="start_date")
        end_date, end = _normalize_datetime(item.get("end_date"), field_name="end_date")
        if end < start:
            reasons.append("end_date:before_start_date")
    except _CandidateValueError as exc:
        reasons.append(_extract_error_code(exc))
        start_date = None
        end_date = None

    try:
        value = _finite_number(item.get("value"), field_name="value")
    except _CandidateValueError:
        reasons.append("missing_or_nonnumeric_value")
        value = None
    try:
        source_day_label = _required_string(item.get("day"), field_name="day")
    except _CandidateValueError as exc:
        reasons.append(_extract_error_code(exc))
        source_day_label = None
    try:
        added_by = _required_string(item.get("added_by"), field_name="added_by")
    except _CandidateValueError as exc:
        reasons.append(_extract_error_code(exc))
        added_by = None

    if reasons:
        return _CandidateOutcome(None, None, _unique_sorted(reasons))
    assert start_date is not None and end_date is not None
    assert value is not None and source_day_label is not None and added_by is not None
    record = _project_registered_record(
        context.source_feature,
        {
            "start_date": start_date,
            "end_date": end_date,
            "value": value,
            "source_day_label": source_day_label,
            "added_by": added_by,
            "collecting_method_version": context.version,
            "outer_id": context.outer_id,
        },
    )
    return _CandidateOutcome(context.source_feature, record)


def _decode_step_count_legacy_candidate(item: Any, context: _CandidateContext) -> _CandidateOutcome:
    reasons: list[str] = []
    try:
        outer_datetime, _ = _normalize_datetime(context.outer_datetime, field_name="outer_datetime")
    except _CandidateValueError as exc:
        reasons.append(_extract_error_code(exc))
        outer_datetime = None
    try:
        value = _integer(item, field_name="value")
    except _CandidateValueError as exc:
        reasons.append(_extract_error_code(exc))
        value = None
    if reasons:
        return _CandidateOutcome(None, None, _unique_sorted(reasons))
    assert outer_datetime is not None and value is not None
    record = _project_registered_record(
        "StepCount",
        {
            "outer_datetime": outer_datetime,
            "summary_date": None,
            "start_date": None,
            "end_date": None,
            "value": value,
            "raw_value": None,
            "aggregation_source": None,
            "origin_data_source_id": None,
            "app_name": None,
            "app_package_name": None,
            "device_manufacturer": None,
            "device_model": None,
            "device_type": None,
            "device_uid": None,
            "collecting_method_version": context.version,
            "outer_id": context.outer_id,
            "payload_index": 0,
        },
    )
    return _CandidateOutcome("StepCount", record)


def _decode_step_count_alternative_candidate(item: Any, context: _CandidateContext) -> _CandidateOutcome:
    if not isinstance(item, Mapping):
        return _CandidateOutcome(None, None, ("non_dict_list_item",))
    contract = get_source_feature_version_spec("StepCount", context.version)
    reasons = list(_unexpected_fields(item, contract.expected_source_fields))

    try:
        outer_datetime, _ = _normalize_datetime(context.outer_datetime, field_name="outer_datetime")
    except _CandidateValueError as exc:
        reasons.append(_extract_error_code(exc))
        outer_datetime = None

    try:
        aggregation_source = _required_string(item.get("source"), field_name="source", nonempty=True)
    except _CandidateValueError as exc:
        reasons.append(_extract_error_code(exc))
        aggregation_source = None

    summaries = item.get("steps")
    raw_steps = item.get("raw_steps")
    if not isinstance(summaries, list):
        reasons.append("steps:missing_or_nonlist")
        summaries = []
    if not isinstance(raw_steps, list):
        reasons.append("raw_steps:missing_or_nonlist")
        raw_steps = []
    if len(summaries) != 1:
        reasons.append(f"steps:expected_one_item_found_{len(summaries)}")
    if len(raw_steps) != 1:
        reasons.append(f"raw_steps:expected_one_item_found_{len(raw_steps)}")

    summary = summaries[0] if len(summaries) == 1 else None
    raw = raw_steps[0] if len(raw_steps) == 1 else None
    if summary is not None and not isinstance(summary, Mapping):
        reasons.append("steps:non_dict_item")
        summary = None
    if raw is not None and not isinstance(raw, Mapping):
        reasons.append("raw_steps:non_dict_item")
        raw = None

    summary_date: str | None = None
    value: int | None = None
    if isinstance(summary, Mapping):
        reasons.extend(_unexpected_fields(summary, _STEP_SUMMARY_FIELDS, prefix="steps."))
        try:
            summary_date = _normalize_date(summary.get("date"), field_name="steps.date")
        except _CandidateValueError as exc:
            reasons.append(_extract_error_code(exc))
        try:
            value = _integer(summary.get("value"), field_name="steps.value")
        except _CandidateValueError as exc:
            reasons.append(_extract_error_code(exc))

    start_date: str | None = None
    end_date: str | None = None
    raw_value: int | None = None
    origin_data_source_id: str | None = None
    app_name: str | None = None
    app_package_name: str | None = None
    device_manufacturer: str | None = None
    device_model: str | None = None
    device_type: str | None = None
    device_uid: str | None = None
    if isinstance(raw, Mapping):
        reasons.extend(_unexpected_fields(raw, _STEP_RAW_FIELDS, prefix="raw_steps."))
        try:
            start_date, start = _normalize_epoch_milliseconds(raw.get("start_date"), field_name="raw_steps.start_date")
            end_date, end = _normalize_epoch_milliseconds(raw.get("end_date"), field_name="raw_steps.end_date")
            if end < start:
                reasons.append("raw_steps.end_date:before_start_date")
        except _CandidateValueError as exc:
            reasons.append(_extract_error_code(exc))
        try:
            raw_value = _integer(raw.get("steps"), field_name="raw_steps.steps")
        except _CandidateValueError as exc:
            reasons.append(_extract_error_code(exc))
        if raw.get("data_type_name") != STEP_COUNT_DATA_TYPE_NAME:
            reasons.append("raw_steps.data_type_name:registry_constant_mismatch")
        if raw.get("data_source_id") != STEP_COUNT_DATA_SOURCE_ID:
            reasons.append("raw_steps.data_source_id:registry_constant_mismatch")
        for field_name in (
            "origin_data_source_id",
            "app_name",
            "app_package_name",
            "device_manufacturer",
            "device_model",
            "device_type",
            "device_uid",
        ):
            try:
                value_optional = _optional_string(raw, field_name)
            except _CandidateValueError as exc:
                reasons.append(_extract_error_code(exc))
                value_optional = None
            if field_name == "origin_data_source_id":
                origin_data_source_id = value_optional
            elif field_name == "app_name":
                app_name = value_optional
            elif field_name == "app_package_name":
                app_package_name = value_optional
            elif field_name == "device_manufacturer":
                device_manufacturer = value_optional
            elif field_name == "device_model":
                device_model = value_optional
            elif field_name == "device_type":
                device_type = value_optional
            elif field_name == "device_uid":
                device_uid = value_optional

    if reasons:
        return _CandidateOutcome(None, None, _unique_sorted(reasons))

    assert outer_datetime is not None
    assert aggregation_source is not None
    assert summary_date is not None
    assert start_date is not None and end_date is not None
    assert value is not None and raw_value is not None
    record = _project_registered_record(
        "StepCount",
        {
            "outer_datetime": outer_datetime,
            "summary_date": summary_date,
            "start_date": start_date,
            "end_date": end_date,
            "value": value,
            "raw_value": raw_value,
            "aggregation_source": aggregation_source,
            "origin_data_source_id": origin_data_source_id,
            "app_name": app_name,
            "app_package_name": app_package_name,
            "device_manufacturer": device_manufacturer,
            "device_model": device_model,
            "device_type": device_type,
            "device_uid": device_uid,
            "collecting_method_version": context.version,
            "outer_id": context.outer_id,
            "payload_index": context.payload_index,
        },
    )
    return _CandidateOutcome("StepCount", record)


def _candidate_items_for_contract(payload: Any, feature: str, version: str) -> tuple[list[Any], tuple[str, ...]]:
    """
    Project a decoded payload into source items without losing countable evidence. Serialization failures are
    handled before this function. Envelope or cardinality drift is returned as candidate-level reasons so every
    visible item is still counted exactly rather than disappearing into a row-level decode failure.
    """

    contract = get_source_feature_version_spec(feature, version)
    reasons: list[str] = []
    if contract.envelope is PayloadEnvelope.LIST:
        if isinstance(payload, list):
            items = list(payload)
        else:
            items = [payload]
            reasons.append(f"payload_envelope:expected_list_found_{type(payload).__name__}")
    elif contract.envelope is PayloadEnvelope.DICT:
        if isinstance(payload, dict):
            items = [payload]
        else:
            items = [payload]
            reasons.append(f"payload_envelope:expected_dict_found_{type(payload).__name__}")
    elif contract.envelope is PayloadEnvelope.SCALAR_INTEGER:
        items = [payload]
        if isinstance(payload, bool) or not isinstance(payload, int):
            reasons.append(f"payload_envelope:expected_integer_found_{type(payload).__name__}")
    else:  # pragma: no cover - registry enum exhaustiveness
        raise AssertionError(f"Unsupported payload envelope: {contract.envelope}")

    if len(items) < contract.minimum_items:
        reasons.append(f"payload_cardinality:below_minimum_{contract.minimum_items}_found_{len(items)}")
    if contract.maximum_items is not None and len(items) > contract.maximum_items:
        reasons.append(f"payload_cardinality:above_maximum_{contract.maximum_items}_found_{len(items)}")
    return items, _unique_sorted(reasons)


def _decode_candidate(item: Any, context: _CandidateContext) -> _CandidateOutcome:
    contract = get_source_feature_version_spec(context.source_feature, context.version)
    if contract.decoder.value == "activity_items":
        return _decode_activity_candidate(item, context)
    if contract.decoder.value == "interval_total":
        return _decode_interval_total_candidate(item, context)
    if contract.decoder.value == "anthropometric":
        return _decode_anthropometric_candidate(item, context)
    if contract.decoder.value == "step_count_legacy":
        return _decode_step_count_legacy_candidate(item, context)
    if contract.decoder.value == "step_count_alternatives":
        return _decode_step_count_alternative_candidate(item, context)
    raise AssertionError(f"Unhandled decoder kind: {contract.decoder}")
