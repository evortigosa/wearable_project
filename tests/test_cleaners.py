"""
Wearable Data Processing and Modeling project
"""


from wearable_project.processing.cleaners import clean_feature


def base(**updates):
    row = {
        "participant_id": "P1", "feature": "StepCount", "data_source": "AppleHealthkit",
        "collecting_method_version": "2.0", "record_id": None, "source_id": None,
        "source_name": None, "metadata": None, "metadata_raw": None,
        "start_date_raw": "2024-01-02T02:00:00+0200", "end_date_raw": "2024-01-02T03:00:00+0200",
        "datetime_raw": "2024-01-01 00:00:00", "created_at_raw": "2024-02-01T00:00:00Z",
        "updated_at_raw": "2024-02-01T00:00:00Z", "source_file": "2024-1.csv",
        "source_month": "2024-01", "source_file_sha256": "x", "outer_id": "o",
        "outer_row_number": 2, "payload_index": 0, "quality_flags_raw": [],
    }
    row.update(updates)
    return row


def test_boundary_revision_uses_local_date_copy():
    rows = [base(value=10), base(value=100, datetime_raw="2024-01-02 00:00:00", outer_id="new", payload_index=1)]
    result = clean_feature("StepCount", rows)
    assert len(result.dataframe) == 1
    row = result.dataframe.iloc[0]
    assert row["value"] == 100
    assert "boundary_revision_resolved" in row["quality_flags"]


def test_sleep_states_are_not_mode_reduced():
    rows = [
        base(feature="Sleep", record_id="A", value="DEEP"),
        base(feature="Sleep", record_id="B", value="REM", payload_index=1),
    ]
    frame = clean_feature("Sleep", rows).dataframe
    assert set(frame["value"]) == {"DEEP", "REM"}


def test_waist_duplicates_and_canonical_unit():
    common = dict(
        feature="WaistCircumference", value=38.976377952755904, source_id="com.apple.Health",
        start_date_raw="2019-10-17T20:47:00+0300", end_date_raw="2019-10-17T20:47:00+0300",
        datetime_raw="2019-10-17 00:00:00",
    )
    rows = [
        base(record_id="A", **common), base(record_id="B", payload_index=1, **common),
        base(feature="WaistCircumference", record_id="C", value=38.976377952755904,
             source_id="com.apple.Health", start_date_raw="2019-10-17T20:46:00+0300",
             end_date_raw="2019-10-17T20:46:00+0300", datetime_raw="2019-10-17 00:00:00", payload_index=2),
    ]
    result = clean_feature("WaistCircumference", rows)
    assert len(result.dataframe) == 2
    assert result.exact_duplicates_removed == 1
    row = result.dataframe[result.dataframe["start_date"].str.contains("17:47")].iloc[0]
    assert abs(row["canonical_value"] - 0.99) < 1e-12
    assert row["occurrence_count"] == 2


def test_blood_pressure_pairs_remain_measured_pairs():
    rows = [
        base(feature="BloodPressure", record_id="A", blood_pressure_systolic_value=139,
             blood_pressure_diastolic_value=79, start_date_raw="2024-01-02T10:00:00+0200",
             end_date_raw="2024-01-02T10:00:00+0200"),
        base(feature="BloodPressure", record_id="B", blood_pressure_systolic_value=150,
             blood_pressure_diastolic_value=75, start_date_raw="2024-01-02T10:01:00+0200",
             end_date_raw="2024-01-02T10:01:00+0200", payload_index=1),
    ]
    frame = clean_feature("BloodPressure", rows).dataframe
    assert set(zip(frame["blood_pressure_systolic_value"], frame["blood_pressure_diastolic_value"])) == {(139, 79), (150, 75)}


def test_activity_summary_is_not_averaged():
    shared = dict(
        feature="ActivitySummary", start_date_raw=None, end_date_raw=None,
        datetime_raw="2024-01-01 00:00:00", apple_stand_hours=1, apple_exercise_time=0,
        apple_stand_hours_goal=12, apple_exercise_time_goal=30, active_energy_burned_goal=500,
    )
    frame = clean_feature("ActivitySummary", [
        base(active_energy_burned=2, payload_index=0, **shared),
        base(active_energy_burned=8, payload_index=1, **shared),
    ]).dataframe
    assert list(frame["active_energy_burned"]) == [2.0, 8.0]


def test_iana_local_time_is_preserved_when_metadata_zone_exists():
    row = base(
        feature="BloodGlucose", record_id="G", value=8.0,
        start_date_raw="2024-01-01T12:00:00+0200",
        end_date_raw="2024-01-01T12:00:00+0200",
        time_zone="Asia/Jerusalem",
    )
    frame = clean_feature("BloodGlucose", [row]).dataframe
    assert frame.iloc[0]["start_date"] == "2024-01-01T10:00:00Z"
    assert frame.iloc[0]["utc_offset_minutes"] == 120
    assert frame.iloc[0]["time_zone"] == "Asia/Jerusalem"
    assert "start_date_iana" not in frame.columns


def test_compact_output_omits_folder_and_filename_identity_columns():
    frame = clean_feature("StepCount", [base(value=100)]).dataframe
    assert list(frame.columns[:8]) == [
        "start_date", "end_date", "value", "datetime", "created_at",
        "updated_at", "data_source", "collecting_method_version",
    ]
    for redundant in (
        "event_id", "participant_id", "feature", "start_date_local",
        "end_date_local", "start_date_raw", "end_date_raw", "datetime_raw",
        "duration_seconds", "source_file", "source_month", "outer_id",
        "metadata_raw", "metadata_present",
    ):
        assert redundant not in frame.columns


def test_compact_existing_row_preserves_boundary_revision_logic():
    first = clean_feature("StepCount", [base(value=10)]).dataframe.iloc[0].to_dict()
    first.update({"participant_id": "P1", "feature": "StepCount", "_from_existing_output": True})
    recovered = base(
        value=100,
        datetime_raw="2024-01-02 00:00:00",
        outer_id="new",
        payload_index=1,
    )
    result = clean_feature("StepCount", [first, recovered])
    assert len(result.dataframe) == 1
    assert result.dataframe.iloc[0]["value"] == 100
    assert "boundary_revision_resolved" in result.dataframe.iloc[0]["quality_flags"]


def test_metadata_output_contains_only_unflattened_residual_keys():
    row = base(
        feature="HeartRate",
        record_id="H",
        value=72,
        metadata='{"h_k_metadata_key_heart_rate_motion_context":0,"custom_key":"kept"}',
        heart_rate_motion_context=0,
    )
    frame = clean_feature("HeartRate", [row]).dataframe
    assert frame.iloc[0]["heart_rate_motion_context"] == 0
    assert frame.iloc[0]["metadata"] == '{"custom_key":"kept"}'
    assert "metadata_raw" not in frame.columns
