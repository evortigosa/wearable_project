"""
Wearable Data Processing and Modeling project
"""


from wearable_project.processing.parser import decode_payload, normalize_metadata
from wearable_project.processing.parser import SourceFile, build_event_record
from pathlib import Path

def test_decode_repeated_layers():
    raw = '"[{\'value\': 3, \'start_date\': \'2024-01-01T00:00:00+0000\'}]"'
    assert decode_payload(raw)[0]["value"] == 3


def test_decode_outer_quotes_with_inner_unescaped_double_quote():
    raw = '"[{\'value\': 1, \'metadata\': {\'food\': "תמר מג\'הול"}}]"'
    assert decode_payload(raw)[0]["metadata"]["food"] == "תמר מג'הול"


def test_metadata_spacing_bug_is_normalized():
    metadata, flags = normalize_metadata({
        "trend _rate": -0.4,
        "modified _date": "x",
        "withings _user _identifier": "y",
    })
    assert metadata == {"trend_rate": -0.4, "modified_date": "x", "withings_user_identifier": "y"}
    assert flags == []


def test_metadata_absent_and_empty_are_distinct():
    outer = {
        "id": "outer", "participant_id": "P1", "data_source": "AppleHealthkit",
        "collecting_method_version": "2.0", "datetime": "2024-01-01 00:00:00",
        "created_at": "2024-02-01T00:00:00Z", "updated_at": "2024-02-01T00:00:00Z",
    }
    source = SourceFile(Path("2024-1.csv"), "2024-01", "abc", 1)
    absent = build_event_record({"value": 1}, outer, "P1", "X", source, 2, 0)
    empty = build_event_record({"value": 1, "metadata": {}}, outer, "P1", "X", source, 3, 0)
    assert absent["metadata_present"] is False and absent["metadata"] is None
    assert empty["metadata_present"] is True and empty["metadata"] == "{}"


def test_decode_outer_json_string_preserves_escaped_newline_for_python_literal():
    inner = (
        "[{'id': 'D043B1D5-C4A5-458C-B5C0-4DEF37881E5D', "
        "'value': 145.5050930420192, "
        "'metadata': {'comments': 'משקל - ללא כותרת\\n"
        "(עודכן משינוי המשקל בדף הפרופיל)'}, "
        "'start_date': '2020-06-28T20:43:18.215+0300', "
        "'end_date': '2020-06-28T20:43:18.215+0300'}]"
    )
    raw = f'"{inner}"'
    decoded = decode_payload(raw)
    assert decoded[0]["value"] == 145.5050930420192
    assert decoded[0]["metadata"]["comments"] == ("משקל - ללא כותרת\n(עודכן משינוי המשקל בדף הפרופיל)")


def test_decode_json_escaped_structured_string_still_uses_layered_fallback():
    raw = '"[{\\"value\\": 3, \\"start_date\\": \\"2024-01-01T00:00:00+0000\\"}]"'
    assert decode_payload(raw)[0]["value"] == 3


def test_parse_month_file_handles_quoted_python_payload_with_escaped_newline(tmp_path):
    import csv

    from wearable_project.processing.parser import SourceFile, parse_month_file, sha256_file

    path = tmp_path / "2020-6.csv"
    inner = (
        "[{'id': 'R1', 'value': 145.5050930420192, "
        "'end_date': '2020-06-28T20:43:18.215+0300', "
        "'metadata': {'comments': 'משקל - ללא כותרת\\n"
        "(עודכן משינוי המשקל בדף הפרופיל)', 'obj_i_d1': 0}, "
        "'source_id': 'com.VisionVision.calories', "
        "'start_date': '2020-06-28T20:43:18.215+0300', "
        "'source_name': 'קלוריות'}]"
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "id", "participant_id", "data_source", "name", "datetime", "data",
            "collecting_method_version", "created_at", "updated_at",
        ])
        writer.writeheader()
        writer.writerow({
            "id": "outer-1",
            "participant_id": "P1",
            "data_source": "AppleHealthkit",
            "name": "Weight",
            "datetime": "2020-06-28 00:00:00",
            "data": f'"{inner}"',
            "collecting_method_version": "2.0",
            "created_at": "2020-07-01T00:00:00Z",
            "updated_at": "2020-07-01T00:00:00Z",
        })

    source = SourceFile(path, "2020-06", sha256_file(path), path.stat().st_size)
    result = parse_month_file(source, "P1")

    assert result.payload_decode_failures == 0
    assert result.payloads_decoded == 1
    assert result.payload_items_seen == 1
    record = result.records_by_feature["Weight"][0]
    assert record["value"] == 145.5050930420192
    assert "עודכן משינוי המשקל" in record["metadata"]
