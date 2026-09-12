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
