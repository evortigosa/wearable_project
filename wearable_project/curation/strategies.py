"""
Wearable Data Processing and Modeling project
Named strategy catalog referenced by curation feature policies. The alpha release declares strategy
identifiers and validates references. The later curation engine will implement these strategies without
changing policy names, which keeps policy fingerprints and audit trails stable.
"""


from __future__ import annotations


CONVERSION_STRATEGIES: dict[str, str] = {
    "not_applicable": "No numerical conversion applies.",
    "resolved_mass_to_kilograms": "Convert the resolved kg/lb mass unit to kilograms.",
    "resolved_length_to_metres": "Convert the resolved m/cm/in length unit to metres.",
    "resolved_temperature_to_celsius": "Convert the resolved Celsius/Fahrenheit unit to Celsius.",
    "resolved_glucose_to_mmol_l": "Convert the resolved mg/dL or mmol/L glucose unit to mmol/L.",
    "resolved_energy_to_kcal": "Convert the resolved kcal or kJ energy unit to kilocalories.",
    "resolved_distance_to_metres": "Convert the resolved m/km/mi distance unit to metres.",
    "identity": "Retain the numeric value in the same resolved unit.",
    "none": "No canonical conversion is defined.",
    "fraction_to_percent": "Multiply a HealthKit 0.0-1.0 percent quantity by 100.",
    "seconds_to_milliseconds": "Multiply seconds by 1000 to represent milliseconds.",
    "pounds_to_kilograms": "Multiply pounds by 0.45359237.",
    "ounces_to_kilograms": "Multiply ounces by 0.028349523125.",
    "kilograms_identity": "Retain kilograms.",
    "grams_to_kilograms": "Divide grams by 1000.",
    "inches_to_metres": "Multiply inches by 0.0254.",
    "centimetres_to_metres": "Divide centimetres by 100.",
    "metres_identity": "Retain metres.",
    "miles_to_metres": "Multiply statute miles by 1609.344.",
    "kilometres_to_metres": "Multiply kilometres by 1000.",
    "fahrenheit_to_celsius": "Convert degrees Fahrenheit to Celsius.",
    "celsius_identity": "Retain degrees Celsius.",
    "mg_dl_to_mmol_l_glucose": "Divide mg/dL glucose by 18.0182.",
    "mmol_l_identity": "Retain mmol/L.",
    "kj_to_kcal": "Divide kilojoules by 4.184.",
    "kcal_identity": "Retain kilocalories.",
}


UNIT_RESOLUTION_STRATEGIES: dict[str, str] = {
    "not_applicable": "The feature has no numerical unit requiring resolution.",
    "fixed_source_convention": "Use the reviewed exporter/source convention.",
    "healthkit_percent_fraction": "Interpret HealthKit percent as a 0.0-1.0 fraction.",
    "participant_source_epoch_mass": "Infer mass unit within participant/source epochs.",
    "participant_source_epoch_length": "Infer length unit within participant/source epochs.",
    "participant_source_epoch_temperature": "Infer Celsius/Fahrenheit within source epochs.",
    "participant_source_epoch_glucose": "Infer mg/dL versus mmol/L within source epochs.",
    "participant_source_epoch_energy": "Infer kcal versus kJ within source epochs.",
    "participant_source_epoch_distance": "Infer metres/kilometres/miles within source epochs.",
    "cohort_exporter_celsius": "Use full-cohort evidence that the exporter emits body temperature in degrees Celsius.",
    "cohort_exporter_metres": "Use full-cohort evidence that the exporter emits the distance feature in metres.",
    "cohort_exporter_seconds": "Use full-cohort evidence that the exporter emits the temporal quantity in seconds.",
    "cohort_exporter_inches": "Use full-cohort evidence that the exporter emits the anthropometric length in inches.",
    "cohort_hrv_seconds": "Use cohort evidence that exported HRV is represented in seconds.",
    "source_specific_peak_flow": "Infer peak-flow volume/time unit from source and value regime.",
    "waveform_unit_unresolved": "Retain ECG waveform amplitude in raw exported units.",
}


ACQUISITION_STRATEGIES: dict[str, str] = {
    "preserve_source_context": "Preserve source/device context without assigning a stronger method.",
    "device_or_application_estimate": "Classify automated aggregate/summary values as estimates.",
    "heart_rate_source_and_motion_context": "Use source and motion-context metadata.",
    "user_entered_or_imported": "Use user-entered metadata and source identity.",
    "cgm_source_status_trend": "Preserve CGM source, device, status, trend and time-zone context.",
    "vo2max_test_type": "Use VO2Max test-type metadata and source identity.",
    "bac_source_regime": "Distinguish calculator estimates from manual/Health imports.",
    "activity_summary_system_object": "Classify ActivitySummary as a HealthKit daily summary object.",
    "ecg_source_device_algorithm": "Use ECG source, device, algorithm and classification metadata.",
    "sleep_source_state": "Use source/device and sleep-state context.",
    "nutrition_source_and_user_entry": "Use food source, UUID and user-entry context.",
    "mindful_session_source": "Preserve source and native session interval.",
}


SOURCE_EPOCH_STRATEGIES: dict[str, str] = {
    "source_id_device_version_time_zone": "Split epochs on source, device/version or IANA time-zone changes.",
    "source_id_device": "Split epochs on source/device changes.",
    "source_id_only": "Split epochs on source ID changes.",
    "none": "No source-epoch logic is required.",
}


SOURCE_PRIORITY_STRATEGIES: dict[str, str] = {
    "preserve_all": "Do not rank sources during native curation.",
    "preserve_all_mark_conflicts": "Preserve sources and mark unresolved conflicts.",
    "not_applicable": "Source priority does not apply.",
}


OCCURRENCE_IDENTITY_STRATEGIES: dict[str, str] = {
    "inherit_native_event_identity": "Use the Milestone 1 native occurrence identity.",
    "inherit_native_summary_identity": "Use outer summary occurrence and payload index.",
    "inherit_native_waveform_identity": "Use ECG record identity and native signal event.",
}


DUPLICATE_STRATEGIES: dict[str, str] = {
    "inherit_native_record_id_then_content": "Trust Milestone 1 record-ID/content reconciliation.",
    "inherit_native_record_id_only": "Trust Milestone 1 record-ID-only reconciliation.",
    "inherit_native_summary_occurrence": "Preserve ActivitySummary occurrences.",
    "inherit_native_sleep_occurrence": "Preserve distinct state intervals after exact duplicate reconciliation.",
    "inherit_native_vector_occurrence": "Preserve coupled multivariate readings.",
    "inherit_native_waveform_occurrence": "Preserve each waveform event.",
}


REVISION_STRATEGIES: dict[str, str] = {
    "inherit_native_interval_revision": "Use Milestone 1 Apple interval-boundary revision resolution.",
    "inherit_native_record_revision": "Use Milestone 1 same-record revision handling.",
    "walking_heart_rate_replaceable_estimate": "Retain native rows and recognize replaceable Apple estimates.",
    "activity_summary_unresolved_date_assignment": "Do not select one sliding summary item yet.",
    "none": "No additional revision strategy is declared.",
}


RULE_STRATEGIES: dict[str, str] = {
    "required_measurements_present": "Check required measurement columns and non-null values.",
    "finite_numeric": "Check that numeric values are finite.",
    "timestamp_order": "Check start <= end.",
    "nonnegative_measurement": "Flag negative values where the measurement cannot be negative.",
    "positive_measurement": "Flag non-positive values where a positive value is expected.",
    "point_duration_zero": "Check that a point measurement has zero native duration.",
    "interval_duration_positive": "Check that an interval expected to carry support has positive duration.",
    "unresolved_same_interval_conflict": "Detect retained same-interval conflicts from Milestone 1.",
    "unit_resolved_for_canonical_value": "Require resolved unit before writing a canonical value.",
    "unit_epoch_discontinuity": "Detect abrupt unit-scale changes within participant/source history.",
    "manual_entry_context": "Retain and expose manual-entry provenance.",
    "source_epoch_transition": "Record source/device/time-zone epoch transitions.",
    "heart_rate_motion_context_vocabulary": "Validate but preserve motion-context values.",
    "heart_rate_summary_interval_context": "Flag long summary intervals as summaries, not dense observations.",
    "hrv_sdnn_context": "Retain SDNN algorithm/version context and unit confidence.",
    "sleep_state_vocabulary": "Validate known HealthKit sleep states without lexical selection.",
    "sleep_same_state_overlap": "Detect overlapping intervals carrying the same sleep state.",
    "sleep_detailed_state_overlap": "Detect overlaps among detailed sleep stages.",
    "sleep_inbed_nesting": "Evaluate detailed states relative to in-bed support without requiring full coverage.",
    "blood_pressure_pair_complete": "Require systolic and diastolic components in the same event.",
    "blood_pressure_pair_order": "Flag systolic values not greater than diastolic values.",
    "activity_summary_date_ambiguity": "Flag summary items whose calendar assignment is unresolved.",
    "activity_summary_goal_nonnegative": "Check summary and goal fields for nonnegative values.",
    "ecg_waveform_shape": "Validate nested waveform structure.",
    "ecg_sample_count_consistency": "Compare waveform length with frequency and duration.",
    "ecg_relative_time_monotonic": "Check monotonic waveform relative time.",
    "ecg_finite_amplitude": "Check waveform amplitudes for finite values.",
    "cgm_status_vocabulary": "Validate but preserve CGM status values.",
    "cgm_trend_vocabulary": "Validate but preserve CGM trend arrows and rates.",
    "cgm_time_zone_context": "Preserve and validate IANA time-zone context.",
    "pulse_ox_source_limitations": "Flag missing device/source context needed for interpretation.",
    "temperature_sensor_location_context": "Flag missing temperature sensor/location context.",
    "peak_flow_session_context": "Flag lack of maneuver/session quality context.",
    "vo2max_test_type_context": "Flag missing or unknown VO2Max test type.",
    "bac_calculator_estimate": "Identify calculator-generated BAC values.",
    "nutrition_preserve_distinct_uuid": "Preserve equal values with distinct UUIDs.",
    "nutrition_nonnegative": "Flag negative nutrient/energy amounts.",
    "mindful_interval_valid": "Validate mindful-session start and end.",
    "bmi_screening_context": "Record that BMI is a screening measure, not a diagnosis.",
    "waist_protocol_context": "Flag unknown anatomical measurement protocol.",
    "unknown_feature_policy": "Mark an unknown feature as review/exclude without conversion.",
}


CROSS_FEATURE_STRATEGIES: dict[str, str] = {
    "bmi_height_weight_consistency": "Compare BMI with temporally matched canonical height and weight.",
    "lean_mass_weight_body_fat_consistency": "Check lean mass against weight and body-fat fraction.",
    "none": "No cross-feature relationship is required.",
}


RESAMPLING_STRATEGIES: dict[str, str] = {
    "extensive_overlap_sum": "Allocate interval totals by exact overlap and sum contributions.",
    "intensive_window_distribution": "Summarize native observations while retaining support counts.",
    "point_retain": "Retain point observations; no artificial duration or interpolation by default.",
    "state_interval_union": "Compute per-state union/coverage without categorical mode.",
    "daily_context_only": "Keep daily summary at daily resolution unless explicitly joined as context.",
    "long_summary_reference": "Reference long summary events; prohibit default broadcasting.",
    "waveform_native_only": "Keep signal-native events; ordinary window resampling is unsupported.",
    "duration_interval_coverage": "Represent duration-event overlap without inventing values.",
    "nutrition_event_retain": "Retain discrete nutrition events; do not spread amounts by default.",
    "unknown_unsupported": "Do not resample an unknown feature automatically.",
}


ALL_STRATEGY_CATALOGS: dict[str, dict[str, str]] = {
    "conversion": CONVERSION_STRATEGIES,
    "unit_resolution": UNIT_RESOLUTION_STRATEGIES,
    "acquisition": ACQUISITION_STRATEGIES,
    "source_epoch": SOURCE_EPOCH_STRATEGIES,
    "source_priority": SOURCE_PRIORITY_STRATEGIES,
    "occurrence_identity": OCCURRENCE_IDENTITY_STRATEGIES,
    "duplicate": DUPLICATE_STRATEGIES,
    "revision": REVISION_STRATEGIES,
    "rule": RULE_STRATEGIES,
    "cross_feature": CROSS_FEATURE_STRATEGIES,
    "resampling": RESAMPLING_STRATEGIES,
}
