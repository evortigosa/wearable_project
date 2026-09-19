"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit ActivitySummary.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class ActivitySummaryLoader(AppleHealthFeatureLoader):
    feature_name = "ActivitySummary"

    date_column = "datetime"
    date_semantics = "outer_summary_datetime_not_canonical_summary_day"

