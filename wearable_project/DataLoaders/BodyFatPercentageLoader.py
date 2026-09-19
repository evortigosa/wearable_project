"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit BodyFatPercentage.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class BodyFatPercentageLoader(AppleHealthFeatureLoader):
    feature_name = "BodyFatPercentage"

