"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit WalkingHeartRate.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class WalkingHeartRateLoader(AppleHealthFeatureLoader):
    feature_name = "WalkingHeartRate"

