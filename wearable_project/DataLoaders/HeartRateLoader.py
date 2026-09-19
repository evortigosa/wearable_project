"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit HeartRate.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class HeartRateLoader(AppleHealthFeatureLoader):
    feature_name = "HeartRate"

