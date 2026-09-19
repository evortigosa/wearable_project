"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit HeartRateVariability.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class HeartRateVariabilityLoader(AppleHealthFeatureLoader):
    feature_name = "HeartRateVariability"

