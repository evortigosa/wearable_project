"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit RestingHeartRate.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class RestingHeartRateLoader(AppleHealthFeatureLoader):
    feature_name = "RestingHeartRate"

