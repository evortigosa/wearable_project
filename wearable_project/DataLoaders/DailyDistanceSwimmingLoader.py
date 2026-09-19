"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit DailyDistanceSwimming.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class DailyDistanceSwimmingLoader(AppleHealthFeatureLoader):
    feature_name = "DailyDistanceSwimming"

