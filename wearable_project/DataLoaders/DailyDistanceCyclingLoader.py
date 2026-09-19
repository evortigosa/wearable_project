"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit DailyDistanceCycling.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class DailyDistanceCyclingLoader(AppleHealthFeatureLoader):
    feature_name = "DailyDistanceCycling"

