"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit DistanceWalkingRunning.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class DistanceWalkingRunningLoader(AppleHealthFeatureLoader):
    feature_name = "DistanceWalkingRunning"

