"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit StepCount.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class StepCountLoader(AppleHealthFeatureLoader):
    feature_name = "StepCount"

