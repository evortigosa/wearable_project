"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit Sleep.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class SleepLoader(AppleHealthFeatureLoader):
    feature_name = "Sleep"

