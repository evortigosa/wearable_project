"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit Weight.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class WeightLoader(AppleHealthFeatureLoader):
    feature_name = "Weight"

