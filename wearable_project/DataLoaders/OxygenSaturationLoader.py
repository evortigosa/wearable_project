"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit OxygenSaturation.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class OxygenSaturationLoader(AppleHealthFeatureLoader):
    feature_name = "OxygenSaturation"

