"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit Electrocardiogram.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class ElectrocardiogramLoader(AppleHealthFeatureLoader):
    feature_name = "Electrocardiogram"

