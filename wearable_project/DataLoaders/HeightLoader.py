"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit Height.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class HeightLoader(AppleHealthFeatureLoader):
    feature_name = "Height"

