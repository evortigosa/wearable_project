"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit TotalFat.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class TotalFatLoader(AppleHealthFeatureLoader):
    feature_name = "TotalFat"

