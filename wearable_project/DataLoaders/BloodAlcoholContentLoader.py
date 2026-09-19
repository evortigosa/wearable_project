"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit BloodAlcoholContent.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class BloodAlcoholContentLoader(AppleHealthFeatureLoader):
    feature_name = "BloodAlcoholContent"

