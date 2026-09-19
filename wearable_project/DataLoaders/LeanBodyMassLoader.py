"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit LeanBodyMass.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class LeanBodyMassLoader(AppleHealthFeatureLoader):
    feature_name = "LeanBodyMass"

