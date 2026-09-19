"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit EnergyConsumed.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class EnergyConsumedLoader(AppleHealthFeatureLoader):
    feature_name = "EnergyConsumed"

