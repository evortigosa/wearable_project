"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit BasalEnergyBurned.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class BasalEnergyBurnedLoader(AppleHealthFeatureLoader):
    feature_name = "BasalEnergyBurned"

