"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit ActiveEnergyBurned.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class ActiveEnergyBurnedLoader(AppleHealthFeatureLoader):
    feature_name = "ActiveEnergyBurned"

