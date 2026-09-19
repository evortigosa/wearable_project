"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit BloodPressure.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class BloodPressureLoader(AppleHealthFeatureLoader):
    feature_name = "BloodPressure"

