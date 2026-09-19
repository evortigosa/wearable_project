"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit BloodGlucose.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class BloodGlucoseLoader(AppleHealthFeatureLoader):
    feature_name = "BloodGlucose"

