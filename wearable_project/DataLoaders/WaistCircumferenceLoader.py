"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit WaistCircumference.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class WaistCircumferenceLoader(AppleHealthFeatureLoader):
    feature_name = "WaistCircumference"

