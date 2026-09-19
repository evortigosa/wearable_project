"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit BMI.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class BMILoader(AppleHealthFeatureLoader):
    feature_name = "BMI"

