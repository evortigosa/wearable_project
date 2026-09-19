"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit BodyTemperature.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class BodyTemperatureLoader(AppleHealthFeatureLoader):
    feature_name = "BodyTemperature"

