"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit Vo2Max.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class Vo2MaxLoader(AppleHealthFeatureLoader):
    feature_name = "Vo2Max"

