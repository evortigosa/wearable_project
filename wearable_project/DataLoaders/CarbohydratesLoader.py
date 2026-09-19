"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit Carbohydrates.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class CarbohydratesLoader(AppleHealthFeatureLoader):
    feature_name = "Carbohydrates"

