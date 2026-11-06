"""
Wearable Data Processing and Modeling project
Source-aware HPP loader for Height. Apple HealthKit remains the backward-compatible default;
pass source="google_fitness" to load the Google Fitness product.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class HeightLoader(AppleHealthFeatureLoader):
    feature_name = "Height"

