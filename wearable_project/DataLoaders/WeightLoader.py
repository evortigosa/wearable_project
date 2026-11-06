"""
Wearable Data Processing and Modeling project
Source-aware HPP loader for Weight. Apple HealthKit remains the backward-compatible default; pass
source="google_fitness" to load the Google Fitness product.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class WeightLoader(AppleHealthFeatureLoader):
    feature_name = "Weight"

