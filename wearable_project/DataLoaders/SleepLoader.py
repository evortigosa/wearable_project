"""
Wearable Data Processing and Modeling project
Source-aware HPP loader for Sleep. Apple HealthKit remains the backward-compatible default;
pass source="google_fitness" to load the Google Fitness product.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class SleepLoader(AppleHealthFeatureLoader):
    feature_name = "Sleep"

