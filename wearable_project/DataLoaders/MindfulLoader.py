"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit Mindful.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class MindfulLoader(AppleHealthFeatureLoader):
    feature_name = "Mindful"

