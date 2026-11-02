"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Calories.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class CaloriesLoader(WearableFeatureLoader):
    feature_name = "Calories"
    default_source = WearableSource.GOOGLE_FITNESS
