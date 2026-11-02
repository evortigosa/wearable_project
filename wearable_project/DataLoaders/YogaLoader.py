"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Yoga.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class YogaLoader(WearableFeatureLoader):
    feature_name = "Yoga"
    default_source = WearableSource.GOOGLE_FITNESS
