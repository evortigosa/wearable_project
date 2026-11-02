"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness BikingMountain.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class BikingMountainLoader(WearableFeatureLoader):
    feature_name = "BikingMountain"
    default_source = WearableSource.GOOGLE_FITNESS
