"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness BikingStationary.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class BikingStationaryLoader(WearableFeatureLoader):
    feature_name = "BikingStationary"
    default_source = WearableSource.GOOGLE_FITNESS
