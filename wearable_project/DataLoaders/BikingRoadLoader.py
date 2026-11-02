"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness BikingRoad.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class BikingRoadLoader(WearableFeatureLoader):
    feature_name = "BikingRoad"
    default_source = WearableSource.GOOGLE_FITNESS
