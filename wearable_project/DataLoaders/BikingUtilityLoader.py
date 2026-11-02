"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness BikingUtility.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class BikingUtilityLoader(WearableFeatureLoader):
    feature_name = "BikingUtility"
    default_source = WearableSource.GOOGLE_FITNESS
