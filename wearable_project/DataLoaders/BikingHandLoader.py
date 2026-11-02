"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness BikingHand.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class BikingHandLoader(WearableFeatureLoader):
    feature_name = "BikingHand"
    default_source = WearableSource.GOOGLE_FITNESS
