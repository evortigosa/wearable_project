"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness BikingSpinning.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class BikingSpinningLoader(WearableFeatureLoader):
    feature_name = "BikingSpinning"
    default_source = WearableSource.GOOGLE_FITNESS
