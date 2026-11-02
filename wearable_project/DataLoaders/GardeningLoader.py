"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Gardening.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class GardeningLoader(WearableFeatureLoader):
    feature_name = "Gardening"
    default_source = WearableSource.GOOGLE_FITNESS
