"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Distance.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class DistanceLoader(WearableFeatureLoader):
    feature_name = "Distance"
    default_source = WearableSource.GOOGLE_FITNESS
