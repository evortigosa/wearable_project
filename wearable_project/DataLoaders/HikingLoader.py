"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Hiking.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class HikingLoader(WearableFeatureLoader):
    feature_name = "Hiking"
    default_source = WearableSource.GOOGLE_FITNESS
