"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness MartialArts.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class MartialArtsLoader(WearableFeatureLoader):
    feature_name = "MartialArts"
    default_source = WearableSource.GOOGLE_FITNESS
