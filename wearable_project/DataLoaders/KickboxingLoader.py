"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Kickboxing.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class KickboxingLoader(WearableFeatureLoader):
    feature_name = "Kickboxing"
    default_source = WearableSource.GOOGLE_FITNESS
