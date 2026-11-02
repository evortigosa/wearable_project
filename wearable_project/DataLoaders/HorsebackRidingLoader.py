"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness HorsebackRiding.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class HorsebackRidingLoader(WearableFeatureLoader):
    feature_name = "HorsebackRiding"
    default_source = WearableSource.GOOGLE_FITNESS
