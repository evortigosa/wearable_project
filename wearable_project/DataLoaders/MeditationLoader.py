"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Meditation.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class MeditationLoader(WearableFeatureLoader):
    feature_name = "Meditation"
    default_source = WearableSource.GOOGLE_FITNESS
