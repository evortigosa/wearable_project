"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Pilates.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class PilatesLoader(WearableFeatureLoader):
    feature_name = "Pilates"
    default_source = WearableSource.GOOGLE_FITNESS
