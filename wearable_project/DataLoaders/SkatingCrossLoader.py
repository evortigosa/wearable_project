"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness SkatingCross.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class SkatingCrossLoader(WearableFeatureLoader):
    feature_name = "SkatingCross"
    default_source = WearableSource.GOOGLE_FITNESS
