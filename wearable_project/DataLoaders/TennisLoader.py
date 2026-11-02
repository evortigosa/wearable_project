"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Tennis.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class TennisLoader(WearableFeatureLoader):
    feature_name = "Tennis"
    default_source = WearableSource.GOOGLE_FITNESS
