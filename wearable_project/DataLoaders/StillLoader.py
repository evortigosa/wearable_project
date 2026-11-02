"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Still.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class StillLoader(WearableFeatureLoader):
    feature_name = "Still"
    default_source = WearableSource.GOOGLE_FITNESS
