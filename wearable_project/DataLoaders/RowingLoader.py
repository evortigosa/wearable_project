"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Rowing.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class RowingLoader(WearableFeatureLoader):
    feature_name = "Rowing"
    default_source = WearableSource.GOOGLE_FITNESS
