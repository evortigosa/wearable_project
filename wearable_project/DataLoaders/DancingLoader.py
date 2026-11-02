"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Dancing.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class DancingLoader(WearableFeatureLoader):
    feature_name = "Dancing"
    default_source = WearableSource.GOOGLE_FITNESS
