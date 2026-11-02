"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Boxing.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class BoxingLoader(WearableFeatureLoader):
    feature_name = "Boxing"
    default_source = WearableSource.GOOGLE_FITNESS
