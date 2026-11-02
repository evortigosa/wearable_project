"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Walking.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class WalkingLoader(WearableFeatureLoader):
    feature_name = "Walking"
    default_source = WearableSource.GOOGLE_FITNESS
