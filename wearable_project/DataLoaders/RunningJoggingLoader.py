"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness RunningJogging.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class RunningJoggingLoader(WearableFeatureLoader):
    feature_name = "RunningJogging"
    default_source = WearableSource.GOOGLE_FITNESS
