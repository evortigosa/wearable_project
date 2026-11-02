"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness JumpRope.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class JumpRopeLoader(WearableFeatureLoader):
    feature_name = "JumpRope"
    default_source = WearableSource.GOOGLE_FITNESS
