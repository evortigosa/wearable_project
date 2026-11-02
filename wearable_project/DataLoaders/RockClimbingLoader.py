"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness RockClimbing.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class RockClimbingLoader(WearableFeatureLoader):
    feature_name = "RockClimbing"
    default_source = WearableSource.GOOGLE_FITNESS
