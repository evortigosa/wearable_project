"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness StairClimbing.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class StairClimbingLoader(WearableFeatureLoader):
    feature_name = "StairClimbing"
    default_source = WearableSource.GOOGLE_FITNESS
