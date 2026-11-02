"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness RunningSand.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class RunningSandLoader(WearableFeatureLoader):
    feature_name = "RunningSand"
    default_source = WearableSource.GOOGLE_FITNESS
