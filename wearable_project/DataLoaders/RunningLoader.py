"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Running.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class RunningLoader(WearableFeatureLoader):
    feature_name = "Running"
    default_source = WearableSource.GOOGLE_FITNESS
