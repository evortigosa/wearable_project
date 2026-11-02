"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness RunningTreadmill.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class RunningTreadmillLoader(WearableFeatureLoader):
    feature_name = "RunningTreadmill"
    default_source = WearableSource.GOOGLE_FITNESS
