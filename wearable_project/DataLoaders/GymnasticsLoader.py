"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Gymnastics.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class GymnasticsLoader(WearableFeatureLoader):
    feature_name = "Gymnastics"
    default_source = WearableSource.GOOGLE_FITNESS
