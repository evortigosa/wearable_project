"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Aerobics.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class AerobicsLoader(WearableFeatureLoader):
    feature_name = "Aerobics"
    default_source = WearableSource.GOOGLE_FITNESS
