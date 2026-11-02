"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Biking.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class BikingLoader(WearableFeatureLoader):
    feature_name = "Biking"
    default_source = WearableSource.GOOGLE_FITNESS
