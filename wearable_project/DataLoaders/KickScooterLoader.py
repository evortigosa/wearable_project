"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness KickScooter.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class KickScooterLoader(WearableFeatureLoader):
    feature_name = "KickScooter"
    default_source = WearableSource.GOOGLE_FITNESS
