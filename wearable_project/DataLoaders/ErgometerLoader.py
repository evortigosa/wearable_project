"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Ergometer.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class ErgometerLoader(WearableFeatureLoader):
    feature_name = "Ergometer"
    default_source = WearableSource.GOOGLE_FITNESS
