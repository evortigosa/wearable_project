"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Elliptical.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class EllipticalLoader(WearableFeatureLoader):
    feature_name = "Elliptical"
    default_source = WearableSource.GOOGLE_FITNESS
