"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Crossfit.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class CrossfitLoader(WearableFeatureLoader):
    feature_name = "Crossfit"
    default_source = WearableSource.GOOGLE_FITNESS
