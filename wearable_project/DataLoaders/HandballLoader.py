"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Handball.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class HandballLoader(WearableFeatureLoader):
    feature_name = "Handball"
    default_source = WearableSource.GOOGLE_FITNESS
