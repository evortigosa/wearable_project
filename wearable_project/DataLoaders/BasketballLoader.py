"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Basketball.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class BasketballLoader(WearableFeatureLoader):
    feature_name = "Basketball"
    default_source = WearableSource.GOOGLE_FITNESS
