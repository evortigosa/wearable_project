"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness UnknownActivity.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class UnknownActivityLoader(WearableFeatureLoader):
    feature_name = "UnknownActivity"
    default_source = WearableSource.GOOGLE_FITNESS
