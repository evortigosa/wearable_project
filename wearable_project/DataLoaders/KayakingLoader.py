"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Kayaking.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class KayakingLoader(WearableFeatureLoader):
    feature_name = "Kayaking"
    default_source = WearableSource.GOOGLE_FITNESS
