"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Calisthenics.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class CalisthenicsLoader(WearableFeatureLoader):
    feature_name = "Calisthenics"
    default_source = WearableSource.GOOGLE_FITNESS
