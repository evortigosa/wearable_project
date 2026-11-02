"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness OtherActivity.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class OtherActivityLoader(WearableFeatureLoader):
    feature_name = "OtherActivity"
    default_source = WearableSource.GOOGLE_FITNESS
