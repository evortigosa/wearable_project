"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness VolleyballBeach.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class VolleyballBeachLoader(WearableFeatureLoader):
    feature_name = "VolleyballBeach"
    default_source = WearableSource.GOOGLE_FITNESS
