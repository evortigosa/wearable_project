"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Volleyball.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class VolleyballLoader(WearableFeatureLoader):
    feature_name = "Volleyball"
    default_source = WearableSource.GOOGLE_FITNESS
