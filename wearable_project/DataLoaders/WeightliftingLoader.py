"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness Weightlifting.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class WeightliftingLoader(WearableFeatureLoader):
    feature_name = "Weightlifting"
    default_source = WearableSource.GOOGLE_FITNESS
