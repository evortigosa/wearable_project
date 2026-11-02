"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness SwimmingOpenWater.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class SwimmingOpenWaterLoader(WearableFeatureLoader):
    feature_name = "SwimmingOpenWater"
    default_source = WearableSource.GOOGLE_FITNESS
