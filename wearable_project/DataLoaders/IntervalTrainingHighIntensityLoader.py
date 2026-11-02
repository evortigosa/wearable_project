"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness IntervalTrainingHighIntensity.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class IntervalTrainingHighIntensityLoader(WearableFeatureLoader):
    feature_name = "IntervalTrainingHighIntensity"
    default_source = WearableSource.GOOGLE_FITNESS
