"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness IntervalTraining.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class IntervalTrainingLoader(WearableFeatureLoader):
    feature_name = "IntervalTraining"
    default_source = WearableSource.GOOGLE_FITNESS
