"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness StrengthTraining.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class StrengthTrainingLoader(WearableFeatureLoader):
    feature_name = "StrengthTraining"
    default_source = WearableSource.GOOGLE_FITNESS
