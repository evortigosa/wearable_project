"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness CircuitTraining.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class CircuitTrainingLoader(WearableFeatureLoader):
    feature_name = "CircuitTraining"
    default_source = WearableSource.GOOGLE_FITNESS
