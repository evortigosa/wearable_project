"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness RowingMachine.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class RowingMachineLoader(WearableFeatureLoader):
    feature_name = "RowingMachine"
    default_source = WearableSource.GOOGLE_FITNESS
