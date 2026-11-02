"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness StairClimbingMachine.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class StairClimbingMachineLoader(WearableFeatureLoader):
    feature_name = "StairClimbingMachine"
    default_source = WearableSource.GOOGLE_FITNESS
