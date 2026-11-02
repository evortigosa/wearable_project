"""
Wearable Data Processing and Modeling project
HPP-style DataLoader for Google Fitness SkiingCrossCountry.
"""


from wearable_project.DataLoaders._base import WearableFeatureLoader
from wearable_project.DataLoaders._sources import WearableSource


class SkiingCrossCountryLoader(WearableFeatureLoader):
    feature_name = "SkiingCrossCountry"
    default_source = WearableSource.GOOGLE_FITNESS
