"""
Wearable Data Processing and Modeling project
Feature-specific DataLoaders.
Loaders consume validated processed feature tables rather than raw HealthKit exports.
"""


from importlib import import_module
from typing import Any
from wearable_project.DataLoaders._base import (
    AppleHealthFeatureLoader, DEFAULT_CURATED_ROOT, DEFAULT_NATIVE_ROOT, LoaderData,
)
from wearable_project.DataLoaders.ActiveEnergyBurnedLoader import ActiveEnergyBurnedLoader
from wearable_project.DataLoaders.ActivitySummaryLoader import ActivitySummaryLoader
from wearable_project.DataLoaders.BMILoader import BMILoader
from wearable_project.DataLoaders.BasalEnergyBurnedLoader import BasalEnergyBurnedLoader
from wearable_project.DataLoaders.BloodAlcoholContentLoader import BloodAlcoholContentLoader
from wearable_project.DataLoaders.BloodGlucoseLoader import BloodGlucoseLoader
from wearable_project.DataLoaders.BloodPressureLoader import BloodPressureLoader
from wearable_project.DataLoaders.BodyFatPercentageLoader import BodyFatPercentageLoader
from wearable_project.DataLoaders.BodyTemperatureLoader import BodyTemperatureLoader
from wearable_project.DataLoaders.CarbohydratesLoader import CarbohydratesLoader
from wearable_project.DataLoaders.DailyDistanceCyclingLoader import DailyDistanceCyclingLoader
from wearable_project.DataLoaders.DailyDistanceSwimmingLoader import DailyDistanceSwimmingLoader
from wearable_project.DataLoaders.DistanceWalkingRunningLoader import DistanceWalkingRunningLoader
from wearable_project.DataLoaders.ElectrocardiogramLoader import ElectrocardiogramLoader
from wearable_project.DataLoaders.EnergyConsumedLoader import EnergyConsumedLoader
from wearable_project.DataLoaders.FlightsClimbedLoader import FlightsClimbedLoader
from wearable_project.DataLoaders.HeartRateLoader import HeartRateLoader
from wearable_project.DataLoaders.HeartRateVariabilityLoader import HeartRateVariabilityLoader
from wearable_project.DataLoaders.HeightLoader import HeightLoader
from wearable_project.DataLoaders.LeanBodyMassLoader import LeanBodyMassLoader
from wearable_project.DataLoaders.MindfulLoader import MindfulLoader
from wearable_project.DataLoaders.OxygenSaturationLoader import OxygenSaturationLoader
from wearable_project.DataLoaders.PeakFlowLoader import PeakFlowLoader
from wearable_project.DataLoaders.ProteinLoader import ProteinLoader
from wearable_project.DataLoaders.RespiratoryRateLoader import RespiratoryRateLoader
from wearable_project.DataLoaders.RestingHeartRateLoader import RestingHeartRateLoader
from wearable_project.DataLoaders.SleepLoader import SleepLoader
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader
from wearable_project.DataLoaders.TotalFatLoader import TotalFatLoader
from wearable_project.DataLoaders.Vo2MaxLoader import Vo2MaxLoader
from wearable_project.DataLoaders.WaistCircumferenceLoader import WaistCircumferenceLoader
from wearable_project.DataLoaders.WalkingHeartRateLoader import WalkingHeartRateLoader
from wearable_project.DataLoaders.WeightLoader import WeightLoader


# ``info`` reads the processing and curation registries, which the feature loaders themselves do not
# need. Resolving these three names lazily keeps ``from wearable_project.DataLoaders.StepCountLoader
# import StepCountLoader`` working when only the DataLoaders subpackage is available, while
# ``DataLoaders.info(...)`` still behaves exactly as before.
_LAZY_INFO_EXPORTS = ("InfoReport", "available_features", "info")


def __getattr__(name: str) -> Any:
    if name in _LAZY_INFO_EXPORTS:
        module = import_module("wearable_project.DataLoaders.info")
        # Importing the submodule binds the *module* object to this package's ``info`` attribute.
        # Rebind all three exports afterward so ``DataLoaders.info`` is the function, not the module,
        # and so later lookups resolve from globals without re-entering __getattr__.
        for export in _LAZY_INFO_EXPORTS:
            globals()[export] = getattr(module, export)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "AppleHealthFeatureLoader",
    "LoaderData",
    "DEFAULT_NATIVE_ROOT",
    "DEFAULT_CURATED_ROOT",
    "InfoReport",
    "available_features",
    "info",
    "ActiveEnergyBurnedLoader",
    "ActivitySummaryLoader",
    "BMILoader",
    "BasalEnergyBurnedLoader",
    "BloodAlcoholContentLoader",
    "BloodGlucoseLoader",
    "BloodPressureLoader",
    "BodyFatPercentageLoader",
    "BodyTemperatureLoader",
    "CarbohydratesLoader",
    "DailyDistanceCyclingLoader",
    "DailyDistanceSwimmingLoader",
    "DistanceWalkingRunningLoader",
    "ElectrocardiogramLoader",
    "EnergyConsumedLoader",
    "FlightsClimbedLoader",
    "HeartRateLoader",
    "HeartRateVariabilityLoader",
    "HeightLoader",
    "LeanBodyMassLoader",
    "MindfulLoader",
    "OxygenSaturationLoader",
    "PeakFlowLoader",
    "ProteinLoader",
    "RespiratoryRateLoader",
    "RestingHeartRateLoader",
    "SleepLoader",
    "StepCountLoader",
    "TotalFatLoader",
    "Vo2MaxLoader",
    "WaistCircumferenceLoader",
    "WalkingHeartRateLoader",
    "WeightLoader",
]
