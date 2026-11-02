"""
Wearable Data Processing and Modeling project
Feature-specific DataLoaders.
Loaders consume validated processed feature tables rather than raw HealthKit exports.
"""


from importlib import import_module
from typing import Any
from wearable_project.DataLoaders._base import (
    AppleHealthFeatureLoader, DEFAULT_CURATED_ROOT, DEFAULT_MAX_ROWS, DEFAULT_NATIVE_ROOT, DataLoaderSizeError,
    DataLoaderStateError, LoaderData, WearableFeatureLoader,
)
from wearable_project.DataLoaders._sources import (
    DEFAULT_APPLE_HEALTHKIT_CURATED_ROOT, DEFAULT_APPLE_HEALTHKIT_NATIVE_ROOT, DEFAULT_GOOGLE_FITNESS_CURATED_ROOT,
    DEFAULT_GOOGLE_FITNESS_NATIVE_ROOT, WearableSource,
)
from wearable_project.exceptions import DataLoaderSourceError
from wearable_project.DataLoaders._profile import ProfileReport
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
from wearable_project.DataLoaders.AerobicsLoader import AerobicsLoader
from wearable_project.DataLoaders.BasketballLoader import BasketballLoader
from wearable_project.DataLoaders.BikingLoader import BikingLoader
from wearable_project.DataLoaders.BikingHandLoader import BikingHandLoader
from wearable_project.DataLoaders.BikingMountainLoader import BikingMountainLoader
from wearable_project.DataLoaders.BikingRoadLoader import BikingRoadLoader
from wearable_project.DataLoaders.BikingSpinningLoader import BikingSpinningLoader
from wearable_project.DataLoaders.BikingStationaryLoader import BikingStationaryLoader
from wearable_project.DataLoaders.BikingUtilityLoader import BikingUtilityLoader
from wearable_project.DataLoaders.BoxingLoader import BoxingLoader
from wearable_project.DataLoaders.CalisthenicsLoader import CalisthenicsLoader
from wearable_project.DataLoaders.CaloriesLoader import CaloriesLoader
from wearable_project.DataLoaders.CircuitTrainingLoader import CircuitTrainingLoader
from wearable_project.DataLoaders.CrossfitLoader import CrossfitLoader
from wearable_project.DataLoaders.DancingLoader import DancingLoader
from wearable_project.DataLoaders.DistanceLoader import DistanceLoader
from wearable_project.DataLoaders.EllipticalLoader import EllipticalLoader
from wearable_project.DataLoaders.ErgometerLoader import ErgometerLoader
from wearable_project.DataLoaders.GardeningLoader import GardeningLoader
from wearable_project.DataLoaders.GuidedBreathingLoader import GuidedBreathingLoader
from wearable_project.DataLoaders.GymnasticsLoader import GymnasticsLoader
from wearable_project.DataLoaders.HandballLoader import HandballLoader
from wearable_project.DataLoaders.HikingLoader import HikingLoader
from wearable_project.DataLoaders.HorsebackRidingLoader import HorsebackRidingLoader
from wearable_project.DataLoaders.InVehicleLoader import InVehicleLoader
from wearable_project.DataLoaders.IntervalTrainingLoader import IntervalTrainingLoader
from wearable_project.DataLoaders.IntervalTrainingHighIntensityLoader import IntervalTrainingHighIntensityLoader
from wearable_project.DataLoaders.JumpRopeLoader import JumpRopeLoader
from wearable_project.DataLoaders.KayakingLoader import KayakingLoader
from wearable_project.DataLoaders.KickScooterLoader import KickScooterLoader
from wearable_project.DataLoaders.KickboxingLoader import KickboxingLoader
from wearable_project.DataLoaders.MartialArtsLoader import MartialArtsLoader
from wearable_project.DataLoaders.MeditationLoader import MeditationLoader
from wearable_project.DataLoaders.OtherActivityLoader import OtherActivityLoader
from wearable_project.DataLoaders.PilatesLoader import PilatesLoader
from wearable_project.DataLoaders.RockClimbingLoader import RockClimbingLoader
from wearable_project.DataLoaders.RowingLoader import RowingLoader
from wearable_project.DataLoaders.RowingMachineLoader import RowingMachineLoader
from wearable_project.DataLoaders.RunningLoader import RunningLoader
from wearable_project.DataLoaders.RunningJoggingLoader import RunningJoggingLoader
from wearable_project.DataLoaders.RunningSandLoader import RunningSandLoader
from wearable_project.DataLoaders.RunningTreadmillLoader import RunningTreadmillLoader
from wearable_project.DataLoaders.SkatingCrossLoader import SkatingCrossLoader
from wearable_project.DataLoaders.SkiingCrossCountryLoader import SkiingCrossCountryLoader
from wearable_project.DataLoaders.StairClimbingLoader import StairClimbingLoader
from wearable_project.DataLoaders.StairClimbingMachineLoader import StairClimbingMachineLoader
from wearable_project.DataLoaders.StillLoader import StillLoader
from wearable_project.DataLoaders.StrengthTrainingLoader import StrengthTrainingLoader
from wearable_project.DataLoaders.SwimmingLoader import SwimmingLoader
from wearable_project.DataLoaders.SwimmingOpenWaterLoader import SwimmingOpenWaterLoader
from wearable_project.DataLoaders.SwimmingPoolLoader import SwimmingPoolLoader
from wearable_project.DataLoaders.TennisLoader import TennisLoader
from wearable_project.DataLoaders.UnknownActivityLoader import UnknownActivityLoader
from wearable_project.DataLoaders.VolleyballLoader import VolleyballLoader
from wearable_project.DataLoaders.VolleyballBeachLoader import VolleyballBeachLoader
from wearable_project.DataLoaders.VolleyballIndoorLoader import VolleyballIndoorLoader
from wearable_project.DataLoaders.WalkingLoader import WalkingLoader
from wearable_project.DataLoaders.WalkingFitnessLoader import WalkingFitnessLoader
from wearable_project.DataLoaders.WalkingNordicLoader import WalkingNordicLoader
from wearable_project.DataLoaders.WalkingTreadmillLoader import WalkingTreadmillLoader
from wearable_project.DataLoaders.WeightliftingLoader import WeightliftingLoader
from wearable_project.DataLoaders.YogaLoader import YogaLoader
from wearable_project.DataLoaders.ZumbaLoader import ZumbaLoader


# ``info`` reads the processing and curation registries, which the feature loaders themselves do not
# need. Resolving these names lazily keeps ``from wearable_project.DataLoaders.StepCountLoader
# import StepCountLoader`` working when only the DataLoaders subpackage is available, while
# ``DataLoaders.info(...)`` still behaves exactly as before.
_LAZY_INFO_EXPORTS = ("InfoReport", "available_features", "available_sources", "info")


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
    "WearableFeatureLoader",
    "WearableSource",
    "DEFAULT_MAX_ROWS",
    "DataLoaderSizeError",
    "DataLoaderStateError",
    "DataLoaderSourceError",
    "LoaderData",
    "ProfileReport",
    "DEFAULT_NATIVE_ROOT",
    "DEFAULT_CURATED_ROOT",
    "DEFAULT_APPLE_HEALTHKIT_NATIVE_ROOT",
    "DEFAULT_APPLE_HEALTHKIT_CURATED_ROOT",
    "DEFAULT_GOOGLE_FITNESS_NATIVE_ROOT",
    "DEFAULT_GOOGLE_FITNESS_CURATED_ROOT",
    "InfoReport",
    "available_features",
    "available_sources",
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
    "AerobicsLoader",
    "BasketballLoader",
    "BikingLoader",
    "BikingHandLoader",
    "BikingMountainLoader",
    "BikingRoadLoader",
    "BikingSpinningLoader",
    "BikingStationaryLoader",
    "BikingUtilityLoader",
    "BoxingLoader",
    "CalisthenicsLoader",
    "CaloriesLoader",
    "CircuitTrainingLoader",
    "CrossfitLoader",
    "DancingLoader",
    "DistanceLoader",
    "EllipticalLoader",
    "ErgometerLoader",
    "GardeningLoader",
    "GuidedBreathingLoader",
    "GymnasticsLoader",
    "HandballLoader",
    "HikingLoader",
    "HorsebackRidingLoader",
    "InVehicleLoader",
    "IntervalTrainingLoader",
    "IntervalTrainingHighIntensityLoader",
    "JumpRopeLoader",
    "KayakingLoader",
    "KickScooterLoader",
    "KickboxingLoader",
    "MartialArtsLoader",
    "MeditationLoader",
    "OtherActivityLoader",
    "PilatesLoader",
    "RockClimbingLoader",
    "RowingLoader",
    "RowingMachineLoader",
    "RunningLoader",
    "RunningJoggingLoader",
    "RunningSandLoader",
    "RunningTreadmillLoader",
    "SkatingCrossLoader",
    "SkiingCrossCountryLoader",
    "StairClimbingLoader",
    "StairClimbingMachineLoader",
    "StillLoader",
    "StrengthTrainingLoader",
    "SwimmingLoader",
    "SwimmingOpenWaterLoader",
    "SwimmingPoolLoader",
    "TennisLoader",
    "UnknownActivityLoader",
    "VolleyballLoader",
    "VolleyballBeachLoader",
    "VolleyballIndoorLoader",
    "WalkingLoader",
    "WalkingFitnessLoader",
    "WalkingNordicLoader",
    "WalkingTreadmillLoader",
    "WeightliftingLoader",
    "YogaLoader",
    "ZumbaLoader",
]
