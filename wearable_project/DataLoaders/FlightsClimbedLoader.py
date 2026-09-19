"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit FlightsClimbed.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class FlightsClimbedLoader(AppleHealthFeatureLoader):
    feature_name = "FlightsClimbed"

