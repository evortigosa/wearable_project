"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit PeakFlow.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class PeakFlowLoader(AppleHealthFeatureLoader):
    feature_name = "PeakFlow"

