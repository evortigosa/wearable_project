"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit RespiratoryRate.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class RespiratoryRateLoader(AppleHealthFeatureLoader):
    feature_name = "RespiratoryRate"

