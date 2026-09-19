"""
Wearable Data Processing and Modeling project
HPP-style loader for Apple HealthKit Protein.
"""


from wearable_project.DataLoaders._base import AppleHealthFeatureLoader


class ProteinLoader(AppleHealthFeatureLoader):
    feature_name = "Protein"

