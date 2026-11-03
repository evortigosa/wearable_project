"""
Wearable Data Processing and Modeling project
Backward-compatible Apple HealthKit stored-unit API. The implementation lives in ``DataLoaders.apple_healthkit.units``
so the shared loader engine and Google Fitness paths do not import Apple registries. Existing internal and laboratory
imports from ``DataLoaders._units`` remain valid.
"""


from wearable_project.DataLoaders.apple_healthkit.units import StoredUnit, _curation_declaration, stored_unit


__all__ = ["StoredUnit", "stored_unit"]
