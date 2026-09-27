"""
Wearable Data Processing and Modeling project
Shared runtime, filesystem, serialization, statistics and plotting utilities.
``utils.info()`` explains the statistics and figure tools (``data_statistics``, ``data_summaries``,
``domain_metrics`` and ``data_plots``); ``utils.info("plot_agp")`` explains one of them.
"""


from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    # Resolved lazily, so that importing utils stays cheap. The info module is callable, so utils.info(...) works
    # whether wearable_project.utils.info was imported first.
    if name == "info":
        return import_module("wearable_project.utils.info")
    if name == "InfoReport":
        return import_module("wearable_project.DataLoaders.info").InfoReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
