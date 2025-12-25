# drug_stats/__init__.py

__version__ = "0.1.0"

# Expose main classes for easier access
from .sampling.split_methods import DataSplitter
from .performance.metrics import MetricCalculator
from .statistics.engine import StatisticalValidator
from .visualization.plots import plot_mcsim, plot_simultaneous_ci
from .visualization.metrics_plot import MetricVisualizer

# Define what is available when running 'from drug_stats import *'
__all__ = [
    "DataSplitter",
    "MetricCalculator",
    "StatisticalValidator",
    "plot_mcsim",
    "plot_simultaneous_ci",
    "MetricVisualizer"
]