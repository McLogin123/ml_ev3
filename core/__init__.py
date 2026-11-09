"""Núcleo del framework: ingesta, EDA y preprocesamiento."""

from .eda import EDAnalyzer
from .loader import DataLoadError, TabularLoader
from .schema import Dataset, TaskType

__all__ = [
    "DataLoadError",
    "Dataset",
    "EDAnalyzer",
    "TabularLoader",
    "TaskType",
]
