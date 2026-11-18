"""Núcleo del framework: ingesta, EDA y preprocesamiento."""

from .eda import EDAnalyzer
from .loader import DataLoadError, TabularLoader
from .preprocessor import PreparedData, Preprocessor, prepare_dataset
from .schema import Dataset, TaskType

__all__ = [
    "DataLoadError",
    "Dataset",
    "EDAnalyzer",
    "PreparedData",
    "Preprocessor",
    "TabularLoader",
    "TaskType",
    "prepare_dataset",
]
