"""Núcleo del framework: ingesta, EDA y preprocesamiento."""

from .loader import DataLoadError, TabularLoader
from .schema import Dataset, TaskType

__all__ = ["DataLoadError", "Dataset", "TabularLoader", "TaskType"]
