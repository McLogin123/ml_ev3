"""Modelos supervisados: lineales, árboles, splines y prototipos."""

from .linear_models import LinearRegression, LogisticRegression
from .tree_models import BaggingClassifier, DecisionTree
from .spline_models import MARSRegressor
from .prototype_models import DANN, KNeighborsModel, LVQ1

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "DecisionTree",
    "BaggingClassifier",
    "MARSRegressor",
    "KNeighborsModel",
    "LVQ1",
    "DANN",
]
