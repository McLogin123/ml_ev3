"""Modelos de aprendizaje supervisado y no supervisado."""

from .base import (
    BaseModel,
    BaseSupervisedModel,
    BaseUnsupervisedModel,
    ModelNotFittedError,
)
from .supervised import (
    BaggingClassifier,
    DANN,
    DecisionTree,
    KNeighborsModel,
    LinearRegression,
    LogisticRegression,
    LVQ1,
    GAMRegressor,
    MARSRegressor,
)
from .unsupervised import AgglomerativeClustering, KMeans

__all__ = [
    "BaseModel",
    "BaseSupervisedModel",
    "BaseUnsupervisedModel",
    "ModelNotFittedError",
    "LinearRegression",
    "LogisticRegression",
    "DecisionTree",
    "BaggingClassifier",
    "MARSRegressor",
    "GAMRegressor",
    "KNeighborsModel",
    "LVQ1",
    "DANN",
    "KMeans",
    "AgglomerativeClustering",
]
