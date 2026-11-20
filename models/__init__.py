"""Modelos de aprendizaje supervisado y no supervisado."""

from .base import (
    BaseModel,
    BaseSupervisedModel,
    BaseUnsupervisedModel,
    ModelNotFittedError,
)

__all__ = [
    "BaseModel",
    "BaseSupervisedModel",
    "BaseUnsupervisedModel",
    "ModelNotFittedError",
]
