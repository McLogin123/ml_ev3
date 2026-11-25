"""Interfaces comunes de los modelos supervisados y no supervisados

Este módulo prepara la estructura para que main.py trate a cualquier modelo de la misma
forma: fit, predict y, cuando tiene sentido, score
"""

from abc import ABC, abstractmethod

from core.schema import TaskType
from evaluation.metrics import accuracy_score, r2_score


class ModelNotFittedError(Exception):
    """Error al predecir con un modelo que todavía no fue entrenado"""


class BaseModel(ABC):
    """Contrato mínimo compartido por todos los modelos

    Atributos:
        requires_scaling : bool
            False por defecto; los modelos sensibles a la escala lo cambian
            a True y el pipeline decide el escalado con este valor
    """

    requires_scaling = False

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, X, y=None):
        """Entrena el modelo con los datos de entrada"""

    @abstractmethod
    def predict(self, X):
        """Devuelve una predicción para cada fila de X"""

    def _check_is_fitted(self):
        """Lanza un error claro si se intenta predecir antes de entrenar"""
        if not self.is_fitted:
            raise ModelNotFittedError(
                "El modelo no está ajustado. Llame a fit() antes de predecir."
            )


class BaseSupervisedModel(BaseModel, ABC):
    """Contrato de los modelos que aprenden de etiquetas

    Parametros: 
        task : TaskType or str
            Clasificación o regresión. Define qué métrica usa score()
    """

    def __init__(self, task):
        super().__init__()
        self.task = TaskType(task)

    @abstractmethod
    def fit(self, X, y):
        """Entrena el modelo con datos X y etiquetas y"""

    def score(self, X, y):
        """Devuelve accuracy en clasificación o R² en regresión

        Parametros
            X : array-like
                Muestras a evaluar
            y : array-like
                Etiquetas reales

        Retorna:
            float
                Puntuación del modelo
        """
        self._check_is_fitted()
        predictions = self.predict(X)

        if self.task == TaskType.CLASSIFICATION:
            return accuracy_score(y, predictions)
        if self.task == TaskType.REGRESSION:
            return r2_score(y, predictions)

        raise ValueError(f"Tarea no soportada para score(): {self.task}")


class BaseUnsupervisedModel(BaseModel, ABC):
    """Contrato de los modelos que no usan etiquetas

    Atributos:
        labels_ : numpy.ndarray or None
            Cluster asignado a cada muestra de entrenamiento, si aplica
    """

    def __init__(self):
        super().__init__()
        self.labels_ = None

    @abstractmethod
    def fit(self, X):
        """Encuentra estructura en los datos sin usar etiquetas"""
