"""Pruebas de las interfaces base de los modelos."""

import numpy as np
import pytest

from core.schema import TaskType
from models.base import (
    BaseModel,
    BaseSupervisedModel,
    BaseUnsupervisedModel,
    ModelNotFittedError,
)


class DummyClassifier(BaseSupervisedModel):
    """Modelo de prueba que siempre predice una misma clase."""

    def __init__(self, constant=1):
        super().__init__(TaskType.CLASSIFICATION)
        self.constant = constant

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        self.is_fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        return np.full(len(X), self.constant)


class DummyRegressor(BaseSupervisedModel):
    """Modelo de prueba que predice la media vista en entrenamiento."""

    def __init__(self):
        super().__init__(TaskType.REGRESSION)

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        self.is_fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        return np.full(len(X), self.mean_)


class DummyUnsupervised(BaseUnsupervisedModel):
    """Modelo de prueba que asigna todas las filas al cluster 0."""

    def fit(self, X):
        self.labels_ = np.zeros(len(X), dtype=int)
        self.is_fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        return np.zeros(len(X), dtype=int)


class BadTaskModel(BaseSupervisedModel):
    """Modelo de prueba para validar la tarea recibida."""

    def __init__(self, task):
        super().__init__(task)

    def fit(self, X, y):
        self.is_fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        return np.zeros(len(X), dtype=int)


def test_clases_abstractas_no_se_instancian():
    # Ejecutar y verificar
    with pytest.raises(TypeError):
        BaseModel()
    with pytest.raises(TypeError):
        BaseSupervisedModel(TaskType.CLASSIFICATION)
    with pytest.raises(TypeError):
        BaseUnsupervisedModel()


def test_herencia_de_un_modelo_concreto():
    # Preparar
    model = DummyClassifier()

    # Verificar
    assert isinstance(model, BaseModel)
    assert isinstance(model, BaseSupervisedModel)


def test_fit_marca_el_modelo_como_ajustado():
    # Preparar
    model = DummyClassifier()

    # Ejecutar
    result = model.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))

    # Verificar
    assert result is model
    assert model.is_fitted is True
    assert model.n_features_ == 2


def test_predecir_antes_de_fit_falla():
    # Preparar
    model = DummyClassifier()

    # Ejecutar y verificar
    with pytest.raises(ModelNotFittedError, match="no está ajustado"):
        model.predict(np.array([[1, 2]]))


def test_score_antes_de_fit_falla():
    # Preparar
    model = DummyRegressor()

    # Ejecutar y verificar
    with pytest.raises(ModelNotFittedError):
        model.score(np.array([[1]]), np.array([1]))


def test_score_de_clasificacion_usa_accuracy():
    # Preparar: la clase 1 aparece 2 de 4 veces.
    model = DummyClassifier(constant=1)
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 1, 0])

    # Ejecutar
    model.fit(X, y)
    result = model.score(X, y)

    # Verificar
    assert result == pytest.approx(0.5)


def test_score_de_regresion_usa_r2():
    # Preparar
    model = DummyRegressor()
    X = np.array([[0], [1], [2]])
    y = np.array([1.0, 2.0, 3.0])

    # Ejecutar
    model.fit(X, y)
    result = model.score(X, y)

    # Verificar
    assert result == pytest.approx(0.0)


def test_tarea_desconocida_falla():
    # Ejecutar y verificar
    with pytest.raises(ValueError):
        BadTaskModel("cluster")


def test_score_con_tarea_no_soportada_falla():
    # Preparar
    model = BadTaskModel(TaskType.UNSUPERVISED)
    X = np.array([[0], [1]])
    model.fit(X, np.array([0, 1]))

    # Ejecutar y verificar
    with pytest.raises(ValueError, match="Tarea no soportada"):
        model.score(X, np.array([0, 1]))


def test_modelo_no_supervisado_guarda_etiquetas():
    # Preparar
    model = DummyUnsupervised()

    # Ejecutar
    model.fit(np.array([[0, 0], [1, 1], [2, 2]]))

    # Verificar
    assert model.is_fitted is True
    assert model.labels_.tolist() == [0, 0, 0]


def test_modelo_no_supervisado_predice_cluster():
    # Preparar
    model = DummyUnsupervised()
    model.fit(np.array([[0, 0], [1, 1]]))

    # Ejecutar
    result = model.predict(np.array([[9, 9], [8, 8]]))

    # Verificar
    assert result.tolist() == [0, 0]


def test_modelo_no_supervisado_predice_antes_de_fit_falla():
    # Preparar
    model = DummyUnsupervised()

    # Ejecutar y verificar
    with pytest.raises(ModelNotFittedError):
        model.predict(np.array([[0, 0]]))
