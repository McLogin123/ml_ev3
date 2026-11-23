"""Pruebas de los modelos lineales."""

import numpy as np
import pytest

from models.base import ModelNotFittedError
from models.supervised.linear_models import LinearRegression, LogisticRegression


def test_linear_regression_recovers_line():
    # Preparar
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    # Ejecutar
    model = LinearRegression().fit(X, y)
    predictions = model.predict(X)

    # Verificar
    np.testing.assert_allclose(model.weights_, [1.0, 2.0], atol=1e-10)
    np.testing.assert_allclose(predictions, y, atol=1e-10)
    assert model.score(X, y) == pytest.approx(1.0)


def test_linear_regression_without_intercept():
    # Preparar
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])

    # Ejecutar
    model = LinearRegression(fit_intercept=False).fit(X, y)

    # Verificar
    assert model.intercept_ == 0.0
    np.testing.assert_allclose(model.predict(X), y, atol=1e-10)


def test_linear_regression_rejects_different_feature_count():
    # Preparar
    model = LinearRegression().fit(np.array([[0.0], [1.0]]), np.array([1.0, 2.0]))

    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="número de columnas"):
        model.predict(np.array([[1.0, 2.0]]))


def test_logistic_regression_separable_data():
    # Preparar
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    # Ejecutar
    model = LogisticRegression(learning_rate=0.5, n_iterations=200).fit(X, y)
    probabilities = model.predict_proba(X)

    # Verificar
    np.testing.assert_array_equal(model.predict(X), y)
    assert probabilities.shape == (4, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_logistic_regression_rejects_non_binary_labels():
    # Preparar
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 2])

    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="etiquetas 0 y 1"):
        LogisticRegression().fit(X, y)


def test_predict_before_fit_raises():
    # Preparar
    model = LinearRegression()

    # Ejecutar / Verificar
    with pytest.raises(ModelNotFittedError):
        model.predict(np.array([[1.0]]))
