"""Pruebas de árboles CART y Bagging."""

import numpy as np
import pytest

from core.schema import TaskType
from models.supervised.tree_models import BaggingClassifier, DecisionTree


def test_decision_tree_regression_learns_linear_relation():
    # Preparar
    X = np.linspace(0.0, 4.0, 9).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0

    # Ejecutar
    model = DecisionTree(TaskType.REGRESSION, max_depth=None).fit(X, y)
    predictions = model.predict(X)

    # Verificar
    np.testing.assert_allclose(predictions, y, atol=1e-10)
    assert model.score(X, y) == pytest.approx(1.0)


def test_decision_tree_classification_solves_xor():
    # Preparar
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Ejecutar
    model = DecisionTree(TaskType.CLASSIFICATION, max_depth=2).fit(X, y)

    # Verificar
    np.testing.assert_array_equal(model.predict(X), y)
    assert model.score(X, y) == 1.0


def test_decision_tree_max_depth_zero_returns_constant():
    # Preparar
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 5.0, 9.0])

    # Ejecutar
    model = DecisionTree(TaskType.REGRESSION, max_depth=0).fit(X, y)

    # Verificar
    np.testing.assert_allclose(model.predict(X), [5.0, 5.0, 5.0])


def test_decision_tree_reports_feature_importance():
    # Preparar: la primera feature es ruido y la segunda decide la etiqueta
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=60), np.arange(60, dtype=float)])
    y = (np.arange(60) >= 30).astype(int)

    # Ejecutar
    model = DecisionTree(TaskType.CLASSIFICATION, max_depth=3).fit(X, y)

    # Verificar
    assert model.feature_importances_[1] > model.feature_importances_[0]
    np.testing.assert_allclose(model.feature_importances_.sum(), 1.0)


def test_decision_tree_rejects_bad_target_shape():
    # Preparar
    X = np.array([[0.0], [1.0]])

    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="unidimensional"):
        DecisionTree(TaskType.REGRESSION).fit(X, np.array([[1.0, 2.0]]))


def test_bagging_classifier_uses_majority_vote():
    # Preparar
    X = np.array(
        [
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
            [3.0, 3.0], [3.1, 3.0], [3.0, 3.1],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    # Ejecutar
    model = BaggingClassifier(
        n_estimators=7, max_depth=2, random_state=42
    ).fit(X, y)

    # Verificar
    np.testing.assert_array_equal(model.predict(X), y)
    assert len(model.estimators_) == 7
    np.testing.assert_allclose(model.feature_importances_.sum(), 1.0)


def test_bagging_classifier_rejects_zero_estimators():
    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="n_estimators"):
        BaggingClassifier(n_estimators=0)
