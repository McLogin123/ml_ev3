"""Pruebas de modelos basados en prototipos."""

import numpy as np
import pytest

from core.schema import TaskType
from models.supervised.prototype_models import DANN, KNeighborsModel, LVQ1


def test_knn_classifier_predicts_nearest_cluster():
    # Preparar
    X = np.array([[0.0, 0.0], [0.2, 0.0], [4.0, 4.0], [4.2, 4.0]])
    y = np.array(["A", "A", "B", "B"])

    # Ejecutar
    model = KNeighborsModel(TaskType.CLASSIFICATION, n_neighbors=1).fit(X, y)

    # Verificar
    np.testing.assert_array_equal(
        model.predict(np.array([[0.1, 0.1], [4.1, 4.1]])), ["A", "B"]
    )


def test_knn_regressor_averages_neighbors():
    # Preparar
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 2.0, 4.0, 6.0])

    # Ejecutar
    model = KNeighborsModel(TaskType.REGRESSION, n_neighbors=2).fit(X, y)

    # Verificar
    np.testing.assert_allclose(model.predict(np.array([[0.5], [2.5]])), [1.0, 5.0])


def test_knn_rejects_too_many_neighbors():
    # Preparar
    X = np.array([[0.0], [1.0]])

    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="n_neighbors"):
        KNeighborsModel(TaskType.REGRESSION, n_neighbors=3).fit(X, np.array([0.0, 1.0]))


def test_lvq1_separates_two_clusters():
    # Preparar
    X = np.array(
        [
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
            [5.0, 5.0], [5.1, 5.0], [5.0, 5.1],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    # Ejecutar
    model = LVQ1(
        n_prototypes_per_class=1,
        learning_rate=0.2,
        n_epochs=50,
        random_state=7,
    ).fit(X, y)

    # Verificar
    np.testing.assert_array_equal(model.predict(X), y)
    assert model.prototypes_.shape == (2, 2)


def test_lvq1_is_reproducible_with_same_seed():
    # Preparar
    X = np.array([[0.0], [0.2], [3.0], [3.2]])
    y = np.array([0, 0, 1, 1])

    # Ejecutar
    first = LVQ1(n_epochs=10, random_state=3).fit(X, y)
    second = LVQ1(n_epochs=10, random_state=3).fit(X, y)

    # Verificar
    np.testing.assert_allclose(first.prototypes_, second.prototypes_)


def test_dann_trains_and_predicts_target_domain():
    # Preparar: fuente etiquetada y objetivo sin etiquetar con el mismo significado
    X_source = np.array([[-2.0, 0.0], [-2.2, 0.1], [2.0, 0.0], [2.2, 0.1]])
    y_source = np.array([0, 0, 1, 1])
    X_target = np.array([[-2.0, 2.0], [-2.2, 2.1], [2.0, 2.0], [2.2, 2.1]])
    X = np.vstack([X_source, X_target])
    y = np.concatenate([y_source, np.full(len(X_target), -1)])
    domains = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # Ejecutar
    model = DANN(
        hidden_size=8,
        learning_rate=0.05,
        n_epochs=300,
        domain_weight=1.0,
        random_state=0,
    ).fit(X, y, domains)

    # Verificar
    probabilities = model.predict_proba(X_target)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_array_equal(model.predict(X_target), [0, 0, 1, 1])
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_dann_rejects_unlabeled_only_data():
    # Preparar
    X = np.array([[0.0], [1.0]])

    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="etiquetada"):
        DANN().fit(X, np.array([-1, -1]), np.array([0, 1]))


def test_dann_rejects_unknown_domain():
    # Preparar
    X = np.array([[0.0], [1.0]])

    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="domains"):
        DANN().fit(X, np.array([0, 1]), np.array([0, 2]))
