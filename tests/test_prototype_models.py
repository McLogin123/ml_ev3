"""Pruebas de modelos basados en prototipos."""

import numpy as np
import pytest

from core.schema import TaskType
from models.supervised.prototype_models import DANN, KNeighborsModel, LVQ1


def test_todos_los_prototipos_requieren_escalado():
    # Preparar / Ejecutar / Verificar
    assert KNeighborsModel(TaskType.CLASSIFICATION).requires_scaling is True
    assert LVQ1().requires_scaling is True
    assert DANN().requires_scaling is True


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


def test_knn_distance_weights_favor_the_closest_voter():
    # Preparar: un vecino de A muy cerca y tres de B más lejos
    X = np.array([[0.1, 0.0], [1.0, 0.0], [1.1, 0.0], [1.2, 0.0]])
    y = np.array(["A", "B", "B", "B"])
    query = np.array([[0.0, 0.0]])

    # Ejecutar
    weighted = KNeighborsModel(
        TaskType.CLASSIFICATION, n_neighbors=4, weights="distance"
    ).fit(X, y)
    uniform = KNeighborsModel(
        TaskType.CLASSIFICATION, n_neighbors=4, weights="uniform"
    ).fit(X, y)

    # Verificar
    assert weighted.predict(query)[0] == "A"
    assert uniform.predict(query)[0] == "B"


def test_knn_rejects_unknown_weights():
    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="weights"):
        KNeighborsModel(TaskType.CLASSIFICATION, weights="manhattan")


def test_dann_deforma_la_metrica_local():
    # Preparar: la vecindad se extiende en x pero es casi plana en y
    X = np.array([[x, 0.0] for x in range(-5, 6)] + [[0.0, 3.0], [0.0, 3.1]])
    y = np.array([0] * 11 + [1, 1])

    # Ejecutar
    model = DANN(n_neighbors=5).fit(X, y)
    metric = model.local_metric(np.array([0.0, 0.0]))

    # Verificar
    assert metric.shape == (2, 2)
    np.testing.assert_allclose(metric, metric.T, atol=1e-12)
    assert metric[1, 1] > metric[0, 0]


def test_dann_clasifica_con_metrica_adaptada():
    # Preparar
    X = np.array(
        [[-2.0, 0.0], [-2.1, 0.1], [-1.9, -0.1], [2.0, 0.0], [2.1, 0.1], [1.9, -0.1]]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    # Ejecutar
    model = DANN(n_neighbors=3, n_votes=3, shrinkage=0.01).fit(X, y)
    probabilities = model.predict_proba(X)

    # Verificar
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_array_equal(model.predict(X), y)


def test_dann_rechaza_parametros_invalidos():
    # Preparar / Ejecutar / Verificar
    with pytest.raises(ValueError, match="n_neighbors"):
        DANN(n_neighbors=1)
    with pytest.raises(ValueError, match="n_votes"):
        DANN(n_votes=0)
    with pytest.raises(ValueError, match="shrinkage"):
        DANN(shrinkage=-0.1)
    with pytest.raises(ValueError, match="n_neighbors"):
        DANN(n_neighbors=3).fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
