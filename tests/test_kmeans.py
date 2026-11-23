"""Pruebas del modelo K-Means."""

import numpy as np
import pytest

from evaluation.metrics import inertia_score
from models.base import ModelNotFittedError
from models.unsupervised.kmeans_engine import KMeans


def _make_blobs():
    """Tres grupos separados para probar asignación y centros."""
    X = np.vstack(
        [
            np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]),
            np.array([[6.0, 6.0], [6.1, 6.0], [6.0, 6.1]]),
            np.array([[0.0, 6.0], [0.1, 6.0], [0.0, 6.1]]),
        ]
    )
    y = np.repeat([0, 1, 2], 3)
    return X, y


def test_kmeans_recovers_three_separated_groups():
    # Preparar
    X, y = _make_blobs()

    # Ejecutar
    model = KMeans(n_clusters=3, n_init=5, random_state=1).fit(X)

    # Verificar
    group_labels = []
    for group in (0, 1, 2):
        labels_in_group = np.unique(model.labels_[y == group])
        assert len(labels_in_group) == 1
        group_labels.append(labels_in_group[0])
    assert len(set(group_labels)) == 3
    np.testing.assert_array_equal(model.predict(X), model.labels_)


def test_kmeans_predict_assigns_new_samples_to_their_group():
    # Preparar
    X, y = _make_blobs()
    model = KMeans(n_clusters=3, n_init=5, random_state=2).fit(X)
    label_by_group = {group: model.labels_[y == group][0] for group in (0, 1, 2)}
    new_X = np.array([[0.05, 0.05], [6.05, 6.05], [0.05, 6.05]])

    # Ejecutar
    predictions = model.predict(new_X)

    # Verificar
    np.testing.assert_array_equal(
        predictions,
        [label_by_group[0], label_by_group[1], label_by_group[2]],
    )


def test_kmeans_fit_predict_returns_training_labels():
    # Preparar
    X, _ = _make_blobs()
    model = KMeans(n_clusters=3, random_state=3)

    # Ejecutar
    labels = model.fit_predict(X)

    # Verificar
    np.testing.assert_array_equal(labels, model.labels_)
    assert model.is_fitted


def test_kmeans_inertia_decreases_with_more_clusters():
    # Preparar
    X, _ = _make_blobs()

    # Ejecutar
    one_cluster = KMeans(n_clusters=1, n_init=3, random_state=4).fit(X)
    three_clusters = KMeans(n_clusters=3, n_init=3, random_state=4).fit(X)

    # Verificar
    assert three_clusters.inertia_ < one_cluster.inertia_


def test_kmeans_inertia_matches_metric_module():
    # Preparar
    X, _ = _make_blobs()
    model = KMeans(n_clusters=3, n_init=3, random_state=5).fit(X)

    # Ejecutar
    expected = inertia_score(X, model.labels_, model.cluster_centers_)

    # Verificar
    assert model.inertia_ == pytest.approx(expected)


def test_kmeans_is_reproducible_with_same_seed():
    # Preparar
    X, _ = _make_blobs()

    # Ejecutar
    first = KMeans(n_clusters=3, random_state=6).fit(X)
    second = KMeans(n_clusters=3, random_state=6).fit(X)

    # Verificar
    np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_)
    np.testing.assert_array_equal(first.labels_, second.labels_)


def test_kmeans_handles_empty_clusters_on_constant_data():
    # Preparar: todos los puntos iguales obliga a tener clusters vacíos
    X = np.zeros((5, 2))

    # Ejecutar
    model = KMeans(n_clusters=3, n_init=1, random_state=7).fit(X)

    # Verificar
    assert model.cluster_centers_.shape == (3, 2)
    assert np.isfinite(model.cluster_centers_).all()
    assert model.labels_.shape == (5,)
    assert model.inertia_ == pytest.approx(0.0)


def test_kmeans_rejects_more_clusters_than_samples():
    # Preparar
    X = np.array([[0.0], [1.0]])
    model = KMeans(n_clusters=3)

    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="n_clusters"):
        model.fit(X)


def test_kmeans_rejects_non_two_dimensional_X():
    # Preparar
    model = KMeans(n_clusters=2)

    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="bidimensional"):
        model.fit(np.array([0.0, 1.0, 2.0]))


def test_kmeans_rejects_wrong_feature_count_in_predict():
    # Preparar
    model = KMeans(n_clusters=2, random_state=8).fit(np.array([[0.0], [1.0]]))

    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="misma forma"):
        model.predict(np.array([[0.0, 1.0]]))


def test_kmeans_predict_before_fit_raises():
    # Preparar
    model = KMeans(n_clusters=2)

    # Ejecutar / Verificar
    with pytest.raises(ModelNotFittedError):
        model.predict(np.array([[0.0]]))
