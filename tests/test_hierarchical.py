"""Pruebas del clustering jerárquico aglomerativo."""

import numpy as np
import pytest

from models.base import ModelNotFittedError
from models.unsupervised.clustering_engine import AgglomerativeClustering


def _make_blobs():
    """Tres grupos separados para probar asignación final."""
    X = np.vstack(
        [
            np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]),
            np.array([[6.0, 6.0], [6.1, 6.0], [6.0, 6.1]]),
            np.array([[0.0, 6.0], [0.1, 6.0], [0.0, 6.1]]),
        ]
    )
    y = np.repeat([0, 1, 2], 3)
    return X, y


def test_complete_linkage_uses_maximum_distance():
    # Preparar
    X = np.array([[0.0], [3.0], [10.0]])

    # Ejecutar
    model = AgglomerativeClustering(n_clusters=1, linkage="complete").fit(X)

    # Verificar
    np.testing.assert_allclose(model.linkage_matrix_[:, 2], [3.0, 10.0])


def test_ward_linkage_uses_variance_increase():
    # Preparar
    X = np.array([[0.0], [3.0], [10.0]])
    expected_first = 0.5 * (3.0**2)
    expected_second = (2.0 / 3.0) * ((10.0 - 1.5) ** 2)

    # Ejecutar
    model = AgglomerativeClustering(n_clusters=1, linkage="ward").fit(X)

    # Verificar
    np.testing.assert_allclose(
        model.linkage_matrix_[:, 2], [expected_first, expected_second]
    )


def test_linkage_matrix_records_new_ids_and_sizes():
    # Preparar
    X = np.array([[0.0], [3.0], [10.0]])

    # Ejecutar
    model = AgglomerativeClustering(n_clusters=1).fit(X)

    # Verificar
    assert model.linkage_matrix_.shape == (2, 4)
    np.testing.assert_array_equal(model.linkage_matrix_[0, :2], [0, 1])
    assert model.linkage_matrix_[0, 3] == 2
    np.testing.assert_array_equal(model.linkage_matrix_[1, :2], [2, 3])
    assert model.linkage_matrix_[1, 3] == 3


def test_agglomerative_recovers_three_separated_groups():
    # Preparar
    X, y = _make_blobs()

    # Ejecutar
    model = AgglomerativeClustering(n_clusters=3, linkage="complete").fit(X)

    # Verificar
    group_labels = []
    for group in (0, 1, 2):
        labels_in_group = np.unique(model.labels_[y == group])
        assert len(labels_in_group) == 1
        group_labels.append(labels_in_group[0])
    assert len(set(group_labels)) == 3


def test_agglomerative_predict_uses_nearest_training_sample():
    # Preparar
    X, y = _make_blobs()
    model = AgglomerativeClustering(n_clusters=3, linkage="complete").fit(X)
    label_by_group = {group: model.labels_[y == group][0] for group in (0, 1, 2)}
    new_X = np.array([[0.05, 0.05], [6.05, 6.05], [0.05, 6.05]])

    # Ejecutar
    predictions = model.predict(new_X)

    # Verificar
    np.testing.assert_array_equal(
        predictions,
        [label_by_group[0], label_by_group[1], label_by_group[2]],
    )


def test_agglomerative_with_one_cluster_has_no_merges_when_single_sample():
    # Preparar
    X = np.array([[1.0, 2.0]])

    # Ejecutar
    model = AgglomerativeClustering(n_clusters=1).fit(X)

    # Verificar
    np.testing.assert_array_equal(model.labels_, [0])
    assert model.linkage_matrix_.shape == (0, 4)


def test_agglomerative_rejects_unknown_linkage():
    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="linkage"):
        AgglomerativeClustering(linkage="desconocido")


def test_agglomerative_rejects_more_clusters_than_samples():
    # Preparar
    X = np.array([[0.0], [1.0]])

    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="n_clusters"):
        AgglomerativeClustering(n_clusters=3).fit(X)


def test_agglomerative_rejects_wrong_feature_count_in_predict():
    # Preparar
    model = AgglomerativeClustering(n_clusters=1).fit(np.array([[0.0], [1.0]]))

    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="misma forma"):
        model.predict(np.array([[0.0, 1.0]]))


def test_agglomerative_predict_before_fit_raises():
    # Preparar
    model = AgglomerativeClustering(n_clusters=2)

    # Ejecutar / Verificar
    with pytest.raises(ModelNotFittedError):
        model.predict(np.array([[0.0]]))
