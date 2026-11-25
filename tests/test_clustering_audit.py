"""Pruebas de la auditoría de particionamiento."""

import numpy as np
import pytest

from evaluation.clustering_audit import audit_partitioning, find_elbow


def _three_clusters():
    """Nube con tres grupos bien separados"""
    rng = np.random.default_rng(0)
    return np.vstack([
        rng.normal([0.0, 0.0], 0.1, (20, 2)),
        rng.normal([5.0, 5.0], 0.1, (20, 2)),
        rng.normal([0.0, 5.0], 0.1, (20, 2)),
    ])


def test_find_elbow_detects_kneedle_point():
    # Ejecutar
    result = find_elbow(range(1, 7), [100.0, 40.0, 20.0, 15.0, 13.0, 12.0])

    # Verificar
    assert result == 3


def test_find_elbow_handles_flat_curve_and_bad_inputs():
    # Ejecutar / Verificar
    assert find_elbow([2, 3, 4], [5.0, 5.0, 5.0]) is None
    with pytest.raises(ValueError, match="al menos 3 puntos"):
        find_elbow([2, 3], [5.0, 4.0])
    with pytest.raises(ValueError, match="misma longitud"):
        find_elbow([2, 3, 4], [5.0, 4.0])


def test_audit_partitioning_reports_every_index():
    # Preparar
    X = _three_clusters()

    # Ejecutar
    audit = audit_partitioning(X, [2, 3, 4], seed=42)

    # Verificar
    assert audit["k_values"] == [2, 3, 4]
    for key in (
        "inertias", "silhouette", "calinski_harabasz", "davies_bouldin"
    ):
        assert len(audit[key]) == 3
    assert audit["inertias"] == sorted(audit["inertias"], reverse=True)
    assert audit["selected_k"] == 3
    assert audit["selection_rule"] == "silhouette"
    assert audit["elbow_k"] in (2, 3, 4, None)
    assert all(value is not None for value in audit["silhouette"])


def test_audit_partitioning_converts_infinite_indices_to_none():
    # Preparar: dos grupos sin dispersión interna dan CH infinito.
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    # Ejecutar
    audit = audit_partitioning(X, [2], seed=42)

    # Verificar
    assert audit["calinski_harabasz"] == [None]
    assert audit["davies_bouldin"] == [0.0]
    assert audit["silhouette"] == [1.0]


def test_audit_partitioning_skips_indices_of_single_cluster():
    # Ejecutar: datos constantes producen un solo clúster para todo k.
    audit = audit_partitioning(np.zeros((6, 2)), [2, 3], seed=42)

    # Verificar
    assert audit["silhouette"] == [None, None]
    assert audit["selected_k"] == 2
    assert len(audit["warnings"]) == 2


def test_audit_partitioning_rejects_bad_matrix():
    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="bidimensional"):
        audit_partitioning(np.array([1.0, 2.0]), [2])
