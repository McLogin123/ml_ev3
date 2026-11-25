"""Pruebas de la proyección PCA en 2D."""

import numpy as np
import pytest

from models.unsupervised import pca_project


def test_pca_project_uses_same_components_for_centers():
    # Preparar
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    centers = np.array([[1.0, 1.0], [3.0, 3.0]])

    # Ejecutar
    projected, projected_centers, ratios = pca_project(X, centers)

    # Verificar
    assert projected.shape == (4, 2)
    assert projected_centers.shape == (2, 2)
    assert ratios[0] == pytest.approx(1.0)
    assert ratios[1] == pytest.approx(0.0)
    assert np.allclose(projected[:, 1], 0.0, atol=1e-12)
    assert np.allclose(projected_centers[0], projected[1], atol=1e-12)


def test_pca_project_pads_second_axis_with_one_feature():
    # Ejecutar
    projected, centers, ratios = pca_project(
        np.array([[0.0], [1.0], [2.0]]), np.array([[1.0]])
    )

    # Verificar
    assert projected.shape == (3, 2)
    assert centers.shape == (1, 2)
    assert np.allclose(projected[:, 1], 0.0)
    assert np.allclose(centers[:, 1], 0.0)
    assert ratios[0] == pytest.approx(1.0)


def test_pca_project_without_centers():
    # Ejecutar
    projected, centers, ratios = pca_project(np.array([[0.0, 1.0], [2.0, 3.0]]))

    # Verificar
    assert projected.shape == (2, 2)
    assert centers is None
    assert sum(ratios) == pytest.approx(1.0)


def test_pca_project_with_one_sample_still_returns_two_axes():
    # Ejecutar
    projected, centers, ratios = pca_project(
        np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]])
    )

    # Verificar
    assert projected.shape == (1, 2)
    assert centers.shape == (1, 2)
    assert len(ratios) == 2


def test_pca_project_rejects_bad_input():
    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="bidimensional"):
        pca_project(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="centers debe tener forma"):
        pca_project(np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([[1.0]]))
    with pytest.raises(ValueError, match="n_components"):
        pca_project(np.array([[0.0, 1.0], [2.0, 3.0]]), n_components=3)
