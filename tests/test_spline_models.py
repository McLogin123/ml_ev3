"""Pruebas del modelo MARS aditivo."""

import numpy as np
import pytest

from models.base import ModelNotFittedError
from models.supervised.spline_models import GAMRegressor, MARSRegressor


def test_mars_represents_absolute_value():
    # Preparar
    X = np.linspace(-3.0, 3.0, 13).reshape(-1, 1)
    y = np.abs(X.ravel())

    # Ejecutar
    model = MARSRegressor(max_terms=7, min_samples_leaf=2).fit(X, y)

    # Verificar
    np.testing.assert_allclose(model.predict(X), y, atol=1e-10)
    assert (0, 0.0, 1) in model.basis_
    assert (0, 0.0, -1) in model.basis_


def test_mars_with_one_term_predicts_mean():
    # Preparar
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 3.0, 5.0])

    # Ejecutar
    model = MARSRegressor(max_terms=1).fit(X, y)

    # Verificar
    assert model.basis_ == []
    np.testing.assert_allclose(model.predict(X), [3.0, 3.0, 3.0])


def test_mars_approximates_smooth_curve():
    # Preparar
    X = np.linspace(0.0, 5.0, 25).reshape(-1, 1)
    y = np.sin(X.ravel())

    # Ejecutar
    model = MARSRegressor(max_terms=8, min_samples_leaf=2).fit(X, y)

    # Verificar
    assert model.score(X, y) > 0.95
    assert len(model.basis_) > 0


def test_gam_approximates_additive_function():
    # Preparar: la salida es suma de una curva en x0 y una recta en x1
    x0 = np.linspace(0.0, 5.0, 40)
    x1 = np.linspace(-2.0, 2.0, 40)
    X = np.column_stack([x0, x1])
    y = np.sin(x0) + 2.0 * x1

    # Ejecutar
    model = GAMRegressor(n_knots=6).fit(X, y)

    # Verificar
    assert model.score(X, y) > 0.95
    assert len(model.basis_) == 12


def test_gam_rejects_invalid_n_knots():
    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="n_knots"):
        GAMRegressor(n_knots=0)


def test_mars_rejects_invalid_max_terms():
    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="max_terms"):
        MARSRegressor(max_terms=0)


def test_mars_predict_before_fit_raises():
    # Preparar
    model = MARSRegressor()

    # Ejecutar / Verificar
    with pytest.raises(ModelNotFittedError):
        model.predict(np.array([[1.0]]))
