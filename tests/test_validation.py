"""Pruebas de validación cruzada y búsqueda de hiperparámetros."""

import numpy as np
import pytest

from core.schema import TaskType
from evaluation.validation import KFold, StratifiedKFold, cross_validate, grid_search
from models.supervised.tree_models import DecisionTree


def test_stratified_kfold_conserva_clases_y_cubre_todas_las_filas():
    # Preparar
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # Ejecutar
    folds = list(StratifiedKFold(n_splits=4, random_state=1).split(np.zeros((8, 2)), y))

    # Verificar
    assert len(folds) == 4
    test_indices = np.concatenate([test for _, test in folds])
    np.testing.assert_array_equal(np.sort(test_indices), np.arange(8))
    for _, test in folds:
        assert sorted(y[test]) == [0, 1]


def test_kfold_cubre_todas_las_filas():
    # Ejecutar
    folds = list(KFold(n_splits=3, random_state=2).split(np.zeros((9, 2))))

    # Verificar
    test_indices = np.concatenate([test for _, test in folds])
    np.testing.assert_array_equal(np.sort(test_indices), np.arange(9))


def test_cross_validate_clasificacion_devuelve_scores():
    # Preparar
    X = np.array([[0.0], [0.1], [2.0], [2.1]] * 3)
    y = np.array([0, 0, 1, 1] * 3)

    # Ejecutar
    result = cross_validate(
        lambda: DecisionTree(TaskType.CLASSIFICATION, max_depth=2),
        X,
        y,
        TaskType.CLASSIFICATION,
        n_splits=3,
        random_state=4,
    )

    # Verificar
    assert len(result["scores"]) == 3
    assert result["mean"] == pytest.approx(1.0)
    assert result["std"] == pytest.approx(0.0)


def test_cross_validate_regresion_devuelve_r2():
    # Preparar
    X = np.linspace(0.0, 4.0, 20).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0

    # Ejecutar
    result = cross_validate(
        lambda: DecisionTree(TaskType.REGRESSION, max_depth=3),
        X,
        y,
        TaskType.REGRESSION,
        n_splits=4,
        random_state=5,
    )

    # Verificar
    assert len(result["scores"]) == 4
    assert result["mean"] > 0.8


def test_grid_search_elige_la_mejor_profundidad():
    # Preparar: XOR necesita profundidad 2; con max_depth=1 no se puede separar
    X = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float), (25, 1))
    y = np.tile(np.array([0, 1, 1, 0]), 25)

    # Ejecutar
    result = grid_search(
        lambda **params: DecisionTree(TaskType.CLASSIFICATION, **params),
        {"max_depth": [1, 3]},
        X,
        y,
        TaskType.CLASSIFICATION,
        n_splits=4,
        random_state=6,
    )

    # Verificar
    assert result["best_params"] == {"max_depth": 3}
    assert result["best_score"] == pytest.approx(1.0)
    assert len(result["results"]) == 2


def test_validacion_rechaza_folds_insuficientes():
    # Ejecutar / Verificar
    with pytest.raises(ValueError, match="n_splits"):
        StratifiedKFold(n_splits=1)
    with pytest.raises(ValueError, match="Cada clase"):
        list(
            StratifiedKFold(n_splits=3).split(
                np.zeros((3, 1)), np.array([0, 0, 1])
            )
        )
    with pytest.raises(ValueError, match="param_grid"):
        grid_search(lambda: None, {}, np.zeros((2, 1)), np.array([0, 1]), TaskType.CLASSIFICATION)
