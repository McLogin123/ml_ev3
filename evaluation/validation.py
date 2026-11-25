"""Validación cruzada y búsqueda de hiperparámetros sin scikit-learn

Las particiones se hacen sobre el conjunto de entrenamiento: el conjunto de
prueba nunca interviene en la selección del modelo
"""

import itertools

import numpy as np

from core.schema import TaskType
from evaluation.metrics import accuracy_score, r2_score


class StratifiedKFold:
    """Divide en folds conservando la proporción de clases

    Parametros:
        n_splits : int, default=5
            Número de particiones
        shuffle : bool, default=True
            Si se mezclan las filas antes de repartirlas
        random_state : int, default=None
            Semilla de la mezcla
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits debe ser al menos 2.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        """Genera pares (índices_de_train, índices_de_test)"""
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError("y debe ser unidimensional.")
        if len(labels) != len(X):
            raise ValueError("X e y tienen distinto número de filas.")
        _, counts = np.unique(labels, return_counts=True)
        if np.min(counts) < self.n_splits:
            raise ValueError("Cada clase debe tener al menos n_splits muestras.")

        generator = np.random.default_rng(self.random_state)
        folds = [[] for _ in range(self.n_splits)]
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            if self.shuffle:
                generator.shuffle(indices)
            for position, index in enumerate(indices):
                folds[position % self.n_splits].append(int(index))

        all_indices = np.arange(len(labels))
        for test in folds:
            test_indices = np.asarray(test, dtype=int)
            yield np.setdiff1d(all_indices, test_indices), test_indices


class KFold:
    """Divide en folds sin tener en cuenta etiquetas; se usa en regresión"""

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits debe ser al menos 2.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        """Genera pares (índices_de_train, índices_de_test)"""
        n_rows = len(X)
        if self.n_splits > n_rows:
            raise ValueError("n_splits no puede superar el número de muestras.")
        indices = np.arange(n_rows)
        if self.shuffle:
            np.random.default_rng(self.random_state).shuffle(indices)
        all_indices = np.arange(n_rows)
        for test_indices in np.array_split(indices, self.n_splits):
            yield np.setdiff1d(all_indices, test_indices), test_indices


def cross_validate(factory, X, y, task, n_splits=5, random_state=None):
    """Evalúa un modelo nuevo en cada fold y devuelve las puntuaciones

    factory debe ser una función sin argumentos que devuelva un modelo virgen
    """
    matrix = np.asarray(X, dtype=float)
    labels = np.asarray(y)
    splitter_class = StratifiedKFold if TaskType(task) == TaskType.CLASSIFICATION else KFold
    splitter = splitter_class(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    scores = []
    for train_indices, test_indices in splitter.split(matrix, labels):
        model = factory().fit(matrix[train_indices], labels[train_indices])
        predictions = model.predict(matrix[test_indices])
        if TaskType(task) == TaskType.CLASSIFICATION:
            scores.append(float(accuracy_score(labels[test_indices], predictions)))
        else:
            scores.append(float(r2_score(labels[test_indices], predictions)))
    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }


def grid_search(factory, param_grid, X, y, task, n_splits=3, random_state=None):
    """Prueba todas las combinaciones de hiperparámetros con validación cruzada

    factory debe aceptar los parámetros como argumentos nombrados y devolver
    un modelo virgen. Devuelve la mejor combinación y el resultado de cada una
    """
    if not param_grid:
        raise ValueError("param_grid no puede estar vacío.")
    names = list(param_grid)
    results = []
    for values in itertools.product(*[param_grid[name] for name in names]):
        params = dict(zip(names, values))
        scores = cross_validate(
            lambda: factory(**params),
            X,
            y,
            task,
            n_splits=n_splits,
            random_state=random_state,
        )
        results.append({"params": params, "mean": scores["mean"], "std": scores["std"]})

    results.sort(key=lambda row: row["mean"], reverse=True)
    return {
        "best_params": results[0]["params"],
        "best_score": results[0]["mean"],
        "results": results,
    }
