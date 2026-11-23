"""Modelos basados en árboles de decisión implementados desde cero

Incluye un CART binario para regresión y clasificación, y un ensemble Bagging
que combina varios árboles entrenados con muestras bootstrap
"""

import numpy as np

from core.schema import TaskType
from models.base import BaseSupervisedModel


class _TreeNode:
    """Nodo interno de un árbol: guarda la regla de división y sus hijos"""

    def __init__(self, feature_index, threshold, left, right):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right


class _LeafNode:
    """Nodo terminal de un árbol: guarda una predicción constante"""

    def __init__(self, prediction):
        self.prediction = prediction


class DecisionTree(BaseSupervisedModel):
    """Árbol CART binario

    Parametros:
        task : TaskType
            REGRESSION o CLASSIFICATION
        max_depth : int, default=5
            Profundidad máxima. None elimina el límite
        min_samples_split : int, default=2
            Mínimo número de filas para intentar una división
        min_samples_leaf : int, default=1
            Mínimo número de filas en cada hoja
        max_features : int or None, default=None
            Número de features candidatas por división. None usa todas

    Notes
    -----
    Selecciona el mejor umbral entre puntos medios de valores consecutivos de cada feature
    En regresión minimiza MSE; en clasificación minimiza Gini
    """

    def __init__(
        self,
        task,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
    ):
        super().__init__(task)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def fit(self, X, y):
        """Construye el árbol recursivamente con los datos de entrenamiento"""
        matrix = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        if matrix.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")
        if labels.ndim != 1 or len(labels) != len(matrix):
            raise ValueError("y debe ser unidimensional y tener la longitud de X.")

        self.n_features_in_ = matrix.shape[1]
        self.classes_ = np.unique(labels) if self.task == TaskType.CLASSIFICATION else None
        self.root_ = self._build_tree(matrix, labels, depth=0)
        self.is_fitted = True
        return self

    def _build_tree(self, X, y, depth):
        """Crea una hoja o prueba la mejor división disponible"""
        prediction = self._leaf_prediction(y)
        limit_reached = (
            self.max_depth is not None and depth >= self.max_depth
        ) or len(y) < self.min_samples_split
        if limit_reached:
            return _LeafNode(prediction)

        best = self._best_split(X, y)
        if best is None:
            return _LeafNode(prediction)

        feature_index, threshold = best
        mask = X[:, feature_index] <= threshold
        left = self._build_tree(X[mask], y[mask], depth + 1)
        right = self._build_tree(X[~mask], y[~mask], depth + 1)
        return _TreeNode(feature_index, threshold, left, right)

    def _best_split(self, X, y):
        """Devuelve (feature, threshold) con menor impureza o None"""
        candidate_features = range(self.n_features_in_)
        if self.max_features is not None:
            if self.max_features < 1:
                raise ValueError("max_features debe ser al menos 1.")
            candidate_features = np.random.choice(
                self.n_features_in_,
                size=min(self.max_features, self.n_features_in_),
                replace=False,
            )

        best_gain = -np.inf
        best_split = None
        for feature_index in candidate_features:
            values = X[:, feature_index]
            ordered = np.unique(values)
            thresholds = (ordered[:-1] + ordered[1:]) / 2.0
            for threshold in thresholds:
                mask = values <= threshold
                left_count = int(np.count_nonzero(mask))
                right_count = len(y) - left_count
                if (
                    left_count < self.min_samples_leaf
                    or right_count < self.min_samples_leaf
                ):
                    continue

                current_impurity = self._impurity(y)
                left_impurity = self._impurity(y[mask])
                right_impurity = self._impurity(y[~mask])
                gain = current_impurity - (
                    left_count * left_impurity + right_count * right_impurity
                ) / len(y)

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, float(threshold))
        return best_split

    def _impurity(self, y):
        """MSE para regresión; Gini para clasificación"""
        if len(y) == 0:
            return 0.0
        if self.task == TaskType.REGRESSION:
            return float(np.mean((y.astype(float) - np.mean(y.astype(float))) ** 2))

        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return float(1.0 - np.sum(proportions**2))

    def _leaf_prediction(self, y):
        """Media en regresión; clase mayoritaria en clasificación"""
        if self.task == TaskType.REGRESSION:
            return float(np.mean(y.astype(float)))
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        """Recorre el árbol y devuelve la predicción de cada fila"""
        self._check_is_fitted()
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError("X debe tener la misma forma que en fit().")

        predictions = [_predict_row(row, self.root_) for row in matrix]
        if self.task == TaskType.REGRESSION:
            return np.asarray(predictions, dtype=float)
        return np.asarray(predictions)


def _predict_row(row, node):
    """Aplica recursivamente las reglas del árbol a una sola fila"""
    if isinstance(node, _LeafNode):
        return node.prediction
    if row[node.feature_index] <= node.threshold:
        return _predict_row(row, node.left)
    return _predict_row(row, node.right)


class BaggingClassifier(BaseSupervisedModel):
    """Ensemble de árboles entrenados con muestras bootstrap

    Parametros:
        n_estimators : int, default=10
            Número de árboles
        max_depth : int, default=5
            Profundidad máxima de cada árbol
        max_features : int, default=None
            Features candidatas por división en cada árbol
        random_state : int, default=None
            Semilla para las muestras bootstrap

    Notes
    -----
    Cada árbol ve una muestra con reposición del tamaño original

    La predicción final es el voto mayoritario; en empate gana la clase que aparece primero
    en classes_
    """

    def __init__(self, n_estimators=10, max_depth=5, max_features=None, random_state=None):
        super().__init__(TaskType.CLASSIFICATION)
        if n_estimators < 1:
            raise ValueError("n_estimators debe ser al menos 1.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        """Entrena cada árbol con un bootstrap distinto"""
        matrix = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        if matrix.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")
        if labels.ndim != 1 or len(labels) != len(matrix):
            raise ValueError("y debe ser unidimensional y tener la longitud de X.")

        generator = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(labels)
        self.n_features_in_ = matrix.shape[1]
        self.estimators_ = []
        for _ in range(self.n_estimators):
            indices = generator.integers(0, len(matrix), size=len(matrix))
            tree = DecisionTree(
                TaskType.CLASSIFICATION,
                max_depth=self.max_depth,
                max_features=self.max_features,
            )
            tree.fit(matrix[indices], labels[indices])
            self.estimators_.append(tree)

        self.is_fitted = True
        return self

    def predict(self, X):
        """Devuelve la clase con más votos entre los árboles"""
        self._check_is_fitted()
        predictions = np.column_stack(
            [tree.predict(X) for tree in self.estimators_]
        )

        final = []
        for row in predictions:
            votes = [int(np.count_nonzero(row == label)) for label in self.classes_]
            final.append(self.classes_[int(np.argmax(votes))])
        return np.asarray(final)
