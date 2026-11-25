"""Modelos basados en prototipos: k-NN, LVQ1 y DANN

Todos están implementados con álgebra de NumPy
DANN adapta la métrica con una covarianza local elipsoidal Sigma(x0)
"""

import numpy as np

from core.schema import TaskType
from models.base import BaseSupervisedModel


def _distances(reference, X):
    """Distancias euclídeas de cada fila de X a cada fila de reference"""
    difference = X[:, np.newaxis, :] - reference[np.newaxis, :, :]
    return np.sqrt(np.sum(difference**2, axis=2))


class KNeighborsModel(BaseSupervisedModel):
    """Vecinos más cercanos para clasificación o regresión

    Parametros:
        task : TaskType
            "CLASSIFICATION" o "REGRESSION"
        n_neighbors : int, default=5
            Número de vecinos que votan o se promedian
        weights : {"distance", "uniform"}, default="distance"
            "distance" pesa cada vecino con la inversa de su distancia euclidiana;
            "uniform" da el mismo peso a todos

    Notes
    -----
    No hay entrenamiento: fit solo guarda la matriz y las etiquetas.
    En empate de clasificación gana la clase menor según np.unique
    """

    requires_scaling = True

    def __init__(self, task, n_neighbors=5, weights="distance"):
        super().__init__(task)
        if n_neighbors < 1:
            raise ValueError("n_neighbors debe ser al menos 1.")
        if weights not in ("distance", "uniform"):
            raise ValueError("weights debe ser 'distance' u 'uniform'.")
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        """Guarda los ejemplos de entrenamiento"""
        matrix = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        if matrix.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")
        if labels.ndim != 1 or len(labels) != len(matrix):
            raise ValueError("y debe ser unidimensional y tener la longitud de X.")
        if self.n_neighbors > len(matrix):
            raise ValueError("n_neighbors no puede superar el número de muestras.")

        self.X_train_ = matrix.copy()
        self.y_train_ = labels.copy()
        self.n_features_in_ = matrix.shape[1]
        self.classes_ = np.unique(labels) if self.task == TaskType.CLASSIFICATION else None
        self.is_fitted = True
        return self

    def predict(self, X):
        """Predice usando los vecinos más cercanos a cada fila"""
        self._check_is_fitted()
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError("X debe tener la misma forma que en fit().")

        distances = _distances(self.X_train_, matrix)
        neighbor_indices = np.argsort(distances, axis=1, kind="stable")[
            :, : self.n_neighbors
        ]
        neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
        neighbor_labels = self.y_train_[neighbor_indices]

        result = []
        for labels, row_distances in zip(neighbor_labels, neighbor_distances):
            if self.weights == "distance" and np.all(row_distances > 0):
                row_weights = 1.0 / row_distances
            else:
                row_weights = np.ones(len(labels))
            if self.task == TaskType.REGRESSION:
                result.append(
                    float(np.average(labels.astype(float), weights=row_weights))
                )
            else:
                votes = [
                    float(np.sum(row_weights[labels == label]))
                    for label in self.classes_
                ]
                result.append(self.classes_[int(np.argmax(votes))])
        return np.asarray(result, dtype=float if self.task == TaskType.REGRESSION else None)


class LVQ1(BaseSupervisedModel):
    """Learning Vector Quantization 1: prototipos que compiten por cada fila

    Parametros:
        n_prototypes_per_class : int, default=1
            Prototipos iniciales por clase
        learning_rate : float, default=0.1
            Tamaño del movimiento del prototipo ganador
        n_epochs : int, default=50
            Pasadas completas sobre los datos
        random_state : int, default=None
            Semilla para elegir y recorrer ejemplos

    Notes
    -----
    Si el prototipo ganador tiene la clase correcta se acerca a la muestra;
    si no, se aleja. La regla es p <- p ± eta (x - p)
    """

    requires_scaling = True

    def __init__(
        self,
        n_prototypes_per_class=1,
        learning_rate=0.1,
        n_epochs=50,
        random_state=None,
    ):
        super().__init__(TaskType.CLASSIFICATION)
        if n_prototypes_per_class < 1:
            raise ValueError("n_prototypes_per_class debe ser al menos 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate debe ser mayor que 0.")
        if n_epochs < 1:
            raise ValueError("n_epochs debe ser al menos 1.")

        self.n_prototypes_per_class = n_prototypes_per_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        """Inicializa prototipos y aplica la regla LVQ1 época por época"""
        matrix = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        if matrix.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")
        if labels.ndim != 1 or len(labels) != len(matrix):
            raise ValueError("y debe ser unidimensional y tener la longitud de X.")

        generator = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(labels)
        self.n_features_in_ = matrix.shape[1]
        prototypes = []
        prototype_labels = []
        for label in self.classes_:
            indices = np.flatnonzero(labels == label)
            chosen = generator.choice(
                indices, size=self.n_prototypes_per_class, replace=True
            )
            for index in chosen:
                prototypes.append(matrix[index].copy())
                prototype_labels.append(label)

        self.prototypes_ = np.asarray(prototypes, dtype=float)
        self.prototype_labels_ = np.asarray(prototype_labels)

        for _ in range(self.n_epochs):
            order = generator.permutation(len(matrix))
            for index in order:
                sample = matrix[index]
                distances = np.linalg.norm(self.prototypes_ - sample, axis=1)
                winner = int(np.argmin(distances))
                direction = 1.0 if prototype_labels[winner] == labels[index] else -1.0
                self.prototypes_[winner] += (
                    direction * self.learning_rate * (sample - self.prototypes_[winner])
                )

        self.is_fitted = True
        return self

    def predict(self, X):
        """Asigna la clase del prototipo más cercano"""
        self._check_is_fitted()
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError("X debe tener la misma forma que en fit().")

        distances = _distances(self.prototypes_, matrix)
        winners = np.argmin(distances, axis=1)
        return self.prototype_labels_[winners]


class DANN(BaseSupervisedModel):
    """Discriminant Adaptive Nearest-Neighbor con métrica elipsoidal local

    Parametros:
        n_neighbors : int, default=5
            Vecinos usados para estimar la covarianza local Sigma(x0)
        n_votes : int, default=3
            Vecinos finales que votan con la métrica adaptada
        shrinkage : float, default=0.01
            Fracción del autovalor mayor que se añade a todos los autovalores
            para evitar una matriz singular

    Notes
    -----
    Para cada x0: se toman sus k vecinos euclídeos, se calcula la covarianza
    local y se invierte con contracción de autovalores. Esa métrica Sigma(x0)
    estira el espacio en las direcciones de poca varianza y lo comprime en las
    de mucha, de modo que la vecindad final sea elipsoidal en vez de esférica.
    La clase se decide por voto ponderado por la inversa de la distancia.
    """

    requires_scaling = True

    def __init__(self, n_neighbors=5, n_votes=3, shrinkage=0.01):
        super().__init__(TaskType.CLASSIFICATION)
        if n_neighbors < 2:
            raise ValueError("n_neighbors debe ser al menos 2.")
        if n_votes < 1:
            raise ValueError("n_votes debe ser al menos 1.")
        if shrinkage < 0:
            raise ValueError("shrinkage debe ser mayor o igual que 0.")

        self.n_neighbors = n_neighbors
        self.n_votes = n_votes
        self.shrinkage = shrinkage

    def fit(self, X, y):
        """Guarda los ejemplos etiquetados; la métrica se calcula al predecir"""
        matrix = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        if matrix.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")
        if labels.ndim != 1 or len(labels) != len(matrix):
            raise ValueError("y debe ser unidimensional y tener la longitud de X.")
        if self.n_neighbors > len(matrix):
            raise ValueError("n_neighbors no puede superar el número de muestras.")

        self.X_train_ = matrix.copy()
        self.y_train_ = labels.copy()
        self.classes_ = np.unique(labels)
        self.n_features_in_ = matrix.shape[1]
        self.is_fitted = True
        return self

    def local_metric(self, x0):
        """Devuelve la matriz elipsoidal Sigma(x0) de tamaño n_features"""
        self._check_is_fitted()
        point = np.asarray(x0, dtype=float)
        if point.shape != (self.n_features_in_,):
            raise ValueError("x0 debe tener la misma longitud que las features.")

        distances = _distances(self.X_train_, point.reshape(1, -1)).ravel()
        neighbors = np.argsort(distances, kind="stable")[: self.n_neighbors]
        local = self.X_train_[neighbors]
        covariance = np.atleast_2d(np.cov(local, rowvar=False))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        ridge = max(float(np.max(eigenvalues)), 0.0) * self.shrinkage + 1e-12
        inverse = 1.0 / (np.maximum(eigenvalues, 0.0) + ridge)
        return (eigenvectors * inverse) @ eigenvectors.T

    def predict_proba(self, X):
        """Devuelve probabilidades de clase según el voto de la vecindad adaptada"""
        self._check_is_fitted()
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError("X debe tener la misma forma que en fit().")

        probabilities = np.zeros((len(matrix), len(self.classes_)), dtype=float)
        for row_index, point in enumerate(matrix):
            metric = self.local_metric(point)
            difference = self.X_train_ - point
            quadratic = np.einsum("ij,jk,ik->i", difference, metric, difference)
            distances = np.sqrt(np.maximum(quadratic, 0.0))
            nearest = np.argsort(distances, kind="stable")[: self.n_votes]
            weights = 1.0 / np.maximum(distances[nearest], 1e-12)
            for class_index, label in enumerate(self.classes_):
                mask = self.y_train_[nearest] == label
                probabilities[row_index, class_index] = np.sum(weights[mask])
        return probabilities / np.maximum(
            probabilities.sum(axis=1, keepdims=True), 1e-12
        )

    def predict(self, X):
        """Devuelve la clase con mayor probabilidad"""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
