"""Clustering jerárquico aglomerativo implementado con NumPy

Enlace completo y Ward
"""

import numpy as np

from models.base import BaseUnsupervisedModel


def _pairwise_distances(X):
    """Matriz de distancias euclídeas entre todas las filas de X"""
    difference = X[:, np.newaxis, :] - X[None, :, :]
    distances = np.sqrt(np.sum(difference**2, axis=2))
    np.fill_diagonal(distances, 0.0)
    return distances


def _cross_squared_distances(reference, X):
    """Distancias euclídeas al cuadrado de cada fila de X a cada reference"""
    difference = X[:, np.newaxis, :] - reference[np.newaxis, :, :]
    return np.sum(difference**2, axis=2)


class AgglomerativeClustering(BaseUnsupervisedModel):
    """Clustering jerárquico

    Parametros:
        n_clusters : int, default=2
            Número final de clusters
        linkage : {"complete", "ward"}, default="complete"
            Criterio para medir la distancia entre dos clusters

    Atributos:
        labels_ : numpy.ndarray
            Cluster asignado a cada muestra de entrenamiento
        linkage_matrix_ : numpy.ndarray
            Fusiones con forma (n_samples - n_clusters, 4). Cada fila es [id_a, id_b, distancia, n_muestras].
            Los ids menores que n_samples son muestras originales; los demás son fusiones nuevas

    Notes
    -----
    En predict no existe una regla natural para muestras nuevas.
    Se asigna la etiqueta de la muestra de entrenamiento más cercana; es una aproximación
    explícita y no recomputa la jerarquía
    """

    def __init__(self, n_clusters=2, linkage="complete"):
        super().__init__()
        if n_clusters < 1:
            raise ValueError("n_clusters debe ser al menos 1.")
        valid_linkages = ("complete", "ward")
        if linkage not in valid_linkages:
            raise ValueError(f"linkage debe ser uno de {valid_linkages}.")

        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        """Construye la jerarquía y devuelve las etiquetas finales"""
        matrix = self._check_X(X)
        n_samples = len(matrix)
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters no puede superar el número de muestras.")

        distances = _pairwise_distances(matrix)
        clusters = {index: [index] for index in range(n_samples)}
        next_id = n_samples
        merges = []

        while len(clusters) > self.n_clusters:
            ids = sorted(clusters)
            best_value = np.inf
            best_pair = None
            for position_a in range(len(ids)):
                for position_b in range(position_a + 1, len(ids)):
                    id_a, id_b = ids[position_a], ids[position_b]
                    value = self._cluster_distance(
                        distances, matrix, clusters[id_a], clusters[id_b]
                    )
                    if value < best_value:
                        best_value = value
                        best_pair = (id_a, id_b)

            id_a, id_b = best_pair
            merged = clusters.pop(id_a) + clusters.pop(id_b)
            clusters[next_id] = merged
            merges.append([id_a, id_b, best_value, len(merged)])
            next_id += 1

        labels = np.empty(n_samples, dtype=int)
        for label, members in enumerate(clusters.values()):
            labels[members] = label

        self.linkage_matrix_ = (
            np.asarray(merges, dtype=float)
            if merges
            else np.empty((0, 4), dtype=float)
        )
        self.labels_ = labels
        self.X_train_ = matrix.copy()
        self.n_features_in_ = matrix.shape[1]
        self.is_fitted = True
        return self

    @staticmethod
    def _check_X(X):
        """Valida la matriz de entrada y la devuelve como float"""
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")
        if len(matrix) == 0:
            raise ValueError("X debe contener al menos una muestra.")
        return matrix

    def _cluster_distance(self, distances, X, members_a, members_b):
        """Distancia entre dos clusters según el enlace elegido"""
        if self.linkage == "ward":
            mean_a = X[members_a].mean(axis=0)
            mean_b = X[members_b].mean(axis=0)
            difference = mean_a - mean_b
            factor = len(members_a) * len(members_b) / (len(members_a) + len(members_b))
            return float(factor * np.dot(difference, difference))

        subset = distances[np.ix_(members_a, members_b)]
        return float(np.max(subset))

    def predict(self, X):
        """Asigna la clase del ejemplo de entrenamiento más cercano"""
        self._check_is_fitted()
        matrix = self._check_X(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("X debe tener la misma forma que en fit().")

        distances = _cross_squared_distances(self.X_train_, matrix)
        nearest = np.argmin(distances, axis=1)
        return self.labels_[nearest]

    def fit_predict(self, X):
        """Entrena y devuelve la etiqueta de cluster de cada fila"""
        return self.fit(X).labels_
