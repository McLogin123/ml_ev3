"""K-Means implementado con NumPy"""

from models.base import BaseUnsupervisedModel
import numpy as np

def _squared_distances(reference, X):
    """Distancias euclídeas al cuadrado de cada fila de X a cada reference"""
    difference = X[:, np.newaxis, :] - reference[np.newaxis, :, :]
    return np.sum(difference**2, axis=2)


def _initial_centers(X, n_clusters, generator):
    """Elige centros iniciales con el procedimiento k-means++"""
    n_samples = len(X)
    chosen = [int(generator.integers(n_samples))]
    closest = _squared_distances(X[chosen], X).ravel()

    for _ in range(1, n_clusters):
        total = float(np.sum(closest))
        if total > 0:
            probabilities = closest / total
            index = int(generator.choice(n_samples, p=probabilities))
        else:
            # Todos los puntos son iguales a un centro ya elegido.
            remaining = np.setdiff1d(np.arange(n_samples), np.array(chosen))
            index = (
                int(generator.choice(remaining))
                if len(remaining) > 0
                else int(generator.integers(n_samples))
            )

        chosen.append(index)
        distances_to_new = _squared_distances(X[[index]], X).ravel()
        closest = np.minimum(closest, distances_to_new)

    return X[chosen].copy()


class KMeans(BaseUnsupervisedModel):
    """Clustering por k-medias con iteraciones de Lloyd

    Parametros:
        n_clusters : int
            Número de clusters buscados
        max_iter : int, default=100
            Número máximo de iteraciones de Lloyd por repetición
        tol : float, default=1e-4
            Criterio de parada: se detiene cuando el mayor movimiento de los centros
            es menor o igual que este valor
        n_init : int, default=10
            Número de inicializaciones. Se conserva la de menor inercia
        random_state : int, default=None
            Semilla para la selección de centros y el manejo de clusters vacíos

    Notes
    -----
    La inicialización es k-means++:
        El primer centro es aleatorio y cada centro siguiente se elige con probabilidad
        proporcional a su distancia al centro más cercano ya seleccionado. Esto reduce el
        riesgo de caer en un mínimo local pobre

    Si un cluster queda vacío, su centro se reubica en el punto más alejado de
    su propio centro asignado. Así el algoritmo no produce centros NaN
    """

    requires_scaling = True

    def __init__(self, n_clusters, max_iter=100, tol=1e-4, n_init=10, random_state=None):
        super().__init__()
        if n_clusters < 1:
            raise ValueError("n_clusters debe ser al menos 1.")
        if max_iter < 1:
            raise ValueError("max_iter debe ser al menos 1.")
        if tol < 0:
            raise ValueError("tol no puede ser negativo.")
        if n_init < 1:
            raise ValueError("n_init debe ser al menos 1.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X):
        """Entrena el modelo y conserva la mejor repetición"""
        matrix = self._check_X(X)
        if self.n_clusters > len(matrix):
            raise ValueError("n_clusters no puede superar el número de muestras.")

        generator = np.random.default_rng(self.random_state)
        best_inertia = np.inf
        best_result = None

        for _ in range(self.n_init):
            centers = _initial_centers(matrix, self.n_clusters, generator)
            labels, centers, inertia, n_iter = self._lloyd(matrix, centers)
            if inertia < best_inertia:
                best_inertia = inertia
                best_result = (centers, labels, n_iter)

        centers, labels, n_iter = best_result
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = n_iter
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

    def _lloyd(self, X, centers):
        """Alterna asignación y actualización hasta convergencia"""
        n_iter = 0
        for iteration in range(1, self.max_iter + 1):
            distances = _squared_distances(centers, X)
            labels = np.argmin(distances, axis=1)

            new_centers = centers.copy()
            distances_to_assigned = distances[np.arange(len(X)), labels]
            for cluster in range(self.n_clusters):
                mask = labels == cluster
                if np.any(mask):
                    new_centers[cluster] = np.mean(X[mask], axis=0)
                else:
                    farthest = int(np.argmax(distances_to_assigned))
                    new_centers[cluster] = X[farthest].copy()

            movement = np.max(np.linalg.norm(new_centers - centers, axis=1))
            centers = new_centers
            n_iter = iteration
            if movement <= self.tol:
                break

        final_distances = _squared_distances(centers, X)
        labels = np.argmin(final_distances, axis=1)
        inertia = float(np.sum(final_distances[np.arange(len(X)), labels]))
        return labels, centers, inertia, n_iter

    def predict(self, X):
        """Asigna cada fila al cluster con el centro más cercano"""
        self._check_is_fitted()
        matrix = self._check_X(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("X debe tener la misma forma que en fit().")

        distances = _squared_distances(self.cluster_centers_, matrix)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """Entrena y devuelve la etiqueta de cluster de cada fila"""
        return self.fit(X).labels_
