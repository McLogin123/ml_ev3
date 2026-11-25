"""Proyección PCA en 2D para visualizar particiones"""

import numpy as np


def pca_project(X, centers=None, n_components=2):
    """Proyecta muestras y centroides con la misma transformación PCA

    Parametros:
        X : array-like
            Matriz de muestras (n, d)
        centers : array-like, optional
            Centroides en el mismo espacio que X. Se proyectan con la media y
            los componentes aprendidos de X, no con un PCA propio
        n_components : int, default=2
            Número de ejes devueltos (1 o 2). Si hay menos componentes, se
            rellena con ceros

    Retorna:
        tuple
            (muestras_proyectadas, centroides_proyectados, varianza_explicada)
            Los centroides proyectados son None si no se entregaron
    """
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2 or len(matrix) == 0 or matrix.shape[1] == 0:
        raise ValueError("X debe ser una matriz bidimensional no vacía.")
    if n_components not in (1, 2):
        raise ValueError("n_components debe ser 1 o 2.")

    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    total = float(np.sum(singular_values ** 2))
    ratios = np.zeros(n_components)
    if total > 0:
        ratios[:len(singular_values)] = singular_values[:n_components] ** 2 / total

    def _pad(values):
        """Añade ejes artificiales cuando la SVD devuelve menos componentes"""
        if values.shape[1] >= n_components:
            return values[:, :n_components]
        missing = n_components - values.shape[1]
        return np.column_stack([values, np.zeros((len(values), missing))])

    projected = _pad(centered @ components.T)
    projected_centers = None
    if centers is not None:
        centers_array = np.asarray(centers, dtype=float)
        if centers_array.ndim != 2 or centers_array.shape[1] != matrix.shape[1]:
            raise ValueError("centers debe tener forma (n_clusters, n_features).")
        projected_centers = _pad((centers_array - mean) @ components.T)

    return projected, projected_centers, np.asarray(ratios, dtype=float)
