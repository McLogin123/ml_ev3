"""Splines adaptativos tipo MARS para regresión

MARS (Multivariate Adaptive Regression Splines) construye una suma de funciones
de bisagra max(0, x - k) y max(0, k - x)
 
Aquí se usa la versión aditiva (sin términos de interacción), equivalente a un GAM con nodos elegidos automáticamente
"""

import numpy as np

from core.schema import TaskType
from models.base import BaseSupervisedModel


def _hinge(values, knot, sign):
    """Devuelve max(0, x-k) si sign=1 o max(0, k-x) si sign=-1"""
    if sign == 1:
        return np.maximum(0.0, values - knot)
    return np.maximum(0.0, knot - values)


class MARSRegressor(BaseSupervisedModel):
    """Regresión aditiva con bases de bisagra elegidas por búsqueda forward

    Parametros:
        max_terms : int, default=10
            Número máximo de términos contando la columna constante
        min_samples_leaf : int, default=5
            Mínimo de filas que deben quedar a cada lado de un nodo candidato

    Notes
    -----
    En cada paso se prueba un nodo por cada valor distinto de cada feature y
    se añade la base que reduce más la suma de errores cuadrados. 
    Los pesos se recalculan con lstsq porque las bases pueden ser redundantes
    """

    def __init__(self, max_terms=10, min_samples_leaf=5):
        super().__init__(TaskType.REGRESSION)
        if max_terms < 1:
            raise ValueError("max_terms debe ser al menos 1.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf debe ser al menos 1.")
        self.max_terms = max_terms
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        """Selecciona las bases de bisagra y ajusta sus coeficientes"""
        matrix = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")
        if labels.ndim != 1 or len(labels) != len(matrix):
            raise ValueError("y debe ser unidimensional y tener la longitud de X.")

        self.n_features_in_ = matrix.shape[1]
        self.basis_ = []
        self.used_knots_ = set()
        best_rss = float(np.sum((labels - np.mean(labels)) ** 2))
        self.coefficients_ = np.array([np.mean(labels)])

        while len(self.basis_) + 1 < self.max_terms:
            candidate, candidate_rss = self._best_candidate(matrix, labels)
            if candidate is None or candidate_rss >= best_rss:
                break
            self.basis_.append(candidate)
            self.used_knots_.add(candidate)
            best_rss = candidate_rss
            self.coefficients_ = np.linalg.lstsq(
                self._evaluate_basis(matrix), labels, rcond=None
            )[0]

        self.is_fitted = True
        return self

    def _best_candidate(self, X, y):
        """Devuelve ((feature, knot, sign), rss) o (None, inf)"""
        best_candidate = None
        best_rss = np.inf

        for feature_index in range(self.n_features_in_):
            values = X[:, feature_index]
            for knot in np.unique(values):
                for sign in (-1, 1):
                    candidate = (feature_index, float(knot), sign)
                    if candidate in self.used_knots_:
                        continue

                    column = _hinge(values, knot, sign)
                    mask = column > 0
                    if (
                        np.count_nonzero(mask) < self.min_samples_leaf
                        or np.count_nonzero(~mask) < self.min_samples_leaf
                    ):
                        continue

                    design = np.hstack(
                        [self._evaluate_basis(X), column.reshape(-1, 1)]
                    )
                    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
                    residuals = y - design @ coefficients
                    rss = float(residuals @ residuals)

                    if rss < best_rss:
                        best_rss = rss
                        best_candidate = candidate
        return best_candidate, best_rss

    def _evaluate_basis(self, X):
        """Construye la matriz de diseño: constante más bases seleccionadas"""
        columns = [np.ones(len(X))]
        for feature_index, knot, sign in self.basis_:
            columns.append(_hinge(X[:, feature_index], knot, sign).reshape(-1, 1))
        return np.column_stack(columns)

    def predict(self, X):
        """Devuelve la suma de las bases activas ponderadas"""
        self._check_is_fitted()
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError("X debe tener la misma forma que en fit().")
        return self._evaluate_basis(matrix) @ self.coefficients_


class GAMRegressor(BaseSupervisedModel):
    """Modelo aditivo generalizado con bases de bisagra en cuantiles fijos

    Parametros:
        n_knots : int, default=5
            Nodos por feature, repartidos entre los cuantiles 1/(n_knots+1)...n/(n_knots+1)

    Notes
    -----
    Es un GAM aditivo: la predicción es una constante más una suma de funciones
    de una sola feature. A diferencia de MARS, los nodos no se buscan con una
    pasada forward: se fijan con cuantiles de train y solo se ajustan los
    coeficientes por mínimos cuadrados
    """

    def __init__(self, n_knots=5):
        super().__init__(TaskType.REGRESSION)
        if n_knots < 1:
            raise ValueError("n_knots debe ser al menos 1.")
        self.n_knots = n_knots

    def fit(self, X, y):
        """Fija los nodos con cuantiles de train y ajusta los coeficientes"""
        matrix = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")
        if labels.ndim != 1 or len(labels) != len(matrix):
            raise ValueError("y debe ser unidimensional y tener la longitud de X.")

        self.n_features_in_ = matrix.shape[1]
        quantiles = np.linspace(0.0, 1.0, self.n_knots + 2)[1:-1]
        self.basis_ = []
        for feature_index in range(self.n_features_in_):
            for knot in np.quantile(matrix[:, feature_index], quantiles):
                self.basis_.append((feature_index, float(knot), 1))
        self.coefficients_ = np.linalg.lstsq(
            self._evaluate_basis(matrix), labels, rcond=None
        )[0]
        self.is_fitted = True
        return self

    def _evaluate_basis(self, X):
        """Construye la matriz de diseño: constante más todas las bisagras"""
        columns = [np.ones(len(X))]
        for feature_index, knot, sign in self.basis_:
            columns.append(_hinge(X[:, feature_index], knot, sign).reshape(-1, 1))
        return np.column_stack(columns)

    def predict(self, X):
        """Devuelve la suma de las contribuciones aditivas de cada feature"""
        self._check_is_fitted()
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError("X debe tener la misma forma que en fit().")
        return self._evaluate_basis(matrix) @ self.coefficients_
