"""Modelos lineales implementados con álgebra de NumPy

Incluye regresión lineal por mínimos cuadrados y regresión logística por descenso de gradiente.
Ninguno usa scikit-learn
"""

import numpy as np

from core.schema import TaskType
from models.base import BaseSupervisedModel


def _design_matrix(X, fit_intercept):
    """Convierte X en float y añade una columna de unos si procede"""
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("X debe ser una matriz bidimensional.")

    if fit_intercept:
        return np.hstack([np.ones((len(matrix), 1)), matrix])
    return matrix


def _check_target(y, n_rows):
    """Valida el vector de etiquetas y lo devuelve como float unidimensional"""
    labels = np.asarray(y, dtype=float)
    if labels.ndim != 1:
        raise ValueError("y debe ser unidimensional.")
    if len(labels) != n_rows:
        raise ValueError("X e y tienen distinto número de filas.")
    return labels


def _sigmoid(z):
    """Sigmoide recortada para evitar desbordamiento en exponenciales"""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LinearRegression(BaseSupervisedModel):
    """Regresión lineal por mínimos cuadrados

    Parametros:
        fit_intercept : bool, default=True
            Si es True, se añade una columna de unos para aprender el término
            independiente

    Notes
    -----
    Resuelve "w = pinv(X) · y". La seudoinversa tolera columnas redundantes,
    algo habitual cuando las categóricas se codifican con one-hot completo
    """

    def __init__(self, fit_intercept=True):
        super().__init__(TaskType.REGRESSION)
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """Estima los pesos minimizando la suma de errores cuadrados"""
        matrix = _design_matrix(X, self.fit_intercept)
        labels = _check_target(y, len(matrix))

        self.weights_ = np.linalg.pinv(matrix) @ labels
        self.intercept_ = float(self.weights_[0]) if self.fit_intercept else 0.0
        self.n_features_in_ = matrix.shape[1] - int(self.fit_intercept)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Devuelve la predicción numérica de cada fila"""
        self._check_is_fitted()
        matrix = _design_matrix(X, self.fit_intercept)
        if matrix.shape[1] != len(self.weights_):
            raise ValueError("X tiene distinto número de columnas que en fit().")
        return matrix @ self.weights_


class LogisticRegression(BaseSupervisedModel):
    """Regresión logística binaria entrenada con descenso de gradiente

    Parametros:
        learning_rate : float, default=0.1
            Tamaño del paso del gradiente
        n_iterations : int, default=1000
            Número de pasos de optimización
        fit_intercept : bool, default=True
            Si es True, se aprende también el término independiente

    Notes
    -----
    Las etiquetas deben estar codificadas como 0 y 1. Conviene que las features estén
    escaladas; el preprocesador ya lo hace por defecto
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000, fit_intercept=True):
        super().__init__(TaskType.CLASSIFICATION)

        if learning_rate <= 0:
            raise ValueError("learning_rate debe ser mayor que 0.")
        if n_iterations < 1:
            raise ValueError("n_iterations debe ser al menos 1.")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """Minimiza la entropía cruzada binaria con gradientes"""
        matrix = _design_matrix(X, self.fit_intercept)
        labels = _check_target(y, len(matrix))

        unique = np.unique(labels)
        if not np.isin(unique, [0, 1]).all():
            raise ValueError("La regresión logística requiere etiquetas 0 y 1.")

        self.weights_ = np.zeros(matrix.shape[1])
        self.loss_history_ = []
        n_rows = len(matrix)

        for _ in range(self.n_iterations):
            probabilities = _sigmoid(matrix @ self.weights_)
            clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
            loss = -np.mean(
                labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)
            )
            gradient = matrix.T @ (probabilities - labels) / n_rows
            self.weights_ -= self.learning_rate * gradient
            self.loss_history_.append(float(loss))

        self.intercept_ = float(self.weights_[0]) if self.fit_intercept else 0.0
        self.n_features_in_ = matrix.shape[1] - int(self.fit_intercept)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """Devuelve [P(clase 0), P(clase 1)] para cada fila"""
        self._check_is_fitted()
        matrix = _design_matrix(X, self.fit_intercept)
        if matrix.shape[1] != len(self.weights_):
            raise ValueError("X tiene distinto número de columnas que en fit().")

        positive = _sigmoid(matrix @ self.weights_)
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        """Clasifica con umbral 0.5 sobre la probabilidad de la clase 1"""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(int)
