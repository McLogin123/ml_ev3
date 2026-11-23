"""Modelos basados en prototipos: k-NN, LVQ1 y DANN

Todos están implementados con álgebra de NumPy
DANN incluye una versión explicitamente derivada del gradiente inverso (gradient reversal)
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

    Notes
    -----
    No hay entrenamiento: fit solo guarda la matriz y las etiquetas. 
    En empate de clasificación gana la clase menor según np.unique
    """

    def __init__(self, task, n_neighbors=5):
        super().__init__(task)
        if n_neighbors < 1:
            raise ValueError("n_neighbors debe ser al menos 1.")
        self.n_neighbors = n_neighbors

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
        neighbor_labels = self.y_train_[neighbor_indices]

        if self.task == TaskType.REGRESSION:
            return np.mean(neighbor_labels.astype(float), axis=1)

        result = []
        for row in neighbor_labels:
            votes = [int(np.count_nonzero(row == label)) for label in self.classes_]
            result.append(self.classes_[int(np.argmax(votes))])
        return np.asarray(result)


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
    """Domain-Adversarial Neural Network entrenada con NumPy

    Parametros:
        hidden_size : int, default=10
            Neuronas de la capa oculta compartida
        learning_rate : float, default=0.01
            Tamaño del paso de gradiente
        n_epochs : int, default=200
            Pasadas de gradiente completo sobre todos los datos
        domain_weight : float, default=1.0
            Factor lambda del gradiente inverso
        random_state : int, default=None
            Semilla para los pesos iniciales

    Notes
    -----
    fit(X, y, domains) recibe todas las muestras (fuente y objetivo),
    etiquetas de clase con -1 en las objetivo no etiquetadas y un vector domains con 0=fuente, 1=objetivo

    La capa compartida h = tanh(X W1 + b1) alimenta dos salidas: el
    clasificador de etiqueta y el clasificador de dominio

    En la retropropagación el gradiente del dominio se multiplica por -domain_weight antes de sumarse
    al gradiente de etiqueta; así el extractor aprende representaciones que confunden al
    clasificador de dominio sin dañar la clasificación
    """

    def __init__(
        self,
        hidden_size=10,
        learning_rate=0.01,
        n_epochs=200,
        domain_weight=1.0,
        random_state=None,
    ):
        super().__init__(TaskType.CLASSIFICATION)
        if hidden_size < 1:
            raise ValueError("hidden_size debe ser al menos 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate debe ser mayor que 0.")
        if n_epochs < 1:
            raise ValueError("n_epochs debe ser al menos 1.")

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.domain_weight = domain_weight
        self.random_state = random_state

    def fit(self, X, y, domains):
        """Entrena de forma conjunta el clasificador y el adversario de dominio"""
        matrix = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=int)
        domain_labels = np.asarray(domains, dtype=int)
        if matrix.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")
        if labels.ndim != 1 or domain_labels.ndim != 1:
            raise ValueError("y y domains deben ser unidimensionales.")
        if len(labels) != len(matrix) or len(domain_labels) != len(matrix):
            raise ValueError("X, y y domains deben tener la misma longitud.")
        if not np.isin(domain_labels, [0, 1]).all():
            raise ValueError("domains solo puede contener 0 (fuente) o 1 (objetivo).")

        labeled = labels >= 0
        if not np.any(labeled):
            raise ValueError("Debe haber al menos una muestra etiquetada (y >= 0).")
        n_rows, n_features = matrix.shape
        n_classes = int(np.max(labels[labeled])) + 1
        if not np.array_equal(np.unique(labels[labeled]), np.arange(n_classes)):
            raise ValueError("Las etiquetas deben estar codificadas desde 0 sin huecos.")
        generator = np.random.default_rng(self.random_state)
        scale = np.sqrt(1.0 / n_features)

        self.classes_ = np.arange(n_classes)
        self.n_features_in_ = n_features
        self.W1_ = generator.normal(0, scale, (n_features, self.hidden_size))
        self.b1_ = np.zeros(self.hidden_size)
        self.Wy_ = generator.normal(
            0, np.sqrt(1.0 / self.hidden_size), (self.hidden_size, n_classes)
        )
        self.by_ = np.zeros(n_classes)
        self.Wd_ = generator.normal(
            0, np.sqrt(1.0 / self.hidden_size), (self.hidden_size, 1)
        )
        self.bd_ = np.zeros(1)
        self.loss_history_ = []

        one_hot = np.zeros((n_rows, n_classes))
        one_hot[labeled, labels[labeled]] = 1.0

        for _ in range(self.n_epochs):
            hidden_input = matrix @ self.W1_ + self.b1_
            hidden = np.tanh(hidden_input)

            label_logits = hidden @ self.Wy_ + self.by_
            probabilities = self._softmax(label_logits)

            domain_logits = (hidden @ self.Wd_ + self.bd_).ravel()
            domain_probabilities = self._sigmoid(domain_logits)

            clipped_label = np.clip(probabilities, 1e-12, 1.0)
            clipped_domain = np.clip(domain_probabilities, 1e-12, 1 - 1e-12)
            label_loss = -np.mean(
                np.sum(one_hot[labeled] * np.log(clipped_label[labeled]), axis=1)
            )
            domain_loss = -np.mean(
                domain_labels * np.log(clipped_domain)
                + (1 - domain_labels) * np.log(1 - clipped_domain)
            )
            self.loss_history_.append(float(label_loss + domain_loss))

            d_label_logits = probabilities - one_hot
            d_label_logits[~labeled] = 0.0
            d_label_logits /= np.count_nonzero(labeled)

            d_domain_logits = (domain_probabilities - domain_labels).reshape(-1, 1)
            d_domain_logits /= n_rows

            d_Wy = hidden.T @ d_label_logits
            d_by = np.sum(d_label_logits, axis=0)
            d_Wd = hidden.T @ d_domain_logits
            d_bd = np.sum(d_domain_logits, axis=0)

            d_hidden = d_label_logits @ self.Wy_.T
            d_hidden -= self.domain_weight * (d_domain_logits @ self.Wd_.T)
            d_hidden_input = d_hidden * (1.0 - hidden**2)
            d_W1 = matrix.T @ d_hidden_input / n_rows
            d_b1 = np.sum(d_hidden_input, axis=0)

            self.W1_ -= self.learning_rate * d_W1
            self.b1_ -= self.learning_rate * d_b1
            self.Wy_ -= self.learning_rate * d_Wy
            self.by_ -= self.learning_rate * d_by
            self.Wd_ -= self.learning_rate * d_Wd
            self.bd_ -= self.learning_rate * d_bd

        self.is_fitted = True
        return self

    @staticmethod
    def _softmax(logits):
        """Softmax estable por filas"""
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    @staticmethod
    def _sigmoid(values):
        """Sigmoide recortada para evitar desbordamiento"""
        values = np.clip(values, -500, 500)
        return 1.0 / (1.0 + np.exp(-values))

    def predict_proba(self, X):
        """Devuelve probabilidades de clase con la capa compartida aprendida"""
        self._check_is_fitted()
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError("X debe tener la misma forma que en fit().")
        hidden = np.tanh(matrix @ self.W1_ + self.b1_)
        return self._softmax(hidden @ self.Wy_ + self.by_)

    def predict(self, X):
        """Devuelve la clase de mayor probabilidad"""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
