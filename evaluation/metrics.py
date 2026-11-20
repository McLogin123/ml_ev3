"""Métricas formales de clasificación, regresión y clustering

Todas las funciones reciben arreglos o secuencias y devuelven números o diccionarios de 
Python, para que el reporte pueda guardarlos como JSON
"""

import numpy as np


def _check_vectors(y_true, y_pred):
    """Valida dos vectores de la misma longitud y los devuelve como arrays"""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true e y_pred deben ser unidimensionales.")
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true e y_pred tienen tamaños distintos: {true.shape} y {pred.shape}."
        )
    if true.size == 0:
        raise ValueError("Se necesita al menos una muestra para calcular métricas.")
    return true, pred


def _check_clustering(X, labels):
    """Valida la matriz de datos y el vector de etiquetas de cluster"""
    matrix = np.asarray(X, dtype=float)
    values = np.asarray(labels)

    if matrix.ndim != 2:
        raise ValueError("X debe ser una matriz bidimensional.")
    if values.ndim != 1:
        raise ValueError("labels debe ser unidimensional.")
    if matrix.shape[0] != values.shape[0]:
        raise ValueError("X y labels tienen distinto número de filas.")
    if matrix.shape[0] == 0:
        raise ValueError("Se necesita al menos una muestra.")
    return matrix, values


def _cluster_centers(X, labels):
    """Devuelve las etiquetas únicas ordenadas y el centro de cada cluster"""
    unique = np.unique(labels)
    centers = []
    for value in unique:
        group = X[labels == value]
        centers.append(group.mean(axis=0))
    return unique, np.asarray(centers, dtype=float)


def _validate_cluster_count(labels):
    """Exige al menos 2 clusters y menos clusters que muestras"""
    n_clusters = len(np.unique(labels))
    n_samples = len(labels)
    if n_clusters < 2 or n_clusters >= n_samples:
        raise ValueError(
            "La métrica necesita al menos 2 clusters y menos clusters que muestras."
        )


def accuracy_score(y_true, y_pred) -> float:
    """Fracción de predicciones correctas

    Parametros:
        y_true, y_pred : array-like
            Etiquetas reales y predichas

    Retorna:
        float
            Valor entre 0 y 1
    """
    true, pred = _check_vectors(y_true, y_pred)
    return float(np.mean(true == pred))


def confusion_matrix(y_true, y_pred, labels=None):
    """Matriz de confusión: filas son clases reales, columnas predichas

    Parametros
        y_true, y_pred : array-like
            Etiquetas reales y predichas
        labels : array-like, optional
            Clases y orden deseado. Por defecto se toman todas las clases presentes, ordenadas

    Retorna:
        numpy.ndarray
            Matriz cuadrada de enteros
    """
    true, pred = _check_vectors(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([true, pred]))
    else:
        labels = np.asarray(labels)
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("labels debe ser un vector no vacío.")
        if len(np.unique(labels)) != len(labels):
            raise ValueError("labels no puede contener valores repetidos.")

    positions = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for actual, predicted in zip(true, pred):
        if actual not in positions or predicted not in positions:
            raise ValueError(
                "Hay etiquetas en y_true o y_pred que no aparecen en labels."
            )
        matrix[positions[actual], positions[predicted]] += 1

    return matrix


def classification_report(y_true, y_pred, labels=None):
    """Métricas por clase, macro, ponderadas y exactitud global

    Parametros
        y_true, y_pred : array-like
            Etiquetas reales y predichas
        labels : array-like, optional
            Clases y orden deseado

    Retorna:
        dict
            Diccionario con accuracy, labels, classes, macro_avg y weighted_avg
    """
    true, pred = _check_vectors(y_true, y_pred)
    matrix = confusion_matrix(true, pred, labels)

    if labels is None:
        used_labels = np.unique(np.concatenate([true, pred]))
    else:
        used_labels = np.asarray(labels)

    classes = {}
    for index, label in enumerate(used_labels):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        support = tp + fn

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator > 0 else 0.0

        classes[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }

    fields = ("precision", "recall", "f1")
    macro = {}
    weighted = {}
    total_support = sum(item["support"] for item in classes.values())

    for field in fields:
        per_class = [item[field] for item in classes.values()]
        macro[field] = float(np.mean(per_class))

        total = 0.0
        for item in classes.values():
            total += item[field] * item["support"]
        weighted[field] = float(total / total_support)

    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "labels": [str(label) for label in used_labels],
        "classes": classes,
        "macro_avg": macro,
        "weighted_avg": weighted,
    }


def _reduce_report(report, field: str, average):
    """Extrae un campo del reporte según el tipo de promedio pedido"""
    if average is None:
        return {name: item[field] for name, item in report["classes"].items()}
    if average == "macro":
        return report["macro_avg"][field]
    if average == "weighted":
        return report["weighted_avg"][field]
    raise ValueError("average debe ser None, 'macro' o 'weighted'.")


def precision_score(y_true, y_pred, average="macro", labels=None):
    """Precisión: de los predichos como clase, cuántos lo eran realmente"""
    report = classification_report(y_true, y_pred, labels)
    return _reduce_report(report, "precision", average)


def recall_score(y_true, y_pred, average="macro", labels=None):
    """Recall: de los reales de una clase, cuántos se detectaron"""
    report = classification_report(y_true, y_pred, labels)
    return _reduce_report(report, "recall", average)


def f1_score(y_true, y_pred, average="macro", labels=None):
    """F1: media armónica de precisión y recall"""
    report = classification_report(y_true, y_pred, labels)
    return _reduce_report(report, "f1", average)


def _check_regression(y_true, y_pred):
    """Convierte los vectores de regresión a float después de validarlos"""
    true, pred = _check_vectors(y_true, y_pred)
    return true.astype(float), pred.astype(float)


def mae(y_true, y_pred) -> float:
    """Error absoluto medio: media de |y_real - y_predicho|"""
    true, pred = _check_regression(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def mse(y_true, y_pred) -> float:
    """Error cuadrático medio: media de (y_real - y_predicho)²"""
    true, pred = _check_regression(y_true, y_pred)
    return float(np.mean((true - pred) ** 2))


def rmse(y_true, y_pred) -> float:
    """Raíz del error cuadrático medio, en las unidades de y"""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true, y_pred) -> float:
    """Coeficiente de determinación R².

    R² = 1 - SS_res / SS_tot. Si el target es constante (SS_tot = 0),
    devuelve 1.0 cuando la predicción es perfecta y 0.0 en caso contrario
    """
    true, pred = _check_regression(y_true, y_pred)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def regression_report(y_true, y_pred):
    """Resumen de regresión con MAE, MSE, RMSE y R²."""
    return {
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "n_samples": int(len(np.asarray(y_true))),
    }


def inertia_score(X, labels, centers=None) -> float:
    """Suma de distancias cuadradas de cada punto a su centro de cluster

    Parametros
        X : array-like
            Matriz de muestras
        labels : array-like
            Cluster asignado a cada muestra
        centers : array-like, optional
            Centros en el mismo orden que las etiquetas únicas. Si se omite, se calculan como la media de cada cluster

    Retorna:
        float
            Inercia. Cuanto menor, más compactos son los clusters
    """
    matrix, values = _check_clustering(X, labels)
    unique = np.unique(values)

    if centers is None:
        _, used_centers = _cluster_centers(matrix, values)
    else:
        used_centers = np.asarray(centers, dtype=float)
        if used_centers.ndim != 2 or used_centers.shape[1] != matrix.shape[1]:
            raise ValueError("centers debe tener forma (n_clusters, n_features).")
        if used_centers.shape[0] != len(unique):
            raise ValueError("El número de centros debe coincidir con los clusters.")

    inertia = 0.0
    for index, value in enumerate(unique):
        group = matrix[values == value]
        difference = group - used_centers[index]
        inertia += float(np.sum(difference ** 2))
    return inertia


def silhouette_score(X, labels) -> float:
    """Silhouette medio: compacidad interna frente a separación externa

    Para cada muestra i:

    s(i) = (b(i) - a(i)) / max(a(i), b(i))

    donde a(i) es la distancia media a su propio cluster y b(i) la
    menor distancia media a otro cluster. Va de -1 a 1; cuanto mayor, mejor

    Notes
    -----
    Complejidad O(n²): calcula todas las distancias. Es suficiente para los datasets de este
    proyecto y evita una vectorización difícil de defender
    """
    matrix, values = _check_clustering(X, labels)
    _validate_cluster_count(values)

    unique = np.unique(values)
    n_samples = len(matrix)
    scores = []

    for i in range(n_samples):
        own = values[i]
        same = np.where((values == own) & (np.arange(n_samples) != i))[0]

        if len(same) == 0:
            scores.append(0.0)
            continue

        a = float(np.mean(np.linalg.norm(matrix[same] - matrix[i], axis=1)))
        other_means = []
        for value in unique:
            if value == own:
                continue
            group = matrix[values == value]
            other_means.append(float(np.mean(np.linalg.norm(group - matrix[i], axis=1))))

        b = min(other_means)
        denominator = max(a, b)
        scores.append((b - a) / denominator if denominator > 0 else 0.0)

    return float(np.mean(scores))


def calinski_harabasz_score(X, labels) -> float:
    """Índice de Calinski-Harabasz: dispersión entre clusters / dentro

    CH = (SS_between / (k - 1)) / (SS_within / (n - k))

    Cuanto mayor, mejor. Si no hay dispersión interna devuelve inf
    """
    matrix, values = _check_clustering(X, labels)
    _validate_cluster_count(values)

    n_samples = len(matrix)
    n_clusters = len(np.unique(values))
    overall = matrix.mean(axis=0)
    ss_between = 0.0
    ss_within = 0.0

    for value in np.unique(values):
        group = matrix[values == value]
        center = group.mean(axis=0)
        ss_between += len(group) * float(np.sum((center - overall) ** 2))
        ss_within += float(np.sum((group - center) ** 2))

    if ss_within == 0:
        return float("inf")

    return float((ss_between / (n_clusters - 1)) / (ss_within / (n_samples - n_clusters)))


def davies_bouldin_score(X, labels) -> float:
    """Índice de Davies-Bouldin: solapamiento medio entre pares de clusters

    Para cada cluster se calcula
    R_i = max_j (spread_i + spread_j) / distancia(centro_i, centro_j)

    El resultado es la media de R_i. Cuanto menor, mejor
    """
    matrix, values = _check_clustering(X, labels)
    _validate_cluster_count(values)

    unique, centers = _cluster_centers(matrix, values)
    n_clusters = len(unique)
    spreads = []

    for index, value in enumerate(unique):
        group = matrix[values == value]
        spreads.append(float(np.mean(np.linalg.norm(group - centers[index], axis=1))))

    ratios = []
    for i in range(n_clusters):
        worst = 0.0
        for j in range(n_clusters):
            if i == j:
                continue
            distance = float(np.linalg.norm(centers[i] - centers[j]))
            ratio = float("inf") if distance == 0 else (spreads[i] + spreads[j]) / distance
            worst = max(worst, ratio)
        ratios.append(worst)

    return float(np.mean(ratios))


def clustering_report(X, labels, centers=None):
    """Resumen de clustering con inercia y métricas externas válidas

    Si no hay al menos 2 clusters y menos clusters que muestras, las métricas
    que no están definidas se devuelven como None y se añade un aviso
    """
    matrix, values = _check_clustering(X, labels)
    n_clusters = len(np.unique(values))
    n_samples = len(matrix)

    report = {
        "n_samples": int(n_samples),
        "n_clusters": int(n_clusters),
        "inertia": inertia_score(matrix, values, centers),
    }

    if 2 <= n_clusters < n_samples:
        report["silhouette"] = silhouette_score(matrix, values)
        report["calinski_harabasz"] = calinski_harabasz_score(matrix, values)
        report["davies_bouldin"] = davies_bouldin_score(matrix, values)
        report["warnings"] = []
    else:
        report["silhouette"] = None
        report["calinski_harabasz"] = None
        report["davies_bouldin"] = None
        report["warnings"] = [
            "Silhouette, CH y DB requieren al menos 2 clusters y menos "
            "clusters que muestras. Solo se informa la inercia."
        ]

    return report
