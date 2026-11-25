"""Auditoría de particionamiento: selección de K y métricas por clúster"""

import numpy as np

from evaluation.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from models import KMeans


def _finite(value):
    """Devuelve None para infinitos, porque JSON no los puede representar"""
    return None if not np.isfinite(value) else float(value)


def find_elbow(k_values, inertias):
    """Detecta el codo de la curva de inercia con Kneedle simplificado

    Parametros:
        k_values : array-like
            Números de clústeres evaluados, en orden creciente
        inertias : array-like
            Inercia de K-Means para cada k

    Retorna:
        int or None
            k del punto más alejado de la recta que une los extremos normalizados
            None si la curva es plana o hay menos de tres puntos
    """
    k = np.asarray(k_values, dtype=float)
    inertia = np.asarray(inertias, dtype=float)
    if k.ndim != 1 or inertia.ndim != 1 or len(k) != len(inertia):
        raise ValueError("k_values e inertias deben tener la misma longitud.")
    if len(k) < 3:
        raise ValueError("find_elbow necesita al menos 3 puntos.")
    if k.max() == k.min() or inertia.max() == inertia.min():
        return None

    x = (k - k.min()) / (k.max() - k.min())
    y = (inertia - inertia.min()) / (inertia.max() - inertia.min())
    slope = (y[-1] - y[0]) / (x[-1] - x[0])
    intercept = y[0] - slope * x[0]
    distances = np.abs(slope * x - y + intercept) / np.sqrt(slope ** 2 + 1)
    distances[0] = distances[-1] = -1.0
    return int(k[int(np.argmax(distances))])


def audit_partitioning(X, k_values, seed=42):
    """Ajusta K-Means para cada k y reúne las señales de selección.

    Parametros:
        X : array-like
            Matriz de muestras ya preprocesada
        k_values : iterable
            Valores de k a evaluar; se descartan los fuera de [2, n_muestras-1]
        seed : int, default=42
            Semilla de K-Means para que el barrido sea reproducible

    Retorna:
        dict
            Series por k (inercia, silueta, Calinski-Harabasz y Davies-Bouldin),
            codo, k seleccionado por silueta máxima y avisos
    """
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2 or len(matrix) == 0 or matrix.shape[1] == 0:
        raise ValueError("X debe ser una matriz bidimensional no vacía.")

    warnings = []
    requested_k = list(k_values)
    valid_k = [int(k) for k in requested_k if 2 <= k < len(matrix)]
    if len(valid_k) != len(requested_k):
        warnings.append("Se descartaron valores de k fuera de [2, n_muestras-1].")

    audit = {
        "k_values": [],
        "inertias": [],
        "silhouette": [],
        "calinski_harabasz": [],
        "davies_bouldin": [],
        "elbow_k": None,
        "selected_k": None,
        "selection_rule": None,
    }

    for k in valid_k:
        model = KMeans(n_clusters=k, random_state=seed).fit(matrix)
        labels = model.labels_
        audit["k_values"].append(k)
        audit["inertias"].append(float(model.inertia_))

        if len(np.unique(labels)) < 2:
            audit["silhouette"].append(None)
            audit["calinski_harabasz"].append(None)
            audit["davies_bouldin"].append(None)
            warnings.append(f"k={k} produjo un solo clúster; se omiten sus índices.")
            continue

        audit["silhouette"].append(float(silhouette_score(matrix, labels)))
        audit["calinski_harabasz"].append(
            _finite(calinski_harabasz_score(matrix, labels))
        )
        audit["davies_bouldin"].append(
            _finite(davies_bouldin_score(matrix, labels))
        )

    if audit["k_values"]:
        if len(audit["k_values"]) >= 3:
            audit["elbow_k"] = find_elbow(
                audit["k_values"], audit["inertias"]
            )
            if audit["elbow_k"] is None:
                warnings.append("La curva de inercia es plana: no hay codo.")
        scores = [
            -np.inf if value is None else value for value in audit["silhouette"]
        ]
        best = int(np.argmax(scores))
        audit["selected_k"] = audit["k_values"][best]
        audit["selection_rule"] = "silhouette"
        if audit["elbow_k"] not in (None, audit["selected_k"]):
            warnings.append(
                f"El codo sugiere k={audit['elbow_k']} y la silueta "
                f"k={audit['selected_k']}."
            )
    else:
        warnings.append("No hay valores de k válidos para auditar.")

    audit["warnings"] = warnings
    return audit
