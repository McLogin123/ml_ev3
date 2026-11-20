"""Pruebas de las métricas de clasificación, regresión y clustering."""

import numpy as np
import pytest

from evaluation.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    clustering_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    inertia_score,
    mae,
    mse,
    precision_score,
    r2_score,
    recall_score,
    regression_report,
    rmse,
    silhouette_score,
)


def test_accuracy_simple():
    # Preparar: acierta 2 de 3.
    real = [1, 0, 1]
    predicho = [1, 0, 0]

    # Ejecutar
    result = accuracy_score(real, predicho)

    # Verificar
    assert result == pytest.approx(2 / 3)


def test_matriz_de_confusion():
    # Preparar
    real = [0, 1, 1, 0]
    predicho = [0, 1, 0, 0]

    # Ejecutar
    matrix = confusion_matrix(real, predicho)

    # Verificar: filas reales, columnas predichas.
    assert matrix.tolist() == [[2, 0], [1, 1]]


def test_matriz_de_confusion_con_orden_explicito():
    # Preparar
    real = [0, 1, 1, 0]
    predicho = [0, 1, 0, 0]

    # Ejecutar
    matrix = confusion_matrix(real, predicho, labels=[1, 0])

    # Verificar
    assert matrix.tolist() == [[1, 1], [0, 2]]


def test_precision_recall_f1_por_clase():
    # Preparar: para la clase 1 hay TP=1, FP=0, FN=1.
    real = [0, 1, 1, 0]
    predicho = [0, 1, 0, 0]

    # Ejecutar
    precision = precision_score(real, predicho, average=None)
    recall = recall_score(real, predicho, average=None)
    f1 = f1_score(real, predicho, average=None)

    # Verificar
    assert precision["1"] == pytest.approx(1.0)
    assert recall["1"] == pytest.approx(0.5)
    assert f1["1"] == pytest.approx(2 / 3)


def test_precision_y_recall_con_division_por_cero():
    # Preparar: no se predice ni se observa la clase 1.
    real = [0, 0]
    predicho = [0, 0]

    # Ejecutar: se fuerzan las etiquetas 0 y 1 para ver la clase ausente.
    report = classification_report(real, predicho, labels=[0, 1])

    # Verificar
    assert report["classes"]["1"]["precision"] == 0.0
    assert report["classes"]["1"]["recall"] == 0.0
    assert report["classes"]["1"]["support"] == 0


def test_promedios_macro_y_ponderado():
    # Preparar: clase 0 con support 3, clase 1 con support 1.
    real = [0, 0, 0, 1]
    predicho = [0, 0, 1, 1]

    # Ejecutar
    report = classification_report(real, predicho)

    # Verificar
    assert report["accuracy"] == pytest.approx(0.75)
    assert report["classes"]["0"]["precision"] == pytest.approx(1.0)
    assert report["classes"]["0"]["recall"] == pytest.approx(2 / 3)
    assert report["classes"]["0"]["f1"] == pytest.approx(0.8)
    assert report["classes"]["1"]["precision"] == pytest.approx(0.5)
    assert report["macro_avg"]["precision"] == pytest.approx(0.75)
    assert report["weighted_avg"]["precision"] == pytest.approx(0.875)


def test_funciones_aceptan_tipos_de_promedio():
    # Preparar
    real = [0, 0, 0, 1]
    predicho = [0, 0, 1, 1]

    # Ejecutar y verificar
    assert precision_score(real, predicho, average="macro") == pytest.approx(0.75)
    assert precision_score(real, predicho, average="weighted") == pytest.approx(0.875)


def test_promedio_desconocido_falla():
    # Preparar
    real = [0, 1]
    predicho = [0, 1]

    # Ejecutar y verificar
    with pytest.raises(ValueError, match="average"):
        f1_score(real, predicho, average="medio")


def test_etiquetas_explicitas_incompletas_fallan():
    # Preparar: y_pred contiene la clase 1, pero labels solo tiene la 0.
    real = [0, 0]
    predicho = [0, 1]

    # Ejecutar y verificar
    with pytest.raises(ValueError, match="labels"):
        confusion_matrix(real, predicho, labels=[0])


def test_clase_solo_en_prediccion_aparece_en_matriz():
    # Preparar
    real = [0, 0]
    predicho = [0, 1]

    # Ejecutar
    matrix = confusion_matrix(real, predicho)

    # Verificar
    assert matrix.shape == (2, 2)
    assert accuracy_score(real, predicho) == 0.5


def test_vectores_de_distinto_tamano_fallan():
    # Ejecutar y verificar
    with pytest.raises(ValueError, match="tamaños"):
        accuracy_score([0, 1], [0])


def test_vectores_vacios_fallan():
    # Ejecutar y verificar
    with pytest.raises(ValueError, match="al menos una muestra"):
        mae([], [])


def test_mae_mse_y_rmse():
    # Preparar: error 2 en una de tres muestras.
    real = [1, 2, 3]
    predicho = [1, 4, 3]

    # Ejecutar
    result_mae = mae(real, predicho)
    result_mse = mse(real, predicho)
    result_rmse = rmse(real, predicho)

    # Verificar
    assert result_mae == pytest.approx(2 / 3)
    assert result_mse == pytest.approx(4 / 3)
    assert result_rmse == pytest.approx(np.sqrt(4 / 3))


def test_r2_perfecto():
    # Ejecutar y verificar
    assert r2_score([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_r2_igual_a_la_media():
    # Preparar: predecir siempre la media no explica varianza.
    real = [1, 2, 3]

    # Ejecutar y verificar
    assert r2_score(real, [2, 2, 2]) == pytest.approx(0.0)


def test_r2_puede_ser_negativo():
    # Ejecutar y verificar
    assert r2_score([1, 2, 3], [1, 4, 3]) == pytest.approx(-1.0)


def test_r2_con_target_constante():
    # Ejecutar y verificar
    assert r2_score([5, 5, 5], [5, 5, 5]) == 1.0
    assert r2_score([5, 5, 5], [5, 5, 6]) == 0.0


def test_reporte_de_regresion():
    # Ejecutar
    report = regression_report([1, 2, 3], [1, 4, 3])

    # Verificar
    assert report["mse"] == pytest.approx(4 / 3)
    assert report["r2"] == pytest.approx(-1.0)
    assert report["n_samples"] == 3


def test_inercia_calcula_centros():
    # Preparar: dos clusters, el primero con centro (0.5, 0.5).
    X = [[0, 0], [1, 1], [10, 10]]
    labels = [0, 0, 1]

    # Ejecutar
    result = inertia_score(X, labels)

    # Verificar: 0.5 + 0.5 + 0 = 1.
    assert result == pytest.approx(1.0)


def test_inercia_acepta_centros_explicitos():
    # Preparar
    X = [[0, 0], [1, 1], [10, 10]]
    labels = [0, 0, 1]

    # Ejecutar
    result = inertia_score(X, labels, centers=[[0.5, 0.5], [10, 10]])

    # Verificar
    assert result == pytest.approx(1.0)


def test_inercia_rechaza_centros_incorrectos():
    # Preparar
    X = [[0, 0], [1, 1], [10, 10]]
    labels = [0, 0, 1]

    # Ejecutar y verificar
    with pytest.raises(ValueError, match="centros"):
        inertia_score(X, labels, centers=[[0, 0]])


def test_silhouette_de_clusters_separados():
    # Preparar: dos pares de puntos muy separados.
    X = [[0, 0], [1, 0], [10, 10], [11, 10]]
    labels = [0, 0, 1, 1]

    # Ejecutar
    result = silhouette_score(X, labels)

    # Verificar
    assert 0.8 < result <= 1.0


def test_silhouette_con_cluster_de_una_muestra():
    # Preparar: el cluster 1 tiene una sola muestra.
    X = [[0, 0], [1, 0], [2, 0], [10, 10]]
    labels = [0, 0, 0, 1]

    # Ejecutar
    result = silhouette_score(X, labels)

    # Verificar
    assert -1.0 <= result <= 1.0


def test_silhouette_rechaza_un_solo_cluster():
    # Ejecutar y verificar
    with pytest.raises(ValueError, match="al menos 2 clusters"):
        silhouette_score([[0, 0], [1, 1]], [0, 0])


def test_silhouette_rechaza_mas_clusters_que_muestras():
    # Ejecutar y verificar
    with pytest.raises(ValueError, match="menos clusters"):
        silhouette_score([[0, 0], [1, 1]], [0, 1])


def test_calinski_harabasz_positivo():
    # Preparar
    X = [[0, 0], [1, 0], [10, 10], [11, 10]]
    labels = [0, 0, 1, 1]

    # Ejecutar
    result = calinski_harabasz_score(X, labels)

    # Verificar
    assert result > 0


def test_calinski_harabasz_sin_dispersion_interna():
    # Preparar: puntos repetidos dentro de cada cluster.
    X = [[0, 0], [0, 0], [10, 10], [10, 10]]
    labels = [0, 0, 1, 1]

    # Ejecutar y verificar
    assert calinski_harabasz_score(X, labels) == float("inf")


def test_davies_bouldin_finito():
    # Preparar
    X = [[0, 0], [1, 0], [10, 10], [11, 10]]
    labels = [0, 0, 1, 1]

    # Ejecutar
    result = davies_bouldin_score(X, labels)

    # Verificar
    assert np.isfinite(result)
    assert result > 0


def test_reporte_de_clustering():
    # Preparar
    X = [[0, 0], [1, 0], [10, 10], [11, 10]]
    labels = [0, 0, 1, 1]

    # Ejecutar
    report = clustering_report(X, labels)

    # Verificar
    assert report["n_clusters"] == 2
    assert report["warnings"] == []
    assert "silhouette" in report
    assert "calinski_harabasz" in report
    assert "davies_bouldin" in report


def test_reporte_de_clustering_con_un_cluster():
    # Preparar
    X = [[0, 0], [1, 1], [2, 2]]

    # Ejecutar
    report = clustering_report(X, [0, 0, 0])

    # Verificar
    assert report["silhouette"] is None
    assert report["davies_bouldin"] is None
    assert report["warnings"]


def test_X_bidimensional_requerido():
    # Ejecutar y verificar
    with pytest.raises(ValueError, match="bidimensional"):
        clustering_report([0, 1, 2], [0, 0, 1])


def test_longitud_de_labels_requerida():
    # Ejecutar y verificar
    with pytest.raises(ValueError, match="distinto número de filas"):
        clustering_report([[0, 0], [1, 1]], [0])
