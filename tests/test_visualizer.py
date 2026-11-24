"""Pruebas de las visualizaciones de alta resolución."""

import struct
from pathlib import Path

import numpy as np
import pytest

from evaluation.visualizer import Visualizer


@pytest.fixture
def visualizer(tmp_path, monkeypatch):
    """Aísla cada prueba en una carpeta temporal sin rutas absolutas fijas."""
    monkeypatch.chdir(tmp_path)
    return Visualizer()


def _assert_png(path):
    """Comprueba que el archivo existe y declara al menos 300 DPI."""
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0

    data = path.read_bytes()
    index = data.find(b"pHYs")
    assert index != -1
    pixels_per_meter = struct.unpack(">I", data[index + 4 : index + 8])[0]
    unit = data[index + 12]
    assert unit == 1
    assert round(pixels_per_meter * 0.0254) >= 300


def _sample_report():
    """Reporte mínimo con los bloques que consumen los gráficos."""
    return {
        "missing": {
            "columns": {
                "age": {"pct": 25.0},
                "income": {"pct": 10.0},
            }
        },
        "correlations": {
            "pearson": {
                "age": {"age": 1.0, "income": 0.8},
                "income": {"age": 0.8, "income": 1.0},
            }
        },
        "target": {
            "type": "classification",
            "classes": {"A": 3, "B": 1},
            "imbalance": 3.0,
        },
    }


def test_default_dpi_is_300_and_rejects_lower_values():
    # Preparar / Ejecutar / Verificar
    assert Visualizer().dpi == 300
    with pytest.raises(ValueError, match="dpi"):
        Visualizer(dpi=200)


def test_rejects_absolute_paths(tmp_path, visualizer):
    # Preparar / Ejecutar / Verificar
    with pytest.raises(ValueError, match="relativa"):
        Visualizer(output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="relativa"):
        visualizer.plot_missing_values(_sample_report(), str(tmp_path / "plot.png"))


def test_plot_missing_values_creates_300_dpi_png(visualizer):
    # Ejecutar
    path = visualizer.plot_missing_values(_sample_report())

    # Verificar
    assert path == Path("reports/figures/missing_values.png")
    _assert_png(path)


def test_plot_missing_values_without_missing_data(visualizer):
    # Preparar
    report = {"missing": {"columns": {}}}

    # Ejecutar
    path = visualizer.plot_missing_values(report, "sin_faltantes.png")

    # Verificar
    _assert_png(path)


def test_plot_correlation_heatmap_creates_png(visualizer):
    # Ejecutar
    path = visualizer.plot_correlation_heatmap(_sample_report())

    # Verificar
    _assert_png(path)


def test_plot_correlation_heatmap_with_one_numeric_column(visualizer):
    # Preparar
    report = {"correlations": {"pearson": {"age": {"age": 1.0}}}}

    # Ejecutar
    path = visualizer.plot_correlation_heatmap(report, "una_columna.png")

    # Verificar
    _assert_png(path)


def test_plot_target_distribution_classification(visualizer):
    # Ejecutar
    path = visualizer.plot_target_distribution(_sample_report())

    # Verificar
    _assert_png(path)


def test_plot_target_distribution_regression(visualizer):
    # Preparar
    report = {
        "target": {
            "type": "regression",
            "correlations": [
                {"feature": "age", "pearson": 0.7},
                {"feature": "income", "pearson": -0.4},
            ],
        }
    }

    # Ejecutar
    path = visualizer.plot_target_distribution(report, "regresion.png")

    # Verificar
    _assert_png(path)


def test_plot_target_distribution_without_target(visualizer):
    # Preparar
    report = {"target": None}

    # Ejecutar
    path = visualizer.plot_target_distribution(report, "sin_objetivo.png")

    # Verificar
    _assert_png(path)


def test_plot_confusion_matrix_creates_png(visualizer):
    # Preparar
    matrix = np.array([[3, 1], [0, 4]])

    # Ejecutar
    path = visualizer.plot_confusion_matrix(matrix, ["A", "B"])

    # Verificar
    _assert_png(path)


def test_plot_confusion_matrix_rejects_non_square_matrix(visualizer):
    # Preparar / Ejecutar / Verificar
    with pytest.raises(ValueError, match="cuadrada"):
        visualizer.plot_confusion_matrix(np.array([[1, 2, 3]]), ["A"])


def test_plot_clusters_creates_png(visualizer):
    # Preparar
    X = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
    labels = np.array([0, 0, 1])
    centers = np.array([[0.05, 0.0], [5.0, 5.0]])

    # Ejecutar
    path = visualizer.plot_clusters(X, labels, centers)

    # Verificar
    _assert_png(path)


def test_plot_clusters_rejects_wrong_label_length(visualizer):
    # Preparar / Ejecutar / Verificar
    with pytest.raises(ValueError, match="longitud"):
        visualizer.plot_clusters(np.array([[0.0], [1.0]]), np.array([0]))


def test_plot_dendrogram_creates_png(visualizer):
    # Preparar
    linkage = np.array([[0.0, 1.0, 2.0, 2.0], [2.0, 3.0, 7.0, 3.0]])

    # Ejecutar
    path = visualizer.plot_dendrogram(linkage, ["a", "b", "c"])

    # Verificar
    _assert_png(path)


def test_plot_dendrogram_with_single_sample(visualizer):
    # Ejecutar
    path = visualizer.plot_dendrogram(np.empty((0, 4)))

    # Verificar
    _assert_png(path)


def test_plot_metric_comparison_creates_png(visualizer):
    # Ejecutar
    path = visualizer.plot_metric_comparison({"KMeans": 0.6, "Jerárquico": 0.7})

    # Verificar
    _assert_png(path)


def test_plot_metric_comparison_rejects_empty_scores(visualizer):
    # Preparar / Ejecutar / Verificar
    with pytest.raises(ValueError, match="vacío"):
        visualizer.plot_metric_comparison({})


def _classification_scores():
    """Tres clases con probabilidades one-vs-rest ya normalizadas."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_scores = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.6, 0.3],
            [0.1, 0.2, 0.7],
            [0.2, 0.1, 0.7],
        ]
    )
    return y_true, y_scores


def test_plot_confusion_matrix_normalized(visualizer):
    # Ejecutar
    path = visualizer.plot_confusion_matrix(
        np.array([[3, 1], [0, 4]]), ["A", "B"], normalize=True
    )

    # Verificar
    _assert_png(path)


def test_plot_roc_curves_multiclass(visualizer):
    # Preparar
    y_true, y_scores = _classification_scores()

    # Ejecutar
    path = visualizer.plot_roc_curves(y_true, y_scores, [0, 1, 2])

    # Verificar
    _assert_png(path)


def test_plot_roc_curves_rejects_wrong_shape(visualizer):
    # Preparar / Ejecutar / Verificar
    with pytest.raises(ValueError, match="forma"):
        visualizer.plot_roc_curves(np.array([0, 1]), np.array([[0.8], [0.2], [0.5]]))


def test_plot_precision_recall_curves(visualizer):
    # Preparar
    y_true, y_scores = _classification_scores()

    # Ejecutar
    path = visualizer.plot_precision_recall_curves(y_true, y_scores, [0, 1, 2])

    # Verificar
    _assert_png(path)


def test_plot_learning_curves_by_sample_size(visualizer):
    # Ejecutar
    path = visualizer.plot_learning_curves(
        [100, 200, 300], [0.5, 0.3, 0.25], [0.55, 0.35, 0.28]
    )

    # Verificar
    _assert_png(path)


def test_plot_learning_curves_by_hyperparameter(visualizer):
    # Ejecutar
    path = visualizer.plot_learning_curves(
        [1, 2, 3], [0.6, 0.4, 0.35], x_label="max_depth"
    )

    # Verificar
    _assert_png(path)


def test_plot_predicted_vs_actual_with_confidence_band(visualizer):
    # Preparar
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.2, 1.8, 3.3, 3.7])

    # Ejecutar
    path = visualizer.plot_predicted_vs_actual(y_true, y_pred)

    # Verificar
    _assert_png(path)


def test_plot_residuals(visualizer):
    # Preparar
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.2, 1.8, 3.3, 3.7])

    # Ejecutar
    path = visualizer.plot_residuals(y_true, y_pred)

    # Verificar
    _assert_png(path)


def test_plot_regression_metrics(visualizer):
    # Ejecutar
    path = visualizer.plot_regression_metrics(
        {"r2": 0.91, "rmse": 0.42, "mae": 0.31}
    )

    # Verificar
    _assert_png(path)


def test_plot_regression_metrics_requires_all_fields(visualizer):
    # Preparar / Ejecutar / Verificar
    with pytest.raises(ValueError, match="r2, rmse y mae"):
        visualizer.plot_regression_metrics({"r2": 0.9})


def test_plot_inertia_curve_annotates_values(visualizer):
    # Ejecutar
    path = visualizer.plot_inertia_curve([1, 2, 3], [10.0, 4.5, 3.0])

    # Verificar
    _assert_png(path)


def test_plot_silhouette_sorted_by_cluster(visualizer):
    # Preparar
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    labels = np.array([0, 0, 1, 1])

    # Ejecutar
    path = visualizer.plot_silhouette(X, labels)

    # Verificar
    _assert_png(path)


def test_plot_silhouette_with_one_cluster(visualizer):
    # Ejecutar
    path = visualizer.plot_silhouette(np.array([[0.0], [1.0]]), np.array([0, 0]))

    # Verificar
    _assert_png(path)


def test_plot_cluster_centers_with_samples(visualizer):
    # Preparar
    centers = np.array([[0.0, 0.0], [5.0, 5.0]])
    X = np.array([[0.1, 0.0], [4.9, 5.1]])

    # Ejecutar
    path = visualizer.plot_cluster_centers(centers, X)

    # Verificar
    _assert_png(path)
