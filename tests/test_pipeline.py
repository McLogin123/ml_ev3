"""Pruebas de integración del CLI y del reporte en JSON."""

import json
import struct
from pathlib import Path

import pytest

from main import build_parser, main, run_pipeline

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Trabaja con rutas relativas dentro de una carpeta temporal."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    return tmp_path


def _dataset(name, workspace):
    """Copia un dataset de ejemplo al espacio de trabajo temporal."""
    destination = workspace / "data" / name
    destination.write_bytes((_DATA_DIR / name).read_bytes())
    return f"data/{name}"


def _output(workspace, data):
    """Carpeta de salida esperada para un dataset."""
    return workspace / "reports" / Path(data).stem


def _assert_png(path):
    """Comprueba que el PNG declara al menos 300 DPI."""
    assert path.exists() and path.stat().st_size > 0
    data = path.read_bytes()
    index = data.find(b"pHYs")
    assert index != -1
    pixels_per_meter = struct.unpack(">I", data[index + 4 : index + 8])[0]
    assert round(pixels_per_meter * 0.0254) >= 300


def _options(data, **extra):
    """Opciones mínimas para ejecutar el pipeline."""
    values = {"data": data, "random_state": 42}
    values.update(extra)
    return values


def test_build_parser_defaults():
    # Ejecutar
    options = vars(build_parser().parse_args([]))

    # Verificar
    assert options["data"] == "data/ejemplo_clasificacion.csv"
    assert options["test_size"] == 0.2
    assert options["random_state"] == 42
    assert options["n_clusters"] is None
    assert options["cv"] == 0
    assert options["grid"] is False


def test_run_pipeline_classification(workspace):
    # Preparar
    data = _dataset("ejemplo_clasificacion.csv", workspace)

    # Ejecutar
    report = run_pipeline(_options(data, target="target"))

    # Verificar
    assert report["dataset"]["task"] == "classification"
    assert report["model"]["name"] == "bagging"
    assert report["model"]["scaled"] is False
    assert "accuracy" in report["metrics"]
    assert report["precision"] == report["metrics"]["accuracy"]
    assert report["precision_metric"] == "accuracy"
    assert report["validation"] == {}
    assert sum(report["feature_importances"].values()) == pytest.approx(1.0)
    assert len(report["metrics"]["confusion_matrix"]) == 3
    names = [Path(path).name for path in report["figures"]]
    assert {"confusion_matrix.png", "roc_curves.png", "precision_recall_curves.png"} <= set(
        names
    )
    assert {"correlation_heatmap.png", "spearman_heatmap.png"} <= set(names)
    assert "learning_curves.png" in names
    for path in report["figures"]:
        _assert_png(Path(path))


def test_run_pipeline_without_probabilities_skips_roc(workspace):
    # Preparar
    data = _dataset("ejemplo_clasificacion.csv", workspace)

    # Ejecutar
    report = run_pipeline(_options(data, target="target", model="tree"))

    # Verificar
    names = [Path(path).name for path in report["figures"]]
    assert "roc_curves.png" not in names
    assert any("probabilidades" in warning for warning in report["warnings"])


def test_run_pipeline_knn_scales_and_tree_does_not(workspace):
    # Preparar
    data = _dataset("ejemplo_clasificacion.csv", workspace)

    # Ejecutar
    knn = run_pipeline(_options(data, target="target", model="knn"))
    tree = run_pipeline(_options(data, target="target", model="tree"))

    # Verificar
    assert knn["model"]["scaled"] is True
    assert tree["model"]["scaled"] is False


def test_run_pipeline_cv_and_grid_search(workspace):
    # Preparar
    data = _dataset("ejemplo_clasificacion.csv", workspace)
    options = _options(data, target="target", model="tree", cv=3, grid=True)

    # Ejecutar
    report = run_pipeline(options)

    # Verificar
    validation = report["validation"]
    assert len(validation["cv"]["scores"]) == 3
    assert 0.0 <= validation["cv"]["mean"] <= 1.0
    assert validation["grid"]["best_params"] in [
        {"max_depth": 2}, {"max_depth": 4}, {"max_depth": 6}
    ]
    assert len(validation["grid"]["results"]) == 3
    saved = json.loads((_output(workspace, data) / "report.json").read_text("utf-8"))
    assert saved["validation"] == validation


def test_run_pipeline_lvq1_dann_and_gam(workspace):
    # Preparar
    classification = _dataset("ejemplo_clasificacion.csv", workspace)
    regression = _dataset("ejemplo_regresion.csv", workspace)

    # Ejecutar
    lvq1 = run_pipeline(_options(classification, target="target", model="lvq1"))
    dann = run_pipeline(_options(classification, target="target", model="dann"))
    gam = run_pipeline(_options(regression, target="target", model="gam"))

    # Verificar
    assert "accuracy" in lvq1["metrics"]
    assert "accuracy" in dann["metrics"]
    assert "r2" in gam["metrics"]
    assert gam["model"]["scaled"] is False


def test_run_pipeline_regression(workspace):
    # Preparar
    data = _dataset("ejemplo_regresion.csv", workspace)

    # Ejecutar
    report = run_pipeline(_options(data, target="target"))

    # Verificar
    assert report["dataset"]["task"] == "regression"
    assert {"r2", "rmse", "mae"} <= set(report["metrics"])
    assert report["precision"] == report["metrics"]["r2"]
    assert report["precision_metric"] == "R²"
    names = [Path(path).name for path in report["figures"]]
    assert {"predicted_vs_actual.png", "residuals.png", "regression_metrics.png"} <= set(
        names
    )
    assert "learning_curves.png" in names


def test_run_pipeline_kmeans(workspace):
    # Preparar
    data = _dataset("ejemplo_clusters.csv", workspace)

    # Ejecutar
    report = run_pipeline(_options(data, task="unsupervised"))

    # Verificar
    assert report["dataset"]["task"] == "unsupervised"
    assert report["model"]["scaled"] is True
    assert "silhouette" in report["metrics"]
    assert report["precision"] == report["metrics"]["silhouette"]
    assert report["precision_metric"] == "silueta"
    names = [Path(path).name for path in report["figures"]]
    assert {
        "inertia_curve.png", "silhouette.png", "cluster_centers.png",
        "pca_projection.png",
    } <= set(names)
    assert "dendrogram.png" not in names

    audit = report["clustering_audit"]
    assert audit["k_values"] == [2, 3, 4, 5, 6]
    assert all(len(audit[key]) == 5 for key in (
        "inertias", "silhouette", "calinski_harabasz", "davies_bouldin"
    ))
    assert report["model"]["n_clusters"] == audit["selected_k"]
    assert len(audit["pca_explained_variance"]) == 2
    saved = json.loads((_output(workspace, data) / "report.json").read_text("utf-8"))
    assert saved["clustering_audit"] == audit


def test_run_pipeline_forces_k_outside_the_sweep(workspace):
    # Preparar
    data = _dataset("ejemplo_clusters.csv", workspace)

    # Ejecutar
    report = run_pipeline(_options(
        data, task="unsupervised", n_clusters=7, max_k=3
    ))

    # Verificar
    assert report["model"]["n_clusters"] == 7
    assert report["clustering_audit"]["selection_rule"] == "flag --n-clusters"
    assert report["clustering_audit"]["k_values"] == [2, 3]
    assert any("fuera del barrido" in warning for warning in report["warnings"])


def test_run_pipeline_hierarchical_draws_partial_dendrogram(workspace):
    # Preparar: el linkage tiene menos filas que muestras-1.
    data = _dataset("ejemplo_clusters.csv", workspace)

    # Ejecutar
    report = run_pipeline(_options(data, task="unsupervised", model="hierarchical"))

    # Verificar
    assert report["model"]["scaled"] is True
    names = [Path(path).name for path in report["figures"]]
    assert {"dendrogram.png", "pca_projection.png"} <= set(names)
    assert report["model"]["n_clusters"] == report["clustering_audit"]["selected_k"]
    _assert_png(_output(workspace, data) / "figures/dendrogram.png")


def test_run_pipeline_rejects_unknown_model(workspace):
    # Preparar
    data = _dataset("ejemplo_clasificacion.csv", workspace)

    # Preparar / Ejecutar / Verificar
    with pytest.raises(ValueError, match="no válido"):
        run_pipeline(_options(data, target="target", model="desconocido"))


def test_report_folder_is_named_after_dataset(workspace):
    # Preparar
    data = _dataset("ejemplo_clasificacion.csv", workspace)

    # Ejecutar
    run_pipeline(_options(data, target="target"))

    # Verificar
    destination = _output(workspace, data)
    assert (destination / "report.json").exists()
    assert (destination / "figures").is_dir()
    assert not (destination / "reporte_final.md").exists()
    assert not (destination / "eda.json").exists()
    assert list((workspace / "reports").iterdir()) == [destination]


def test_repeated_runs_get_separate_folders(workspace):
    # Preparar
    data = _dataset("ejemplo_clasificacion.csv", workspace)

    # Ejecutar: la segunda corrida cambia las opciones, como en el caso del usuario.
    first = run_pipeline(_options(data, target="target"))
    second = run_pipeline(_options(data, target="target", task="classification"))

    # Verificar
    assert first["output"] == "reports/ejemplo_clasificacion"
    assert second["output"] == "reports/ejemplo_clasificacion_1"
    assert (workspace / first["output"] / "report.json").exists()
    assert (workspace / second["output"] / "report.json").exists()
    assert all("ejemplo_clasificacion/" in path for path in first["figures"])
    assert all("ejemplo_clasificacion_1/" in path for path in second["figures"])


def test_report_json_contains_results_and_figures(workspace):
    # Preparar
    data = _dataset("ejemplo_regresion.csv", workspace)

    # Ejecutar
    report = run_pipeline(_options(data, target="target"))

    # Verificar
    saved = json.loads((_output(workspace, data) / "report.json").read_text(encoding="utf-8"))
    assert saved["metrics"] == report["metrics"]
    assert saved["precision"] == report["metrics"]["r2"]
    assert saved["precision_metric"] == "R²"
    assert "class_diagram" not in saved
    assert all(path.startswith("reports/ejemplo_regresion/figures/") for path in saved["figures"])


def test_main_prints_summary(capsys, workspace):
    # Preparar
    data = _dataset("ejemplo_clasificacion.csv", workspace)

    # Ejecutar
    code = main(["--data", data, "--target", "target"])
    first_output = capsys.readouterr().out
    code_again = main(["--data", data, "--target", "target"])
    second_output = capsys.readouterr().out

    # Verificar
    assert code == 0 and code_again == 0
    assert "Tarea: classification" in first_output
    assert "Precisión (accuracy):" in first_output
    assert "reports/ejemplo_clasificacion/report.json" in first_output
    assert "reports/ejemplo_clasificacion_1/report.json" in second_output
