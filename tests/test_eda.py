"""Pruebas del análisis exploratorio; usan tablas construidas en memoria."""

import json
from pathlib import Path

import pandas as pd

from core.eda import EDAnalyzer
from core.schema import Dataset, TaskType


def _dataset(frame, target=None, task=TaskType.UNSUPERVISED):
    """Envuelve un DataFrame en el contrato que espera el EDA."""
    return Dataset(
        frame=frame,
        target=target,
        task=task,
        source=Path("memoria.csv"),
    )


def test_general_cuenta_filas_columnas_y_duplicados():
    # Preparar: la fila (2, 2) aparece dos veces.
    frame = pd.DataFrame({"a": [1, 2, 2, 3], "b": [1, 2, 2, 3]})

    # Ejecutar
    report = EDAnalyzer(_dataset(frame)).run()

    # Verificar
    assert report["general"]["n_rows"] == 4
    assert report["general"]["n_cols"] == 2
    assert report["general"]["n_duplicates"] == 1
    assert report["general"]["missing_cells"] == 0


def test_faltantes_por_columna():
    # Preparar: 'a' tiene dos nulos de cuatro filas.
    frame = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})

    # Ejecutar
    report = EDAnalyzer(_dataset(frame)).run()

    # Verificar
    assert report["missing"]["columns"]["a"]["count"] == 2
    assert report["missing"]["columns"]["a"]["pct"] == 50.0
    assert "b" not in report["missing"]["columns"]
    assert report["general"]["missing_cells"] == 2


def test_estadisticos_numericos():
    # Preparar
    frame = pd.DataFrame({"x": [1, 2, 3, 4]})

    # Ejecutar
    report = EDAnalyzer(_dataset(frame)).run()

    # Verificar
    stats = report["numeric"]["x"]
    assert stats["count"] == 4
    assert stats["mean"] == 2.5
    assert stats["median"] == 2.5
    assert stats["std"] == 1.291


def test_booleano_se_trata_como_categorico():
    # Preparar
    frame = pd.DataFrame({"flag": [True, False, True], "x": [1, 2, 3]})

    # Ejecutar
    report = EDAnalyzer(_dataset(frame)).run()

    # Verificar
    assert "flag" not in report["numeric"]
    assert report["categorical"]["flag"]["n_unique"] == 2


def test_resumen_categorico():
    # Preparar: 'rojo' aparece dos veces de tres.
    frame = pd.DataFrame({"color": ["rojo", "rojo", "azul"]})

    # Ejecutar
    report = EDAnalyzer(_dataset(frame)).run()

    # Verificar
    info = report["categorical"]["color"]
    assert info["n_unique"] == 2
    assert info["top"] == "rojo"
    assert info["top_freq"] == 2
    assert info["top_pct"] == 66.67


def test_detecta_outlier_con_regla_iqr():
    # Preparar: 100 queda fuera del rango [-1, 7].
    frame = pd.DataFrame({"x": [1, 2, 3, 4, 100]})

    # Ejecutar
    report = EDAnalyzer(_dataset(frame)).run()

    # Verificar
    info = report["outliers"]["x"]
    assert info["low"] == -1.0
    assert info["high"] == 7.0
    assert info["count"] == 1


def test_correlacion_y_aviso_de_redundancia():
    # Preparar: y es el doble de x, correlación perfecta.
    frame = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})

    # Ejecutar
    report = EDAnalyzer(_dataset(frame)).run()

    # Verificar
    assert report["correlations"]["pearson"]["x"]["y"] == 1.0
    assert len(report["correlations"]["strong_pairs"]) == 1
    assert any("correlacionadas" in text for text in report["warnings"])


def test_target_de_clasificacion_y_desequilibrio():
    # Preparar: cinco casos de 'a' y uno de 'b'.
    frame = pd.DataFrame(
        {"x": [1, 2, 3, 4, 5, 6], "target": ["a", "a", "a", "a", "a", "b"]}
    )
    dataset = _dataset(frame, target="target", task=TaskType.CLASSIFICATION)

    # Ejecutar
    report = EDAnalyzer(dataset).run()

    # Verificar
    target = report["target"]
    assert target["type"] == "classification"
    assert target["n_classes"] == 2
    assert target["classes"] == {"a": 5, "b": 1}
    assert target["imbalance"] == 5.0
    assert any("desequilibradas" in text for text in report["warnings"])


def test_target_de_regresion_y_correlaciones():
    # Preparar: 'y' es el doble de 'x'.
    frame = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    dataset = _dataset(frame, target="y", task=TaskType.REGRESSION)

    # Ejecutar
    report = EDAnalyzer(dataset).run()

    # Verificar
    target = report["target"]
    assert target["type"] == "regression"
    assert target["stats"]["mean"] == 5.0
    assert target["correlations"] == [{"feature": "x", "pearson": 1.0}]


def test_sin_target_no_analiza_objetivo():
    # Preparar
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    # Ejecutar
    report = EDAnalyzer(_dataset(frame)).run()

    # Verificar
    assert report["target"] is None
    assert report["correlations"]["pearson"]["x"]["y"] == 1.0


def test_avisa_columna_constante():
    # Preparar: 'x' siempre vale 1.
    frame = pd.DataFrame({"x": [1, 1, 1, 1], "feature": [1, 2, 3, 4]})

    # Ejecutar
    report = EDAnalyzer(_dataset(frame)).run()

    # Verificar
    assert any("constante" in text and "'x'" in text for text in report["warnings"])


def test_avisa_categoria_de_alta_cardinalidad():
    # Preparar: 60 valores distintos, más que el umbral de 50.
    frame = pd.DataFrame(
        {
            "categoria": [f"v{i}" for i in range(60)],
            "valor": list(range(60)),
        }
    )

    # Ejecutar
    report = EDAnalyzer(_dataset(frame)).run()

    # Verificar
    assert report["categorical"]["categoria"]["n_unique"] == 60
    assert any("valores distintos" in text for text in report["warnings"])


def test_avisa_columna_completamente_vacia():
    # Preparar
    frame = pd.DataFrame({"vacia": [None, None, None], "x": [1, 2, 3]})

    # Ejecutar
    report = EDAnalyzer(_dataset(frame)).run()

    # Verificar
    assert any("completamente vacía" in text for text in report["warnings"])


def test_guarda_json_sin_haber_ejecutado_run(tmp_path):
    # Preparar
    frame = pd.DataFrame({"x": [1, 2, 3]})
    analyzer = EDAnalyzer(_dataset(frame))
    path = tmp_path / "reports" / "eda.json"

    # Ejecutar
    written = analyzer.save_json(path)

    # Verificar
    assert written == path
    assert path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["general"]["n_rows"] == 3
    assert saved["target"] is None
