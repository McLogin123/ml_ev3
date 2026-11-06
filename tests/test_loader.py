"""Pruebas del módulo de ingesta; no dependen de datasets externos."""

import pandas as pd
import pytest

from core.loader import DataLoadError, TabularLoader
from core.schema import TaskType


def _write_csv(tmp_path, text: str, name: str = "data.csv", encoding="utf-8"):
    """Guarda un CSV temporal y devuelve su ruta."""
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


def test_detecta_target_y_clasificacion(tmp_path):
    # Preparar
    path = _write_csv(tmp_path, "a,b,target\n1,2,0\n2,3,1\n3,4,0\n")

    # Ejecutar
    dataset = TabularLoader().load(path)

    # Verificar
    assert dataset.target == "target"
    assert dataset.task == TaskType.CLASSIFICATION
    assert list(dataset.features().columns) == ["a", "b"]
    assert dataset.warnings


def test_detecta_regresion_por_cardinalidad(tmp_path):
    # Preparar: 60 valores distintos en 'y', más de max_classes.
    rows = "\n".join(f"{i},{i * 1.5}" for i in range(60))
    path = _write_csv(tmp_path, f"x,y\n{rows}\n", name="regresion.csv")

    # Ejecutar
    dataset = TabularLoader().load(path)

    # Verificar
    assert dataset.target == "y"
    assert dataset.task == TaskType.REGRESSION


def test_target_explicito_inexistente_falla(tmp_path):
    # Preparar
    path = _write_csv(tmp_path, "a,b\n1,2\n")

    # Ejecutar y verificar
    with pytest.raises(DataLoadError, match="no existe"):
        TabularLoader().load(path, target="objetivo")


def test_sin_evidencia_enruta_a_no_supervisado(tmp_path):
    # Preparar: tres columnas continuas, ninguna con nombre de target.
    rows = "\n".join(f"{i},{i * 2.0},{i * 3.0}" for i in range(30))
    path = _write_csv(tmp_path, f"f1,f2,f3\n{rows}\n", name="sin_etiquetas.csv")

    # Ejecutar
    dataset = TabularLoader().load(path)

    # Verificar
    assert dataset.task == TaskType.UNSUPERVISED
    assert dataset.target is None
    assert dataset.labels() is None
    assert "no supervisado" in dataset.warnings[0].lower()


def test_no_elige_identificadores_como_target(tmp_path):
    # Preparar: 'id' se repite, pero su nombre lo delata como identificador.
    rows = "\n".join(f"{1 + i % 5},{i * 1.0},{i % 2}" for i in range(20))
    path = _write_csv(tmp_path, f"id,f1,label\n{rows}\n", name="ids.csv")

    # Ejecutar
    dataset = TabularLoader().load(path)

    # Verificar
    assert dataset.target == "label"


def test_forzar_no_supervisado(tmp_path):
    # Preparar
    path = _write_csv(tmp_path, "a,target\n1,0\n2,1\n")

    # Ejecutar
    dataset = TabularLoader().load(path, task=TaskType.UNSUPERVISED)

    # Verificar
    assert dataset.target is None
    assert dataset.task == TaskType.UNSUPERVISED


def test_target_y_no_supervisado_incompatibles(tmp_path):
    # Preparar
    path = _write_csv(tmp_path, "a,target\n1,0\n2,1\n")

    # Ejecutar y verificar
    with pytest.raises(DataLoadError, match="task=unsupervised"):
        TabularLoader().load(path, target="target", task=TaskType.UNSUPERVISED)


def test_detecta_punto_y_coma_con_decimales(tmp_path):
    # Preparar: separador ';' y decimales con ','.
    path = _write_csv(
        tmp_path,
        "a;b;objetivo\n1;3,5;0\n2;4,5;1\n3;5,5;0\n",
        name="europeo.csv",
    )

    # Ejecutar
    dataset = TabularLoader().load(path)

    # Verificar
    assert list(dataset.frame.columns) == ["a", "b", "objetivo"]
    assert dataset.frame.shape == (3, 3)
    assert dataset.target == "objetivo"


def test_separador_explicito(tmp_path):
    # Preparar: sin el parámetro, se detectaría ',' por orden de preferencia.
    path = _write_csv(tmp_path, "a;b,c\n1;2,3\n", name="ambos.csv")

    # Ejecutar
    dataset = TabularLoader().load(path, sep=";")

    # Verificar
    assert list(dataset.frame.columns) == ["a", "b,c"]


def test_lee_codificacion_latin1(tmp_path):
    # Preparar
    path = tmp_path / "latin1.csv"
    path.write_bytes(
        "ciudad,precio,objetivo\nMálaga,10,1\nSevilla,20,0\n".encode("latin-1")
    )

    # Ejecutar
    dataset = TabularLoader().load(path)

    # Verificar
    assert dataset.frame.loc[0, "ciudad"] == "Málaga"
    assert dataset.target == "objetivo"


def test_lee_tsv(tmp_path):
    # Preparar
    path = _write_csv(tmp_path, "a\tb\ttarget\n1\t2\t0\n2\t3\t1\n", name="data.tsv")

    # Ejecutar
    dataset = TabularLoader().load(path)

    # Verificar
    assert list(dataset.frame.columns) == ["a", "b", "target"]


def test_lee_parquet(tmp_path):
    # Preparar
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "target": [0, 1, 0]})
    path = tmp_path / "data.parquet"
    frame.to_parquet(path, index=False)

    # Ejecutar
    dataset = TabularLoader().load(path)

    # Verificar
    assert dataset.target == "target"


def test_encabezados_duplicados_falla(tmp_path):
    # Preparar
    path = _write_csv(tmp_path, "a,b,a\n1,2,3\n")

    # Ejecutar y verificar
    with pytest.raises(DataLoadError, match="duplicados"):
        TabularLoader().load(path)


def test_archivo_vacio_falla(tmp_path):
    # Preparar
    path = _write_csv(tmp_path, "")

    # Ejecutar y verificar
    with pytest.raises(DataLoadError, match="vac"):
        TabularLoader().load(path)


def test_formato_no_soportado_falla(tmp_path):
    # Preparar
    path = _write_csv(tmp_path, "{}", name="data.json")

    # Ejecutar y verificar
    with pytest.raises(DataLoadError, match="no soportado"):
        TabularLoader().load(path)


def test_target_constante_falla(tmp_path):
    # Preparar
    path = _write_csv(tmp_path, "a,target\n1,0\n2,0\n", name="constante.csv")

    # Ejecutar y verificar
    with pytest.raises(DataLoadError, match="constante"):
        TabularLoader().load(path, target="target")


def test_tarea_desconocida_falla(tmp_path):
    # Preparar
    path = _write_csv(tmp_path, "a,target\n1,0\n2,1\n")

    # Ejecutar y verificar
    with pytest.raises(DataLoadError, match="Tarea desconocida"):
        TabularLoader().load(path, task="cluster")
