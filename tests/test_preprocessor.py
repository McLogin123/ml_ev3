"""Pruebas del preprocesador: nulos, categóricas, escalado y split."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.loader import TabularLoader
from core.preprocessor import Preprocessor, prepare_dataset
from core.schema import Dataset, TaskType


def _dataset(frame, target=None, task=TaskType.UNSUPERVISED):
    """Envuelve un DataFrame en el contrato que espera el preprocesador."""
    return Dataset(
        frame=frame,
        target=target,
        task=task,
        source=Path("memoria.csv"),
    )


def test_imputa_numericos_con_mediana_de_train():
    # Preparar: la mediana de train (1 y 3) es 2.
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    test = pd.DataFrame({"x": [np.nan]})
    preprocessor = Preprocessor(scale=False)

    # Ejecutar
    X_train = preprocessor.fit_transform(train)
    X_test = preprocessor.transform(test)

    # Verificar
    assert X_train[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert X_test[0, 0] == 2.0


def test_escala_con_media_y_desviacion_de_train():
    # Preparar: media 5 y desviación poblacional 5.
    train = pd.DataFrame({"x": [0.0, 10.0]})
    preprocessor = Preprocessor()

    # Ejecutar
    X_train = preprocessor.fit_transform(train)
    X_test = preprocessor.transform(pd.DataFrame({"x": [20.0]}))

    # Verificar
    assert X_train[:, 0].tolist() == [-1.0, 1.0]
    assert X_test[0, 0] == 3.0


def test_test_no_influye_en_la_escala():
    # Preparar: el valor 1000 no debe cambiar la media ni la desviación.
    preprocessor = Preprocessor()
    preprocessor.fit(pd.DataFrame({"x": [0.0, 10.0]}))

    # Ejecutar
    X_test = preprocessor.transform(pd.DataFrame({"x": [1000.0]}))

    # Verificar: (1000 - 5) / 5 = 199
    assert X_test[0, 0] == 199.0


def test_escalado_desactivado():
    # Preparar
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    preprocessor = Preprocessor(scale=False)

    # Ejecutar
    X_train = preprocessor.fit_transform(train)

    # Verificar
    assert X_train[:, 0].tolist() == [1.0, 2.0, 3.0]


def test_infinitos_se_tratan_como_nulos():
    # Preparar: inf se vuelve NaN y se imputa con la mediana (2).
    train = pd.DataFrame({"x": [1.0, np.inf, 3.0]})
    preprocessor = Preprocessor(scale=False)

    # Ejecutar
    X_train = preprocessor.fit_transform(train)

    # Verificar
    assert X_train[:, 0].tolist() == [1.0, 2.0, 3.0]


def test_one_hot_con_categorias_de_train():
    # Preparar
    train = pd.DataFrame({"color": ["rojo", "azul", "rojo"]})
    preprocessor = Preprocessor(scale=False)

    # Ejecutar
    X_train = preprocessor.fit_transform(train)

    # Verificar: categorías ordenadas alfabéticamente.
    assert preprocessor.feature_names == ["color=azul", "color=rojo"]
    assert X_train.tolist() == [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]


def test_categoria_desconocida_produce_ceros():
    # Preparar
    preprocessor = Preprocessor(scale=False)
    preprocessor.fit(pd.DataFrame({"color": ["rojo", "azul"]}))

    # Ejecutar
    X_test = preprocessor.transform(pd.DataFrame({"color": ["verde"]}))

    # Verificar
    assert X_test.tolist() == [[0.0, 0.0]]


def test_booleano_se_codifica_como_categorico():
    # Preparar
    train = pd.DataFrame({"flag": [True, False, True]})
    preprocessor = Preprocessor(scale=False)

    # Ejecutar
    X_train = preprocessor.fit_transform(train)

    # Verificar
    assert preprocessor.feature_names == ["flag=False", "flag=True"]
    assert X_train.tolist() == [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]


def test_nulo_categorico_usa_la_moda_de_train():
    # Preparar: la moda es 'a'.
    train = pd.DataFrame({"color": ["a", None, "a", "b"]})
    preprocessor = Preprocessor(scale=False)
    preprocessor.fit(train)

    # Ejecutar
    X_test = preprocessor.transform(pd.DataFrame({"color": [None]}))

    # Verificar
    assert X_test.tolist() == [[1.0, 0.0]]


def test_elimina_columnas_constantes():
    # Preparar
    train = pd.DataFrame({"constante": [1, 1, 1], "x": [1, 2, 3]})
    preprocessor = Preprocessor(scale=False)

    # Ejecutar
    X_train = preprocessor.fit_transform(train)

    # Verificar
    assert preprocessor.feature_names == ["x"]
    assert X_train.shape == (3, 1)


def test_puede_conservar_columnas_constantes():
    # Preparar
    train = pd.DataFrame({"constante": [1, 1, 1], "x": [1, 2, 3]})
    preprocessor = Preprocessor(scale=False, drop_constant=False)

    # Ejecutar
    X_train = preprocessor.fit_transform(train)

    # Verificar
    assert preprocessor.feature_names == ["constante", "x"]
    assert X_train[:, 0].tolist() == [1.0, 1.0, 1.0]


def test_columna_vacia_se_elimina_siempre():
    # Preparar
    train = pd.DataFrame({"vacia": [np.nan, np.nan], "x": [1.0, 2.0]})
    preprocessor = Preprocessor(scale=False, drop_constant=False)

    # Ejecutar
    X_train = preprocessor.fit_transform(train)

    # Verificar
    assert preprocessor.feature_names == ["x"]
    assert X_train.shape == (2, 1)


def test_transform_antes_de_fit_falla():
    # Preparar
    preprocessor = Preprocessor()

    # Ejecutar y verificar
    with pytest.raises(ValueError, match="ajustado"):
        preprocessor.transform(pd.DataFrame({"x": [1, 2]}))


def test_transform_sin_columna_requerida_falla():
    # Preparar
    preprocessor = Preprocessor()
    preprocessor.fit(pd.DataFrame({"x": [1, 2]}))

    # Ejecutar y verificar
    with pytest.raises(ValueError, match="Faltan columnas"):
        preprocessor.transform(pd.DataFrame({"y": [1, 2]}))


def test_sin_columnas_utilizables_falla():
    # Preparar
    frame = pd.DataFrame({"constante": [1, 1, 1]})

    # Ejecutar y verificar
    with pytest.raises(ValueError, match="utilizables"):
        Preprocessor().fit(frame)


def test_split_regresion_tamano_esperado():
    # Preparar
    frame = pd.DataFrame(
        {"x": np.arange(100, dtype=float), "target": np.arange(100, dtype=float)}
    )
    dataset = _dataset(frame, target="target", task=TaskType.REGRESSION)

    # Ejecutar
    prepared = prepare_dataset(dataset, test_size=0.25, random_state=1)

    # Verificar
    assert prepared.X_train.shape == (75, 1)
    assert prepared.X_test.shape == (25, 1)
    assert prepared.y_train.dtype == float
    assert prepared.classes is None


def test_split_estratificado_conserva_proporcion():
    # Preparar: 80 casos de 'a' y 20 de 'b'.
    frame = pd.DataFrame(
        {
            "x": np.arange(100, dtype=float),
            "target": ["a"] * 80 + ["b"] * 20,
        }
    )
    dataset = _dataset(frame, target="target", task=TaskType.CLASSIFICATION)

    # Ejecutar
    prepared = prepare_dataset(dataset, test_size=0.2, random_state=1)

    # Verificar: 20 % de cada clase.
    values, counts = np.unique(prepared.y_test, return_counts=True)
    assert dict(zip(values.tolist(), counts.tolist())) == {0: 16, 1: 4}
    assert len(prepared.X_test) == 20


def test_codifica_etiquetas_de_clasificacion():
    # Preparar
    frame = pd.DataFrame(
        {
            "x": np.arange(12, dtype=float),
            "target": ["rojo", "azul"] * 6,
        }
    )
    dataset = _dataset(frame, target="target", task=TaskType.CLASSIFICATION)

    # Ejecutar
    prepared = prepare_dataset(dataset, test_size=0.25, random_state=2)

    # Verificar
    assert prepared.classes == ["azul", "rojo"]
    assert set(prepared.y_train.tolist()) == {0, 1}
    assert prepared.y_train.dtype.kind == "i"


def test_elimina_filas_sin_target():
    # Preparar: una fila no tiene etiqueta.
    frame = pd.DataFrame(
        {"x": [1, 2, 3, 4, 5, 6], "target": [0, 1, None, 0, 1, 0]}
    )
    dataset = _dataset(frame, target="target", task=TaskType.CLASSIFICATION)

    # Ejecutar
    prepared = prepare_dataset(dataset, test_size=0.2, random_state=3)

    # Verificar
    assert prepared.dropped_rows == 1
    assert len(prepared.y_train) + len(prepared.y_test) == 5


def test_no_supervisado_usa_todas_las_filas():
    # Preparar
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    dataset = _dataset(frame)

    # Ejecutar
    prepared = prepare_dataset(dataset, test_size=0.5)

    # Verificar
    assert prepared.X_train.shape == (3, 2)
    assert prepared.X_test is None
    assert prepared.y_train is None
    assert prepared.y_test is None
    assert prepared.classes is None


def test_target_de_regresion_queda_como_float():
    # Preparar
    frame = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "target": [1.5, 2.5, 3.5, 4.5]}
    )
    dataset = _dataset(frame, target="target", task=TaskType.REGRESSION)

    # Ejecutar
    prepared = prepare_dataset(dataset, test_size=0.25, random_state=4)

    # Verificar
    assert prepared.y_train.dtype == float
    assert prepared.classes is None


def test_test_size_invalido_falla():
    # Preparar
    frame = pd.DataFrame({"x": [1.0, 2.0], "target": [0.0, 1.0]})
    dataset = _dataset(frame, target="target", task=TaskType.REGRESSION)

    # Ejecutar y verificar
    with pytest.raises(ValueError, match="test_size"):
        prepare_dataset(dataset, test_size=1.5)


def test_clasificacion_con_una_sola_clase_falla():
    # Preparar
    frame = pd.DataFrame({"x": [1, 2, 3, 4], "target": ["a"] * 4})
    dataset = _dataset(frame, target="target", task=TaskType.CLASSIFICATION)

    # Ejecutar y verificar
    with pytest.raises(ValueError, match="dos clases"):
        prepare_dataset(dataset)


def test_estratificacion_con_clase_de_una_fila_falla():
    # Preparar: 'b' solo aparece una vez.
    frame = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "target": ["a"] * 5 + ["b"]})
    dataset = _dataset(frame, target="target", task=TaskType.CLASSIFICATION)

    # Ejecutar y verificar
    with pytest.raises(ValueError, match="estratificar"):
        prepare_dataset(dataset)


def test_integracion_con_loader(tmp_path):
    # Preparar: 12 filas, una edad nula y tres ciudades.
    rows = []
    for i in range(12):
        age = "" if i == 3 else str(20 + i)
        city = ["Madrid", "Lima", "Bogota"][i % 3]
        rows.append(f"{age},{city},{i % 2}")
    text = "edad,ciudad,target\n" + "\n".join(rows) + "\n"
    path = tmp_path / "datos.csv"
    path.write_text(text, encoding="utf-8")

    # Ejecutar
    dataset = TabularLoader().load(path)
    prepared = prepare_dataset(dataset, test_size=0.25, random_state=5)

    # Verificar
    assert dataset.target == "target"
    assert prepared.task == TaskType.CLASSIFICATION
    assert prepared.X_train.shape[0] + prepared.X_test.shape[0] == 12
    assert not np.isnan(prepared.X_train).any()
    assert any(name.startswith("ciudad=") for name in prepared.feature_names)
