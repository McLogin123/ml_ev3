"""Preprocesamiento: división train/test, nulos, categóricas y escalado

La regla principal es evitar fuga de información: todo se ajusta con el
conjunto de entrenamiento y luego se aplica al de prueba
"""

import numpy as np
import pandas as pd

from .schema import TaskType


def _is_numeric(series) -> bool:
    """Indica si la columna es numérica y no booleana"""
    if pd.api.types.is_bool_dtype(series):
        return False
    return pd.api.types.is_numeric_dtype(series)


def _clean_numeric(series):
    """Convierte a float y transforma infinitos en nulos"""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.replace([np.inf, -np.inf], np.nan)


def _validate_test_size(test_size):
    """Comprueba que la fracción de prueba esté entre 0 y 1"""
    if not 0 < test_size < 1:
        raise ValueError("test_size debe ser un número entre 0 y 1.")


def _random_indices(n_rows: int, test_size: float, rng):
    """Devuelve índices de train y test para un split aleatorio simple"""
    if n_rows < 2:
        raise ValueError("Se necesitan al menos 2 filas para dividir los datos.")

    n_test = max(1, int(round(n_rows * test_size)))
    if n_test >= n_rows:
        n_test = n_rows - 1

    indices = np.arange(n_rows)
    rng.shuffle(indices)
    return indices[n_test:], indices[:n_test]


def _stratified_indices(labels, test_size: float, rng):
    """Divide manteniendo aproximadamente la proporción de cada clase

    Cada clase aporta al menos una fila a test y deja al menos una en train

    Si una clase tiene una sola fila, la estratificación es imposible y se
    lanza un error para que el usuario decida (más datos o stratify=False)
    """
    groups = {}
    for position, value in enumerate(labels.to_numpy()):
        groups.setdefault(value, []).append(position)

    train = []
    test = []
    for value, positions in groups.items():
        if len(positions) < 2:
            raise ValueError(
                f"La clase '{value}' tiene una sola fila y no se puede "
                "estratificar. Añada más datos o use stratify=False."
            )

        positions = np.array(positions)
        rng.shuffle(positions)

        n_test = max(1, int(round(len(positions) * test_size)))
        if n_test >= len(positions):
            n_test = len(positions) - 1

        test.extend(positions[:n_test])
        train.extend(positions[n_test:])

    train = np.array(train)
    test = np.array(test)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _encode_target(task, y_train, y_test):
    """Codifica las clases como enteros y deja regresión como float

    Retorna:
        tuple
            (y_train, y_test, classes). classes es la lista de etiquetas
            originales en el orden de los códigos, o None en regresión
    """
    if task == TaskType.REGRESSION:
        return (
            y_train.to_numpy(dtype=float),
            y_test.to_numpy(dtype=float),
            None,
        )

    values = y_train.unique().tolist()
    if len(values) < 2:
        raise ValueError("La clasificación necesita al menos dos clases en train.")

    try:
        classes = sorted(values)
    except TypeError:
        classes = sorted(values, key=str)

    mapping = {value: index for index, value in enumerate(classes)}
    unknown = [value for value in y_test.unique().tolist() if value not in mapping]
    if unknown:
        raise ValueError(
            f"El conjunto de test contiene clases no vistas en train: {unknown}. "
            "Use stratify=True o revise la división."
        )

    train_encoded = np.array([mapping[value] for value in y_train], dtype=int)
    test_encoded = np.array([mapping[value] for value in y_test], dtype=int)
    return train_encoded, test_encoded, classes


class Preprocessor:
    """Imputa, codifica y escala columnas usando solo datos de entrenamiento

    Parametros:
        scale : bool, default=True
            Si es True estandariza las columnas numéricas con la media y la
            desviación del conjunto de entrenamiento
        drop_constant : bool, default=True
            Si es True elimina columnas constantes o completamente vacías

    Notes
    -----
    - Numéricas: los nulos se reemplazan por la mediana de train
    - Categóricas: los nulos se reemplazan por la moda de train y cada valor
      se convierte en una columna 0/1 (*one-hot*)
    - Una categoría no vista en train produce una fila de ceros en esa columna
    """

    def __init__(self, scale: bool = True, drop_constant: bool = True):
        self.scale = scale
        self.drop_constant = drop_constant
        self.numeric_stats = {}
        self.categorical_stats = {}
        self.used_columns = []
        self.feature_names = None
        self._is_fitted = False

    def fit(self, frame):
        """Aprende medianas, modas, categorías y parámetros de escalado

        Parametros:
            frame : pandas.DataFrame
                Datos de entrenamiento, sin la columna objetivo

        Retorna:
            Preprocessor
                El propio objeto, para poder encadenar fit(...).transform

        Raises
        ------
        ValueError
            Si no queda ninguna columna utilizable
        """
        self.numeric_stats = {}
        self.categorical_stats = {}
        self.used_columns = []
        self.feature_names = []
        self._is_fitted = False

        for name in frame.columns:
            series = frame[name]

            if _is_numeric(series):
                clean = _clean_numeric(series)
                valid = clean.dropna()

                if valid.empty:
                    continue
                if self.drop_constant and valid.nunique() <= 1:
                    continue

                median = float(valid.median())
                filled = clean.fillna(median)
                mean = float(filled.mean())
                std = float(filled.std(ddof=0))

                # std=0 en una columna constante: dividir por 1 la deja en cero.
                if not np.isfinite(std) or std == 0:
                    std = 1.0

                self.numeric_stats[name] = {
                    "median": median,
                    "mean": mean,
                    "std": std,
                }
                self.used_columns.append(name)
                self.feature_names.append(str(name))
                continue

            clean = series.dropna()
            if clean.empty:
                continue
            if self.drop_constant and clean.nunique() <= 1:
                continue

            mode = clean.value_counts().index[0]
            categories = sorted(clean.unique().tolist(), key=str)

            self.categorical_stats[name] = {
                "mode": mode,
                "categories": categories,
            }
            self.used_columns.append(name)
            for category in categories:
                self.feature_names.append(f"{name}={category}")

        if not self.feature_names:
            raise ValueError(
                "No quedan columnas utilizables: todas son constantes o vacías."
            )

        self._is_fitted = True
        return self

    def transform(self, frame):
        """Aplica lo aprendido en fit y devuelve una matriz numérica

        Parametros:
            frame : pandas.DataFrame
                Datos a transformar. Debe contener las columnas usadas en fit
                Las columnas extra se ignoran

        Retorna:
            numpy.ndarray
                Matriz de forma (n_filas, n_features) y tipo float

        Raises
        ------
        ValueError
            Si el preprocesador no fue ajustado o faltan columnas requeridas
        """
        if not self._is_fitted:
            raise ValueError("El preprocesador no está ajustado. Use fit() primero.")

        missing = [name for name in self.used_columns if name not in frame.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")

        blocks = []
        for name in self.used_columns:
            series = frame[name]

            if name in self.numeric_stats:
                stats = self.numeric_stats[name]
                values = _clean_numeric(series).fillna(stats["median"])
                block = values.to_numpy(dtype=float).reshape(-1, 1)

                if self.scale:
                    block = (block - stats["mean"]) / stats["std"]
                blocks.append(block)
                continue

            info = self.categorical_stats[name]
            values = series.fillna(info["mode"])
            block = np.zeros((len(frame), len(info["categories"])), dtype=float)

            for index, category in enumerate(info["categories"]):
                block[:, index] = (values == category).to_numpy(dtype=float)
            blocks.append(block)

        if not blocks:
            return np.empty((len(frame), 0))

        return np.hstack(blocks)

    def fit_transform(self, frame):
        """Ajusta el preprocesador y transforma los mismos datos"""
        return self.fit(frame).transform(frame)


class PreparedData:
    """Resultado del preprocesamiento, listo para los modelos

    Parametros:
        X_train : numpy.ndarray
            Matriz de entrenamiento
        X_test : numpy.ndarray or None
            Matriz de prueba. Es None en aprendizaje no supervisado
        y_train : numpy.ndarray or None
            Etiquetas de entrenamiento codificadas, o valores reales en regresión
        y_test : numpy.ndarray or None
            Etiquetas de prueba, o None si no hay conjunto de prueba
        feature_names : list of str
            Nombre de cada columna de X_train después de la codificación
        target : str or None
            Nombre de la columna objetivo
        task : TaskType
            Tarea del pipeline
        classes : list or None
            Etiquetas originales en el orden de los códigos de clasificación
        dropped_rows : int
            Filas eliminadas por tener el target vacío
        preprocessor : Preprocessor or None
            Preprocesador ajustado, para transformar datos nuevos
    """

    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        target,
        task,
        classes=None,
        dropped_rows=0,
        preprocessor=None,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.target = target
        self.task = task
        self.classes = classes
        self.dropped_rows = dropped_rows
        self.preprocessor = preprocessor


def prepare_dataset(
    dataset,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
    drop_constant: bool = True,
    stratify: bool = True,
):
    """Divide, ajusta y transforma un Dataset completo

    En supervisado devuelve train/test separados y etiquetas codificadas

    En no supervisado transforma todas las filas y deja X_test y "y"
    en None, porque el clustering no necesita holdout para ajustarse

    Parametros:
        dataset : core.schema.Dataset
            Datos ya cargados y validados
        test_size : float, default=0.2
            Fracción de filas reservadas para prueba
        random_state : int, default=42
            Semilla para que la división sea reproducible
        scale : bool, default=True
            Estandariza las columnas numéricas
        drop_constant : bool, default=True
            Elimina columnas constantes o vacías
        stratify : bool, default=True
            Mantiene la proporción de clases en clasificación

    Retorna:
        PreparedData
            Matrices, etiquetas, nombres de features y el preprocesador ajustado

    Raises
    ------
    ValueError
        Si test_size no es válido, hay menos de dos filas o la
        estratificación no es posible
    """
    _validate_test_size(test_size)

    features = dataset.features()
    labels = dataset.labels()
    preprocessor = Preprocessor(scale=scale, drop_constant=drop_constant)

    if not dataset.is_supervised():
        matrix = preprocessor.fit_transform(features)
        return PreparedData(
            X_train=matrix,
            X_test=None,
            y_train=None,
            y_test=None,
            feature_names=preprocessor.feature_names,
            target=dataset.target,
            task=dataset.task,
            classes=None,
            dropped_rows=0,
            preprocessor=preprocessor,
        )

    if labels is None:
        raise ValueError("Un dataset supervisado necesita una columna objetivo.")

    # No se puede entrenar sin etiqueta: se eliminan esas filas antes de dividir.
    valid = labels.notna()
    dropped_rows = int((~valid).sum())
    features = features.loc[valid]
    labels = labels.loc[valid]

    if len(features) < 2:
        raise ValueError("Quedan menos de 2 filas con target para dividir.")

    if dataset.task == TaskType.CLASSIFICATION and labels.nunique() < 2:
        raise ValueError("La clasificación necesita al menos dos clases.")

    rng = np.random.default_rng(random_state)
    if dataset.task == TaskType.CLASSIFICATION and stratify:
        train_index, test_index = _stratified_indices(labels, test_size, rng)
    else:
        train_index, test_index = _random_indices(len(features), test_size, rng)

    X_train = preprocessor.fit_transform(features.iloc[train_index])
    X_test = preprocessor.transform(features.iloc[test_index])

    y_train, y_test, classes = _encode_target(
        dataset.task,
        labels.iloc[train_index],
        labels.iloc[test_index],
    )

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=preprocessor.feature_names,
        target=dataset.target,
        task=dataset.task,
        classes=classes,
        dropped_rows=dropped_rows,
        preprocessor=preprocessor,
    )
