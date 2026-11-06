"""Ingesta de archivos tabulares y detección automática de la tarea.

    Este módulo solo lee y valida.
"""

from pathlib import Path
from .schema import Dataset, TaskType
import pandas as pd
import re

_SEPARADOR_POR_DEFECTO = {".csv": ",", ".tsv": "\t", ".txt": "\t"}
_EXTENSIONES = {".csv", ".tsv", ".txt", ".parquet"}
_CODIFICACIONES = ("utf-8-sig", "utf-8", "latin-1")
_SEPARADORES = (",", ";", "\t", "|")
_CARACTERES_MUESTRA = 8192
_MAX_CLASES = 20
_MIN_FILAS_PARA_ID = 10
_RATIO_ID = 0.95

# Nombres que suelen indicar la columna objetivo.
_PISTAS_TARGET = frozenset({
    "target", "label", "class", "clase", "etiqueta", "objetivo", "y",
    "survived", "churn", "diagnosis", "diagnostico", "species", "quality",
    "default", "fraud", "price", "precio", 
})

# Nombres que suelen indicar un identificador de fila.
_PISTAS_ID = frozenset({"id", "uuid", "code", "codigo", "index", "unnamed", "row"})


class DataLoadError(Exception):
    """Error de carga con un mensaje que explica la causa al usuario"""


def _tokens(name) -> set[str]:
    """Convierte 'customer_id' en {'customer', 'id'}"""
    return {token for token in re.split(r"[^a-z0-9]+", str(name).lower()) if token}


class TabularLoader:
    """Carga CSV, TSV o Parquet y decide qué tarea de aprendizaje ejecutar

    Parametros:
        max_classes : int, default=20
            Si una columna numérica tiene como máximo este número de valores
            distintos, se trata como categórica (clasificación); si tiene más,
            como continua (regresión).

    Notes
    -----
    La detección del target usa el nombre de la columna y su cardinalidad.
    No usa pruebas estadísticas: elegir como target la columna más relacionada
    con las demás sería circular y contaminaría el análisis.
    """

    def __init__(self, max_classes: int = _MAX_CLASES) -> None:
        if max_classes < 2:
            raise ValueError("max_classes debe ser >= 2")
        self.max_classes = max_classes

    def load(self, path, target=None, task=None, sep=None) -> Dataset:
        """Lee un archivo y devuelve un ``Dataset`` listo para el pipeline.

        Parametros:
            path : str or pathlib.Path
                Archivo de entrada (CSV, TSV o Parquet)
            target : str, optional
                Columna objetivo. Si se omite, se intenta detectar
            task : TaskType or str, optional
                Fuerza la tarea. "unsupervised" ignora cualquier target
            sep : str, optional
                Separador explícito para CSV/TSV. Si es None, se detecta

        Retorna:
            Dataset

        Raises:
            DataLoadError
                Si el archivo no existe, está vacío, tiene encabezados repetidos,
                el target no existe o la configuración es contradictoria.
            """
        source = Path(path)

        # Paso 1: leer y validar lo básico.
        if not source.is_file():
            raise DataLoadError(f"No existe el archivo: {source}")
        frame = self._read(source, sep)
        if frame.empty or frame.shape[1] == 0:
            raise DataLoadError(f"{source.name} no contiene filas o columnas útiles.")
        if target is not None and target not in frame.columns:
            raise DataLoadError(
                f"La columna objetivo '{target}' no existe. "
                f"Columnas disponibles: {list(frame.columns)}"
            )

        # Paso 2: normalizar la tarea que pidió el usuario.
        if task is not None:
            try:
                task = TaskType(task)
            except ValueError as error:
                raise DataLoadError(f"Tarea desconocida: {task}") from error

        # Paso 3: decidir target y tarea.
        warnings = []
        if task == TaskType.UNSUPERVISED:
            if target is not None:
                raise DataLoadError(
                    "No se puede indicar target y task=unsupervised a la vez."
                )
            warnings.append("Aprendizaje no supervisado forzado: se ignora el target.")
        elif target is None:
            if task is not None:
                raise DataLoadError(
                    "Una tarea supervisada requiere indicar la columna target."
                )
            target, task, warnings = self._detect(frame)
        elif task is None:
            task = self._task_for(frame[target])

        # Paso 4: avisar si el target parece un identificador.
        if target is not None and task == TaskType.CLASSIFICATION:
            n_classes = int(frame[target].nunique(dropna=True))
            if n_classes > self.max_classes:
                warnings.append(
                    f"Target '{target}' tiene {n_classes} clases "
                    f"(> {self.max_classes}): revise si es un identificador "
                    "o si conviene agrupar categorías."
                )

        return Dataset(
            frame=frame,
            target=target,
            task=task,
            source=source,
            warnings=tuple(warnings),
        )

    def _read(self, source: Path, sep) -> pd.DataFrame:
        """Elige el lector según la extensión y prueba varias codificaciones"""
        extension = source.suffix.lower()
        if extension not in _EXTENSIONES:
            raise DataLoadError(
                f"Formato no soportado '{extension}'. "
                f"Use uno de {sorted(_EXTENSIONES)}."
            )

        if extension == ".parquet":
            try:
                return pd.read_parquet(source)
            except ImportError as error:
                raise DataLoadError(
                    "Parquet requiere 'pyarrow' instalado: pip install pyarrow"
                ) from error

        last_error = None
        for encoding in _CODIFICACIONES:
            try:
                return self._read_delimited(source, encoding, sep)
            except UnicodeDecodeError as error:
                last_error = error
        raise DataLoadError(
            f"No se pudo decodificar {source.name} con ninguna de {_CODIFICACIONES}."
        ) from last_error

    def _read_delimited(self, source: Path, encoding: str, sep) -> pd.DataFrame:
        """Lee un CSV/TSV y valida que los encabezados no estén repetidos."""
        with source.open("r", encoding=encoding) as file:
            sample = file.read(_CARACTERES_MUESTRA)

        lines = [line for line in sample.splitlines() if line.strip()]
        if not lines:
            raise DataLoadError(f"El archivo {source.name} está vacío.")

        if sep is None:
            default = _SEPARADOR_POR_DEFECTO.get(source.suffix.lower(), ",")
            sep = self._detect_separator(lines, default)

        header = [cell.strip().strip('"') for cell in lines[0].split(sep)]
        repeated = sorted({cell for cell in header if header.count(cell) > 1})
        if repeated:
            raise DataLoadError(
                f"Encabezados duplicados en {source.name}: {repeated}. "
                "Renombre las columnas en el archivo de origen."
            )

        return pd.read_csv(source, sep=sep, encoding=encoding)

    def _detect_separator(self, lines, default: str) -> str:
        """Elige el separador que aparece con el mismo conteo en cada línea.

        Un archivo bien delimitado tiene igual número de separadores en todas
        las filas. Un separador que aparece dentro de un texto o de un decimal
        ej: (3,14), rompe esa uniformidad y pierde.
        """
        first_lines = lines[:25]
        if len(first_lines) < 2:
            return default

        best = default
        best_score = -1.0
        for candidate in _SEPARADORES:
            counts = [line.count(candidate) for line in first_lines]
            low = min(counts)
            high = max(counts)
            if low == 0:
                continue
            score = low / high
            if score > best_score:
                best = candidate
                best_score = score
        return best

    def _detect(self, frame):
        """Puntúa las columnas y elige la mejor candidata a target

        Reglas:
            1. Se descartan las columnas constantes y los identificadores.
            2. Una columna discreta vale más que una continua.
            3. Un nombre reconocible (target, label, price…) vale más.
            4. En caso de empate gana la columna que aparece más a la derecha.
        """
        best_score = -1.0
        best_column = None

        for column in frame.columns:
            series = frame[column]
            n_unique = int(series.nunique(dropna=True))
            if n_unique < 2:
                continue
            if self._looks_like_identifier(column, series, n_unique):
                continue

            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_discrete = (not is_numeric) or (n_unique <= self.max_classes)
            has_hint = bool(_tokens(column) & _PISTAS_TARGET)

            if not is_discrete and not has_hint:
                continue

            if is_discrete:
                score = 2.0 if has_hint else 1.0
            else:
                score = 1.5

            if score >= best_score:
                best_score = score
                best_column = str(column)

        if best_column is None:
            return None, TaskType.UNSUPERVISED, [
                "No se encontró una columna candidata a target (categórica de "
                "baja cardinalidad, objetivo continuo con nombre reconocible o "
                "nombre explícito). Se enruta a aprendizaje no supervisado "
                f"(K-Means). Filas={len(frame)}, columnas={frame.shape[1]}."
            ]

        task = self._task_for(frame[best_column])
        return best_column, task, [
            f"Target inferido automáticamente: '{best_column}' ({task.value}). "
            "Use el parámetro target para fijarlo manualmente."
        ]

    def _looks_like_identifier(self, name, series, n_unique: int) -> bool:
        """Descarta columnas que identifican filas en lugar de describirlas.

        Se descartan siempre por nombre (id, uuid, code…). La
        proporción de valores únicos solo se usa con 10 filas o más: en tablas
        diminutas todas las columnas parecen identificadores
        """
        if _tokens(name) & _PISTAS_ID:
            return True
        if len(series) < _MIN_FILAS_PARA_ID:
            return False
        if pd.api.types.is_float_dtype(series):
            return False
        return n_unique >= _RATIO_ID * len(series)

    def _task_for(self, y) -> TaskType:
        """Decide clasificación o regresión mirando el tipo y la cardinalidad"""
        n_unique = int(y.nunique(dropna=True))
        if n_unique < 2:
            raise DataLoadError(f"La columna objetivo '{y.name}' es constante.")
        if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
            return TaskType.CLASSIFICATION
        if n_unique <= self.max_classes:
            return TaskType.CLASSIFICATION
        return TaskType.REGRESSION
