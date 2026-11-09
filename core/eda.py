"""Análisis exploratorio automático.

Este módulo solo describe los datos: no imputa, no escala y no elimina nada.
"""

import json
import pandas as pd
from pathlib import Path
from .schema import TaskType

# Umbrales usados para generar avisos. Están juntos para poder ajustarlos.
_MAX_CATEGORIAS = 50
_MAX_CORRELACION = 0.90
_MAX_DESEQUILIBRIO = 3.0
_AVISO_FALTANTES = 50.0


def _number(value, digits: int = 4):
    """Convierte un valor de pandas en un número JSON o None

    None representa NaN. JSON no admite NaN y algunos valores de pandas
    tampoco se pueden serializar directamente
    """
    if pd.isna(value):
        return None
    return round(float(value), digits)


def _pct(part: int, total: int) -> float:
    """Porcentaje redondeado; devuelve 0.0 si no hay datos"""
    if total == 0:
        return 0.0
    return round(100.0 * part / total, 2)


def _numeric_columns(frame):
    """Nombres de las columnas numéricas reales, sin booleanos.

    En pandas un booleano cuenta como numérico, pero para el EDA se comporta
    como categórico (dos valores posibles)
    """
    names = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            names.append(name)
    return names


def _numeric_stats(series):
    """Resumen de una columna numérica, calculado sin los valores faltantes"""
    clean = series.dropna()
    if clean.empty:
        return {"count": 0}

    return {
        "count": int(clean.count()),
        "mean": _number(clean.mean()),
        "std": _number(clean.std()),
        "min": _number(clean.min()),
        "q1": _number(clean.quantile(0.25)),
        "median": _number(clean.median()),
        "q3": _number(clean.quantile(0.75)),
        "max": _number(clean.max()),
        "skew": _number(clean.skew()),
    }


class EDAnalyzer:
    """Describe un Dataset y produce avisos para el preprocesador

    Parametros:
        dataset : core.schema.Dataset
            Datos ya cargados y validados por el loader

    Notes
    -----
    run() recorre la tabla una vez por bloque de análisis. Se prioriza
    que el código se lea sobre microoptimizaciones que no cambian el resultado
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.report = None

    def run(self) -> dict:
        """Calcula el análisis completo y lo guarda en self.report.

        Retorna:
            dict
                Bloques general, missing, numeric, categorical,
                outliers, correlations, target y warnings
        """
        frame = self.dataset.frame
        numeric = _numeric_columns(frame)

        report = {
            "general": self._general(frame),
            "missing": self._missing(frame),
            "numeric": self._numeric(frame, numeric),
            "categorical": self._categorical(frame, numeric),
            "outliers": self._outliers(frame, numeric),
            "correlations": self._correlations(frame, numeric),
            "target": self._target(frame, numeric),
        }
        report["warnings"] = self._warnings(frame, report)

        self.report = report
        return report

    def save_json(self, path) -> Path:
        """Guarda el último análisis en un archivo JSON legible

        Si run() no se ha llamado, lo ejecuta primero

        Parametros:
            path : str or pathlib.Path
                Archivo de salida Se crean las carpetas que falten

        Retorna:
            pathlib.Path
                Ruta del archivo escrito
        """
        if self.report is None:
            self.run()

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.report, indent=2, ensure_ascii=False)
        destination.write_text(text, encoding="utf-8")
        return destination

    def _general(self, frame) -> dict:
        """Tamaño, celdas faltantes y filas duplicadas de toda la tabla"""
        n_rows = int(len(frame))
        n_cols = int(frame.shape[1])
        n_cells = n_rows * n_cols
        missing_cells = int(frame.isna().sum().sum())
        duplicates = int(frame.duplicated().sum())

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "missing_cells": missing_cells,
            "missing_pct": _pct(missing_cells, n_cells),
            "n_duplicates": duplicates,
            "duplicates_pct": _pct(duplicates, n_rows),
        }

    def _missing(self, frame) -> dict:
        """Conteo de faltantes por columna; solo lista las que tienen alguno"""
        total = int(len(frame))
        columns = {}

        for name in frame.columns:
            count = int(frame[name].isna().sum())
            if count == 0:
                continue
            columns[str(name)] = {
                "count": count,
                "pct": _pct(count, total),
            }

        missing_cells = int(frame.isna().sum().sum())
        n_cells = int(frame.shape[0] * frame.shape[1])
        return {
            "missing_cells": missing_cells,
            "missing_pct": _pct(missing_cells, n_cells),
            "columns": columns,
        }

    def _numeric(self, frame, numeric) -> dict:
        """Estadísticos descriptivos de cada columna numérica"""
        result = {}
        for name in numeric:
            result[str(name)] = _numeric_stats(frame[name])
        return result

    def _categorical(self, frame, numeric) -> dict:
        """Cardinalidad y valor más frecuente de las columnas no numéricas"""
        result = {}
        for name in frame.columns:
            if name in numeric:
                continue

            clean = frame[name].dropna()
            if clean.empty:
                result[str(name)] = {
                    "count": 0,
                    "n_unique": 0,
                    "top": None,
                    "top_freq": 0,
                    "top_pct": 0.0,
                }
                continue

            counts = clean.value_counts()
            top_freq = int(counts.iloc[0])
            result[str(name)] = {
                "count": int(len(clean)),
                "n_unique": int(clean.nunique()),
                "top": str(counts.index[0]),
                "top_freq": top_freq,
                "top_pct": _pct(top_freq, len(clean)),
            }
        return result

    def _outliers(self, frame, numeric) -> dict:
        """Cuenta valores fuera del rango Q1 - 1.5*IQR y Q3 + 1.5*IQR.

        Es la regla de Tukey. No decide si un valor es un error: solo señala
        cuántos quedan lejos del cuerpo central de la distribución
        """
        result = {}
        for name in numeric:
            clean = frame[name].dropna()
            if clean.empty:
                continue

            q1 = clean.quantile(0.25)
            q3 = clean.quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            count = int(((clean < low) | (clean > high)).sum())

            result[str(name)] = {
                "q1": _number(q1),
                "q3": _number(q3),
                "iqr": _number(iqr),
                "low": _number(low),
                "high": _number(high),
                "count": count,
                "pct": _pct(count, len(clean)),
            }
        return result

    def _correlations(self, frame, numeric) -> dict:
        """Correlación de Pearson entre columnas numéricas.

        También devuelve los pares con |r| >= 0.90, que suelen aportar
        información redundante y son candidatos a eliminar en el preprocesado
        """
        if len(numeric) < 2:
            return {"pearson": {}, "strong_pairs": []}

        matrix = frame[numeric].corr()
        pearson = {}
        for row in numeric:
            pearson[str(row)] = {}
            for column in numeric:
                pearson[str(row)][str(column)] = _number(matrix.loc[row, column])

        strong_pairs = []
        for i in range(len(numeric)):
            for j in range(i + 1, len(numeric)):
                value = matrix.loc[numeric[i], numeric[j]]
                if pd.isna(value):
                    continue
                if abs(value) >= _MAX_CORRELACION:
                    strong_pairs.append(
                        {
                            "feature_a": str(numeric[i]),
                            "feature_b": str(numeric[j]),
                            "pearson": _number(value),
                        }
                    )

        return {"pearson": pearson, "strong_pairs": strong_pairs}

    def _target(self, frame, numeric):
        """Analiza la columna objetivo según la tarea, o None si no hay"""
        if not self.dataset.is_supervised() or self.dataset.target is None:
            return None

        target = self.dataset.target
        series = frame[target].dropna()

        if self.dataset.task == TaskType.CLASSIFICATION:
            counts = series.value_counts()
            classes = {}
            for value, count in counts.items():
                classes[str(value)] = int(count)

            ratio = None
            if len(counts) > 0 and int(counts.min()) > 0:
                ratio = _number(int(counts.max()) / int(counts.min()))

            return {
                "type": "classification",
                "n_classes": int(len(counts)),
                "classes": classes,
                "imbalance": ratio,
            }

        correlations = []
        labels = frame[target]
        for name in numeric:
            if name == target:
                continue
            value = _number(frame[name].corr(labels))
            if value is not None:
                correlations.append({"feature": str(name), "pearson": value})
        correlations.sort(key=lambda item: abs(item["pearson"]), reverse=True)

        stats = _numeric_stats(labels)
        return {
            "type": "regression",
            "stats": stats,
            "correlations": correlations,
        }

    def _warnings(self, frame, report) -> list:
        """Convierte los hallazgos en frases accionables para el usuario"""
        warnings = []

        duplicates = report["general"]["n_duplicates"]
        if duplicates > 0:
            warnings.append(
                f"Hay {duplicates} filas duplicadas "
                f"({report['general']['duplicates_pct']}%). "
                "Revise si son errores de captura antes de eliminarlas."
            )

        for name in frame.columns:
            series = frame[name]
            valid = series.dropna()

            if valid.empty:
                warnings.append(
                    f"La columna '{name}' está completamente vacía."
                )
                continue

            if valid.nunique() <= 1:
                warnings.append(
                    f"La columna '{name}' es constante: no aporta información."
                )

            missing = report["missing"]["columns"].get(str(name))
            if missing is not None and missing["pct"] >= _AVISO_FALTANTES:
                warnings.append(
                    f"La columna '{name}' tiene {missing['pct']}% de valores "
                    "faltantes; imputar puede introducir mucho ruido."
                )

        for name, info in report["categorical"].items():
            if info["n_unique"] > _MAX_CATEGORIAS:
                warnings.append(
                    f"La columna categórica '{name}' tiene {info['n_unique']} "
                    "valores distintos; codificarla puede crear demasiadas "
                    "columnas."
                )

        for pair in report["correlations"]["strong_pairs"]:
            warnings.append(
                f"'{pair['feature_a']}' y '{pair['feature_b']}' están muy "
                f"correlacionadas (r={pair['pearson']}); aportan información "
                "parecida."
            )

        target = report["target"]
        if target is not None and target["type"] == "classification":
            ratio = target["imbalance"]
            if ratio is not None and ratio >= _MAX_DESEQUILIBRIO:
                warnings.append(
                    f"Las clases están desequilibradas: la más frecuente tiene "
                    f"{ratio} veces más casos que la menos frecuente. "
                    "Use métricas como F1 o balancee las clases."
                )

        return warnings
