"""Define la estructura y los metodos básicos para consultar el dataset"""
from enum import Enum


class TaskType(str, Enum):
    """Tipo de tarea de aprendizaje que ejecutará el pipeline"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNSUPERVISED = "unsupervised"


class Dataset:
    """Datos ya cargados y la decisión de qué tarea ejecutar

    Parametros:
        frame : pandas.DataFrame
            Tabla original, sin transformar
        target : str or None
            Nombre de la columna objetivo. None significa no supervisado
        task : TaskType
            Tarea decidida por el loader o forzada por el usuario
        source : pathlib.Path
            Archivo del que provienen los datos
        warnings : tuple of str
            Avisos sobre decisiones automáticas (target inferido, etc.)
    """

    def __init__(self, frame, target, task, source, warnings=()):
        self.frame = frame
        self.target = target
        self.task = task
        self.source = source
        self.warnings = tuple(warnings)

    def features(self):
        """Devuelve X: la tabla sin la columna objetivo"""
        if self.target is None:
            return self.frame
        return self.frame.drop(columns=[self.target])

    def labels(self):
        """Devuelve y o None si el dataset no está etiquetado"""
        if self.target is None:
            return None
        return self.frame[self.target]

    def is_supervised(self):
        """Indica si existe un target y la tarea es supervisada"""
        return self.task != TaskType.UNSUPERVISED

    def summary(self):
        """Resumen del dataset para guardar en el reporte"""
        return {
            "source": str(self.source),
            "n_rows": int(len(self.frame)),
            "n_cols": int(self.frame.shape[1]),
            "target": self.target,
            "task": self.task.value,
            "warnings": list(self.warnings),
        }
