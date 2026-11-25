"""Visualizaciones de alta resolución para el pipeline de ML"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Visualizer:
    """Genera gráficos a partir de reportes y resultados del pipeline

    Parametros:
        output_dir : str or pathlib.Path, default="reports/figures"
            Carpeta relativa donde se guardan las imágenes
        dpi : int, default=300
            Resolución de salida. Debe ser al menos 300

    Notes
    -----
    Cada método devuelve la ruta relativa del archivo escrito. 
    Las figuras se cierran después de guardar para no acumular memoria entre gráficos
    """

    def __init__(self, output_dir="reports/figures", dpi=300):
        if dpi < 300:
            raise ValueError("dpi debe ser al menos 300.")
        relative_dir = Path(output_dir)
        if relative_dir.is_absolute():
            raise ValueError("output_dir debe ser una ruta relativa.")

        self.output_dir = relative_dir
        self.dpi = dpi

    def _save(self, figure, filename):
        """Guarda una figura y devuelve su ruta relativa"""
        destination = Path(filename)
        if destination.is_absolute():
            raise ValueError("filename debe ser una ruta relativa.")

        full_path = self.output_dir / destination
        full_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(full_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(figure)
        return full_path

    @staticmethod
    def _empty_axis(axis, message):
        """Deja un mensaje en vez de un gráfico vacío"""
        axis.set_axis_off()
        axis.text(0.5, 0.5, message, ha="center", va="center")
        return axis

    def plot_missing_values(self, eda_report, filename="missing_values.png"):
        """Barra de porcentaje de valores faltantes por columna"""
        columns = eda_report["missing"]["columns"]
        figure, axis = plt.subplots()

        if not columns:
            self._empty_axis(axis, "Sin valores faltantes")
            axis.set_title("Valores faltantes")
        else:
            names = list(columns.keys())
            percentages = [columns[name]["pct"] for name in names]
            axis.bar(names, percentages, color="#4C72B0")
            axis.set_ylabel("Porcentaje (%)")
            axis.set_title("Valores faltantes por columna")
            axis.tick_params(axis="x", rotation=45)

        return self._save(figure, filename)

    def plot_correlation_heatmap(self, eda_report, filename="correlation_heatmap.png"):
        """Mapa de calor anotado de la correlación de Pearson"""
        return self._plot_matrix_heatmap(
            eda_report["correlations"]["pearson"], "Correlación de Pearson", filename
        )

    def plot_spearman_heatmap(self, eda_report, filename="spearman_heatmap.png"):
        """Mapa de calor anotado de la correlación de Spearman"""
        return self._plot_matrix_heatmap(
            eda_report["correlations"]["spearman"], "Correlación de Spearman", filename
        )

    def _plot_matrix_heatmap(self, values, title, filename):
        """Dibuja una matriz cuadrada con sus valores escritos en cada celda"""
        names = list(values.keys())
        figure, axis = plt.subplots()

        if len(names) < 2:
            self._empty_axis(axis, "Se necesitan al menos dos columnas numéricas")
            axis.set_title(title)
            return self._save(figure, filename)

        matrix = np.array(
            [[values[row][column] for column in names] for row in names],
            dtype=float,
        )
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
        axis.set_yticks(range(len(names)), names)
        axis.set_title(title)

        for row in range(len(names)):
            for column in range(len(names)):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        figure.colorbar(image, ax=axis)
        return self._save(figure, filename)

    def plot_target_distribution(self, eda_report, filename="target_distribution.png"):
        """Distribución de clases o correlaciones con el objetivo numérico"""
        target = eda_report["target"]
        figure, axis = plt.subplots()

        if target is None:
            self._empty_axis(axis, "El dataset no tiene objetivo")
            axis.set_title("Objetivo")
        elif target["type"] == "classification":
            classes = target["classes"]
            axis.bar(list(classes.keys()), list(classes.values()), color="#55A868")
            axis.set_ylabel("Casos")
            axis.set_title("Distribución de clases")
            axis.tick_params(axis="x", rotation=45)
        else:
            correlations = target["correlations"]
            if not correlations:
                self._empty_axis(axis, "Sin correlaciones numéricas")
            else:
                names = [item["feature"] for item in correlations]
                values = [abs(item["pearson"]) for item in correlations]
                axis.bar(names, values, color="#C44E52")
                axis.set_ylabel("|r| con el objetivo")
                axis.set_title("Correlación con el objetivo")
                axis.tick_params(axis="x", rotation=45)

        return self._save(figure, filename)

    def plot_confusion_matrix(
        self, matrix, labels, normalize=True, filename="confusion_matrix.png"
    ):
        """Mapa de calor de una matriz de confusión, normalizada por fila"""
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError("La matriz de confusión debe ser cuadrada.")
        if len(labels) != values.shape[0]:
            raise ValueError("labels debe tener una entrada por fila de la matriz.")

        displayed = values.copy()
        if normalize:
            row_sums = displayed.sum(axis=1, keepdims=True)
            displayed = np.divide(
                displayed, row_sums, out=np.zeros_like(displayed), where=row_sums != 0
            )

        figure, axis = plt.subplots()
        image = axis.imshow(displayed, cmap="Blues")
        axis.set_xticks(range(len(labels)), [str(label) for label in labels], rotation=45)
        axis.set_yticks(range(len(labels)), [str(label) for label in labels])
        axis.set_xlabel("Predicho")
        axis.set_ylabel("Real")
        axis.set_title(
            "Matriz de confusión normalizada" if normalize else "Matriz de confusión"
        )

        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                text = (
                    f"{displayed[row, column]:.2f}"
                    if normalize
                    else str(int(values[row, column]))
                )
                axis.text(column, row, text, ha="center", va="center")
        figure.colorbar(image, ax=axis)
        return self._save(figure, filename)

    def plot_clusters(
        self, X, labels, centers=None, filename="clusters.png",
        xlabel="Feature 1", ylabel="Feature 2",
    ):
        """Dispersión de clusters usando las dos primeras columnas de X"""
        points = np.asarray(X, dtype=float)
        assigned = np.asarray(labels)
        if points.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")
        if assigned.ndim != 1 or len(assigned) != len(points):
            raise ValueError("labels debe ser unidimensional y tener la longitud de X.")

        if points.shape[1] == 1:
            plotted = np.column_stack([points[:, 0], np.zeros(len(points))])
            note = " (segundo eje artificial)"
        else:
            plotted = points[:, :2]
            note = " (primeras dos features)" if points.shape[1] > 2 else ""

        figure, axis = plt.subplots()
        axis.scatter(plotted[:, 0], plotted[:, 1], c=assigned, cmap="tab10", s=25)
        if centers is not None:
            centers_array = np.asarray(centers, dtype=float)
            axis.scatter(
                centers_array[:, 0],
                centers_array[:, 1],
                marker="x",
                c="black",
                s=80,
                label="Centros",
            )
            axis.legend()

        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(f"Clusters{note}")
        return self._save(figure, filename)

    def plot_dendrogram(self, linkage_matrix, labels=None, filename="dendrogram.png"):
        """Dibuja un dendrograma a partir de la matriz de enlace"""
        merges = np.asarray(linkage_matrix, dtype=float)
        if merges.size == 0:
            merges = np.empty((0, 4), dtype=float)
        if merges.ndim != 2 or merges.shape[1] != 4:
            raise ValueError("La matriz de enlace debe tener forma (m, 4).")
        # El modelo se detiene en n_clusters: puede haber menos fusiones que
        # muestras-1, así que el número de muestras no se deduce del número de filas.
        if labels is not None:
            n_samples = len(labels)
        elif len(merges):
            n_samples = int(np.max(merges[:, :2])) + 1
        else:
            n_samples = 1
        x_positions = {index: float(index) for index in range(n_samples)}
        y_positions = {index: 0.0 for index in range(n_samples)}

        figure, axis = plt.subplots()
        if len(merges) == 0:
            axis.plot([0], [0], marker="o", color="black")

        for step, row in enumerate(merges):
            id_a, id_b = int(row[0]), int(row[1])
            height = float(row[2])
            x_a, x_b = x_positions[id_a], x_positions[id_b]
            y_a, y_b = y_positions[id_a], y_positions[id_b]

            axis.plot([x_a, x_a], [y_a, height], color="black")
            axis.plot([x_b, x_b], [y_b, height], color="black")
            axis.plot([x_a, x_b], [height, height], color="black")

            new_id = n_samples + step
            x_positions[new_id] = (x_a + x_b) / 2.0
            y_positions[new_id] = height

        if labels is not None:
            axis.set_xticks([x_positions[index] for index in range(n_samples)])
            axis.set_xticklabels([str(label) for label in labels], rotation=90)
        else:
            axis.set_xticks([])

        axis.set_ylabel("Altura de fusión")
        axis.set_title("Dendrograma")
        return self._save(figure, filename)

    def plot_metric_comparison(self, scores, filename="metric_comparison.png"):
        """Barra comparativa de puntuaciones por modelo"""
        if not scores:
            raise ValueError("scores no puede estar vacío.")

        names = list(scores.keys())
        values = [float(scores[name]) for name in names]
        figure, axis = plt.subplots()
        axis.bar(names, values, color="#8172B3")
        axis.set_ylabel("Puntuación")
        axis.set_title("Comparación de modelos")
        axis.tick_params(axis="x", rotation=45)
        return self._save(figure, filename)

    @staticmethod
    def _binary_roc(y_true, scores):
        """Devuelve (fpr, tpr) de un problema binario ya codificado"""
        positives = y_true == 1
        n_positive = int(np.count_nonzero(positives))
        n_negative = len(y_true) - n_positive
        if n_positive == 0 or n_negative == 0:
            return None, None

        order = np.argsort(-scores, kind="stable")
        ordered_positives = positives[order]
        true_positives = np.cumsum(ordered_positives)
        false_positives = np.cumsum(~ordered_positives)

        tpr = np.concatenate([[0.0], true_positives / n_positive])
        fpr = np.concatenate([[0.0], false_positives / n_negative])
        return fpr, tpr

    def plot_roc_curves(self, y_true, y_scores, labels=None, filename="roc_curves.png"):
        """Curvas ROC one-vs-rest con AUC por clase"""
        true = np.asarray(y_true)
        scores = np.asarray(y_scores, dtype=float)
        if true.ndim != 1:
            raise ValueError("y_true debe ser unidimensional.")
        if scores.ndim != 2 or len(scores) != len(true):
            raise ValueError("y_scores debe tener forma (n_muestras, n_clases).")

        if labels is None:
            labels = np.unique(true)
        if scores.shape[1] != len(labels):
            raise ValueError("y_scores debe tener una columna por etiqueta.")

        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar")
        plotted = 0

        for index, label in enumerate(labels):
            binary = (true == label).astype(int)
            fpr, tpr = self._binary_roc(binary, scores[:, index])
            if fpr is None:
                continue
            auc = float(np.trapezoid(tpr, fpr))
            axis.plot(fpr, tpr, label=f"{label} (AUC={auc:.2f})")
            plotted += 1

        if plotted == 0:
            self._empty_axis(axis, "No hay clases con dos casos para dibujar ROC")
        axis.set_xlabel("Tasa de falsos positivos")
        axis.set_ylabel("Tasa de verdaderos positivos")
        axis.set_title("Curvas ROC multiclase")
        axis.legend(loc="lower right")
        return self._save(figure, filename)

    def plot_precision_recall_curves(
        self, y_true, y_scores, labels=None, filename="precision_recall_curves.png"
    ):
        """Curvas precision-recall one-vs-rest por clase"""
        true = np.asarray(y_true)
        scores = np.asarray(y_scores, dtype=float)
        if true.ndim != 1:
            raise ValueError("y_true debe ser unidimensional.")
        if scores.ndim != 2 or len(scores) != len(true):
            raise ValueError("y_scores debe tener forma (n_muestras, n_clases).")

        if labels is None:
            labels = np.unique(true)
        if scores.shape[1] != len(labels):
            raise ValueError("y_scores debe tener una columna por etiqueta.")

        figure, axis = plt.subplots()
        plotted = 0

        for index, label in enumerate(labels):
            binary = (true == label).astype(int)
            positives = binary == 1
            n_positive = int(np.count_nonzero(positives))
            if n_positive == 0:
                continue

            order = np.argsort(-scores[:, index], kind="stable")
            ordered_positives = positives[order]
            true_positives = np.cumsum(ordered_positives)
            false_positives = np.cumsum(~ordered_positives)
            precision = np.concatenate(
                [[1.0], true_positives / (true_positives + false_positives)]
            )
            recall = np.concatenate([[0.0], true_positives / n_positive])
            average_precision = float(np.trapezoid(precision, recall))
            axis.plot(
                recall,
                precision,
                label=f"{label} (AP={average_precision:.2f})",
            )
            plotted += 1

        if plotted == 0:
            self._empty_axis(axis, "No hay clases positivas para dibujar PR")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title("Curvas Precision-Recall")
        axis.legend(loc="lower left")
        return self._save(figure, filename)

    def plot_learning_curves(
        self,
        x_values,
        train_errors,
        validation_errors=None,
        x_label="Tamaño muestral",
        filename="learning_curves.png",
    ):
        """Error frente al tamaño muestral o frente a un hiperparámetro"""
        x = np.asarray(x_values, dtype=float)
        train = np.asarray(train_errors, dtype=float)
        if x.ndim != 1 or train.ndim != 1 or len(x) != len(train):
            raise ValueError("x_values y train_errors deben tener la misma longitud.")

        figure, axis = plt.subplots()
        axis.plot(x, train, marker="o", label="Entrenamiento")

        if validation_errors is not None:
            validation = np.asarray(validation_errors, dtype=float)
            if validation.ndim != 1 or len(validation) != len(x):
                raise ValueError("validation_errors debe tener la misma longitud.")
            axis.plot(x, validation, marker="o", label="Validación")

        axis.set_xlabel(x_label)
        axis.set_ylabel("Error")
        axis.set_title("Curva de aprendizaje")
        axis.legend()
        return self._save(figure, filename)

    @staticmethod
    def _check_regression_vectors(y_true, y_pred):
        """Valida y devuelve dos vectores numéricos de la misma longitud"""
        true = np.asarray(y_true, dtype=float)
        predicted = np.asarray(y_pred, dtype=float)
        if true.ndim != 1 or predicted.ndim != 1 or len(true) != len(predicted):
            raise ValueError("y_true e y_pred deben ser unidimensionales y iguales.")
        return true, predicted

    def plot_predicted_vs_actual(self, y_true, y_pred, filename="predicted_vs_actual.png"):
        """Dispersión real vs. predicho con banda del 95% basada en residuales"""
        true, predicted = self._check_regression_vectors(y_true, y_pred)
        residuals = true - predicted
        margin = (
            1.96 * float(np.std(residuals, ddof=1)) if len(true) > 1 else 0.0
        )

        low = float(np.min(np.concatenate([true, predicted]))) - margin
        high = float(np.max(np.concatenate([true, predicted]))) + margin
        line = np.linspace(low, high, 100)

        figure, axis = plt.subplots()
        axis.fill_between(
            line,
            line - margin,
            line + margin,
            color="#55A868",
            alpha=0.25,
            label="Banda 95%",
        )
        axis.scatter(predicted, true, alpha=0.75)
        axis.plot(line, line, "k--", linewidth=1, label="Predicción perfecta")
        axis.set_xlabel("Predicho")
        axis.set_ylabel("Real")
        axis.set_title("Valores reales vs. predichos")
        axis.legend()
        return self._save(figure, filename)

    def plot_residuals(self, y_true, y_pred, filename="residuals.png"):
        """Histograma de residuales real - predicho"""
        true, predicted = self._check_regression_vectors(y_true, y_pred)
        residuals = true - predicted

        figure, axis = plt.subplots()
        axis.hist(residuals, bins="auto", color="#C44E52")
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Residuo (real - predicho)")
        axis.set_ylabel("Frecuencia")
        axis.set_title("Distribución de residuales")
        return self._save(figure, filename)

    def plot_regression_metrics(self, report, filename="regression_metrics.png"):
        """Muestra R², RMSE y MAE de un reporte de regresión"""
        required = ("r2", "rmse", "mae")
        if report is None or any(field not in report for field in required):
            raise ValueError("report debe contener r2, rmse y mae.")

        figure, axis = plt.subplots()
        axis.set_axis_off()
        text = "\n".join(
            [
                f"R²   = {float(report['r2']):.4f}",
                f"RMSE = {float(report['rmse']):.4f}",
                f"MAE  = {float(report['mae']):.4f}",
            ]
        )
        axis.text(0.05, 0.5, text, fontsize=13, va="center", family="monospace")
        axis.set_title("Métricas de regresión")
        return self._save(figure, filename)

    def plot_inertia_curve(
        self, k_values, inertias, elbow_k=None, filename="inertia_curve.png"
    ):
        """Curva de inercia con valores anotados y codo opcional marcado"""
        k = np.asarray(k_values, dtype=float)
        inertia = np.asarray(inertias, dtype=float)
        if k.ndim != 1 or inertia.ndim != 1 or len(k) != len(inertia):
            raise ValueError("k_values e inertias deben tener la misma longitud.")

        figure, axis = plt.subplots()
        axis.plot(k, inertia, marker="o", color="#4C72B0")
        if elbow_k is not None:
            axis.axvline(
                elbow_k,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Codo k={elbow_k}",
            )
            axis.legend()
        for x, y in zip(k, inertia):
            axis.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )
        axis.set_xlabel("Número de clusters (k)")
        axis.set_ylabel("Inercia")
        axis.set_title("Método del codo")
        return self._save(figure, filename)

    @staticmethod
    def _silhouette_samples(X, labels):
        """Silhouette de cada muestra, con la convención s=0 para clusters unitarios"""
        distances = np.sqrt(
            np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        )
        unique = np.unique(labels)
        values = np.zeros(len(X))

        for index in range(len(X)):
            own_mask = labels == labels[index]
            own_mask[index] = False
            if not np.any(own_mask):
                continue

            intra = float(np.mean(distances[index, own_mask]))
            other_means = [
                float(np.mean(distances[index, labels == label]))
                for label in unique
                if label != labels[index]
            ]
            if not other_means:
                continue

            nearest = min(other_means)
            denominator = max(intra, nearest)
            if denominator > 0:
                values[index] = (nearest - intra) / denominator
        return values

    def plot_silhouette(self, X, labels, filename="silhouette.png"):
        """Siluetas individuales ordenadas por clúster con línea de corte media"""
        points = np.asarray(X, dtype=float)
        assigned = np.asarray(labels)
        if points.ndim != 2:
            raise ValueError("X debe ser una matriz bidimensional.")
        if assigned.ndim != 1 or len(assigned) != len(points):
            raise ValueError("labels debe ser unidimensional y tener la longitud de X.")

        figure, axis = plt.subplots()
        if len(np.unique(assigned)) < 2:
            self._empty_axis(axis, "Se necesitan al menos 2 clusters")
            axis.set_title("Siluetas")
            return self._save(figure, filename)

        values = self._silhouette_samples(points, assigned)
        order = np.lexsort((-values, assigned))
        sorted_values = values[order]
        sorted_labels = assigned[order]

        unique = np.unique(assigned)
        color_by_label = {label: position for position, label in enumerate(unique)}
        colors = [color_by_label[label] for label in sorted_labels]

        colormap = plt.get_cmap("tab10")
        axis.barh(
            range(len(values)),
            sorted_values,
            color=[colormap(value % 10) for value in colors],
        )
        mean_value = float(np.mean(values))
        axis.axvline(
            mean_value,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Media={mean_value:.2f}",
        )
        axis.set_xlabel("Silhouette")
        axis.set_ylabel("Muestras ordenadas por clúster")
        axis.set_title("Siluetas individuales")
        axis.legend()
        return self._save(figure, filename)

    def plot_cluster_centers(self, centers, X=None, filename="cluster_centers.png"):
        """Dispersión 2D de centroides, con muestras opcionales en gris"""
        centers_array = np.asarray(centers, dtype=float)
        if centers_array.ndim != 2 or centers_array.shape[0] == 0:
            raise ValueError("centers debe ser una matriz no vacía.")

        if centers_array.shape[1] == 1:
            plotted_centers = np.column_stack(
                [centers_array[:, 0], np.zeros(len(centers_array))]
            )
        else:
            plotted_centers = centers_array[:, :2]

        figure, axis = plt.subplots()
        if X is not None:
            points = np.asarray(X, dtype=float)
            if points.ndim != 2 or points.shape[0] == 0:
                raise ValueError("X debe ser una matriz no vacía.")
            plotted_points = (
                np.column_stack([points[:, 0], np.zeros(len(points))])
                if points.shape[1] == 1
                else points[:, :2]
            )
            axis.scatter(
                plotted_points[:, 0],
                plotted_points[:, 1],
                color="lightgray",
                s=15,
                label="Muestras",
            )

        axis.scatter(
            plotted_centers[:, 0],
            plotted_centers[:, 1],
            marker="x",
            color="black",
            s=120,
            label="Centroides",
        )
        axis.set_xlabel("Feature 1")
        axis.set_ylabel("Feature 2")
        axis.set_title("Centroides de los clusters")
        axis.legend()
        return self._save(figure, filename)
