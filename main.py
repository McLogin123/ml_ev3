"""CLI y orquestación de extremo a extremo del framework

Lee un archivo tabular, describe los datos, entrena un modelo según la tarea
y guarda métricas y gráficos en una carpeta con el nombre del dataset.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from core import EDAnalyzer, TabularLoader, TaskType, prepare_dataset
from evaluation.clustering_audit import audit_partitioning
from evaluation import (
    Visualizer,
    accuracy_score,
    classification_report,
    clustering_report,
    confusion_matrix,
    cross_validate,
    grid_search,
    regression_report,
    rmse,
)
from models import (
    AgglomerativeClustering,
    BaggingClassifier,
    DANN,
    DecisionTree,
    GAMRegressor,
    KMeans,
    KNeighborsModel,
    LinearRegression,
    LogisticRegression,
    LVQ1,
    MARSRegressor,
)
from models.unsupervised import pca_project

_DEFAULT_MODELS = {
    TaskType.CLASSIFICATION: "bagging",
    TaskType.REGRESSION: "linear",
    TaskType.UNSUPERVISED: "kmeans",
}
_FRACTIONS = (0.4, 0.6, 0.8, 1.0)
_HEADLINE_METRIC = {
    TaskType.CLASSIFICATION: ("accuracy", "accuracy"),
    TaskType.REGRESSION: ("r2", "R²"),
    TaskType.UNSUPERVISED: ("silhouette", "silueta"),
}
_PARAM_GRIDS = {
    "bagging": {"n_estimators": [5, 10]},
    "logistic": {"learning_rate": [0.05, 0.1]},
    "tree": {"max_depth": [2, 4, 6]},
    "knn": {"n_neighbors": [3, 5]},
    "lvq1": {"n_epochs": [20, 50]},
    "dann": {"n_neighbors": [3, 5]},
    "linear": {},
    "mars": {"max_terms": [5, 10]},
    "gam": {"n_knots": [3, 5]},
}


def build_parser():
    """Define los argumentos de la línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Pipeline de ML: carga, EDA, modelo, métricas y gráficos."
    )
    parser.add_argument(
        "--data", default="data/ejemplo_clasificacion.csv", help="Archivo de entrada."
    )
    parser.add_argument("--target", help="Columna objetivo. Se infiere si se omite.")
    parser.add_argument(
        "--task", choices=[task.value for task in TaskType], help="Fuerza la tarea."
    )
    parser.add_argument("--model", help="Modelo. Por defecto el de la tarea.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-clusters", type=int, help="Fuerza k; por defecto lo elige la auditoría."
    )
    parser.add_argument(
        "--max-k", type=int, default=6, help="k máximo del barrido de la auditoría."
    )
    parser.add_argument(
        "--cv", type=int, default=0, metavar="FOLDS",
        help="Validación cruzada con FOLDS particiones; 0 la desactiva.",
    )
    parser.add_argument(
        "--grid", action="store_true", help="Busca hiperparámetros con validación cruzada."
    )
    return parser


def _make_model(task, name, seed, n_clusters, **params):
    """Devuelve un modelo nuevo; se llama también en cada curva de aprendizaje"""
    if task == TaskType.CLASSIFICATION:
        specs = {
            "bagging": (BaggingClassifier, {"random_state": seed}),
            "logistic": (LogisticRegression, {}),
            "tree": (DecisionTree, {"task": task}),
            "knn": (KNeighborsModel, {"task": task}),
            "lvq1": (LVQ1, {"random_state": seed}),
            "dann": (DANN, {}),
        }
    elif task == TaskType.REGRESSION:
        specs = {
            "linear": (LinearRegression, {}),
            "mars": (MARSRegressor, {}),
            "gam": (GAMRegressor, {}),
            "tree": (DecisionTree, {"task": task}),
            "knn": (KNeighborsModel, {"task": task}),
        }
    else:
        specs = {
            "kmeans": (KMeans, {"n_clusters": n_clusters, "random_state": seed}),
            "hierarchical": (AgglomerativeClustering, {"n_clusters": n_clusters}),
        }

    if name not in specs:
        raise ValueError(
            f"Modelo '{name}' no válido para {task.value}. Opciones: {sorted(specs)}."
        )
    model_class, defaults = specs[name]
    return model_class(**{**defaults, **params})


def _classification_scores(model, X, classes):
    """Puntaje por clase para ROC/PR, o None si el modelo no lo expone"""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        if probabilities.shape == (len(X), len(classes)):
            return probabilities

    if hasattr(model, "estimators_"):
        votes = np.column_stack([tree.predict(X) for tree in model.estimators_])
        return np.column_stack([np.mean(votes == label, axis=1) for label in classes])

    return None


def _learning_curves(factory, X_train, y_train, X_test, y_test, task, seed):
    """Error al entrenar con subconjuntos cada vez mayores"""
    order = np.random.default_rng(seed).permutation(len(X_train))
    min_size = 2 * len(np.unique(y_train)) if task == TaskType.CLASSIFICATION else 2
    sizes = []
    train_errors = []
    test_errors = []

    for fraction in _FRACTIONS:
        size = min(len(X_train), max(min_size, int(round(len(X_train) * fraction))))
        subset = order[:size]
        model = factory().fit(X_train[subset], y_train[subset])

        if task == TaskType.CLASSIFICATION:
            train_error = 1.0 - accuracy_score(
                y_train[subset], model.predict(X_train[subset])
            )
            test_error = 1.0 - accuracy_score(y_test, model.predict(X_test))
        else:
            train_error = rmse(y_train[subset], model.predict(X_train[subset]))
            test_error = rmse(y_test, model.predict(X_test))

        sizes.append(size)
        train_errors.append(float(train_error))
        test_errors.append(float(test_error))

    return sizes, train_errors, test_errors


def _next_output(name):
    """Devuelve una carpeta libre: reports/<name>, luego <name>_1, <name>_2..."""
    base = Path("reports") / name
    candidate = base
    index = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{index}")
        index += 1
    return candidate


def run_pipeline(options):
    """Ejecuta el flujo completo y devuelve el reporte como diccionario"""
    seed = options.get("random_state", 42)
    dataset = TabularLoader().load(
        options.get("data", "data/ejemplo_clasificacion.csv"),
        target=options.get("target"),
        task=options.get("task"),
    )
    output = _next_output(dataset.source.stem)
    task = dataset.task
    eda_report = EDAnalyzer(dataset).run()
    warnings = list(dataset.warnings) + list(eda_report["warnings"])

    name = options.get("model") or _DEFAULT_MODELS[task]
    forced_k = options.get("n_clusters")
    # 2 es un k válido mientras la auditoría decide el definitivo.
    n_clusters = forced_k or 2
    factory = lambda: _make_model(task, name, seed, n_clusters)
    model = factory()
    prepared = prepare_dataset(
        dataset,
        test_size=options.get("test_size", 0.2),
        random_state=seed,
        scale=model.requires_scaling,
    )

    validation = {}
    if task != TaskType.UNSUPERVISED:
        folds = options.get("cv", 0)
        if options.get("grid"):
            grid = _PARAM_GRIDS.get(name, {})
            if grid:
                best = grid_search(
                    lambda **params: _make_model(task, name, seed, n_clusters, **params),
                    grid,
                    prepared.X_train,
                    prepared.y_train,
                    task,
                    n_splits=folds or 3,
                    random_state=seed,
                )
                validation["grid"] = best
                factory = lambda: _make_model(
                    task, name, seed, n_clusters, **best["best_params"]
                )
                model = factory()
            else:
                warnings.append(f"El modelo '{name}' no tiene grilla de hiperparámetros.")
        if folds:
            validation["cv"] = cross_validate(
                factory,
                prepared.X_train,
                prepared.y_train,
                task,
                n_splits=folds,
                random_state=seed,
            )

    visualizer = Visualizer(output_dir=output / "figures", dpi=300)
    figures = []

    figures.append(visualizer.plot_missing_values(eda_report))
    figures.append(visualizer.plot_correlation_heatmap(eda_report))
    figures.append(visualizer.plot_spearman_heatmap(eda_report))

    if task == TaskType.UNSUPERVISED:
        X = prepared.X_train
        max_k = options.get("max_k", 6)
        k_values = [k for k in range(2, max_k + 1) if k < len(X)]
        audit = audit_partitioning(X, k_values, seed)
        if forced_k is None:
            if audit["selected_k"] is None:
                n_clusters = min(2, len(X))
                audit["selection_rule"] = "respaldo"
                audit["warnings"].append(
                    f"No hubo k válido en el barrido; se usa k={n_clusters}."
                )
            else:
                n_clusters = audit["selected_k"]
        else:
            n_clusters = forced_k
            audit["selection_rule"] = "flag --n-clusters"
            if forced_k not in audit["k_values"]:
                audit["warnings"].append(
                    f"k={forced_k} queda fuera del barrido 2..{max_k}."
                )
        warnings.extend(audit["warnings"])
        factory = lambda: _make_model(task, name, seed, n_clusters)
        model = factory().fit(X)

        labels = model.labels_
        metrics = clustering_report(X, labels)
        centers = (
            np.asarray(model.cluster_centers_)
            if hasattr(model, "cluster_centers_")
            else np.asarray(
                [X[labels == label].mean(axis=0) for label in np.unique(labels)]
            )
        )
        projected, projected_centers, ratios = pca_project(X, centers)
        audit["pca_explained_variance"] = [float(value) for value in ratios]

        figures.append(
            visualizer.plot_inertia_curve(
                audit["k_values"], audit["inertias"], elbow_k=audit["elbow_k"]
            )
        )
        figures.append(visualizer.plot_silhouette(X, labels))
        figures.append(visualizer.plot_cluster_centers(centers, X))
        figures.append(
            visualizer.plot_clusters(
                projected,
                labels,
                projected_centers,
                filename="pca_projection.png",
                xlabel="PC1",
                ylabel="PC2",
            )
        )
        if hasattr(model, "linkage_matrix_"):
            figures.append(
                visualizer.plot_dendrogram(
                    model.linkage_matrix_, labels=np.arange(len(X))
                )
            )
    else:
        if dataset.is_supervised():
            figures.append(visualizer.plot_target_distribution(eda_report))

        model.fit(prepared.X_train, prepared.y_train)
        predictions = model.predict(prepared.X_test)

        if task == TaskType.CLASSIFICATION:
            classes = list(range(len(prepared.classes)))
            metrics = classification_report(prepared.y_test, predictions, labels=classes)
            matrix = confusion_matrix(prepared.y_test, predictions, labels=classes)
            metrics["confusion_matrix"] = matrix.tolist()
            figure_labels = [str(value) for value in prepared.classes]
            figures.append(visualizer.plot_confusion_matrix(matrix, figure_labels))

            scores = _classification_scores(model, prepared.X_test, classes)
            if scores is None:
                warnings.append(
                    "Este modelo no expone probabilidades: no hay curvas ROC ni PR."
                )
            else:
                figures.append(
                    visualizer.plot_roc_curves(prepared.y_test, scores, classes)
                )
                figures.append(
                    visualizer.plot_precision_recall_curves(
                        prepared.y_test, scores, classes
                    )
                )
        else:
            metrics = regression_report(prepared.y_test, predictions)
            figures.append(
                visualizer.plot_predicted_vs_actual(prepared.y_test, predictions)
            )
            figures.append(visualizer.plot_residuals(prepared.y_test, predictions))
            figures.append(visualizer.plot_regression_metrics(metrics))

        sizes, train_errors, test_errors = _learning_curves(
            factory,
            prepared.X_train,
            prepared.y_train,
            prepared.X_test,
            prepared.y_test,
            task,
            seed,
        )
        figures.append(
            visualizer.plot_learning_curves(sizes, train_errors, test_errors)
        )

    figure_paths = [str(path).replace("\\", "/") for path in figures]
    metric_key, metric_name = _HEADLINE_METRIC[task]
    report = {
        "dataset": dataset.summary(),
        "model": {
            "task": task.value,
            "name": name,
            "n_features": len(prepared.feature_names),
            "test_size": options.get("test_size", 0.2),
            "random_state": seed,
            "scaled": model.requires_scaling,
        },
        "metrics": metrics,
        "precision": metrics[metric_key],
        "precision_metric": metric_name,
        "validation": validation,
        "figures": figure_paths,
        "output": output.as_posix(),
        "warnings": warnings,
    }

    if hasattr(model, "feature_importances_"):
        report["feature_importances"] = {
            feature: float(value)
            for feature, value in zip(prepared.feature_names, model.feature_importances_)
        }
    if task == TaskType.UNSUPERVISED:
        report["model"]["n_clusters"] = n_clusters
        report["clustering_audit"] = audit

    report_path = output / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main(argv=None):
    """Punto de entrada de la línea de comandos"""
    options = vars(build_parser().parse_args(argv))
    report = run_pipeline(options)
    print(f"Tarea: {report['model']['task']}")
    print(f"Modelo: {report['model']['name']}")
    if "n_clusters" in report["model"]:
        print(f"Clusters: {report['model']['n_clusters']}")
    print(f"Precisión ({report['precision_metric']}): {report['precision']:.4f}")
    if "cv" in report["validation"]:
        cv = report["validation"]["cv"]
        print(f"Validación cruzada: {cv['mean']:.4f} ± {cv['std']:.4f}")
    if "grid" in report["validation"]:
        grid = report["validation"]["grid"]
        print(f"Mejores parámetros: {grid['best_params']}")
    print(f"Figuras: {len(report['figures'])}")
    print(f"Reporte: {report['output']}/report.json")
    return 0


if __name__ == "__main__":
    main()
