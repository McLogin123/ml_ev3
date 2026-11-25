## Requisitos

- Python 3.10 o superior (probado en 3.15.2).
- Dependencias fijadas en `requirements.txt`.

## Instalación

```powershell
cd ml_framework
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Ejecución de pruebas

```powershell
.venv\Scripts\python -m pytest -q
```

## Uso del pipeline

```powershell
.venv\Scripts\python.exe main.py --data data/ejemplo_clasificacion.csv
.venv\Scripts\python.exe main.py --data data/ejemplo_regresion.csv --target target
.venv\Scripts\python.exe main.py --data data/ejemplo_clusters.csv --task unsupervised
.venv\Scripts\python.exe main.py --data data/ejemplo_clusters.csv --task unsupervised --n-clusters 4
.venv\Scripts\python.exe main.py --data data/ejemplo_clasificacion.csv --model tree --cv 3 --grid
```

Los resultados se escriben en `reports/<nombre_del_dataset>/`: `report.json` (métricas, precisión y rutas de figuras) y las imágenes en `figures/`
Ver todas las opciones con `main.py --help`

## Estructura

```
ml_framework/
├── core/          # ingesta, EDA y preprocesamiento
├── models/        # supervisado y no supervisado
├── evaluation/    # métricas y visualización
├── data/          # datasets de prueba
├── reports/       # gráficos y métricas generadas
└── tests/
```
