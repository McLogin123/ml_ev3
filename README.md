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
