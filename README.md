# Sistema de Inteligencia de Negocios para Pronóstico de Ventas

## Descripción

Aplicación web que automatiza el análisis de datos de ventas, genera pronósticos con modelos de IA (SARIMA, Prophet) y produce reportes ejecutivos con recomendaciones para la toma de decisiones gerenciales.

## Estructura del Proyecto

```
tesis_app/
├── app.py                    # Aplicación principal
├── modules/
│   ├── __init__.py
│   ├── etl.py                # Pipeline de limpieza y transformación
│   ├── eda.py                # Análisis exploratorio y gráficos
│   ├── modeling.py           # Modelos predictivos (Baseline, Prophet, SARIMA)
│   └── reporting.py          # Generación de reportes PDF
├── .streamlit/
│   └── config.toml           # Configuración visual
├── requirements.txt          # Dependencias de Python
├── packages.txt              # Dependencias del sistema (para Streamlit Cloud)
└── README.md
```

## Despliegue en Streamlit Cloud

1. Sube este proyecto a un repositorio de GitHub.
2. Ve a [share.streamlit.io](https://share.streamlit.io).
3. Conecta tu cuenta de GitHub.
4. Selecciona el repositorio y el archivo `app.py`.
5. Haz clic en "Deploy".

## Ejecución Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Uso

1. Abre la aplicación en el navegador.
2. Sube el archivo CSV exportado del ERP.
3. Ajusta los parámetros en el panel lateral (meta de ventas, umbral de alerta, etc.).
4. Visualiza el dashboard interactivo con KPIs, gráficos y recomendaciones.
5. Descarga el reporte ejecutivo en PDF.
