import streamlit as st
import pandas as pd
import requests
import io
import re
from modules.etl import run_etl
from modules.eda import run_eda
from modules.modeling import run_modeling
from modules.reporting import generate_report

st.set_page_config(layout="wide", page_title="Sistema de Pronóstico de Ventas")

st.title("Sistema de Inteligencia de Negocios para Pronóstico de Ventas")
st.write("Esta aplicación automatiza el análisis de datos de ventas, genera pronósticos y crea reportes ejecutivos.")

# --- Sidebar para parámetros ---
st.sidebar.header("Parámetros de Configuración")

# --- Tipo de uso ---
st.sidebar.subheader("1. Tipo de Uso")
tipo_uso = st.sidebar.radio(
    "¿Qué deseas hacer?",
    ["Primera carga (historial completo)", "Actualización mensual"],
    help="Selecciona 'Primera carga' si es la primera vez que usas el sistema. "
         "Selecciona 'Actualización mensual' si ya tienes un historial y quieres agregar el mes más reciente."
)

if tipo_uso == "Primera carga (historial completo)":
    st.sidebar.info(
        "Sube el archivo completo del ERP con todo el historial de ventas "
        "(idealmente 2-3 años). El sistema limpiará los datos, analizará patrones "
        "y generará pronósticos."
    )
else:
    st.sidebar.info(
        "Sube el archivo actualizado del ERP que incluya el mes más reciente. "
        "El sistema re-procesará todo el historial con los nuevos datos para "
        "actualizar los pronósticos."
    )

# --- Carga de archivo ---
st.sidebar.subheader("2. Cargar Datos")

metodo_carga = st.sidebar.radio(
    "Método de carga:",
    ["Subir archivo", "Cargar desde URL"],
    help="Sube un archivo directamente (hasta 1 GB) o pega un enlace de Google Drive, Dropbox u otra URL."
)

uploaded_file = None
file_type = 'csv'


def _detect_file_type(name):
    """Detecta el tipo de archivo por su nombre."""
    name = name.lower()
    if name.endswith('.csv.gz') or name.endswith('.gz'):
        return 'csv_gz'
    elif name.endswith('.xlsx') or name.endswith('.xls'):
        return 'xlsx'
    else:
        return 'csv'


def _download_gdrive_large(file_id):
    """Descarga archivos grandes de Google Drive manejando la confirmación de virus scan."""
    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"

    # Primera solicitud
    response = session.get(base_url, params={'id': file_id}, stream=True, timeout=30)

    # Buscar token de confirmación para archivos grandes
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            confirm_token = value
            break

    if confirm_token:
        response = session.get(base_url, params={'id': file_id, 'confirm': confirm_token},
                               stream=True, timeout=30)

    # Si no hay token en cookies, intentar con confirm=t
    if response.headers.get('Content-Type', '').startswith('text/html'):
        response = session.get(base_url, params={'id': file_id, 'confirm': 't'},
                               stream=True, timeout=30)

    return response


if metodo_carga == "Subir archivo":
    uploaded_file = st.sidebar.file_uploader(
        "Sube tu archivo de ventas",
        type=["csv", "xlsx", "xls", "gz"],
        help="Formatos: CSV, Excel, CSV comprimido (.csv.gz). Máximo 1 GB."
    )
    if uploaded_file is not None:
        file_type = _detect_file_type(uploaded_file.name)

else:
    url_input = st.sidebar.text_input(
        "Pega la URL del archivo",
        placeholder="https://drive.google.com/file/d/.../view",
        help="Compatible con Google Drive, Dropbox, OneDrive (enlace público)"
    )
    if url_input:
        try:
            with st.spinner("Descargando archivo desde URL... Esto puede tardar unos minutos para archivos grandes."):
                # Google Drive
                gdrive_match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', url_input)
                if gdrive_match:
                    file_id = gdrive_match.group(1)
                    response = _download_gdrive_large(file_id)
                # Dropbox
                elif 'dropbox.com' in url_input:
                    download_url = url_input.replace('dl=0', 'dl=1').replace('www.dropbox.com', 'dl.dropboxusercontent.com')
                    response = requests.get(download_url, timeout=300, stream=True)
                else:
                    response = requests.get(url_input, timeout=300, stream=True)

                response.raise_for_status()

                # Descargar por chunks para archivos grandes
                chunks = []
                total_size = 0
                progress_bar = st.sidebar.progress(0, text="Descargando...")
                content_length = int(response.headers.get('content-length', 0))

                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        chunks.append(chunk)
                        total_size += len(chunk)
                        if content_length > 0:
                            progress = min(total_size / content_length, 1.0)
                            progress_bar.progress(progress, text=f"Descargando... {total_size/1024/1024:.0f} MB")

                progress_bar.empty()
                content = b''.join(chunks)
                uploaded_file = io.BytesIO(content)

                # Detectar tipo
                url_lower = url_input.lower()
                if url_lower.endswith('.csv.gz') or url_lower.endswith('.gz'):
                    file_type = 'csv_gz'
                elif url_lower.endswith('.xlsx') or url_lower.endswith('.xls'):
                    file_type = 'xlsx'
                elif content[:4] == b'PK\x03\x04':
                    file_type = 'xlsx'
                elif content[:2] == b'\x1f\x8b':  # Firma gzip
                    file_type = 'csv_gz'
                else:
                    file_type = 'csv'

                st.sidebar.success(f"Archivo descargado ({total_size/1024/1024:.1f} MB) - Tipo: {file_type}")

        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Error al descargar: {e}")
            st.sidebar.info("Verifica que el enlace sea público ('Cualquier persona con el enlace').")
            uploaded_file = None

# --- Parámetros del modelo ---
st.sidebar.subheader("3. Parámetros del Reporte")
meta_ventas = st.sidebar.number_input("Meta de Ventas Mensual ($)", min_value=1000, value=500000, step=25000)
alerta_cumplimiento = st.sidebar.slider("Umbral de Alerta de Cumplimiento (%)", min_value=50, max_value=100, value=90, format="%d%%")
horizonte_pronostico = st.sidebar.selectbox("Meses a Pronosticar", [3, 6, 12, 18], index=2)
top_n_productos = st.sidebar.slider("Top N Productos a Mostrar", min_value=5, max_value=20, value=10)

# --- Selección de modelos ---
st.sidebar.subheader("4. Modelos Predictivos")
modelos_disponibles = {
    "Baseline (Seasonal Naive)": "baseline",
    "Prophet (Meta)": "prophet",
    "SARIMA (Estadístico)": "sarima",
    "XGBoost (Machine Learning)": "xgboost",
    "LSTM (Deep Learning)": "lstm"
}
modelos_seleccionados = st.sidebar.multiselect(
    "Selecciona los modelos a evaluar",
    options=list(modelos_disponibles.keys()),
    default=list(modelos_disponibles.keys()),
    help="Selecciona al menos 2 modelos para comparar. Se recomienda usar los 5 para una evaluación completa."
)

if len(modelos_seleccionados) < 1:
    st.sidebar.warning("Selecciona al menos 1 modelo.")

modelos_keys = [modelos_disponibles[m] for m in modelos_seleccionados]

if uploaded_file is not None and len(modelos_seleccionados) >= 1:
    try:
        with st.spinner("Procesando archivo... Esto puede tardar varios minutos para archivos grandes."):

            # --- Indicador de tipo de uso ---
            if tipo_uso == "Primera carga (historial completo)":
                st.info("Modo: **Primera carga**. Se procesará todo el historial para configurar el sistema de pronósticos.")
            else:
                st.info("Modo: **Actualización mensual**. Se re-procesará el historial actualizado para generar nuevos pronósticos.")

            # --- 1. ETL ---
            st.write("### 1. Limpieza y Transformación de Datos (ETL)")
            with st.expander("Ver detalles del proceso ETL"):
                df_limpios, df_mensual, df_diario, df_producto, df_familia = run_etl(uploaded_file, st, file_type)

            # --- Validar datos suficientes para modelado ---
            n_meses = len(df_mensual)
            if n_meses < 6:
                st.error(f"Se encontraron solo {n_meses} mes(es) de datos. Se necesitan al menos 6 meses para generar pronósticos confiables.")
                st.warning("Por favor, sube un archivo con más meses de datos históricos.")

                st.write("### 2. Análisis Exploratorio de Datos (EDA)")
                with st.expander("Ver detalles del Análisis Exploratorio"):
                    figs_eda, dfs_eda = run_eda(df_limpios, df_mensual, df_diario, df_producto, df_familia, top_n_productos, st)
                st.stop()

            # --- Resumen de datos cargados ---
            st.success(f"Datos cargados exitosamente: **{len(df_limpios):,}** registros limpios, **{n_meses} meses** de historia ({df_mensual['ds'].min().strftime('%b %Y')} - {df_mensual['ds'].max().strftime('%b %Y')})")

            # --- 2. EDA ---
            st.write("### 2. Análisis Exploratorio de Datos (EDA)")
            with st.expander("Ver detalles del Análisis Exploratorio"):
                figs_eda, dfs_eda = run_eda(df_limpios, df_mensual, df_diario, df_producto, df_familia, top_n_productos, st)

            # --- 3. Modelado ---
            st.write("### 3. Modelado Predictivo")
            st.write(f"Evaluando **{len(modelos_seleccionados)} modelos**: {', '.join(modelos_seleccionados)}")
            with st.expander("Ver detalles del Modelado"):
                df_comparacion, df_pronostico_final, best_model_name, figs_modelos = run_modeling(
                    df_mensual, horizonte_pronostico, st, modelos_keys
                )

            # --- 4. Dashboard Interactivo ---
            st.write("## Dashboard de Resultados")

            # KPIs principales
            col1, col2, col3, col4 = st.columns(4)
            prox_mes_valor = df_pronostico_final['yhat'].iloc[0]
            col1.metric("Pronóstico Próx. Mes", f"${prox_mes_valor:,.2f}")
            col2.metric("Meta de Ventas", f"${meta_ventas:,.2f}")
            cumplimiento = (prox_mes_valor / meta_ventas) * 100
            delta_color = "normal" if cumplimiento >= alerta_cumplimiento else "inverse"
            col3.metric("Cumplimiento vs Meta", f"{cumplimiento:.1f}%",
                        delta=f"{'Sobre' if cumplimiento >= 100 else 'Bajo'} meta",
                        delta_color=delta_color)
            col4.metric("Mejor Modelo", best_model_name)

            # Alerta de cumplimiento
            if cumplimiento < alerta_cumplimiento:
                st.warning(f"El pronóstico ({cumplimiento:.1f}%) está por debajo del umbral de alerta ({alerta_cumplimiento}%). Se recomienda tomar acciones correctivas.")
            elif cumplimiento >= 100:
                st.success(f"El pronóstico supera la meta mensual en un {cumplimiento - 100:.1f}%.")

            # Gráfico principal de pronóstico
            st.plotly_chart(figs_modelos['pronostico_vs_historico'], use_container_width=True)

            # Tabla comparativa de modelos
            st.subheader("Comparación de Modelos")
            st.dataframe(df_comparacion.style.format({
                'MAE': '${:,.2f}',
                'RMSE': '${:,.2f}',
                'MAPE': '{:.2f}%'
            }).highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='#d4edda'), use_container_width=True)

            # Tabla de pronóstico
            st.subheader("Pronóstico Detallado")
            df_pron_display = df_pronostico_final.copy()
            df_pron_display['ds'] = df_pron_display['ds'].dt.strftime('%B %Y')
            df_pron_display = df_pron_display.rename(columns={
                'ds': 'Mes', 'yhat': 'Pronóstico ($)',
                'yhat_lower': 'Límite Inferior ($)', 'yhat_upper': 'Límite Superior ($)'
            })
            st.dataframe(df_pron_display.style.format({
                'Pronóstico ($)': '${:,.2f}',
                'Límite Inferior ($)': '${:,.2f}',
                'Límite Superior ($)': '${:,.2f}'
            }), use_container_width=True)

            # Gráficos de análisis
            st.subheader("Análisis de Productos y Patrones")
            c1, c2 = st.columns(2)
            c1.plotly_chart(figs_eda['top_productos'], use_container_width=True)
            c2.plotly_chart(figs_eda['patron_horario'], use_container_width=True)

            # Gráfico de comparación de modelos si existe
            if 'comparacion_modelos' in figs_modelos:
                st.plotly_chart(figs_modelos['comparacion_modelos'], use_container_width=True)

            # --- 5. Generación de Reporte ---
            st.write("### 4. Generación de Reporte PDF")
            pdf_buffer = generate_report(df_pronostico_final, df_comparacion, figs_eda, figs_modelos, meta_ventas, alerta_cumplimiento, df_limpios, top_n_productos, st)

            st.download_button(
                label="Descargar Reporte Ejecutivo en PDF",
                data=pdf_buffer,
                file_name=f"Reporte_Ventas_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

            # --- Resumen final ---
            st.write("---")
            st.write("### Resumen del Análisis")
            if tipo_uso == "Primera carga (historial completo)":
                st.success(
                    f"**Sistema configurado exitosamente.** Se analizaron {n_meses} meses de historial, "
                    f"se evaluaron {len(modelos_seleccionados)} modelos predictivos, y se seleccionó "
                    f"**{best_model_name}** como el mejor modelo. El próximo mes se pronostica en "
                    f"**${prox_mes_valor:,.2f}**. Para la próxima actualización, selecciona 'Actualización mensual' "
                    f"y sube el archivo con el mes más reciente incluido."
                )
            else:
                ultimo_mes = df_mensual['ds'].max().strftime('%B %Y')
                st.success(
                    f"**Pronósticos actualizados.** Se incorporaron datos hasta {ultimo_mes}. "
                    f"El modelo **{best_model_name}** pronostica **${prox_mes_valor:,.2f}** para el próximo mes "
                    f"({cumplimiento:.1f}% de la meta). Descarga el reporte PDF para presentar en la reunión gerencial."
                )

    except Exception as e:
        st.error(f"Ocurrió un error durante el procesamiento: {e}")
        st.error("Por favor, verifica que el formato del archivo sea el correcto y vuelve a intentarlo.")
        import traceback
        with st.expander("Ver detalles del error"):
            st.code(traceback.format_exc())

else:
    if len(modelos_seleccionados) < 1:
        st.warning("Selecciona al menos 1 modelo predictivo en la barra lateral.")
    else:
        st.info("Por favor, sube un archivo o carga uno desde una URL para comenzar el análisis.")

    with st.expander("Guía de uso del sistema"):
        st.write("""
        **¿Cómo funciona este sistema?**

        Este sistema analiza el historial de ventas de tu empresa para generar pronósticos mensuales
        y recomendaciones accionables para la gerencia.

        **Flujo de trabajo:**

        1. **Primera carga:** Sube el archivo completo del ERP con todo el historial (idealmente 2-3 años).
           El sistema limpiará los datos, analizará patrones y generará pronósticos.

        2. **Actualización mensual:** Cada mes, exporta los datos actualizados del ERP (incluyendo el mes nuevo)
           y súbelos al sistema. Los pronósticos se actualizarán automáticamente.

        **Formatos aceptados:**
        - CSV (separado por punto y coma o coma)
        - CSV comprimido (.csv.gz) - recomendado para archivos grandes
        - Excel (.xlsx, .xls)

        **Tamaño máximo:** 1 GB

        **Carga desde URL (Google Drive, Dropbox, etc.):**
        1. Sube tu archivo a Google Drive
        2. Haz clic derecho > Compartir > "Cualquier persona con el enlace"
        3. Copia el enlace y pégalo en la opción "Cargar desde URL"

        **Modelos disponibles:**
        - **Baseline:** Repite el patrón del año anterior (referencia)
        - **Prophet:** Modelo de Meta que descompone tendencia y estacionalidad
        - **SARIMA:** Modelo estadístico clásico para series temporales
        - **XGBoost:** Machine Learning que aprende de múltiples características
        - **LSTM:** Red neuronal con memoria para patrones complejos
        """)
