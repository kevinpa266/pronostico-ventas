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

# --- Carga de archivo ---
st.sidebar.subheader("1. Cargar Datos")

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
st.sidebar.subheader("2. Parámetros del Reporte")
meta_ventas = st.sidebar.number_input("Meta de Ventas Mensual ($)", min_value=1000, value=500000, step=25000)
alerta_cumplimiento = st.sidebar.slider("Umbral de Alerta de Cumplimiento (%)", min_value=50, max_value=100, value=90, format="%d%%")
horizonte_pronostico = st.sidebar.selectbox("Meses a Pronosticar", [3, 6, 12, 18], index=2)
top_n_productos = st.sidebar.slider("Top N Productos a Mostrar", min_value=5, max_value=20, value=10)

if uploaded_file is not None:
    try:
        with st.spinner("Procesando archivo... Esto puede tardar varios minutos para archivos grandes."):
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

            # --- 2. EDA ---
            st.write("### 2. Análisis Exploratorio de Datos (EDA)")
            with st.expander("Ver detalles del Análisis Exploratorio"):
                figs_eda, dfs_eda = run_eda(df_limpios, df_mensual, df_diario, df_producto, df_familia, top_n_productos, st)

            # --- 3. Modelado ---
            st.write("### 3. Modelado Predictivo")
            with st.expander("Ver detalles del Modelado"):
                df_comparacion, df_pronostico_final, best_model_name, figs_modelos = run_modeling(df_mensual, horizonte_pronostico, st)

            # --- 4. Dashboard Interactivo ---
            st.write("## Dashboard de Resultados")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pronóstico Próx. Mes", f"${df_pronostico_final['yhat'].iloc[0]:,.2f}")
            col2.metric("Meta de Ventas", f"${meta_ventas:,.2f}")
            cumplimiento = (df_pronostico_final['yhat'].iloc[0] / meta_ventas) * 100
            col3.metric("Cumplimiento vs Meta", f"{cumplimiento:.1f}%")
            col4.metric("Promedio Hist. Mensual", f"${df_mensual['y'].mean():,.2f}")

            st.plotly_chart(figs_modelos['pronostico_vs_historico'], use_container_width=True)

            st.subheader("Análisis de Productos y Patrones")
            c1, c2 = st.columns(2)
            c1.plotly_chart(figs_eda['top_productos'], use_container_width=True)
            c2.plotly_chart(figs_eda['patron_horario'], use_container_width=True)

            # --- 5. Generación de Reporte ---
            st.write("### 4. Generación de Reporte PDF")
            pdf_buffer = generate_report(df_pronostico_final, df_comparacion, figs_eda, figs_modelos, meta_ventas, alerta_cumplimiento, df_limpios, top_n_productos, st)

            st.download_button(
                label="Descargar Reporte Ejecutivo en PDF",
                data=pdf_buffer,
                file_name=f"Reporte_Ventas_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"Ocurrió un error durante el procesamiento: {e}")
        st.error("Por favor, verifica que el formato del archivo sea el correcto y vuelve a intentarlo.")
        import traceback
        with st.expander("Ver detalles del error"):
            st.code(traceback.format_exc())

else:
    st.info("Por favor, sube un archivo o carga uno desde una URL para comenzar el análisis.")

    with st.expander("Formatos y opciones de carga"):
        st.write("""
        **Formatos aceptados:**
        - CSV (separado por punto y coma o coma)
        - CSV comprimido (.csv.gz) - recomendado para archivos grandes
        - Excel (.xlsx, .xls)

        **Tamaño máximo:** 1 GB

        **Carga desde URL (Google Drive, Dropbox, etc.):**
        1. Sube tu archivo a Google Drive
        2. Haz clic derecho > Compartir > "Cualquier persona con el enlace"
        3. Copia el enlace y pégalo en la opción "Cargar desde URL"

        **Recomendación para archivos grandes (>200 MB):**
        Usa el archivo pre-limpiado comprimido (.csv.gz) para un procesamiento más rápido.
        """)
