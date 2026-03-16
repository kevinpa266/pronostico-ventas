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

if metodo_carga == "Subir archivo":
    uploaded_file = st.sidebar.file_uploader(
        "Sube tu archivo de ventas (CSV o Excel)",
        type=["csv", "xlsx", "xls"]
    )
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            file_type = 'xlsx'
        else:
            file_type = 'csv'

else:
    url_input = st.sidebar.text_input(
        "Pega la URL del archivo (Google Drive, Dropbox, etc.)",
        placeholder="https://drive.google.com/file/d/.../view"
    )
    if url_input:
        try:
            with st.spinner("Descargando archivo desde URL..."):
                # Convertir URL de Google Drive a enlace de descarga directa
                gdrive_match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', url_input)
                if gdrive_match:
                    file_id = gdrive_match.group(1)
                    download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                # Convertir URL de Dropbox
                elif 'dropbox.com' in url_input:
                    download_url = url_input.replace('dl=0', 'dl=1').replace('www.dropbox.com', 'dl.dropboxusercontent.com')
                else:
                    download_url = url_input

                response = requests.get(download_url, timeout=120, stream=True)
                response.raise_for_status()

                content = response.content
                uploaded_file = io.BytesIO(content)

                # Detectar tipo de archivo por la URL o contenido
                url_lower = url_input.lower()
                if url_lower.endswith('.xlsx') or url_lower.endswith('.xls'):
                    file_type = 'xlsx'
                elif url_lower.endswith('.csv'):
                    file_type = 'csv'
                else:
                    # Intentar detectar por el contenido
                    if content[:4] == b'PK\x03\x04':  # Firma de archivo ZIP/XLSX
                        file_type = 'xlsx'
                    else:
                        file_type = 'csv'

                st.sidebar.success(f"Archivo descargado ({len(content)/1024/1024:.1f} MB)")

        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Error al descargar: {e}")
            st.sidebar.info("Verifica que el enlace sea público y accesible.")
            uploaded_file = None

# --- Parámetros del modelo ---
st.sidebar.subheader("2. Parámetros del Reporte")
meta_ventas = st.sidebar.number_input("Meta de Ventas Mensual ($)", min_value=1000, value=500000, step=25000)
alerta_cumplimiento = st.sidebar.slider("Umbral de Alerta de Cumplimiento (%)", min_value=50, max_value=100, value=90, format="%d%%")
horizonte_pronostico = st.sidebar.selectbox("Meses a Pronosticar", [3, 6, 12, 18], index=2)
top_n_productos = st.sidebar.slider("Top N Productos a Mostrar", min_value=5, max_value=20, value=10)

if uploaded_file is not None:
    try:
        with st.spinner("Procesando archivo... Esto puede tardar varios minutos."):
            # --- 1. ETL ---
            st.write("### 1. Limpieza y Transformación de Datos (ETL)")
            with st.expander("Ver detalles del proceso ETL"):
                df_limpios, df_mensual, df_diario, df_producto, df_familia = run_etl(uploaded_file, st, file_type)

            # --- Validar datos suficientes para modelado ---
            n_meses = len(df_mensual)
            if n_meses < 6:
                st.error(f"Se encontraron solo {n_meses} mes(es) de datos. Se necesitan al menos 6 meses para generar pronósticos confiables.")
                st.warning("Por favor, sube un archivo con más meses de datos históricos.")
                st.info("El ETL y el análisis exploratorio se completaron correctamente. Puedes revisar los detalles arriba.")

                # Mostrar EDA aunque no haya suficientes datos para modelado
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
    st.info("Por favor, sube un archivo CSV o Excel, o carga uno desde una URL, para comenzar el análisis.")
    st.write("**Formatos aceptados:** CSV (separado por punto y coma o coma), Excel (.xlsx, .xls)")
    st.write("**Tamaño máximo:** 1 GB")
    st.write("**Fuentes de URL compatibles:** Google Drive, Dropbox, OneDrive (enlace público)")
