
import streamlit as st
import pandas as pd
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
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV de ventas (datakv.csv)", type=["csv"])

# --- Parámetros del modelo ---
st.sidebar.subheader("2. Parámetros del Reporte")
meta_ventas = st.sidebar.number_input("Meta de Ventas Mensual ($)", min_value=100000, value=500000, step=25000)
alerta_cumplimiento = st.sidebar.slider("Umbral de Alerta de Cumplimiento (%)", min_value=50, max_value=100, value=90, format="%d%%")
horizonte_pronostico = st.sidebar.selectbox("Meses a Pronosticar", [6, 12, 18], index=1)
top_n_productos = st.sidebar.slider("Top N Productos a Mostrar", min_value=5, max_value=20, value=10)

if uploaded_file is not None:
    try:
        with st.spinner("Procesando archivo... Esto puede tardar varios minutos."):
            # --- 1. ETL ---
            st.write("### 1. Limpieza y Transformación de Datos (ETL)")
            with st.expander("Ver detalles del proceso ETL"):
                df_limpios, df_mensual, df_diario, df_producto, df_familia = run_etl(uploaded_file, st)

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
                label="📥 Descargar Reporte Ejecutivo en PDF",
                data=pdf_buffer,
                file_name=f"Reporte_Ventas_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"Ocurrió un error durante el procesamiento: {e}")
        st.error("Por favor, verifica que el formato del archivo CSV sea el correcto y vuelve a intentarlo.")

else:
    st.info("Por favor, sube un archivo CSV para comenzar el análisis.")
