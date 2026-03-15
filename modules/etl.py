import pandas as pd
import numpy as np

COLUMN_NAMES = [
    'DOC_ID', 'DOC_NUM', 'DOC_FECHA', 'FECHA_NEGOCIO', 'CI_NUM', 'DE_NUM',
    'DOC_AUTORIZACION', 'DOC_PUNTO_EMISION', 'DOC_TIPO', 'DOC_ESTADO',
    'D_ITEM', 'D_MAJOR', 'D_FAMILY', 'D_QTY', 'D_TOTAL', 'D_TAX',
    'D_DISCOUNT', 'D_NET', 'D_COST', 'D_PROFIT', 'D_MARGIN',
    'DEST_ID', 'DEST_NAME', 'DEST_CITY', 'DEST_STATE', 'DEST_COUNTRY',
    'DEST_REGION', 'DEST_TYPE', 'DEST_STATUS', 'DEST_OPEN_DATE',
    'DEST_CLOSE_DATE', 'DEST_MANAGER'
]


def run_etl(uploaded_file, st_ref):
    """Ejecuta el pipeline ETL completo."""
    st_ref.write("Cargando archivo CSV...")

    # Intentar con punto y coma primero (formato del ERP)
    try:
        df = pd.read_csv(uploaded_file, sep=';', header=None, names=COLUMN_NAMES,
                         encoding='utf-8', low_memory=False)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';', header=None, names=COLUMN_NAMES,
                         encoding='latin-1', low_memory=False)

    # Verificar si el separador fue correcto
    if df.shape[1] == 1 or df.iloc[:, 1:].isna().all().all():
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=',', header=None, names=COLUMN_NAMES,
                         encoding='utf-8', low_memory=False)

    st_ref.write(f"Archivo cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    # --- Limpieza ---
    st_ref.write("Limpiando datos...")

    # Convertir fechas
    df['DOC_FECHA'] = pd.to_datetime(df['DOC_FECHA'], errors='coerce')
    df['FECHA_NEGOCIO'] = pd.to_datetime(df['FECHA_NEGOCIO'], errors='coerce')

    # Eliminar registros sin fecha
    df = df.dropna(subset=['FECHA_NEGOCIO'])

    # Eliminar anuladas
    if 'DOC_ESTADO' in df.columns:
        df = df[df['DOC_ESTADO'].astype(str).str.upper() != 'ANULADA']

    # Convertir columnas numéricas
    numeric_cols = ['D_QTY', 'D_TOTAL', 'D_TAX', 'D_DISCOUNT', 'D_NET',
                    'D_COST', 'D_PROFIT', 'D_MARGIN']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Crear columnas temporales
    df['anio'] = df['FECHA_NEGOCIO'].dt.year
    df['mes'] = df['FECHA_NEGOCIO'].dt.month
    df['dia'] = df['FECHA_NEGOCIO'].dt.day
    df['hora'] = df['DOC_FECHA'].dt.hour
    df['dia_semana'] = df['FECHA_NEGOCIO'].dt.dayofweek
    df['periodo'] = df['FECHA_NEGOCIO'].dt.to_period('M')

    filas_limpias = df.shape[0]
    st_ref.write(f"Datos limpios: {filas_limpias:,} filas")

    # --- Agregaciones ---
    st_ref.write("Generando tablas agregadas...")

    # Agregado mensual
    df_mensual = df.groupby(df['FECHA_NEGOCIO'].dt.to_period('M')).agg(
        y=('D_TOTAL', 'sum')
    ).reset_index()
    df_mensual.columns = ['ds', 'y']
    df_mensual['ds'] = df_mensual['ds'].dt.to_timestamp()

    # Agregado diario
    df_diario = df.groupby(df['FECHA_NEGOCIO'].dt.date).agg(
        ventas_total=('D_TOTAL', 'sum'),
        transacciones=('DOC_ID', 'nunique')
    ).reset_index()
    df_diario.columns = ['fecha', 'ventas_total', 'transacciones']

    # Agregado por producto y mes
    df_producto = df.groupby(['D_ITEM', df['FECHA_NEGOCIO'].dt.to_period('M')]).agg(
        ventas_total=('D_TOTAL', 'sum'),
        cantidad=('D_QTY', 'sum')
    ).reset_index()

    # Agregado por familia y mes
    df_familia = df.groupby(['D_FAMILY', df['FECHA_NEGOCIO'].dt.to_period('M')]).agg(
        ventas_total=('D_TOTAL', 'sum'),
        cantidad=('D_QTY', 'sum')
    ).reset_index()

    rango = f"{df['FECHA_NEGOCIO'].min().strftime('%Y-%m-%d')} a {df['FECHA_NEGOCIO'].max().strftime('%Y-%m-%d')}"
    st_ref.success(f"ETL completado. Rango de datos: {rango}. Total registros limpios: {filas_limpias:,}")

    return df, df_mensual, df_diario, df_producto, df_familia
