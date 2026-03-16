import pandas as pd
import numpy as np

# ============================================================
# ENCABEZADOS DE LAS 32 COLUMNAS (en el orden correcto)
# Estos nombres se obtuvieron del archivo de ejemplo (data.xlsx)
# que sí tenía encabezados. Coinciden con el Notebook 01.
# ============================================================
COLUMN_HEADERS = [
    'DOC_ID',               # 0  - ID único del documento/factura
    'DOC_NUM',              # 1  - Número del documento
    'DOC_FECHA',            # 2  - Fecha y hora exacta de la transacción
    'FECHA_NEGOCIO',        # 3  - Fecha de negocio (solo fecha, sin hora)
    'CL_NUM',               # 4  - Número de cliente
    'DE_NUM',               # 5  - Número de departamento/punto de venta
    'DOC_AUTORIZACION',     # 6  - Número de autorización del documento
    'DOC_PUNTO_EMISION',    # 7  - Punto de emisión
    'DOC_SERIE_SUCURSAL',   # 8  - Serie de la sucursal
    'DOC_SERIE_PEMISION',   # 9  - Serie del punto de emisión
    'DOC_ANULADA',          # 10 - Indicador de anulación (0=válida, 1=anulada)
    'DOC_MOTIVO',           # 11 - Motivo de anulación (si aplica)
    'DOC_BASE_CERO',        # 12 - Base imponible con tarifa 0%
    'DOC_BASE_DIF_CERO',    # 13 - Base imponible con tarifa diferente de 0%
    'DOC_BASE_NO_IVA',      # 14 - Base no sujeta a IVA
    'DOC_BASE_EXENTO',      # 15 - Base exenta de IVA
    'DOC_MONTO_IVA',        # 16 - Monto total de IVA
    'DOC_PROPINA',          # 17 - Propina
    'DOC_IVA_PORC',         # 18 - Porcentaje de IVA aplicado
    'DOC_COMPANIA',         # 19 - Nombre de la compañía
    'DOC_RUC_COMPANIA',     # 20 - RUC de la compañía
    'D_NUM',                # 21 - Número de línea de detalle
    'D_TIPO',               # 22 - Tipo de ítem (M = menú/producto)
    'D_CODIGO',             # 23 - Código del producto
    'D_ITEM',               # 24 - Nombre/descripción del producto
    'D_FAMILY',             # 25 - Código de familia del producto
    'D_MAJOR',              # 26 - Categoría mayor (1=Alimentos, 2=Bebidas, 6=Destino)
    'D_PRICE_LEVEL',        # 27 - Nivel de precio
    'D_CANTIDAD',           # 28 - Cantidad vendida
    'D_VALOR',              # 29 - Valor de la venta ($)
    'D_IMPUESTO',           # 30 - Indicador de impuesto
    'D_ORDEN'               # 31 - Número de orden (generalmente vacío)
]


def run_etl(uploaded_file, st_ref, file_type='csv'):
    """Ejecuta el pipeline ETL completo. Acepta CSV o Excel."""
    st_ref.write("Cargando archivo...")

    if file_type == 'xlsx':
        # --- Cargar Excel ---
        df = pd.read_excel(uploaded_file, header=None)
        # Detectar si la primera fila son encabezados
        first_row = df.iloc[0].astype(str).tolist()
        if 'DOC_ID' in first_row or 'doc_id' in [x.lower() for x in first_row]:
            # El Excel tiene encabezados, descartarlos y usar los nuestros
            df = df.iloc[1:].reset_index(drop=True)
        if df.shape[1] == len(COLUMN_HEADERS):
            df.columns = COLUMN_HEADERS
        else:
            st_ref.warning(f"El archivo tiene {df.shape[1]} columnas, se esperaban {len(COLUMN_HEADERS)}.")
            # Asignar nombres a las columnas disponibles
            df.columns = COLUMN_HEADERS[:df.shape[1]]
    else:
        # --- Cargar CSV ---
        # Intentar con punto y coma primero (formato del ERP)
        try:
            df = pd.read_csv(uploaded_file, sep=';', header=None, names=COLUMN_HEADERS,
                             encoding='utf-8', low_memory=False)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';', header=None, names=COLUMN_HEADERS,
                             encoding='latin-1', low_memory=False)

        # Verificar si el separador fue correcto
        if df.shape[1] == 1 or df.iloc[:, 1:].isna().all().all():
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=',', header=None, names=COLUMN_HEADERS,
                             encoding='utf-8', low_memory=False)

        # Detectar si la primera fila son encabezados (por si el CSV los tiene)
        first_row = df.iloc[0].astype(str).tolist()
        if 'DOC_ID' in first_row or 'doc_id' in [x.lower() for x in first_row]:
            df = df.iloc[1:].reset_index(drop=True)

    st_ref.write(f"Archivo cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    # --- Limpieza ---
    st_ref.write("Limpiando datos...")

    # Convertir fechas
    df['DOC_FECHA'] = pd.to_datetime(df['DOC_FECHA'], errors='coerce')
    df['FECHA_NEGOCIO'] = pd.to_datetime(df['FECHA_NEGOCIO'], errors='coerce')

    # Eliminar registros sin fecha
    df = df.dropna(subset=['FECHA_NEGOCIO'])

    # Eliminar columna D_ORDEN (generalmente vacía)
    if 'D_ORDEN' in df.columns:
        df = df.drop(columns=['D_ORDEN'], errors='ignore')

    # Filtrar transacciones anuladas (DOC_ANULADA: 0=válida, 1=anulada)
    if 'DOC_ANULADA' in df.columns:
        df['DOC_ANULADA'] = pd.to_numeric(df['DOC_ANULADA'], errors='coerce').fillna(0).astype(int)
        filas_antes = len(df)
        df = df[df['DOC_ANULADA'] == 0]
        eliminadas = filas_antes - len(df)
        if eliminadas > 0:
            st_ref.write(f"  Transacciones anuladas eliminadas: {eliminadas:,}")

    # Filtrar registros no comerciales (D_MAJOR == 6 son destinos/países)
    if 'D_MAJOR' in df.columns:
        df['D_MAJOR'] = pd.to_numeric(df['D_MAJOR'], errors='coerce')
        filas_antes = len(df)
        df = df[df['D_MAJOR'] != 6]
        eliminadas = filas_antes - len(df)
        if eliminadas > 0:
            st_ref.write(f"  Registros no comerciales (D_MAJOR=6) eliminados: {eliminadas:,}")

    # Convertir columnas numéricas
    numeric_cols = ['D_CANTIDAD', 'D_VALOR', 'D_IMPUESTO', 'DOC_BASE_CERO',
                    'DOC_BASE_DIF_CERO', 'DOC_BASE_NO_IVA', 'DOC_BASE_EXENTO',
                    'DOC_MONTO_IVA', 'DOC_PROPINA']
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

    # Agregado mensual (D_VALOR es la columna de ventas)
    df_mensual = df.groupby(df['FECHA_NEGOCIO'].dt.to_period('M')).agg(
        y=('D_VALOR', 'sum')
    ).reset_index()
    df_mensual.columns = ['ds', 'y']
    df_mensual['ds'] = df_mensual['ds'].dt.to_timestamp()

    # Agregado diario
    df_diario = df.groupby(df['FECHA_NEGOCIO'].dt.date).agg(
        ventas_total=('D_VALOR', 'sum'),
        transacciones=('DOC_ID', 'nunique')
    ).reset_index()
    df_diario.columns = ['fecha', 'ventas_total', 'transacciones']

    # Agregado por producto y mes
    df_producto = df.groupby(['D_ITEM', df['FECHA_NEGOCIO'].dt.to_period('M')]).agg(
        ventas_total=('D_VALOR', 'sum'),
        cantidad=('D_CANTIDAD', 'sum')
    ).reset_index()

    # Agregado por familia y mes
    df_familia = df.groupby(['D_FAMILY', df['FECHA_NEGOCIO'].dt.to_period('M')]).agg(
        ventas_total=('D_VALOR', 'sum'),
        cantidad=('D_CANTIDAD', 'sum')
    ).reset_index()

    rango = f"{df['FECHA_NEGOCIO'].min().strftime('%Y-%m-%d')} a {df['FECHA_NEGOCIO'].max().strftime('%Y-%m-%d')}"
    st_ref.success(f"ETL completado. Rango de datos: {rango}. Total registros limpios: {filas_limpias:,}")

    return df, df_mensual, df_diario, df_producto, df_familia
