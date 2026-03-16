import pandas as pd
import numpy as np
import io
import gzip

# ============================================================
# ENCABEZADOS DE LAS 32 COLUMNAS (en el orden correcto)
# Estos nombres se obtuvieron del archivo de ejemplo (data.xlsx)
# que sí tenía encabezados. Coinciden con el Notebook 01.
# ============================================================
COLUMN_HEADERS = [
    'DOC_ID', 'DOC_NUM', 'DOC_FECHA', 'FECHA_NEGOCIO', 'CL_NUM', 'DE_NUM',
    'DOC_AUTORIZACION', 'DOC_PUNTO_EMISION', 'DOC_SERIE_SUCURSAL', 'DOC_SERIE_PEMISION',
    'DOC_ANULADA', 'DOC_MOTIVO', 'DOC_BASE_CERO', 'DOC_BASE_DIF_CERO', 'DOC_BASE_NO_IVA',
    'DOC_BASE_EXENTO', 'DOC_MONTO_IVA', 'DOC_PROPINA', 'DOC_IVA_PORC', 'DOC_COMPANIA',
    'DOC_RUC_COMPANIA', 'D_NUM', 'D_TIPO', 'D_CODIGO', 'D_ITEM', 'D_FAMILY', 'D_MAJOR',
    'D_PRICE_LEVEL', 'D_CANTIDAD', 'D_VALOR', 'D_IMPUESTO', 'D_ORDEN'
]

# Columnas esenciales para el análisis (reduce uso de memoria)
ESSENTIAL_COLS_IDX = [0, 2, 3, 10, 24, 25, 26, 28, 29]
ESSENTIAL_COLS_NAMES = ['DOC_ID', 'DOC_FECHA', 'FECHA_NEGOCIO', 'DOC_ANULADA',
                        'D_ITEM', 'D_FAMILY', 'D_MAJOR', 'D_CANTIDAD', 'D_VALOR']


def _detect_and_load_excel(uploaded_file):
    """Carga un archivo Excel detectando automáticamente si tiene encabezados."""
    df_with_header = pd.read_excel(uploaded_file, header=0)
    cols_upper = [str(c).upper().strip() for c in df_with_header.columns]

    known_cols = {'DOC_ID', 'DOC_NUM', 'DOC_FECHA', 'FECHA_NEGOCIO', 'D_VALOR', 'D_ITEM'}
    if known_cols.intersection(set(cols_upper)):
        df_with_header.columns = cols_upper
        return df_with_header
    else:
        uploaded_file.seek(0) if hasattr(uploaded_file, 'seek') else None
        df = pd.read_excel(uploaded_file, header=None)
        if df.shape[1] == len(COLUMN_HEADERS):
            df.columns = COLUMN_HEADERS
        elif df.shape[1] < len(COLUMN_HEADERS):
            df.columns = COLUMN_HEADERS[:df.shape[1]]
        else:
            df.columns = COLUMN_HEADERS + [f'extra_{i}' for i in range(df.shape[1] - len(COLUMN_HEADERS))]
        return df


def _detect_csv_format(uploaded_file):
    """Detecta el formato del CSV: separador, si tiene encabezados, y número de columnas."""
    if hasattr(uploaded_file, 'seek'):
        uploaded_file.seek(0)

    # Leer las primeras líneas
    if hasattr(uploaded_file, 'read'):
        sample = uploaded_file.read(4096)
        uploaded_file.seek(0)
        if isinstance(sample, bytes):
            sample = sample.decode('utf-8', errors='replace')
    else:
        sample = str(uploaded_file)

    first_line = sample.split('\n')[0]

    # Detectar separador
    if first_line.count(';') > first_line.count(','):
        sep = ';'
    else:
        sep = ','

    # Detectar si tiene encabezados
    has_header = 'DOC_ID' in first_line.upper() or 'FECHA_NEGOCIO' in first_line.upper()

    # Contar columnas
    n_cols = len(first_line.split(sep))

    return sep, has_header, n_cols


def _load_csv_chunked(uploaded_file, st_ref, sep, has_header, n_cols):
    """Carga un CSV grande por chunks, limpiando cada chunk para ahorrar memoria."""
    st_ref.write("Procesando archivo grande por bloques para optimizar memoria...")

    # Determinar nombres de columnas y columnas a usar
    if n_cols == 32 and not has_header:
        # Archivo original del ERP con 32 columnas sin encabezados
        usecols = ESSENTIAL_COLS_IDX
        col_names = ESSENTIAL_COLS_NAMES
        header_param = None
        names_param = COLUMN_HEADERS
    elif has_header:
        usecols = None
        col_names = None
        header_param = 0
        names_param = None
    else:
        usecols = None
        col_names = None
        header_param = None
        names_param = COLUMN_HEADERS[:n_cols] if n_cols <= len(COLUMN_HEADERS) else None

    chunks_list = []
    total_raw = 0
    total_clean = 0
    chunk_size = 300000

    try:
        reader = pd.read_csv(
            uploaded_file, sep=sep, header=header_param, names=names_param,
            usecols=usecols, encoding='utf-8', low_memory=False,
            chunksize=chunk_size
        )
    except Exception:
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
        reader = pd.read_csv(
            uploaded_file, sep=sep, header=header_param, names=names_param,
            usecols=usecols, encoding='latin-1', low_memory=False,
            chunksize=chunk_size
        )

    for i, chunk in enumerate(reader):
        total_raw += len(chunk)

        # Si usamos usecols con índices, renombrar columnas
        if col_names and len(chunk.columns) == len(col_names):
            chunk.columns = col_names

        # Limpieza rápida del chunk
        chunk = _clean_chunk(chunk)
        total_clean += len(chunk)
        chunks_list.append(chunk)

        if (i + 1) % 5 == 0:
            st_ref.write(f"  Procesados {total_raw:,} registros ({total_clean:,} válidos)...")

    df = pd.concat(chunks_list, ignore_index=True)
    st_ref.write(f"Archivo procesado: {total_raw:,} registros leídos, {total_clean:,} válidos")
    return df


def _clean_chunk(df):
    """Limpia un chunk de datos: convierte tipos, filtra anulados y no comerciales."""
    # Convertir fechas
    if 'DOC_FECHA' in df.columns:
        df['DOC_FECHA'] = pd.to_datetime(df['DOC_FECHA'], errors='coerce')
    if 'FECHA_NEGOCIO' in df.columns:
        df['FECHA_NEGOCIO'] = pd.to_datetime(df['FECHA_NEGOCIO'], errors='coerce')
        df = df.dropna(subset=['FECHA_NEGOCIO'])

    # Filtrar anuladas
    if 'DOC_ANULADA' in df.columns:
        df['DOC_ANULADA'] = pd.to_numeric(df['DOC_ANULADA'], errors='coerce').fillna(0).astype(int)
        df = df[df['DOC_ANULADA'] == 0]

    # Filtrar no comerciales
    if 'D_MAJOR' in df.columns:
        df['D_MAJOR'] = pd.to_numeric(df['D_MAJOR'], errors='coerce')
        df = df[df['D_MAJOR'] != 6]

    # Convertir numéricos
    for col in ['D_CANTIDAD', 'D_VALOR']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Filtrar valores <= 0
    if 'D_VALOR' in df.columns:
        df = df[df['D_VALOR'] > 0]

    return df


def _load_pre_cleaned_csv(uploaded_file, st_ref):
    """Carga un CSV que ya fue pre-limpiado (tiene encabezados con nombres conocidos)."""
    st_ref.write("Detectado archivo pre-limpiado. Cargando...")

    chunks_list = []
    total = 0
    for chunk in pd.read_csv(uploaded_file, chunksize=300000, low_memory=False):
        total += len(chunk)
        chunks_list.append(chunk)
        if len(chunks_list) % 5 == 0:
            st_ref.write(f"  Cargados {total:,} registros...")

    df = pd.concat(chunks_list, ignore_index=True)
    st_ref.write(f"Archivo cargado: {total:,} registros")
    return df


def run_etl(uploaded_file, st_ref, file_type='csv'):
    """Ejecuta el pipeline ETL completo. Acepta CSV, CSV.GZ o Excel."""
    st_ref.write("Cargando archivo...")

    if file_type == 'xlsx':
        df = _detect_and_load_excel(uploaded_file)
        df = _clean_chunk(df)
    elif file_type == 'csv_gz':
        # Archivo comprimido con gzip
        if hasattr(uploaded_file, 'read'):
            content = uploaded_file.read()
            decompressed = gzip.decompress(content)
            uploaded_file = io.BytesIO(decompressed)
        sep, has_header, n_cols = _detect_csv_format(uploaded_file)
        if has_header:
            df = _load_pre_cleaned_csv(uploaded_file, st_ref)
        else:
            df = _load_csv_chunked(uploaded_file, st_ref, sep, has_header, n_cols)
            df = _clean_chunk(df)
    else:
        # CSV normal
        sep, has_header, n_cols = _detect_csv_format(uploaded_file)

        # Si es un archivo pre-limpiado (tiene encabezados conocidos y pocas columnas)
        if has_header and n_cols <= 12:
            df = _load_pre_cleaned_csv(uploaded_file, st_ref)
            # Asegurar tipos correctos
            if 'FECHA_NEGOCIO' in df.columns:
                df['FECHA_NEGOCIO'] = pd.to_datetime(df['FECHA_NEGOCIO'], errors='coerce')
            if 'DOC_FECHA' in df.columns:
                df['DOC_FECHA'] = pd.to_datetime(df['DOC_FECHA'], errors='coerce')
            for col in ['D_CANTIDAD', 'D_VALOR']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            # Archivo grande del ERP - procesar por chunks
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
            df = _load_csv_chunked(uploaded_file, st_ref, sep, has_header, n_cols)

    st_ref.write(f"Datos después de limpieza: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    if df.shape[0] == 0:
        raise ValueError("No quedaron registros después de la limpieza. Verifica el archivo de datos.")

    # Eliminar columna D_ORDEN si existe
    if 'D_ORDEN' in df.columns:
        df = df.drop(columns=['D_ORDEN'], errors='ignore')

    # Crear columnas temporales
    if 'FECHA_NEGOCIO' in df.columns:
        df['anio'] = df['FECHA_NEGOCIO'].dt.year
        df['mes'] = df['FECHA_NEGOCIO'].dt.month
        df['dia'] = df['FECHA_NEGOCIO'].dt.day
        df['dia_semana'] = df['FECHA_NEGOCIO'].dt.dayofweek
        df['periodo'] = df['FECHA_NEGOCIO'].dt.to_period('M')

    if 'DOC_FECHA' in df.columns:
        df['hora'] = df['DOC_FECHA'].dt.hour.fillna(0).astype(int)
    else:
        df['hora'] = 0

    filas_limpias = df.shape[0]
    st_ref.write(f"Datos limpios: {filas_limpias:,} filas")

    # --- Agregaciones ---
    st_ref.write("Generando tablas agregadas...")

    # Agregado mensual
    df_mensual = df.groupby(df['FECHA_NEGOCIO'].dt.to_period('M')).agg(
        y=('D_VALOR', 'sum')
    ).reset_index()
    df_mensual.columns = ['ds', 'y']
    df_mensual['ds'] = df_mensual['ds'].dt.to_timestamp()
    df_mensual = df_mensual[df_mensual['y'] > 0].reset_index(drop=True)

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
    n_meses = len(df_mensual)
    st_ref.success(f"ETL completado. Rango: {rango}. Registros limpios: {filas_limpias:,}. Meses: {n_meses}")

    return df, df_mensual, df_diario, df_producto, df_familia
