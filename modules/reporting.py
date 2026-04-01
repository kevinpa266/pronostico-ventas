import pandas as pd
import numpy as np
import io
import tempfile
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI, funciona en servidores
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _plot_pronostico(df_mensual_hist, df_pronostico, tmp_dir):
    """Genera gráfico de pronóstico vs histórico con matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)

    # Histórico
    ax.plot(df_mensual_hist['ds'], df_mensual_hist['y'], 'o-',
            color='#2E86AB', linewidth=2, markersize=4, label='Ventas Históricas')

    # Pronóstico
    ax.plot(df_pronostico['ds'], df_pronostico['yhat'], 's--',
            color='#E8630A', linewidth=2, markersize=5, label='Pronóstico')

    # Intervalo de confianza
    if 'yhat_lower' in df_pronostico.columns and 'yhat_upper' in df_pronostico.columns:
        ax.fill_between(df_pronostico['ds'],
                        df_pronostico['yhat_lower'], df_pronostico['yhat_upper'],
                        alpha=0.2, color='#E8630A', label='Intervalo de Confianza')

    ax.set_title('Pronóstico de Ventas vs. Histórico', fontsize=13, fontweight='bold', color='#333')
    ax.set_xlabel('Mes', fontsize=10)
    ax.set_ylabel('Ventas ($)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    path = os.path.join(tmp_dir, 'pronostico.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_top_productos(df_limpios, top_n, tmp_dir):
    """Genera gráfico de top productos con matplotlib."""
    df_top = df_limpios.groupby('D_ITEM')['D_VALOR'].sum().reset_index()
    df_top = df_top.sort_values('D_VALOR', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(df_top)))
    ax.barh(df_top['D_ITEM'].astype(str), df_top['D_VALOR'], color=colors)
    ax.set_title(f'Top {top_n} Productos por Ventas', fontsize=13, fontweight='bold', color='#333')
    ax.set_xlabel('Ventas Totales ($)', fontsize=10)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(tmp_dir, 'top_productos.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_patron_horario(df_limpios, tmp_dir):
    """Genera gráfico de patrón horario con matplotlib."""
    df_horario = df_limpios.groupby('hora')['D_VALOR'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    colors = plt.cm.GnBu(np.linspace(0.3, 0.9, len(df_horario)))
    ax.bar(df_horario['hora'], df_horario['D_VALOR'], color=colors)
    ax.set_title('Distribución de Ventas por Hora del Día', fontsize=13, fontweight='bold', color='#333')
    ax.set_xlabel('Hora', fontsize=10)
    ax.set_ylabel('Ventas Totales ($)', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(tmp_dir, 'patron_horario.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_yoy(df_limpios, tmp_dir):
    """Genera gráfico año sobre año con matplotlib."""
    df_yoy = df_limpios.copy()
    df_yoy['anio'] = df_yoy['FECHA_NEGOCIO'].dt.year
    df_yoy['mes'] = df_yoy['FECHA_NEGOCIO'].dt.month
    df_yoy_agg = df_yoy.groupby(['anio', 'mes'])['D_VALOR'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    colors = ['#2E86AB', '#E8630A', '#A23B72', '#F18F01', '#155724']
    for i, (anio, grupo) in enumerate(df_yoy_agg.groupby('anio')):
        color = colors[i % len(colors)]
        ax.plot(grupo['mes'], grupo['D_VALOR'], 'o-', color=color,
                linewidth=2, markersize=5, label=str(anio))

    ax.set_title('Comparación de Ventas: Año sobre Año', fontsize=13, fontweight='bold', color='#333')
    ax.set_xlabel('Mes', fontsize=10)
    ax.set_ylabel('Ventas ($)', fontsize=10)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                        'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(tmp_dir, 'yoy.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_recommendations(df_pronostico, df_comparacion, df_limpios, meta_ventas,
                             alerta_cumplimiento, top_n, patron_horario_df):
    """Genera recomendaciones basadas en reglas de negocio."""
    recomendaciones = []

    # --- Regla 1: Pronóstico vs Meta ---
    pronostico_prox = df_pronostico['yhat'].iloc[0]
    cumplimiento = (pronostico_prox / meta_ventas) * 100

    if cumplimiento < alerta_cumplimiento:
        recomendaciones.append({
            'tipo': 'Alerta',
            'titulo': 'Pronostico por Debajo de la Meta',
            'texto': f'El pronostico del proximo mes (${pronostico_prox:,.2f}) representa solo el '
                     f'{cumplimiento:.1f}% de la meta (${meta_ventas:,.2f}). Se recomienda implementar '
                     f'estrategias de impulso de ventas como promociones o combos.'
        })
    else:
        recomendaciones.append({
            'tipo': 'Positivo',
            'titulo': 'Pronostico Alineado con la Meta',
            'texto': f'El pronostico del proximo mes (${pronostico_prox:,.2f}) alcanza el '
                     f'{cumplimiento:.1f}% de la meta. Se recomienda mantener las estrategias actuales.'
        })

    # --- Regla 2: Tendencia ---
    pronostico_promedio = df_pronostico['yhat'].mean()
    historico_promedio = df_limpios.groupby(df_limpios['FECHA_NEGOCIO'].dt.to_period('M'))['D_VALOR'].sum().mean()

    if pronostico_promedio < historico_promedio * 0.95:
        recomendaciones.append({
            'tipo': 'Alerta',
            'titulo': 'Tendencia de Ventas a la Baja',
            'texto': f'El promedio pronosticado (${pronostico_promedio:,.2f}) es inferior al promedio '
                     f'historico (${historico_promedio:,.2f}). Se sugiere revisar factores como cambios '
                     f'en rutas aereas, competencia o estacionalidad.'
        })
    else:
        recomendaciones.append({
            'tipo': 'Positivo',
            'titulo': 'Tendencia de Ventas Estable o al Alza',
            'texto': f'El promedio pronosticado (${pronostico_promedio:,.2f}) se mantiene alineado o '
                     f'por encima del promedio historico (${historico_promedio:,.2f}).'
        })

    # --- Regla 3: Concentración de productos (Pareto) ---
    df_pareto = df_limpios.groupby('D_ITEM')['D_VALOR'].sum().reset_index()
    df_pareto = df_pareto.sort_values('D_VALOR', ascending=False)
    df_pareto['pct_acum'] = (df_pareto['D_VALOR'].cumsum() / df_pareto['D_VALOR'].sum()) * 100
    n_80 = df_pareto[df_pareto['pct_acum'] <= 80].shape[0]
    total_prod = len(df_pareto)
    top5 = ", ".join(df_pareto['D_ITEM'].head(5).tolist())

    recomendaciones.append({
        'tipo': 'Estrategico',
        'titulo': 'Concentracion de Productos (Pareto)',
        'texto': f'El 80% de las ventas se concentra en {n_80} de {total_prod} productos '
                 f'({(n_80/total_prod*100):.1f}%). Los 5 principales son: {top5}. '
                 f'Se recomienda asegurar el abastecimiento de estos productos clave.'
    })

    # --- Regla 4: Productos de baja rotación ---
    ventas_por_prod = df_limpios.groupby('D_ITEM')['D_VALOR'].sum()
    umbral_bajo = ventas_por_prod.quantile(0.1)
    n_baja_rotacion = (ventas_por_prod <= umbral_bajo).sum()

    recomendaciones.append({
        'tipo': 'Operativo',
        'titulo': 'Productos de Baja Rotacion',
        'texto': f'Se identificaron {n_baja_rotacion} productos con ventas por debajo del percentil 10 '
                 f'(menos de ${umbral_bajo:,.2f} en el periodo). Evaluar si conviene discontinuarlos '
                 f'para liberar espacio en inventario.'
    })

    # --- Regla 5: Patrón Horario ---
    if patron_horario_df is not None and len(patron_horario_df) > 0:
        horas_pico = patron_horario_df.nlargest(5, 'ventas_total')
        pct_pico = (horas_pico['ventas_total'].sum() / patron_horario_df['ventas_total'].sum() * 100)
        horas_lista = sorted(horas_pico['hora'].astype(int).tolist())
        horas_texto = ', '.join([f'{h}:00' for h in horas_lista])

        recomendaciones.append({
            'tipo': 'Operativo',
            'titulo': 'Optimizacion de Personal por Horario',
            'texto': f'El {pct_pico:.0f}% de las ventas se concentra en las horas: {horas_texto}. '
                     f'Se identifican dos picos principales (manana y tarde), coincidentes con los '
                     f'horarios de mayor flujo de vuelos. Reforzar la dotacion de personal en estas '
                     f'horas especificas.'
        })

    # --- Regla 6: Mejor modelo ---
    mejor = df_comparacion.loc[df_comparacion['MAE'].idxmin()]
    recomendaciones.append({
        'tipo': 'Estrategico',
        'titulo': 'Precision del Modelo Predictivo',
        'texto': f'El modelo {mejor["Modelo"]} obtuvo un MAPE de {mejor["MAPE"]:.2f}%, lo cual indica '
                 f'un margen de error aceptable para la planificacion. Se recomienda re-entrenar el '
                 f'modelo mensualmente con datos actualizados para mantener su precision.'
    })

    return recomendaciones


def generate_report(df_pronostico, df_comparacion, figs_eda, figs_modelos,
                    meta_ventas, alerta_cumplimiento, df_limpios, top_n, st_ref):
    """Genera el reporte ejecutivo en PDF usando fpdf2 y matplotlib para gráficos."""
    from fpdf import FPDF

    st_ref.write("Generando reporte ejecutivo...")

    # Obtener patrón horario
    patron_horario_df = None
    if 'patron_horario' in figs_eda:
        patron_horario_df = df_limpios.groupby('hora')['D_VALOR'].sum().reset_index()
        patron_horario_df.columns = ['hora', 'ventas_total']

    # Generar recomendaciones
    recomendaciones = generate_recommendations(
        df_pronostico, df_comparacion, df_limpios, meta_ventas,
        alerta_cumplimiento, top_n, patron_horario_df
    )

    # --- Generar gráficos con matplotlib (no necesita Kaleido ni Chrome) ---
    tmp_dir = tempfile.mkdtemp()
    chart_files = {}

    try:
        # Obtener datos históricos mensuales para el gráfico de pronóstico
        df_mensual_hist = df_limpios.groupby(df_limpios['FECHA_NEGOCIO'].dt.to_period('M')).agg(
            y=('D_VALOR', 'sum')
        ).reset_index()
        df_mensual_hist.columns = ['ds', 'y']
        df_mensual_hist['ds'] = df_mensual_hist['ds'].dt.to_timestamp()

        chart_files['pronostico'] = _plot_pronostico(df_mensual_hist, df_pronostico, tmp_dir)
        st_ref.write("  Gráfico de pronóstico generado.")

        chart_files['top_productos'] = _plot_top_productos(df_limpios, top_n, tmp_dir)
        st_ref.write("  Gráfico de productos generado.")

        chart_files['patron_horario'] = _plot_patron_horario(df_limpios, tmp_dir)
        st_ref.write("  Gráfico de patrón horario generado.")

        chart_files['yoy'] = _plot_yoy(df_limpios, tmp_dir)
        st_ref.write("  Gráfico año sobre año generado.")

    except Exception as e:
        st_ref.warning(f"Error al generar algunos gráficos: {e}")

    # KPIs
    pronostico_prox = df_pronostico['yhat'].iloc[0]
    cumplimiento = (pronostico_prox / meta_ventas) * 100
    promedio_hist = df_limpios.groupby(df_limpios['FECHA_NEGOCIO'].dt.to_period('M'))['D_VALOR'].sum().mean()

    fecha_gen = datetime.now().strftime('%Y-%m-%d %H:%M')
    periodo = f"{df_pronostico['ds'].iloc[0].strftime('%Y-%m')} a {df_pronostico['ds'].iloc[-1].strftime('%Y-%m')}"

    # --- Crear PDF con fpdf2 ---
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Encabezado
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 12, 'Reporte Ejecutivo de Ventas y Pronostico', new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f'Generado el: {fecha_gen}', new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.cell(0, 6, f'Periodo del Pronostico: {periodo}', new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_draw_color(46, 134, 171)
    pdf.set_line_width(1)
    pdf.line(10, pdf.get_y() + 3, 200, pdf.get_y() + 3)
    pdf.ln(10)

    # --- Resumen Ejecutivo (KPIs) ---
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Resumen Ejecutivo', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    kpi_data = [
        ('Pronostico Prox. Mes', f'${pronostico_prox:,.2f}'),
        ('Meta de Ventas', f'${meta_ventas:,.2f}'),
        ('Cumplimiento', f'{cumplimiento:.1f}%'),
        ('Promedio Hist. Mensual', f'${promedio_hist:,.2f}')
    ]

    col_width = 45
    start_x = 10
    for i, (label, value) in enumerate(kpi_data):
        x = start_x + i * col_width
        pdf.set_xy(x, pdf.get_y())
        pdf.set_fill_color(249, 249, 249)
        pdf.set_draw_color(241, 143, 1)
        pdf.rect(x, pdf.get_y(), col_width - 2, 18, style='DF')
        pdf.set_fill_color(241, 143, 1)
        pdf.rect(x, pdf.get_y(), 2, 18, style='F')
        pdf.set_xy(x + 4, pdf.get_y() + 2)
        pdf.set_font('Helvetica', '', 7)
        pdf.set_text_color(136, 136, 136)
        pdf.cell(col_width - 6, 4, label, new_x="LMARGIN", new_y="NEXT")
        pdf.set_xy(x + 4, pdf.get_y())
        pdf.set_font('Helvetica', 'B', 11)
        if 'Cumplimiento' in label:
            if cumplimiento >= alerta_cumplimiento:
                pdf.set_text_color(21, 87, 36)
            else:
                pdf.set_text_color(114, 28, 36)
        else:
            pdf.set_text_color(51, 51, 51)
        pdf.cell(col_width - 6, 6, value)

    pdf.ln(22)

    # --- Gráfico de Pronóstico ---
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Pronostico de Ventas vs. Historico', new_x="LMARGIN", new_y="NEXT")
    if 'pronostico' in chart_files:
        pdf.image(chart_files['pronostico'], x=10, w=190)
    pdf.ln(5)

    # --- Tabla de Comparación de Modelos ---
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Comparacion de Modelos', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(46, 134, 171)
    pdf.set_text_color(255, 255, 255)
    col_widths_models = [40, 50, 50, 50]
    headers_models = ['Modelo', 'MAE', 'RMSE', 'MAPE']
    for i, h in enumerate(headers_models):
        pdf.cell(col_widths_models[i], 8, h, border=1, fill=True, align='C')
    pdf.ln()

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(51, 51, 51)
    for _, row in df_comparacion.iterrows():
        pdf.cell(col_widths_models[0], 7, str(row['Modelo']), border=1, align='C')
        pdf.cell(col_widths_models[1], 7, f"${row['MAE']:,.2f}", border=1, align='C')
        pdf.cell(col_widths_models[2], 7, f"${row['RMSE']:,.2f}", border=1, align='C')
        pdf.cell(col_widths_models[3], 7, f"{row['MAPE']:.2f}%", border=1, align='C')
        pdf.ln()
    pdf.ln(5)

    # --- Interpretación de Métricas ---
    mejor_modelo = df_comparacion.loc[df_comparacion['MAE'].idxmin()]
    mejor_nombre = mejor_modelo['Modelo']
    mejor_mae = mejor_modelo['MAE']
    mejor_rmse = mejor_modelo['RMSE']
    mejor_mape = mejor_modelo['MAPE']

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 8, f'Modelo Seleccionado: {mejor_nombre}', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(51, 51, 51)

    # Interpretación del MAE
    pdf.multi_cell(0, 5,
        f'MAE (Error Absoluto Medio): ${mejor_mae:,.2f}. Esto significa que, en promedio, '
        f'el modelo se equivoca en ${mejor_mae:,.2f} por mes respecto al valor real de ventas. '
        f'Es decir, si el pronostico indica $500,000, las ventas reales podrian estar entre '
        f'${500000 - mejor_mae:,.2f} y ${500000 + mejor_mae:,.2f} aproximadamente.')
    pdf.ln(2)

    # Interpretación del MAPE
    if mejor_mape <= 10:
        calidad_mape = 'alta precision (excelente)'
    elif mejor_mape <= 20:
        calidad_mape = 'buena precision (aceptable para planificacion)'
    elif mejor_mape <= 30:
        calidad_mape = 'precision moderada (usar con precaucion)'
    else:
        calidad_mape = 'precision baja (se recomienda revisar los datos o el modelo)'

    pdf.multi_cell(0, 5,
        f'MAPE (Error Porcentual Absoluto Medio): {mejor_mape:.2f}%. Esto indica que el modelo '
        f'tiene un margen de error promedio del {mejor_mape:.2f}% en sus predicciones, lo cual '
        f'se considera {calidad_mape}.')
    pdf.ln(2)

    # Interpretación del Cumplimiento
    if cumplimiento < 70:
        texto_cumpl = (
            f'Cumplimiento Proyectado: {cumplimiento:.1f}%. El pronostico del proximo mes '
            f'(${pronostico_prox:,.2f}) esta significativamente por debajo de la meta '
            f'(${meta_ventas:,.2f}). Esto puede deberse a estacionalidad (meses de baja demanda '
            f'como febrero o septiembre), reduccion en el flujo de pasajeros, o a que la meta '
            f'necesita ajustarse segun el patron historico de ventas.')
    elif cumplimiento < 90:
        texto_cumpl = (
            f'Cumplimiento Proyectado: {cumplimiento:.1f}%. El pronostico del proximo mes '
            f'(${pronostico_prox:,.2f}) esta ligeramente por debajo de la meta '
            f'(${meta_ventas:,.2f}). Se recomienda implementar estrategias de impulso como '
            f'promociones o combos para cerrar la brecha.')
    elif cumplimiento <= 110:
        texto_cumpl = (
            f'Cumplimiento Proyectado: {cumplimiento:.1f}%. El pronostico del proximo mes '
            f'(${pronostico_prox:,.2f}) esta alineado con la meta (${meta_ventas:,.2f}). '
            f'Se recomienda mantener las estrategias actuales.')
    else:
        texto_cumpl = (
            f'Cumplimiento Proyectado: {cumplimiento:.1f}%. El pronostico del proximo mes '
            f'(${pronostico_prox:,.2f}) supera la meta (${meta_ventas:,.2f}). Esto es '
            f'consistente con meses de alta temporada (como julio). Considerar ajustar la meta '
            f'al alza o aprovechar el excedente para reforzar inventario.')

    pdf.multi_cell(0, 5, texto_cumpl)
    pdf.ln(2)

    # Interpretación de los intervalos de confianza
    if df_pronostico['yhat_lower'].notna().any():
        rango_inf = df_pronostico['yhat_lower'].iloc[0]
        rango_sup = df_pronostico['yhat_upper'].iloc[0]
        pdf.multi_cell(0, 5,
            f'Intervalo de Confianza (95%): Para el proximo mes, las ventas se estiman entre '
            f'${rango_inf:,.2f} (limite inferior) y ${rango_sup:,.2f} (limite superior). '
            f'Esto significa que existe un 95% de probabilidad de que las ventas reales '
            f'se encuentren dentro de este rango.')
        pdf.ln(2)

    pdf.ln(3)

    # --- Tabla de Pronóstico ---
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, f'Tabla de Pronostico a {len(df_pronostico)} Meses', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(46, 134, 171)
    pdf.set_text_color(255, 255, 255)
    col_widths_pron = [35, 50, 50, 50]
    headers_pron = ['Mes', 'Pronostico ($)', 'Limite Inferior', 'Limite Superior']
    for i, h in enumerate(headers_pron):
        pdf.cell(col_widths_pron[i], 8, h, border=1, fill=True, align='C')
    pdf.ln()

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(51, 51, 51)
    for _, row in df_pronostico.iterrows():
        lower = f"${row['yhat_lower']:,.2f}" if pd.notna(row.get('yhat_lower')) else "N/A"
        upper = f"${row['yhat_upper']:,.2f}" if pd.notna(row.get('yhat_upper')) else "N/A"
        pdf.cell(col_widths_pron[0], 7, row['ds'].strftime('%Y-%m'), border=1, align='C')
        pdf.cell(col_widths_pron[1], 7, f"${row['yhat']:,.2f}", border=1, align='C')
        pdf.cell(col_widths_pron[2], 7, lower, border=1, align='C')
        pdf.cell(col_widths_pron[3], 7, upper, border=1, align='C')
        pdf.ln()
    pdf.ln(5)

    # --- Gráfico Top Productos ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Analisis de Productos', new_x="LMARGIN", new_y="NEXT")
    if 'top_productos' in chart_files:
        pdf.image(chart_files['top_productos'], x=10, w=190)
    pdf.ln(5)

    # --- Gráfico Patrón Horario ---
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Analisis Operativo: Patron Horario', new_x="LMARGIN", new_y="NEXT")
    if 'patron_horario' in chart_files:
        pdf.image(chart_files['patron_horario'], x=10, w=190)
    pdf.ln(5)

    # --- Gráfico Año sobre Año ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Comparacion Ano sobre Ano', new_x="LMARGIN", new_y="NEXT")
    if 'yoy' in chart_files:
        pdf.image(chart_files['yoy'], x=10, w=190)
    pdf.ln(5)

    # --- Recomendaciones ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Recomendaciones', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    color_map = {
        'Positivo': (212, 237, 218),
        'Alerta': (248, 215, 218),
        'Estrategico': (204, 229, 255),
        'Operativo': (255, 243, 205)
    }
    text_color_map = {
        'Positivo': (21, 87, 36),
        'Alerta': (114, 28, 36),
        'Estrategico': (0, 64, 133),
        'Operativo': (133, 100, 4)
    }

    for rec in recomendaciones:
        tipo = rec['tipo']
        bg = color_map.get(tipo, (240, 240, 240))
        tc = text_color_map.get(tipo, (51, 51, 51))

        pdf.set_fill_color(*bg)
        pdf.set_text_color(*tc)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 8, f"  {rec['titulo']}", new_x="LMARGIN", new_y="NEXT", fill=True, border=1)

        pdf.set_fill_color(250, 250, 250)
        pdf.set_text_color(51, 51, 51)
        pdf.set_font('Helvetica', '', 9)
        pdf.multi_cell(0, 5, f"  {rec['texto']}", border=1, fill=True)
        pdf.ln(3)

    # Footer
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(153, 153, 153)
    pdf.cell(0, 5, 'Reporte generado automaticamente por el Sistema de Inteligencia de Negocios',
             align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, 'Los pronosticos son estimaciones basadas en datos historicos y estan sujetos a variabilidad.',
             align='C')

    # Exportar a buffer
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)

    # Limpiar archivos temporales
    for f in chart_files.values():
        try:
            os.remove(f)
        except Exception:
            pass
    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass

    st_ref.success("Reporte ejecutivo generado exitosamente.")
    return pdf_buffer
