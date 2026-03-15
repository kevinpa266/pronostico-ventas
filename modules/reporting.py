import pandas as pd
import numpy as np
import io
import base64
import tempfile
import os
from datetime import datetime


def fig_to_base64(fig, width=700, height=400):
    """Convierte un gráfico Plotly a imagen base64 para el PDF."""
    img_bytes = fig.to_image(format="png", width=width, height=height)
    return base64.b64encode(img_bytes).decode()


def fig_to_png_bytes(fig, width=700, height=400):
    """Convierte un gráfico Plotly a bytes PNG."""
    return fig.to_image(format="png", width=width, height=height)


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
    historico_promedio = df_limpios.groupby(df_limpios['FECHA_NEGOCIO'].dt.to_period('M'))['D_TOTAL'].sum().mean()

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
    df_pareto = df_limpios.groupby('D_ITEM')['D_TOTAL'].sum().reset_index()
    df_pareto = df_pareto.sort_values('D_TOTAL', ascending=False)
    df_pareto['pct_acum'] = (df_pareto['D_TOTAL'].cumsum() / df_pareto['D_TOTAL'].sum()) * 100
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
    ventas_por_prod = df_limpios.groupby('D_ITEM')['D_TOTAL'].sum()
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
    """Genera el reporte ejecutivo en PDF usando fpdf2."""
    from fpdf import FPDF

    st_ref.write("Generando reporte ejecutivo...")

    # Obtener patrón horario
    patron_horario_df = None
    if 'patron_horario' in figs_eda:
        patron_horario_df = df_limpios.groupby('hora')['D_TOTAL'].sum().reset_index()
        patron_horario_df.columns = ['hora', 'ventas_total']

    # Generar recomendaciones
    recomendaciones = generate_recommendations(
        df_pronostico, df_comparacion, df_limpios, meta_ventas,
        alerta_cumplimiento, top_n, patron_horario_df
    )

    # Guardar gráficos como archivos temporales PNG
    tmp_dir = tempfile.mkdtemp()
    chart_files = {}
    try:
        for name, fig in [('pronostico', figs_modelos.get('pronostico_vs_historico')),
                          ('top_productos', figs_eda.get('top_productos')),
                          ('patron_horario', figs_eda.get('patron_horario')),
                          ('yoy', figs_eda.get('yoy'))]:
            if fig is not None:
                path = os.path.join(tmp_dir, f'{name}.png')
                fig.write_image(path, width=750, height=400)
                chart_files[name] = path
    except Exception as e:
        st_ref.warning(f"No se pudieron generar imagenes para el PDF: {e}. "
                       f"Instala kaleido con: pip install kaleido")

    # KPIs
    pronostico_prox = df_pronostico['yhat'].iloc[0]
    cumplimiento = (pronostico_prox / meta_ventas) * 100
    promedio_hist = df_limpios.groupby(df_limpios['FECHA_NEGOCIO'].dt.to_period('M'))['D_TOTAL'].sum().mean()

    fecha_gen = datetime.now().strftime('%Y-%m-%d %H:%M')
    periodo = f"{df_pronostico['ds'].iloc[0].strftime('%Y-%m')} a {df_pronostico['ds'].iloc[-1].strftime('%Y-%m')}"

    # --- Crear PDF con fpdf2 ---
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Encabezado
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(46, 134, 171)  # #2E86AB
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
    pdf.set_text_color(162, 59, 114)  # #A23B72
    pdf.cell(0, 10, 'Resumen Ejecutivo', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # KPI boxes
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
        # Box background
        pdf.set_fill_color(249, 249, 249)
        pdf.set_draw_color(241, 143, 1)  # #F18F01
        pdf.rect(x, pdf.get_y(), col_width - 2, 18, style='DF')
        # Left border accent
        pdf.set_fill_color(241, 143, 1)
        pdf.rect(x, pdf.get_y(), 2, 18, style='F')
        # Label
        pdf.set_xy(x + 4, pdf.get_y() + 2)
        pdf.set_font('Helvetica', '', 7)
        pdf.set_text_color(136, 136, 136)
        pdf.cell(col_width - 6, 4, label, new_x="LMARGIN", new_y="NEXT")
        # Value
        pdf.set_xy(x + 4, pdf.get_y())
        pdf.set_font('Helvetica', 'B', 11)
        if 'Cumplimiento' in label:
            if cumplimiento >= alerta_cumplimiento:
                pdf.set_text_color(21, 87, 36)  # green
            else:
                pdf.set_text_color(114, 28, 36)  # red
        else:
            pdf.set_text_color(51, 51, 51)
        pdf.cell(col_width - 6, 6, value)

    pdf.ln(22)

    # --- Gráfico de Pronóstico ---
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Pronostico de Ventas vs. Historico', new_x="LMARGIN", new_y="NEXT")
    if 'pronostico' in chart_files:
        pdf.image(chart_files['pronostico'], x=15, w=180)
    pdf.ln(5)

    # --- Tabla de Comparación de Modelos ---
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Comparacion de Modelos', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Header
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(46, 134, 171)
    pdf.set_text_color(255, 255, 255)
    col_widths_models = [40, 50, 50, 50]
    headers_models = ['Modelo', 'MAE', 'RMSE', 'MAPE']
    for i, h in enumerate(headers_models):
        pdf.cell(col_widths_models[i], 8, h, border=1, fill=True, align='C')
    pdf.ln()

    # Rows
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(51, 51, 51)
    for _, row in df_comparacion.iterrows():
        pdf.cell(col_widths_models[0], 7, str(row['Modelo']), border=1, align='C')
        pdf.cell(col_widths_models[1], 7, f"${row['MAE']:,.2f}", border=1, align='C')
        pdf.cell(col_widths_models[2], 7, f"${row['RMSE']:,.2f}", border=1, align='C')
        pdf.cell(col_widths_models[3], 7, f"{row['MAPE']:.2f}%", border=1, align='C')
        pdf.ln()
    pdf.ln(5)

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
        lower = f"${row['yhat_lower']:,.2f}" if pd.notna(row['yhat_lower']) else "N/A"
        upper = f"${row['yhat_upper']:,.2f}" if pd.notna(row['yhat_upper']) else "N/A"
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
        pdf.image(chart_files['top_productos'], x=15, w=180)
    pdf.ln(5)

    # --- Gráfico Patrón Horario ---
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Analisis Operativo: Patron Horario', new_x="LMARGIN", new_y="NEXT")
    if 'patron_horario' in chart_files:
        pdf.image(chart_files['patron_horario'], x=15, w=180)
    pdf.ln(5)

    # --- Gráfico Año sobre Año ---
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Comparacion Ano sobre Ano', new_x="LMARGIN", new_y="NEXT")
    if 'yoy' in chart_files:
        pdf.image(chart_files['yoy'], x=15, w=180)
    pdf.ln(5)

    # --- Recomendaciones ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Recomendaciones', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    color_map = {
        'Positivo': (212, 237, 218),    # green
        'Alerta': (248, 215, 218),      # red
        'Estrategico': (204, 229, 255), # blue
        'Operativo': (255, 243, 205)    # yellow
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

        # Header
        pdf.set_fill_color(*bg)
        pdf.set_text_color(*tc)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 8, f"  {rec['titulo']}", new_x="LMARGIN", new_y="NEXT", fill=True, border=1)

        # Body
        pdf.set_fill_color(250, 250, 250)
        pdf.set_text_color(51, 51, 51)
        pdf.set_font('Helvetica', '', 9)
        pdf.multi_cell(0, 5, f"  {rec['texto']}", border=1, fill=True)
        pdf.ln(3)

    # Footer
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(153, 153, 153)
    pdf.cell(0, 5, 'Reporte generado automaticamente por el Sistema de Inteligencia de Negocios', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, 'Los pronosticos son estimaciones basadas en datos historicos y estan sujetos a variabilidad.', align='C')

    # Exportar a buffer
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)

    # Limpiar archivos temporales
    for f in chart_files.values():
        try:
            os.remove(f)
        except:
            pass
    try:
        os.rmdir(tmp_dir)
    except:
        pass

    st_ref.success("Reporte ejecutivo generado exitosamente.")

    return pdf_buffer
