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
                             alerta_cumplimiento, top_n, patron_horario_df, best_model_name=None):
    """Genera recomendaciones basadas en reglas de negocio."""
    recomendaciones = []

    # --- Regla 1: Pronóstico vs Meta ---
    pronostico_prox = df_pronostico['yhat'].iloc[0]
    cumplimiento = (pronostico_prox / meta_ventas) * 100

    if cumplimiento < alerta_cumplimiento:
        recomendaciones.append({
            'tipo': 'Alerta',
            'titulo': 'Pronóstico por Debajo de la Meta',
            'texto': f'El pronóstico del próximo mes (${pronostico_prox:,.2f}) representa solo el '
                     f'{cumplimiento:.1f}% de la meta (${meta_ventas:,.2f}). Se recomienda implementar '
                     f'estrategias de impulso de ventas como promociones o combos.'
        })
    else:
        recomendaciones.append({
            'tipo': 'Positivo',
            'titulo': 'Pronóstico Alineado con la Meta',
            'texto': f'El pronóstico del próximo mes (${pronostico_prox:,.2f}) alcanza el '
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
                     f'histórico (${historico_promedio:,.2f}). Se sugiere revisar factores como cambios '
                     f'en rutas aéreas, competencia o estacionalidad.'
        })
    else:
        recomendaciones.append({
            'tipo': 'Positivo',
            'titulo': 'Tendencia de Ventas Estable o al Alza',
            'texto': f'El promedio pronosticado (${pronostico_promedio:,.2f}) se mantiene alineado o '
                     f'por encima del promedio histórico (${historico_promedio:,.2f}).'
        })

    # --- Regla 3: Concentración de productos (Pareto) ---
    df_pareto = df_limpios.groupby('D_ITEM')['D_VALOR'].sum().reset_index()
    df_pareto = df_pareto.sort_values('D_VALOR', ascending=False)
    df_pareto['pct_acum'] = (df_pareto['D_VALOR'].cumsum() / df_pareto['D_VALOR'].sum()) * 100
    n_80 = df_pareto[df_pareto['pct_acum'] <= 80].shape[0]
    total_prod = len(df_pareto)
    top5 = ", ".join(df_pareto['D_ITEM'].head(5).tolist())

    recomendaciones.append({
        'tipo': 'Estratégico',
        'titulo': 'Concentración de Productos (Pareto)',
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
        'titulo': 'Productos de Baja Rotación',
        'texto': f'Se identificaron {n_baja_rotacion} productos con ventas por debajo del percentil 10 '
                 f'(menos de ${umbral_bajo:,.2f} en el período). Evaluar si conviene discontinuarlos '
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
            'titulo': 'Optimización de Personal por Horario',
            'texto': f'El {pct_pico:.0f}% de las ventas se concentra en las horas: {horas_texto}. '
                     f'Se identifican dos picos principales (mañana y tarde), coincidentes con los '
                     f'horarios de mayor flujo de vuelos. Reforzar la dotación de personal en estas '
                     f'horas específicas.'
        })

    # --- Regla 6: Mejor modelo ---
    # FIX: Usar best_model_name del criterio de parsimonia en vez de buscar el menor MAE
    if best_model_name and best_model_name in df_comparacion['Modelo'].values:
        mejor = df_comparacion[df_comparacion['Modelo'] == best_model_name].iloc[0]
    else:
        mejor = df_comparacion.loc[df_comparacion['MAE'].idxmin()]

    recomendaciones.append({
        'tipo': 'Estratégico',
        'titulo': 'Precisión del Modelo Predictivo',
        'texto': f'El modelo {mejor["Modelo"]} obtuvo un MAPE de {mejor["MAPE"]:.2f}%, lo cual indica '
                 f'un margen de error aceptable para la planificación. Se recomienda re-entrenar el '
                 f'modelo mensualmente con datos actualizados para mantener su precisión.'
    })

    return recomendaciones


def _setup_pdf_fonts(pdf):
    """Configura fuentes Unicode (DejaVu Sans) para el PDF con soporte de tildes y ñ."""
    # Buscar las fuentes en el directorio fonts/ del proyecto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fonts_dir = os.path.join(base_dir, 'fonts')

    # Si no están en el proyecto, buscar en matplotlib
    if not os.path.exists(os.path.join(fonts_dir, 'DejaVuSans.ttf')):
        try:
            import matplotlib
            mpl_fonts = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
            fonts_dir = mpl_fonts
        except Exception:
            return False

    regular = os.path.join(fonts_dir, 'DejaVuSans.ttf')
    bold = os.path.join(fonts_dir, 'DejaVuSans-Bold.ttf')
    italic = os.path.join(fonts_dir, 'DejaVuSans-Oblique.ttf')

    if os.path.exists(regular):
        pdf.add_font('DejaVu', '', regular, uni=True)
    else:
        return False

    if os.path.exists(bold):
        pdf.add_font('DejaVu', 'B', bold, uni=True)

    if os.path.exists(italic):
        pdf.add_font('DejaVu', 'I', italic, uni=True)

    return True


class ReportPDF:
    """Clase helper para crear un FPDF con footer nativo que no se superpone."""

    @staticmethod
    def create(FONT):
        """Crea una instancia de FPDF con footer automático."""
        from fpdf import FPDF

        class _PDF(FPDF):
            def footer(self):
                self.set_y(-12)
                self.set_font(FONT, 'I', 6)
                self.set_text_color(190, 190, 190)
                self.cell(95, 3,
                          'Reporte generado por el Sistema de Inteligencia de Negocios. '
                          'Los pronósticos son estimaciones sujetas a variabilidad.',
                          align='L')
                self.cell(95, 3, f'Página {self.page_no()} de {{nb}}', align='R')

        pdf = _PDF()
        pdf.alias_nb_pages()
        return pdf


def generate_report(df_pronostico, df_comparacion, figs_eda, figs_modelos,
                    meta_ventas, alerta_cumplimiento, df_limpios, top_n, st_ref,
                    best_model_name=None):
    """Genera el reporte ejecutivo en PDF usando fpdf2 y matplotlib para gráficos."""
    st_ref.write("Generando reporte ejecutivo...")

    # Obtener patrón horario
    patron_horario_df = None
    if 'patron_horario' in figs_eda:
        patron_horario_df = df_limpios.groupby('hora')['D_VALOR'].sum().reset_index()
        patron_horario_df.columns = ['hora', 'ventas_total']

    # Generar recomendaciones (ahora con best_model_name)
    recomendaciones = generate_recommendations(
        df_pronostico, df_comparacion, df_limpios, meta_ventas,
        alerta_cumplimiento, top_n, patron_horario_df, best_model_name
    )

    # --- Generar gráficos con matplotlib ---
    tmp_dir = tempfile.mkdtemp()
    chart_files = {}

    try:
        df_mensual_hist = df_limpios.groupby(
            df_limpios['FECHA_NEGOCIO'].dt.to_period('M')
        )['D_VALOR'].sum().reset_index()
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

    # --- Determinar el modelo seleccionado ---
    # FIX: Usar best_model_name del criterio de parsimonia
    if best_model_name and best_model_name in df_comparacion['Modelo'].values:
        mejor_modelo = df_comparacion[df_comparacion['Modelo'] == best_model_name].iloc[0]
    else:
        mejor_modelo = df_comparacion.loc[df_comparacion['MAE'].idxmin()]

    mejor_nombre = mejor_modelo['Modelo']
    mejor_mae = mejor_modelo['MAE']
    mejor_rmse = mejor_modelo['RMSE']
    mejor_mape = mejor_modelo['MAPE']

    # --- Crear PDF con fpdf2 (subclase con footer nativo) ---
    # Primero detectar fuente disponible
    from fpdf import FPDF as _TempFPDF
    _temp = _TempFPDF()
    has_unicode = _setup_pdf_fonts(_temp)
    FONT = 'DejaVu' if has_unicode else 'Helvetica'
    del _temp

    # Crear PDF con footer automático
    pdf = ReportPDF.create(FONT)
    pdf.set_auto_page_break(auto=True, margin=18)

    # Configurar fuente Unicode en la instancia real
    _setup_pdf_fonts(pdf)

    pdf.add_page()

    # ========== PÁGINA 1: Portada + KPIs + Pronóstico ==========

    # Encabezado
    pdf.set_font(FONT, 'B', 20)
    pdf.set_text_color(46, 134, 171)
    pdf.cell(0, 12, 'Reporte Ejecutivo de Ventas y Pronóstico', new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_font(FONT, '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f'Generado el: {fecha_gen}', new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.cell(0, 6, f'Período del Pronóstico: {periodo}', new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_draw_color(46, 134, 171)
    pdf.set_line_width(1)
    pdf.line(10, pdf.get_y() + 3, 200, pdf.get_y() + 3)
    pdf.ln(10)

    # --- Resumen Ejecutivo (KPIs) ---
    pdf.set_font(FONT, 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Resumen Ejecutivo', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # Explicación de la sección
    pdf.set_font(FONT, '', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4,
        'Esta sección presenta los indicadores clave de rendimiento (KPIs) del sistema de pronóstico. '
        'El "Pronóstico Próx. Mes" es la estimación de ventas generada por el modelo de inteligencia artificial. '
        'La "Meta de Ventas" es el objetivo comercial definido por la gerencia. El "Cumplimiento" indica '
        'qué porcentaje de la meta se espera alcanzar. El "Promedio Hist. Mensual" es el promedio de ventas '
        'de todos los meses del historial, y sirve como referencia de la tendencia general.')
    pdf.ln(3)

    kpi_data = [
        ('Pronóstico Próx. Mes', f'${pronostico_prox:,.2f}'),
        ('Meta de Ventas', f'${meta_ventas:,.2f}'),
        ('Cumplimiento', f'{cumplimiento:.1f}%'),
        ('Promedio Hist. Mensual', f'${promedio_hist:,.2f}')
    ]

    col_width = 45
    start_x = 10
    kpi_y = pdf.get_y()
    for i, (label, value) in enumerate(kpi_data):
        x = start_x + i * col_width
        pdf.set_xy(x, kpi_y)
        pdf.set_fill_color(249, 249, 249)
        pdf.set_draw_color(241, 143, 1)
        pdf.rect(x, kpi_y, col_width - 2, 18, style='DF')
        pdf.set_fill_color(241, 143, 1)
        pdf.rect(x, kpi_y, 2, 18, style='F')
        pdf.set_xy(x + 4, kpi_y + 2)
        pdf.set_font(FONT, '', 7)
        pdf.set_text_color(136, 136, 136)
        pdf.cell(col_width - 6, 4, label, new_x="LMARGIN", new_y="NEXT")
        pdf.set_xy(x + 4, kpi_y + 6)
        pdf.set_font(FONT, 'B', 11)
        if 'Cumplimiento' in label:
            if cumplimiento >= alerta_cumplimiento:
                pdf.set_text_color(21, 87, 36)
            else:
                pdf.set_text_color(114, 28, 36)
        else:
            pdf.set_text_color(51, 51, 51)
        pdf.cell(col_width - 6, 6, value)

    pdf.set_y(kpi_y + 22)

    # --- Gráfico de Pronóstico ---
    pdf.set_font(FONT, 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Pronóstico de Ventas vs. Histórico', new_x="LMARGIN", new_y="NEXT")

    # Explicación del gráfico
    pdf.set_font(FONT, '', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4,
        'El siguiente gráfico muestra la serie temporal de ventas históricas (línea azul) junto con el '
        'pronóstico generado por el modelo (línea naranja punteada). La zona sombreada representa el '
        'intervalo de confianza al 95%, es decir, el rango dentro del cual se espera que las ventas '
        'reales se ubiquen con un 95% de probabilidad.')
    pdf.ln(2)

    if 'pronostico' in chart_files:
        pdf.image(chart_files['pronostico'], x=10, w=190)
    pdf.ln(3)

    # ========== PÁGINA 2: Comparación de Modelos ==========
    pdf.add_page()

    # --- Tabla de Comparación de Modelos ---
    pdf.set_font(FONT, 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Comparación de Modelos', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # Explicación de la sección
    pdf.set_font(FONT, '', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4,
        'Se evaluaron múltiples modelos predictivos para determinar cuál ofrece el mejor rendimiento. '
        'Las métricas utilizadas son: MAE (Error Absoluto Medio), que indica en cuántos dólares se '
        'equivoca el modelo en promedio cada mes; RMSE (Raíz del Error Cuadrático Medio), similar al '
        'MAE pero penaliza más los errores grandes; y MAPE (Error Porcentual Absoluto Medio), que '
        'expresa el error como porcentaje. Un MAPE menor al 20% se considera un buen pronóstico.')
    pdf.ln(3)

    pdf.set_font(FONT, 'B', 9)
    pdf.set_fill_color(46, 134, 171)
    pdf.set_text_color(255, 255, 255)
    col_widths_models = [35, 35, 40, 40, 40]
    headers_models = ['Modelo', 'Tipo', 'MAE', 'RMSE', 'MAPE']
    for i, h in enumerate(headers_models):
        pdf.cell(col_widths_models[i], 8, h, border=1, fill=True, align='C')
    pdf.ln()

    pdf.set_font(FONT, '', 9)
    pdf.set_text_color(51, 51, 51)
    for _, row in df_comparacion.iterrows():
        # Resaltar la fila del modelo seleccionado
        if row['Modelo'] == mejor_nombre:
            pdf.set_fill_color(212, 237, 218)  # Verde claro
            fill = True
        else:
            pdf.set_fill_color(255, 255, 255)
            fill = True
        pdf.cell(col_widths_models[0], 7, str(row['Modelo']), border=1, align='C', fill=fill)
        pdf.cell(col_widths_models[1], 7, str(row.get('Tipo', '')), border=1, align='C', fill=fill)
        pdf.cell(col_widths_models[2], 7, f"${row['MAE']:,.2f}", border=1, align='C', fill=fill)
        pdf.cell(col_widths_models[3], 7, f"${row['RMSE']:,.2f}", border=1, align='C', fill=fill)
        pdf.cell(col_widths_models[4], 7, f"{row['MAPE']:.2f}%", border=1, align='C', fill=fill)
        pdf.ln()
    pdf.ln(5)

    # --- Interpretación de Métricas ---
    pdf.set_font(FONT, 'B', 11)
    pdf.set_text_color(21, 87, 36)
    pdf.cell(0, 8, f'Modelo Seleccionado: {mejor_nombre}', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    pdf.set_font(FONT, '', 9)
    pdf.set_text_color(51, 51, 51)

    # Explicación de por qué se seleccionó este modelo (criterio de parsimonia)
    # Verificar si hay un modelo con menor MAE que no sea el seleccionado
    modelo_menor_mae = df_comparacion.loc[df_comparacion['MAE'].idxmin()]
    if modelo_menor_mae['Modelo'] != mejor_nombre:
        diff_pct = abs(modelo_menor_mae['MAE'] - mejor_mae) / modelo_menor_mae['MAE'] * 100
        pdf.multi_cell(0, 5,
            f'Nota: El modelo {modelo_menor_mae["Modelo"]} obtuvo un MAE ligeramente menor '
            f'(${modelo_menor_mae["MAE"]:,.2f}), pero la diferencia con {mejor_nombre} es de solo '
            f'{diff_pct:.1f}%. Aplicando el principio de parsimonia (Navaja de Occam), cuando dos '
            f'modelos tienen rendimiento equivalente (diferencia menor al 5%), se selecciona el más '
            f'simple e interpretable. {mejor_nombre} ofrece mayor transparencia, menor tiempo de '
            f'entrenamiento y mejor reproducibilidad.')
        pdf.ln(2)

    # Interpretación del MAE
    pdf.multi_cell(0, 5,
        f'MAE (Error Absoluto Medio): ${mejor_mae:,.2f}. Esto significa que, en promedio, '
        f'el modelo se equivoca en ${mejor_mae:,.2f} por mes respecto al valor real de ventas. '
        f'Es decir, si el pronóstico indica $500,000, las ventas reales podrían estar entre '
        f'${500000 - mejor_mae:,.2f} y ${500000 + mejor_mae:,.2f} aproximadamente.')
    pdf.ln(2)

    # Interpretación del MAPE
    if mejor_mape <= 10:
        calidad_mape = 'alta precisión (excelente)'
    elif mejor_mape <= 20:
        calidad_mape = 'buena precisión (aceptable para planificación)'
    elif mejor_mape <= 30:
        calidad_mape = 'precisión moderada (usar con precaución)'
    else:
        calidad_mape = 'precisión baja (se recomienda revisar los datos o el modelo)'

    pdf.multi_cell(0, 5,
        f'MAPE (Error Porcentual Absoluto Medio): {mejor_mape:.2f}%. Esto indica que el modelo '
        f'tiene un margen de error promedio del {mejor_mape:.2f}% en sus predicciones, lo cual '
        f'se considera {calidad_mape}.')
    pdf.ln(2)

    # Interpretación del Cumplimiento
    if cumplimiento < 70:
        texto_cumpl = (
            f'Cumplimiento Proyectado: {cumplimiento:.1f}%. El pronóstico del próximo mes '
            f'(${pronostico_prox:,.2f}) está significativamente por debajo de la meta '
            f'(${meta_ventas:,.2f}). Esto puede deberse a estacionalidad (meses de baja demanda '
            f'como febrero o septiembre), reducción en el flujo de pasajeros, o a que la meta '
            f'necesita ajustarse según el patrón histórico de ventas.')
    elif cumplimiento < 90:
        texto_cumpl = (
            f'Cumplimiento Proyectado: {cumplimiento:.1f}%. El pronóstico del próximo mes '
            f'(${pronostico_prox:,.2f}) está ligeramente por debajo de la meta '
            f'(${meta_ventas:,.2f}). Se recomienda implementar estrategias de impulso como '
            f'promociones o combos para cerrar la brecha.')
    elif cumplimiento <= 110:
        texto_cumpl = (
            f'Cumplimiento Proyectado: {cumplimiento:.1f}%. El pronóstico del próximo mes '
            f'(${pronostico_prox:,.2f}) está alineado con la meta (${meta_ventas:,.2f}). '
            f'Se recomienda mantener las estrategias actuales.')
    else:
        texto_cumpl = (
            f'Cumplimiento Proyectado: {cumplimiento:.1f}%. El pronóstico del próximo mes '
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
            f'Intervalo de Confianza (95%): Para el próximo mes, las ventas se estiman entre '
            f'${rango_inf:,.2f} (límite inferior) y ${rango_sup:,.2f} (límite superior). '
            f'Esto significa que existe un 95% de probabilidad de que las ventas reales '
            f'se encuentren dentro de este rango.')
        pdf.ln(2)

    # --- Tabla de Pronóstico ---
    pdf.ln(3)
    pdf.set_font(FONT, 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, f'Tabla de Pronóstico a {len(df_pronostico)} Meses', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # Explicación de la tabla
    pdf.set_font(FONT, '', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4,
        'La siguiente tabla detalla el pronóstico mes a mes generado por el modelo seleccionado. '
        'La columna "Pronóstico ($)" es la estimación puntual de ventas. Las columnas "Límite Inferior" '
        'y "Límite Superior" definen el intervalo de confianza al 95%: existe un 95% de probabilidad '
        'de que las ventas reales se ubiquen dentro de este rango. A medida que el horizonte de '
        'pronóstico se extiende, el intervalo se amplía reflejando mayor incertidumbre.')
    pdf.ln(3)

    pdf.set_font(FONT, 'B', 9)
    pdf.set_fill_color(46, 134, 171)
    pdf.set_text_color(255, 255, 255)
    col_widths_pron = [35, 50, 50, 50]
    headers_pron = ['Mes', 'Pronóstico ($)', 'Límite Inferior', 'Límite Superior']
    for i, h in enumerate(headers_pron):
        pdf.cell(col_widths_pron[i], 8, h, border=1, fill=True, align='C')
    pdf.ln()

    pdf.set_font(FONT, '', 9)
    pdf.set_text_color(51, 51, 51)
    for _, row in df_pronostico.iterrows():
        lower = f"${row['yhat_lower']:,.2f}" if pd.notna(row.get('yhat_lower')) else "N/A"
        upper = f"${row['yhat_upper']:,.2f}" if pd.notna(row.get('yhat_upper')) else "N/A"
        pdf.cell(col_widths_pron[0], 7, row['ds'].strftime('%Y-%m'), border=1, align='C')
        pdf.cell(col_widths_pron[1], 7, f"${row['yhat']:,.2f}", border=1, align='C')
        pdf.cell(col_widths_pron[2], 7, lower, border=1, align='C')
        pdf.cell(col_widths_pron[3], 7, upper, border=1, align='C')
        pdf.ln()
    pdf.ln(3)

    # ========== PÁGINA 3: Análisis de Productos y Patrón Horario ==========
    pdf.add_page()
    pdf.set_font(FONT, 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Análisis de Productos', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # Explicación de la sección
    pdf.set_font(FONT, '', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4,
        'El siguiente gráfico muestra los productos con mayor volumen de ventas en el período analizado, '
        'ordenados de mayor a menor. Esta visualización aplica el principio de Pareto (regla 80/20): '
        'un pequeño porcentaje de productos genera la mayor parte de los ingresos. Identificar estos '
        'productos clave permite priorizar el abastecimiento y evitar quiebres de inventario en los '
        'artículos más rentables.')
    pdf.ln(3)

    if 'top_productos' in chart_files:
        pdf.image(chart_files['top_productos'], x=10, w=190)
    pdf.ln(5)

    # --- Gráfico Patrón Horario ---
    pdf.set_font(FONT, 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Análisis Operativo: Patrón Horario', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # Explicación de la sección
    pdf.set_font(FONT, '', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4,
        'Este gráfico muestra la distribución de ventas acumuladas por hora del día a lo largo de todo '
        'el período analizado. Las barras más altas indican las horas de mayor actividad comercial. '
        'Esta información es útil para optimizar la dotación de personal, programar reposición de '
        'productos y planificar promociones en los horarios de mayor tráfico de clientes.')
    pdf.ln(3)

    if 'patron_horario' in chart_files:
        pdf.image(chart_files['patron_horario'], x=10, w=190)
    pdf.ln(3)

    # ========== PÁGINA 4: Año sobre Año ==========
    pdf.add_page()
    pdf.set_font(FONT, 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Comparación Año sobre Año', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # Explicación de la sección
    pdf.set_font(FONT, '', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4,
        'Este gráfico compara las ventas mensuales de cada año del historial. Cada línea representa '
        'un año diferente, lo que permite identificar patrones estacionales recurrentes (por ejemplo, '
        'meses que siempre tienen ventas altas o bajas) y evaluar si la empresa está creciendo, '
        'estancándose o decreciendo año tras año. Los meses donde las líneas se separan significativamente '
        'indican cambios importantes en el comportamiento comercial.')
    pdf.ln(3)

    if 'yoy' in chart_files:
        pdf.image(chart_files['yoy'], x=10, w=190)
    pdf.ln(5)

    # ========== PÁGINA 5: Recomendaciones ==========
    pdf.add_page()
    pdf.set_font(FONT, 'B', 14)
    pdf.set_text_color(162, 59, 114)
    pdf.cell(0, 10, 'Recomendaciones', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # Explicación de la sección
    pdf.set_font(FONT, '', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4,
        'Las siguientes recomendaciones se generan automáticamente a partir del análisis de los datos '
        'y los resultados del modelo predictivo. Cada recomendación está clasificada por tipo: '
        '"Positivo" (indicadores favorables), "Alerta" (situaciones que requieren atención), '
        '"Estratégico" (decisiones de mediano/largo plazo) y "Operativo" (acciones inmediatas).')
    pdf.ln(3)

    color_map = {
        'Positivo': (212, 237, 218),
        'Alerta': (248, 215, 218),
        'Estratégico': (204, 229, 255),
        'Operativo': (255, 243, 205)
    }
    text_color_map = {
        'Positivo': (21, 87, 36),
        'Alerta': (114, 28, 36),
        'Estratégico': (0, 64, 133),
        'Operativo': (133, 100, 4)
    }

    for rec in recomendaciones:
        tipo = rec['tipo']
        bg = color_map.get(tipo, (240, 240, 240))
        tc = text_color_map.get(tipo, (51, 51, 51))

        pdf.set_fill_color(*bg)
        pdf.set_text_color(*tc)
        pdf.set_font(FONT, 'B', 10)
        pdf.cell(0, 8, f"  {rec['titulo']}", new_x="LMARGIN", new_y="NEXT", fill=True, border=1)

        pdf.set_fill_color(250, 250, 250)
        pdf.set_text_color(51, 51, 51)
        pdf.set_font(FONT, '', 9)
        pdf.multi_cell(0, 5, f"  {rec['texto']}", border=1, fill=True)
        pdf.ln(3)

    # El footer se genera automáticamente por la subclase _PDF

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
