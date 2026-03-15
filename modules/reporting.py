import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime


def fig_to_base64(fig, width=700, height=400):
    """Convierte un gráfico Plotly a imagen base64 para el PDF."""
    img_bytes = fig.to_image(format="png", width=width, height=height)
    return base64.b64encode(img_bytes).decode()


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
    historico_promedio = df_limpios.groupby(df_limpios['FECHA_NEGOCIO'].dt.to_period('M'))['D_TOTAL'].sum().mean()

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
    df_pareto = df_limpios.groupby('D_ITEM')['D_TOTAL'].sum().reset_index()
    df_pareto = df_pareto.sort_values('D_TOTAL', ascending=False)
    df_pareto['pct_acum'] = (df_pareto['D_TOTAL'].cumsum() / df_pareto['D_TOTAL'].sum()) * 100
    n_80 = df_pareto[df_pareto['pct_acum'] <= 80].shape[0]
    total_prod = len(df_pareto)
    top5 = ", ".join(df_pareto['D_ITEM'].head(5).tolist())

    recomendaciones.append({
        'tipo': 'Estrategico',
        'titulo': 'Concentración de Productos (Pareto)',
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
    mejor = df_comparacion.loc[df_comparacion['MAE'].idxmin()]
    recomendaciones.append({
        'tipo': 'Estrategico',
        'titulo': 'Precisión del Modelo Predictivo',
        'texto': f'El modelo {mejor["Modelo"]} obtuvo un MAPE de {mejor["MAPE"]:.2f}%, lo cual indica '
                 f'un margen de error aceptable para la planificación. Se recomienda re-entrenar el '
                 f'modelo mensualmente con datos actualizados para mantener su precisión.'
    })

    return recomendaciones


def generate_report(df_pronostico, df_comparacion, figs_eda, figs_modelos,
                    meta_ventas, alerta_cumplimiento, df_limpios, top_n, st_ref):
    """Genera el reporte ejecutivo en HTML y lo convierte a PDF."""
    from weasyprint import HTML

    st_ref.write("Generando reporte ejecutivo...")

    # Obtener patrón horario
    patron_horario_df = None
    if 'patron_horario' in figs_eda:
        # Reconstruir desde los datos
        patron_horario_df = df_limpios.groupby('hora')['D_TOTAL'].sum().reset_index()
        patron_horario_df.columns = ['hora', 'ventas_total']

    # Generar recomendaciones
    recomendaciones = generate_recommendations(
        df_pronostico, df_comparacion, df_limpios, meta_ventas,
        alerta_cumplimiento, top_n, patron_horario_df
    )

    # Convertir gráficos a base64
    try:
        img_pronostico = fig_to_base64(figs_modelos['pronostico_vs_historico'], 750, 400)
        img_top = fig_to_base64(figs_eda['top_productos'], 750, 450)
        img_horario = fig_to_base64(figs_eda['patron_horario'], 750, 350)
        img_yoy = fig_to_base64(figs_eda['yoy'], 750, 350)
    except Exception as e:
        st_ref.warning(f"No se pudieron generar imágenes para el PDF: {e}. "
                       f"Instala kaleido con: pip install kaleido")
        img_pronostico = img_top = img_horario = img_yoy = ""

    # KPIs
    pronostico_prox = df_pronostico['yhat'].iloc[0]
    cumplimiento = (pronostico_prox / meta_ventas) * 100
    promedio_hist = df_limpios.groupby(df_limpios['FECHA_NEGOCIO'].dt.to_period('M'))['D_TOTAL'].sum().mean()

    # Tabla de pronóstico
    tabla_pronostico = ""
    for _, row in df_pronostico.iterrows():
        lower = f"${row['yhat_lower']:,.2f}" if pd.notna(row['yhat_lower']) else "N/A"
        upper = f"${row['yhat_upper']:,.2f}" if pd.notna(row['yhat_upper']) else "N/A"
        tabla_pronostico += f"""
        <tr>
            <td>{row['ds'].strftime('%Y-%m')}</td>
            <td>${row['yhat']:,.2f}</td>
            <td>{lower}</td>
            <td>{upper}</td>
        </tr>"""

    # Tabla de comparación de modelos
    tabla_modelos = ""
    for _, row in df_comparacion.iterrows():
        tabla_modelos += f"""
        <tr>
            <td>{row['Modelo']}</td>
            <td>${row['MAE']:,.2f}</td>
            <td>${row['RMSE']:,.2f}</td>
            <td>{row['MAPE']:.2f}%</td>
        </tr>"""

    # Recomendaciones HTML
    recs_html = ""
    for rec in recomendaciones:
        color_class = rec['tipo']
        recs_html += f"""
        <div class="rec-card rec-{color_class}">
            <div class="rec-header">{rec['titulo']}</div>
            <div class="rec-body">{rec['texto']}</div>
        </div>"""

    # Color del cumplimiento
    color_cumpl = '#155724' if cumplimiento >= alerta_cumplimiento else '#721c24'

    fecha_gen = datetime.now().strftime('%Y-%m-%d %H:%M')
    periodo = f"{df_pronostico['ds'].iloc[0].strftime('%Y-%m')} a {df_pronostico['ds'].iloc[-1].strftime('%Y-%m')}"

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Reporte Ejecutivo de Ventas</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        * {{ box-sizing: border-box; }}
        body {{ font-family: 'Roboto', sans-serif; margin: 0; padding: 0;
               background-color: #f4f4f9; color: #333; font-size: 0.9em; }}
        .container {{ max-width: 800px; margin: 20px auto; padding: 30px;
                     background-color: #fff; border-radius: 8px;
                     box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 3px solid #2E86AB;
                  padding-bottom: 15px; margin-bottom: 25px; }}
        .header h1 {{ color: #2E86AB; margin: 0; font-size: 1.8em; }}
        .header p {{ margin: 5px 0; color: #666; font-size: 0.95em; }}
        h2 {{ color: #A23B72; border-bottom: 1px solid #eee; padding-bottom: 8px;
             margin-top: 35px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr);
                    gap: 15px; margin: 20px 0; }}
        .kpi-box {{ background: linear-gradient(135deg, #f9f9f9, #fff);
                   border-left: 5px solid #F18F01; padding: 15px;
                   border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
        .kpi-box h3 {{ margin: 0 0 5px 0; font-size: 0.85em; color: #888;
                      text-transform: uppercase; }}
        .kpi-box .value {{ font-size: 1.4em; font-weight: 700; color: #333; }}
        .chart-section {{ text-align: center; margin: 25px 0; }}
        .chart-section img {{ max-width: 85%; border-radius: 5px;
                             box-shadow: 0 1px 5px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0;
                font-size: 0.95em; }}
        th {{ background-color: #2E86AB; color: white; padding: 10px;
             text-align: left; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .rec-card {{ border: 1px solid #ddd; border-radius: 8px;
                    margin-bottom: 12px; overflow: hidden; }}
        .rec-header {{ padding: 10px 15px; font-weight: 700; font-size: 0.95em; }}
        .rec-body {{ padding: 12px 15px; background-color: #fafafa; line-height: 1.5; }}
        .rec-Positivo .rec-header {{ background-color: #d4edda; color: #155724; }}
        .rec-Alerta .rec-header {{ background-color: #f8d7da; color: #721c24; }}
        .rec-Estrategico .rec-header {{ background-color: #cce5ff; color: #004085; }}
        .rec-Operativo .rec-header {{ background-color: #fff3cd; color: #856404; }}
        .footer {{ text-align: center; margin-top: 30px; padding-top: 15px;
                  border-top: 1px solid #eee; color: #999; font-size: 0.85em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Reporte Ejecutivo de Ventas y Pronóstico</h1>
            <p>Generado el: {fecha_gen}</p>
            <p>Período del Pronóstico: {periodo}</p>
        </div>

        <h2>Resumen Ejecutivo</h2>
        <div class="kpi-grid">
            <div class="kpi-box">
                <h3>Pronóstico Próx. Mes</h3>
                <div class="value">${pronostico_prox:,.2f}</div>
            </div>
            <div class="kpi-box">
                <h3>Meta de Ventas</h3>
                <div class="value">${meta_ventas:,.2f}</div>
            </div>
            <div class="kpi-box">
                <h3>Cumplimiento</h3>
                <div class="value" style="color: {color_cumpl}">{cumplimiento:.1f}%</div>
            </div>
            <div class="kpi-box">
                <h3>Promedio Hist. Mensual</h3>
                <div class="value">${promedio_hist:,.2f}</div>
            </div>
        </div>

        <h2>Pronóstico de Ventas vs. Histórico</h2>
        <div class="chart-section">
            <img src="data:image/png;base64,{img_pronostico}" alt="Pronóstico vs Histórico">
        </div>

        <h2>Comparación de Modelos</h2>
        <table>
            <tr><th>Modelo</th><th>MAE</th><th>RMSE</th><th>MAPE</th></tr>
            {tabla_modelos}
        </table>

        <h2>Tabla de Pronóstico a {len(df_pronostico)} Meses</h2>
        <table>
            <tr><th>Mes</th><th>Pronóstico ($)</th><th>Límite Inferior</th><th>Límite Superior</th></tr>
            {tabla_pronostico}
        </table>

        <h2>Análisis de Productos</h2>
        <div class="chart-section">
            <img src="data:image/png;base64,{img_top}" alt="Top Productos">
        </div>

        <h2>Análisis Operativo: Patrón Horario</h2>
        <div class="chart-section">
            <img src="data:image/png;base64,{img_horario}" alt="Patrón Horario">
        </div>

        <h2>Comparación Año sobre Año</h2>
        <div class="chart-section">
            <img src="data:image/png;base64,{img_yoy}" alt="Año sobre Año">
        </div>

        <h2>Recomendaciones</h2>
        {recs_html}

        <div class="footer">
            <p>Reporte generado automáticamente por el Sistema de Inteligencia de Negocios</p>
            <p>Los pronósticos son estimaciones basadas en datos históricos y están sujetos a variabilidad.</p>
        </div>
    </div>
</body>
</html>"""

    # Generar PDF
    pdf_buffer = io.BytesIO()
    HTML(string=html_content).write_pdf(pdf_buffer)
    pdf_buffer.seek(0)

    st_ref.success("Reporte ejecutivo generado exitosamente.")

    return pdf_buffer
