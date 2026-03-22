import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def run_eda(df_limpios, df_mensual, df_diario, df_producto, df_familia, top_n, st_ref):
    """Ejecuta el análisis exploratorio y genera gráficos interactivos."""
    figs = {}
    dfs = {}

    st_ref.write("Generando análisis exploratorio...")

    # --- 1. Tendencia mensual ---
    fig_tendencia = go.Figure()
    fig_tendencia.add_trace(go.Scatter(
        x=df_mensual['ds'], y=df_mensual['y'],
        mode='lines+markers', name='Ventas Mensuales',
        line=dict(color='#2E86AB', width=2)
    ))
    # Media móvil 3 meses
    df_mensual['ma3'] = df_mensual['y'].rolling(window=3, min_periods=1).mean()
    fig_tendencia.add_trace(go.Scatter(
        x=df_mensual['ds'], y=df_mensual['ma3'],
        mode='lines', name='Media Móvil 3M',
        line=dict(color='#E8630A', width=2, dash='dash')
    ))
    fig_tendencia.update_layout(
        title='Tendencia de Ventas Mensuales',
        xaxis_title='Mes', yaxis_title='Ventas ($)',
        template='plotly_white', height=400
    )
    figs['tendencia_mensual'] = fig_tendencia
    st_ref.plotly_chart(fig_tendencia, width='stretch')

    # --- 2. Estacionalidad mensual ---
    df_estacional = df_limpios.groupby('mes')['D_VALOR'].mean().reset_index()
    df_estacional.columns = ['mes', 'promedio_ventas']
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    df_estacional['mes_nombre'] = df_estacional['mes'].apply(lambda x: meses_nombres[int(x)-1] if 1 <= x <= 12 else str(x))

    fig_estacional = px.bar(
        df_estacional, x='mes_nombre', y='promedio_ventas',
        title='Estacionalidad: Promedio de Ventas por Mes del Año',
        labels={'promedio_ventas': 'Promedio Ventas ($)', 'mes_nombre': 'Mes'},
        color='promedio_ventas', color_continuous_scale='Blues',
        template='plotly_white'
    )
    fig_estacional.update_layout(height=400)
    figs['estacionalidad'] = fig_estacional
    st_ref.plotly_chart(fig_estacional, width='stretch')

    # --- 3. Patrón horario ---
    df_horario = df_limpios.groupby('hora')['D_VALOR'].sum().reset_index()
    df_horario.columns = ['hora', 'ventas_total']

    fig_horario = px.bar(
        df_horario, x='hora', y='ventas_total',
        title='Distribución de Ventas por Hora del Día',
        labels={'ventas_total': 'Ventas Totales ($)', 'hora': 'Hora'},
        color='ventas_total', color_continuous_scale='Teal',
        template='plotly_white'
    )
    fig_horario.update_layout(height=400)
    figs['patron_horario'] = fig_horario
    dfs['patron_horario'] = df_horario

    # --- 4. Top N productos ---
    df_top = df_limpios.groupby('D_ITEM')['D_VALOR'].sum().reset_index()
    df_top.columns = ['D_ITEM', 'ventas_total']
    df_top = df_top.sort_values('ventas_total', ascending=False).head(top_n)

    fig_top = px.bar(
        df_top, x='ventas_total', y='D_ITEM',
        orientation='h',
        title=f'Top {top_n} Productos por Ventas',
        labels={'ventas_total': 'Ventas Totales ($)', 'D_ITEM': 'Producto'},
        color='ventas_total', color_continuous_scale='Oranges',
        template='plotly_white'
    )
    fig_top.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    figs['top_productos'] = fig_top
    dfs['top_productos'] = df_top

    # --- 5. Pareto ---
    df_pareto = df_limpios.groupby('D_ITEM')['D_VALOR'].sum().reset_index()
    df_pareto.columns = ['D_ITEM', 'ventas_total']
    df_pareto = df_pareto.sort_values('ventas_total', ascending=False)
    df_pareto['pct_acumulado'] = (df_pareto['ventas_total'].cumsum() / df_pareto['ventas_total'].sum()) * 100
    df_pareto['ranking'] = range(1, len(df_pareto) + 1)
    n_80 = df_pareto[df_pareto['pct_acumulado'] <= 80].shape[0]
    dfs['pareto'] = df_pareto
    dfs['n_pareto_80'] = n_80
    dfs['total_productos'] = len(df_pareto)

    # --- 6. Comparación año sobre año ---
    df_yoy = df_limpios.copy()
    df_yoy['anio'] = df_yoy['FECHA_NEGOCIO'].dt.year
    df_yoy['mes'] = df_yoy['FECHA_NEGOCIO'].dt.month
    df_yoy_agg = df_yoy.groupby(['anio', 'mes'])['D_VALOR'].sum().reset_index()

    fig_yoy = px.line(
        df_yoy_agg, x='mes', y='D_VALOR', color='anio',
        title='Comparación de Ventas: Año sobre Año',
        labels={'D_VALOR': 'Ventas ($)', 'mes': 'Mes', 'anio': 'Año'},
        template='plotly_white', markers=True
    )
    fig_yoy.update_layout(height=400)
    figs['yoy'] = fig_yoy

    # --- 7. Ventas por familia ---
    df_fam = df_limpios.groupby('D_FAMILY')['D_VALOR'].sum().reset_index()
    df_fam.columns = ['D_FAMILY', 'ventas_total']
    df_fam = df_fam.sort_values('ventas_total', ascending=False).head(10)

    fig_fam = px.pie(
        df_fam, values='ventas_total', names='D_FAMILY',
        title='Distribución de Ventas por Familia',
        template='plotly_white', hole=0.3
    )
    fig_fam.update_layout(height=400)
    figs['familias'] = fig_fam

    # --- 8. Patrón por día de la semana ---
    dias_nombres = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    df_dia_sem = df_limpios.groupby('dia_semana')['D_VALOR'].mean().reset_index()
    df_dia_sem.columns = ['dia_semana', 'promedio_ventas']
    df_dia_sem['dia_nombre'] = df_dia_sem['dia_semana'].apply(lambda x: dias_nombres[int(x)] if 0 <= x <= 6 else str(x))

    fig_dia_sem = px.bar(
        df_dia_sem, x='dia_nombre', y='promedio_ventas',
        title='Promedio de Ventas por Día de la Semana',
        labels={'promedio_ventas': 'Promedio Ventas ($)', 'dia_nombre': 'Día'},
        color='promedio_ventas', color_continuous_scale='Purples',
        template='plotly_white'
    )
    fig_dia_sem.update_layout(height=400)
    figs['dia_semana'] = fig_dia_sem

    st_ref.success("Análisis exploratorio completado.")

    return figs, dfs
