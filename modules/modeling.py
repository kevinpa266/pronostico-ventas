import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def run_modeling(df_mensual, horizonte, st_ref):
    """Ejecuta los 3 modelos predictivos y compara resultados."""
    figs = {}

    # --- Preparar datos ---
    df = df_mensual[['ds', 'y']].copy().sort_values('ds').reset_index(drop=True)

    # División train/test (últimos 12 meses como test)
    n_test = min(12, len(df) // 3)
    train = df.iloc[:-n_test].copy()
    test = df.iloc[-n_test:].copy()

    st_ref.write(f"Datos de entrenamiento: {len(train)} meses | Datos de prueba: {len(test)} meses")

    results = {}

    # =============================================
    # MODELO 1: BASELINE (Seasonal Naive)
    # =============================================
    st_ref.write("Entrenando Modelo 1: Baseline (Seasonal Naive)...")
    try:
        baseline_preds = []
        for i in range(len(test)):
            idx = len(train) - 12 + (i % 12)
            if idx >= 0 and idx < len(train):
                baseline_preds.append(train['y'].iloc[idx])
            else:
                baseline_preds.append(train['y'].mean())

        results['Baseline'] = {
            'y_true': test['y'].values,
            'y_pred': np.array(baseline_preds),
            'has_ci': False
        }
        st_ref.write("Baseline completado.")
    except Exception as e:
        st_ref.warning(f"Error en Baseline: {e}")

    # =============================================
    # MODELO 2: PROPHET
    # =============================================
    st_ref.write("Entrenando Modelo 2: Prophet (con feriados de Ecuador)...")
    try:
        from prophet import Prophet

        # Feriados de Ecuador
        holidays_ec = pd.DataFrame({
            'holiday': ['Anio_Nuevo', 'Carnaval', 'Carnaval', 'Viernes_Santo',
                        'Dia_Trabajo', 'Batalla_Pichincha', 'Independencia_Quito',
                        'Independencia_Guayaquil', 'Dia_Difuntos', 'Navidad'] * 5,
            'ds': pd.to_datetime([
                '2021-01-01', '2021-02-15', '2021-02-16', '2021-04-02',
                '2021-05-01', '2021-05-24', '2021-08-10',
                '2021-10-09', '2021-11-02', '2021-12-25',
                '2022-01-01', '2022-02-28', '2022-03-01', '2022-04-15',
                '2022-05-01', '2022-05-24', '2022-08-10',
                '2022-10-09', '2022-11-02', '2022-12-25',
                '2023-01-01', '2023-02-20', '2023-02-21', '2023-04-07',
                '2023-05-01', '2023-05-24', '2023-08-10',
                '2023-10-09', '2023-11-02', '2023-12-25',
                '2024-01-01', '2024-02-12', '2024-02-13', '2024-03-29',
                '2024-05-01', '2024-05-24', '2024-08-10',
                '2024-10-09', '2024-11-02', '2024-12-25',
                '2025-01-01', '2025-03-03', '2025-03-04', '2025-04-18',
                '2025-05-01', '2025-05-24', '2025-08-10',
                '2025-10-09', '2025-11-02', '2025-12-25',
            ])
        })

        model_prophet = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays_ec,
            changepoint_prior_scale=0.1,
            seasonality_mode='multiplicative'
        )
        model_prophet.fit(train)

        future_test = model_prophet.make_future_dataframe(periods=n_test, freq='MS')
        forecast_test = model_prophet.predict(future_test)
        prophet_preds = forecast_test.tail(n_test)

        results['Prophet'] = {
            'y_true': test['y'].values,
            'y_pred': prophet_preds['yhat'].values,
            'y_lower': prophet_preds['yhat_lower'].values,
            'y_upper': prophet_preds['yhat_upper'].values,
            'has_ci': True
        }
        st_ref.write("Prophet completado.")
    except Exception as e:
        st_ref.warning(f"Error en Prophet: {e}")

    # =============================================
    # MODELO 3: SARIMA
    # =============================================
    st_ref.write("Entrenando Modelo 3: SARIMA (auto_arima)... Esto puede tardar unos minutos.")
    try:
        from pmdarima import auto_arima
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        auto_model = auto_arima(
            train['y'], m=12, seasonal=True, trace=False,
            error_action='ignore', suppress_warnings=True, stepwise=True,
            D=1
        )
        best_order = auto_model.order
        best_seasonal_order = auto_model.seasonal_order

        st_ref.write(f"Mejores parámetros: SARIMA{best_order}{best_seasonal_order}")

        model_sarima = SARIMAX(train['y'], order=best_order, seasonal_order=best_seasonal_order)
        results_sarima_fit = model_sarima.fit(disp=False)
        forecast_sarima = results_sarima_fit.get_forecast(steps=n_test)

        results['SARIMA'] = {
            'y_true': test['y'].values,
            'y_pred': forecast_sarima.predicted_mean.values,
            'y_lower': forecast_sarima.conf_int().iloc[:, 0].values,
            'y_upper': forecast_sarima.conf_int().iloc[:, 1].values,
            'has_ci': True,
            'order': best_order,
            'seasonal_order': best_seasonal_order
        }
        st_ref.write("SARIMA completado.")
    except Exception as e:
        st_ref.warning(f"Error en SARIMA: {e}")

    # =============================================
    # COMPARACIÓN DE MODELOS
    # =============================================
    st_ref.write("Comparando modelos...")
    comparison = []
    for name, res in results.items():
        mae = mean_absolute_error(res['y_true'], res['y_pred'])
        rmse = np.sqrt(mean_squared_error(res['y_true'], res['y_pred']))
        mape = mean_absolute_percentage_error(res['y_true'], res['y_pred'])
        comparison.append({
            'Modelo': name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Intervalos_Confianza': res['has_ci']
        })

    df_comparacion = pd.DataFrame(comparison)
    st_ref.dataframe(df_comparacion.style.format({
        'MAE': '${:,.2f}',
        'RMSE': '${:,.2f}',
        'MAPE': '{:.2f}%'
    }))

    # =============================================
    # SELECCIÓN MULTI-CRITERIO
    # =============================================
    modelos_con_ci = df_comparacion[df_comparacion['Intervalos_Confianza'] == True]
    if len(modelos_con_ci) > 0:
        best_model_name = modelos_con_ci.loc[modelos_con_ci['MAE'].idxmin(), 'Modelo']
    else:
        best_model_name = df_comparacion.loc[df_comparacion['MAE'].idxmin(), 'Modelo']

    st_ref.success(f"Modelo seleccionado: **{best_model_name}** (menor MAE entre modelos con intervalos de confianza)")

    # =============================================
    # PRONÓSTICO FINAL
    # =============================================
    st_ref.write(f"Generando pronóstico final a {horizonte} meses con {best_model_name}...")

    if best_model_name == 'SARIMA':
        order = results['SARIMA']['order']
        seasonal_order = results['SARIMA']['seasonal_order']
        model_final = SARIMAX(df['y'], order=order, seasonal_order=seasonal_order)
        fit_final = model_final.fit(disp=False)
        forecast_final = fit_final.get_forecast(steps=horizonte)

        future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.DateOffset(months=1),
                                     periods=horizonte, freq='MS')
        df_pronostico = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_final.predicted_mean.values,
            'yhat_lower': forecast_final.conf_int().iloc[:, 0].values.clip(min=0),
            'yhat_upper': forecast_final.conf_int().iloc[:, 1].values
        })

    elif best_model_name == 'Prophet':
        model_final_p = Prophet(
            yearly_seasonality=True, weekly_seasonality=False,
            daily_seasonality=False, holidays=holidays_ec,
            changepoint_prior_scale=0.1, seasonality_mode='multiplicative'
        )
        model_final_p.fit(df)
        future_final = model_final_p.make_future_dataframe(periods=horizonte, freq='MS')
        forecast_final_p = model_final_p.predict(future_final)
        forecast_final_p = forecast_final_p.tail(horizonte)

        df_pronostico = pd.DataFrame({
            'ds': forecast_final_p['ds'].values,
            'yhat': forecast_final_p['yhat'].values,
            'yhat_lower': forecast_final_p['yhat_lower'].values.clip(min=0),
            'yhat_upper': forecast_final_p['yhat_upper'].values
        })

    else:  # Baseline
        future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.DateOffset(months=1),
                                     periods=horizonte, freq='MS')
        preds = []
        for i in range(horizonte):
            idx = len(df) - 12 + (i % 12)
            if idx >= 0:
                preds.append(df['y'].iloc[idx])
            else:
                preds.append(df['y'].mean())

        df_pronostico = pd.DataFrame({
            'ds': future_dates,
            'yhat': preds,
            'yhat_lower': [np.nan] * horizonte,
            'yhat_upper': [np.nan] * horizonte
        })

    # --- Gráfico de pronóstico vs histórico ---
    fig_pronostico = go.Figure()
    fig_pronostico.add_trace(go.Scatter(
        x=df['ds'], y=df['y'],
        mode='lines+markers', name='Histórico',
        line=dict(color='#2E86AB', width=2)
    ))
    fig_pronostico.add_trace(go.Scatter(
        x=df_pronostico['ds'], y=df_pronostico['yhat'],
        mode='lines+markers', name=f'Pronóstico ({best_model_name})',
        line=dict(color='#E8630A', width=2)
    ))

    if df_pronostico['yhat_lower'].notna().any():
        fig_pronostico.add_trace(go.Scatter(
            x=pd.concat([df_pronostico['ds'], df_pronostico['ds'][::-1]]),
            y=pd.concat([df_pronostico['yhat_upper'], df_pronostico['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(232, 99, 10, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalo de Confianza (95%)'
        ))

    fig_pronostico.update_layout(
        title=f'Pronóstico de Ventas a {horizonte} Meses ({best_model_name})',
        xaxis_title='Mes', yaxis_title='Ventas ($)',
        template='plotly_white', height=500
    )
    figs['pronostico_vs_historico'] = fig_pronostico

    st_ref.success(f"Pronóstico final generado exitosamente.")

    return df_comparacion, df_pronostico, best_model_name, figs
