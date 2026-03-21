import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def _create_xgboost_features(series, n_lags=12):
    """Crea features para XGBoost a partir de una serie temporal."""
    df_feat = pd.DataFrame({'y': series.values}, index=range(len(series)))

    # Lags
    for lag in range(1, min(n_lags + 1, len(series))):
        df_feat[f'lag_{lag}'] = df_feat['y'].shift(lag)

    # Promedios móviles
    if len(series) >= 3:
        df_feat['rolling_mean_3'] = df_feat['y'].shift(1).rolling(window=3, min_periods=1).mean()
    if len(series) >= 6:
        df_feat['rolling_mean_6'] = df_feat['y'].shift(1).rolling(window=6, min_periods=1).mean()
    if len(series) >= 12:
        df_feat['rolling_mean_12'] = df_feat['y'].shift(1).rolling(window=12, min_periods=1).mean()

    # Desviación estándar móvil
    if len(series) >= 3:
        df_feat['rolling_std_3'] = df_feat['y'].shift(1).rolling(window=3, min_periods=1).std().fillna(0)

    # Features de calendario (mes)
    df_feat['month'] = [(i % 12) + 1 for i in range(len(series))]
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)

    # Tendencia lineal
    df_feat['trend'] = range(len(series))

    # Diferencia
    df_feat['diff_1'] = df_feat['y'].diff(1)

    df_feat = df_feat.dropna()
    return df_feat


def _create_lstm_sequences(data, seq_length):
    """Crea secuencias para LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def run_modeling(df_mensual, horizonte, st_ref, modelos_keys=None):
    """Ejecuta los modelos predictivos seleccionados y compara resultados."""

    # Si no se especifican modelos, usar los 3 originales
    if modelos_keys is None:
        modelos_keys = ['baseline', 'prophet', 'sarima']

    figs = {}

    # --- Preparar datos ---
    df = df_mensual[['ds', 'y']].copy().sort_values('ds').reset_index(drop=True)
    df = df.dropna(subset=['ds', 'y'])
    df['y'] = df['y'].replace([np.inf, -np.inf], np.nan).fillna(0)

    n_total = len(df)
    st_ref.write(f"Total de meses disponibles: **{n_total}**")

    if n_total < 6:
        raise ValueError(f"Se necesitan al menos 6 meses de datos para el modelado. Solo hay {n_total}.")

    # División train/test
    if n_total >= 24:
        n_test = 12
    elif n_total >= 12:
        n_test = max(3, n_total // 4)
    else:
        n_test = max(2, n_total // 3)

    train = df.iloc[:-n_test].copy()
    test = df.iloc[-n_test:].copy()

    st_ref.write(f"Datos de entrenamiento: **{len(train)} meses** | Datos de prueba: **{len(test)} meses**")

    results = {}

    # =============================================
    # MODELO 1: BASELINE (Seasonal Naive)
    # =============================================
    if 'baseline' in modelos_keys:
        st_ref.write("Entrenando Modelo 1: **Baseline (Seasonal Naive)**...")
        try:
            baseline_preds = []
            season_len = min(12, len(train))
            for i in range(len(test)):
                idx = len(train) - season_len + (i % season_len)
                if 0 <= idx < len(train):
                    baseline_preds.append(train['y'].iloc[idx])
                else:
                    baseline_preds.append(train['y'].mean())

            results['Baseline'] = {
                'y_true': test['y'].values,
                'y_pred': np.array(baseline_preds),
                'has_ci': False,
                'tipo': 'Referencia'
            }
            st_ref.write("Baseline completado.")
        except Exception as e:
            st_ref.warning(f"Error en Baseline: {e}")

    # =============================================
    # MODELO 2: PROPHET
    # =============================================
    if 'prophet' in modelos_keys:
        st_ref.write("Entrenando Modelo 2: **Prophet** (con feriados de Ecuador)...")
        try:
            from prophet import Prophet

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

            yearly_season = n_total >= 18
            seasonality_mode = 'multiplicative' if n_total >= 24 else 'additive'

            model_prophet = Prophet(
                yearly_seasonality=yearly_season,
                weekly_seasonality=False,
                daily_seasonality=False,
                holidays=holidays_ec,
                changepoint_prior_scale=0.1,
                seasonality_mode=seasonality_mode
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
                'has_ci': True,
                'tipo': 'Estadístico'
            }
            st_ref.write("Prophet completado.")
        except Exception as e:
            st_ref.warning(f"Error en Prophet: {e}")

    # =============================================
    # MODELO 3: SARIMA
    # =============================================
    if 'sarima' in modelos_keys:
        st_ref.write("Entrenando Modelo 3: **SARIMA** (auto_arima)... Esto puede tardar unos minutos.")
        try:
            from pmdarima import auto_arima
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            m_seasonal = 12 if n_total >= 24 else max(2, n_total // 4)

            auto_model = auto_arima(
                train['y'].values, m=m_seasonal, seasonal=True, trace=False,
                error_action='ignore', suppress_warnings=True, stepwise=True,
                D=1 if n_total >= 24 else 0,
                max_p=3, max_q=3, max_P=2, max_Q=2
            )
            best_order = auto_model.order
            best_seasonal_order = auto_model.seasonal_order

            st_ref.write(f"Mejores parámetros: SARIMA{best_order}{best_seasonal_order}")

            model_sarima = SARIMAX(train['y'].values, order=best_order, seasonal_order=best_seasonal_order)
            results_sarima_fit = model_sarima.fit(disp=False)
            forecast_sarima = results_sarima_fit.get_forecast(steps=n_test)

            results['SARIMA'] = {
                'y_true': test['y'].values,
                'y_pred': forecast_sarima.predicted_mean.values,
                'y_lower': forecast_sarima.conf_int().iloc[:, 0].values,
                'y_upper': forecast_sarima.conf_int().iloc[:, 1].values,
                'has_ci': True,
                'order': best_order,
                'seasonal_order': best_seasonal_order,
                'tipo': 'Estadístico'
            }
            st_ref.write("SARIMA completado.")
        except Exception as e:
            st_ref.warning(f"Error en SARIMA: {e}")

    # =============================================
    # MODELO 4: XGBOOST
    # =============================================
    if 'xgboost' in modelos_keys:
        st_ref.write("Entrenando Modelo 4: **XGBoost** (Machine Learning)...")
        try:
            from xgboost import XGBRegressor

            # Crear features para toda la serie
            n_lags = min(12, len(train) - 1)
            df_features_all = _create_xgboost_features(df['y'], n_lags=n_lags)

            # Separar train y test con features
            n_features = len(df_features_all)
            n_test_feat = min(n_test, n_features // 2)
            train_feat = df_features_all.iloc[:-n_test_feat]
            test_feat = df_features_all.iloc[-n_test_feat:]

            feature_cols = [c for c in df_features_all.columns if c != 'y']
            X_train = train_feat[feature_cols].values
            y_train = train_feat['y'].values
            X_test = test_feat[feature_cols].values
            y_test = test_feat['y'].values

            # Entrenar XGBoost
            model_xgb = XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0
            )
            model_xgb.fit(X_train, y_train)
            xgb_preds = model_xgb.predict(X_test)

            # Importancia de features
            importances = model_xgb.feature_importances_
            feat_imp = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10)

            results['XGBoost'] = {
                'y_true': y_test,
                'y_pred': xgb_preds,
                'has_ci': False,
                'tipo': 'Machine Learning',
                'feature_importance': feat_imp,
                'model': model_xgb,
                'feature_cols': feature_cols,
                'n_lags': n_lags
            }
            st_ref.write("XGBoost completado.")

            # Mostrar importancia de features
            st_ref.write("**Top 5 features más importantes (XGBoost):**")
            for _, row in feat_imp.head(5).iterrows():
                st_ref.write(f"  - {row['Feature']}: {row['Importance']:.3f}")

        except Exception as e:
            st_ref.warning(f"Error en XGBoost: {e}")
            import traceback
            st_ref.code(traceback.format_exc())

    # =============================================
    # MODELO 5: LSTM (PyTorch)
    # =============================================
    if 'lstm' in modelos_keys:
        st_ref.write("Entrenando Modelo 5: **LSTM** (Deep Learning con PyTorch)... Esto puede tardar unos minutos.")
        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import MinMaxScaler

            # Definir modelo LSTM en PyTorch
            class LSTMModel(nn.Module):
                def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
                    super(LSTMModel, self).__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                       batch_first=True, dropout=dropout)
                    self.fc = nn.Linear(hidden_size, 1)

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    out = self.fc(lstm_out[:, -1, :])
                    return out

            # Normalizar datos
            scaler = MinMaxScaler(feature_range=(0, 1))
            y_scaled = scaler.fit_transform(df['y'].values.reshape(-1, 1)).flatten()

            # Crear secuencias
            seq_length = min(6, len(train) // 2)
            X_all, y_all = _create_lstm_sequences(y_scaled, seq_length)

            if len(X_all) < n_test + 2:
                raise ValueError(f"No hay suficientes datos para LSTM con secuencia de {seq_length}")

            X_train_lstm = X_all[:-n_test]
            y_train_lstm = y_all[:-n_test]
            X_test_lstm = X_all[-n_test:]
            y_test_lstm = y_all[-n_test:]

            # Convertir a tensores PyTorch
            X_train_t = torch.FloatTensor(X_train_lstm).unsqueeze(-1)
            y_train_t = torch.FloatTensor(y_train_lstm).unsqueeze(-1)
            X_test_t = torch.FloatTensor(X_test_lstm).unsqueeze(-1)

            # Crear y entrenar modelo
            model_lstm = LSTMModel(input_size=1, hidden_size=50, num_layers=2, dropout=0.2)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.01)

            # Entrenamiento
            model_lstm.train()
            best_loss = float('inf')
            patience_counter = 0
            best_state = None

            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model_lstm(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()

                # Early stopping
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                    best_state = model_lstm.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        break

            if best_state is not None:
                model_lstm.load_state_dict(best_state)

            # Predecir test
            model_lstm.eval()
            with torch.no_grad():
                lstm_preds_scaled = model_lstm(X_test_t).numpy().flatten()

            lstm_preds = scaler.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten()
            y_test_real = scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()

            results['LSTM'] = {
                'y_true': y_test_real,
                'y_pred': lstm_preds,
                'has_ci': False,
                'tipo': 'Deep Learning',
                'model': model_lstm,
                'scaler': scaler,
                'seq_length': seq_length
            }
            st_ref.write("LSTM completado.")
        except ImportError:
            st_ref.warning("PyTorch no está instalado. El modelo LSTM no se pudo entrenar.")
        except Exception as e:
            st_ref.warning(f"Error en LSTM: {e}")
            import traceback
            st_ref.code(traceback.format_exc())

    # =============================================
    # COMPARACIÓN DE MODELOS
    # =============================================
    if len(results) == 0:
        raise ValueError("Ningún modelo pudo entrenarse correctamente. Verifica los datos.")

    st_ref.write("---")
    st_ref.write("### Comparación de Modelos")
    comparison = []
    for name, res in results.items():
        y_true = np.array(res['y_true']).astype(float)
        y_pred = np.array(res['y_pred']).astype(float)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() == 0:
            continue
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
        comparison.append({
            'Modelo': name,
            'Tipo': res.get('tipo', ''),
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
    }).highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='#d4edda'))

    # =============================================
    # SELECCIÓN DEL MEJOR MODELO
    # =============================================
    best_model_name = df_comparacion.loc[df_comparacion['MAE'].idxmin(), 'Modelo']
    best_mae = df_comparacion.loc[df_comparacion['MAE'].idxmin(), 'MAE']
    best_mape = df_comparacion.loc[df_comparacion['MAE'].idxmin(), 'MAPE']

    st_ref.success(
        f"Modelo seleccionado: **{best_model_name}** "
        f"(MAE: ${best_mae:,.2f} | MAPE: {best_mape:.2f}%)"
    )

    # =============================================
    # GRÁFICO COMPARATIVO DE MODELOS
    # =============================================
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=test['ds'].values if len(test) == len(list(results.values())[0]['y_true']) else list(range(len(list(results.values())[0]['y_true']))),
        y=list(results.values())[0]['y_true'],
        mode='lines+markers', name='Valores Reales',
        line=dict(color='black', width=3)
    ))

    colors = ['#2E86AB', '#E8630A', '#28A745', '#DC3545', '#6F42C1']
    for i, (name, res) in enumerate(results.items()):
        x_vals = test['ds'].values[:len(res['y_pred'])] if len(test) >= len(res['y_pred']) else list(range(len(res['y_pred'])))
        fig_comp.add_trace(go.Scatter(
            x=x_vals,
            y=res['y_pred'],
            mode='lines+markers', name=name,
            line=dict(color=colors[i % len(colors)], width=2, dash='dash')
        ))

    fig_comp.update_layout(
        title='Comparación de Modelos vs Valores Reales (Período de Prueba)',
        xaxis_title='Mes', yaxis_title='Ventas ($)',
        template='plotly_white', height=500
    )
    figs['comparacion_modelos'] = fig_comp
    st_ref.plotly_chart(fig_comp, use_container_width=True)

    # Gráfico de barras de métricas
    fig_metricas = make_subplots(rows=1, cols=3, subplot_titles=['MAE ($)', 'RMSE ($)', 'MAPE (%)'])

    for i, metrica in enumerate(['MAE', 'RMSE', 'MAPE']):
        fig_metricas.add_trace(go.Bar(
            x=df_comparacion['Modelo'],
            y=df_comparacion[metrica],
            marker_color=[colors[j % len(colors)] for j in range(len(df_comparacion))],
            name=metrica,
            showlegend=False
        ), row=1, col=i+1)

    fig_metricas.update_layout(
        title='Métricas de Error por Modelo',
        template='plotly_white', height=400
    )
    figs['metricas_modelos'] = fig_metricas

    # =============================================
    # PRONÓSTICO FINAL
    # =============================================
    st_ref.write("---")
    st_ref.write(f"### Pronóstico Final a {horizonte} Meses con {best_model_name}")

    future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.DateOffset(months=1),
                                 periods=horizonte, freq='MS')

    if best_model_name == 'SARIMA':
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        order = results['SARIMA']['order']
        seasonal_order = results['SARIMA']['seasonal_order']
        model_final = SARIMAX(df['y'].values, order=order, seasonal_order=seasonal_order)
        fit_final = model_final.fit(disp=False)
        forecast_final = fit_final.get_forecast(steps=horizonte)

        df_pronostico = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_final.predicted_mean.values,
            'yhat_lower': forecast_final.conf_int().iloc[:, 0].values.clip(min=0),
            'yhat_upper': forecast_final.conf_int().iloc[:, 1].values
        })

    elif best_model_name == 'Prophet':
        from prophet import Prophet
        holidays_ec = results['Prophet'].get('holidays_ec', None)
        model_final_p = Prophet(
            yearly_seasonality=n_total >= 18, weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
            seasonality_mode='multiplicative' if n_total >= 24 else 'additive'
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

    elif best_model_name == 'XGBoost':
        # Re-entrenar con todos los datos
        from xgboost import XGBRegressor
        n_lags = results['XGBoost']['n_lags']
        feature_cols = results['XGBoost']['feature_cols']

        df_feat_full = _create_xgboost_features(df['y'], n_lags=n_lags)
        X_full = df_feat_full[feature_cols].values
        y_full = df_feat_full['y'].values

        model_xgb_final = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0
        )
        model_xgb_final.fit(X_full, y_full)

        # Pronóstico iterativo
        y_extended = list(df['y'].values)
        preds_xgb = []
        for step in range(horizonte):
            temp_series = pd.Series(y_extended)
            temp_feat = _create_xgboost_features(temp_series, n_lags=n_lags)
            last_row = temp_feat[feature_cols].iloc[-1:].values
            pred = model_xgb_final.predict(last_row)[0]
            preds_xgb.append(max(0, pred))
            y_extended.append(pred)

        df_pronostico = pd.DataFrame({
            'ds': future_dates,
            'yhat': preds_xgb,
            'yhat_lower': [np.nan] * horizonte,
            'yhat_upper': [np.nan] * horizonte
        })

    elif best_model_name == 'LSTM':
        import torch
        scaler = results['LSTM']['scaler']
        seq_length = results['LSTM']['seq_length']
        model_lstm_final = results['LSTM']['model']

        y_scaled_full = scaler.transform(df['y'].values.reshape(-1, 1)).flatten()
        current_seq = y_scaled_full[-seq_length:].tolist()

        model_lstm_final.eval()
        preds_lstm = []
        for step in range(horizonte):
            X_input = torch.FloatTensor(np.array(current_seq[-seq_length:])).unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                pred_scaled = model_lstm_final(X_input).numpy().flatten()[0]
            pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
            preds_lstm.append(max(0, pred_real))
            current_seq.append(pred_scaled)

        df_pronostico = pd.DataFrame({
            'ds': future_dates,
            'yhat': preds_lstm,
            'yhat_lower': [np.nan] * horizonte,
            'yhat_upper': [np.nan] * horizonte
        })

    else:  # Baseline
        season_len = min(12, len(df))
        preds = []
        for i in range(horizonte):
            idx = len(df) - season_len + (i % season_len)
            if 0 <= idx < len(df):
                preds.append(df['y'].iloc[idx])
            else:
                preds.append(df['y'].mean())

        df_pronostico = pd.DataFrame({
            'ds': future_dates,
            'yhat': preds,
            'yhat_lower': [np.nan] * horizonte,
            'yhat_upper': [np.nan] * horizonte
        })

    # Limpiar NaN en pronóstico
    df_pronostico['yhat'] = df_pronostico['yhat'].fillna(df['y'].mean())

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

    st_ref.plotly_chart(fig_pronostico, use_container_width=True)
    st_ref.success(f"Pronóstico final generado exitosamente con **{best_model_name}**.")

    return df_comparacion, df_pronostico, best_model_name, figs
