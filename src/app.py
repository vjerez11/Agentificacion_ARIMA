#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: app.py
# Descripción: Interfaz web interactiva con Streamlit
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys

# Añadir directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import TimeSeriesProcessor
from src.arima_utils import ARIMAModel, compare_models, create_comparison_table, forecast_with_intervals
from src.rl_agent import ARIMAAgent
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Agente RL-ARIMA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

@st.cache_data
def load_data(data_path='data/germany_monthly_power.csv'):
    """Carga datos con caché."""
    processor = TimeSeriesProcessor(data_path)
    processor.load_data()
    processor.split_data()
    return processor

@st.cache_resource
def load_rl_agent(data_path='data/germany_monthly_power.csv', model_path='models/arima_dqn_agent.zip'):
    """Carga agente RL con caché."""
    if not os.path.exists(model_path):
        return None
    
    processor = load_data(data_path)
    train_data = processor.train['value'].values
    val_data = processor.val['value'].values
    
    try:
        agent = ARIMAAgent(train_data, val_data)
        agent.load(model_path)
        return agent
    except Exception as e:
        st.error(f"Error al cargar agente RL: {e}")
        return None

def plot_time_series(processor):
    """Visualiza serie temporal completa con división train/val/test."""
    df = processor.df
    train = processor.train
    val = processor.val
    test = processor.test
    
    fig = go.Figure()
    
    # Serie completa
    fig.add_trace(go.Scatter(
        x=train.index, y=train['value'],
        mode='lines+markers',
        name='Train',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=val.index, y=val['value'],
        mode='lines+markers',
        name='Validation',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=test.index, y=test['value'],
        mode='lines+markers',
        name='Test',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title='Serie Temporal: Consumo Eléctrico Alemán (60 meses)',
        xaxis_title='Fecha',
        yaxis_title='Consumo (GWh)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_acf_pacf(series, nlags=12):
    """Visualiza funciones ACF y PACF."""
    acf_vals = acf(series, nlags=nlags, fft=False)
    pacf_vals = pacf(series, nlags=nlags, method='ywm')
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Autocorrelation Function (ACF)', 'Partial Autocorrelation Function (PACF)')
    )
    
    # ACF
    fig.add_trace(
        go.Bar(x=list(range(nlags+1)), y=acf_vals, name='ACF'),
        row=1, col=1
    )
    
    # PACF
    fig.add_trace(
        go.Bar(x=list(range(nlags+1)), y=pacf_vals, name='PACF'),
        row=1, col=2
    )
    
    # Líneas de confianza
    conf_interval = 1.96 / np.sqrt(len(series))
    
    for col in [1, 2]:
        fig.add_hline(y=conf_interval, line_dash="dash", line_color="red", row=1, col=col)
        fig.add_hline(y=-conf_interval, line_dash="dash", line_color="red", row=1, col=col)
    
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def plot_forecast(train, val, forecast, lower, upper, title='Pronóstico ARIMA'):
    """Visualiza pronóstico con intervalos de confianza."""
    fig = go.Figure()
    
    # Datos históricos
    fig.add_trace(go.Scatter(
        x=train.index, y=train['value'],
        mode='lines',
        name='Histórico (Train)',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Datos de validación (real)
    fig.add_trace(go.Scatter(
        x=val.index, y=val['value'],
        mode='lines+markers',
        name='Real (Validation)',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=6)
    ))
    
    # Pronóstico
    fig.add_trace(go.Scatter(
        x=val.index, y=forecast,
        mode='lines+markers',
        name='Pronóstico',
        line=dict(color='#d62728', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Intervalo de confianza 95%
    fig.add_trace(go.Scatter(
        x=val.index.tolist() + val.index.tolist()[::-1],
        y=upper.tolist() + lower.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(214, 39, 40, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='IC 95%',
        showlegend=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Fecha',
        yaxis_title='Consumo (GWh)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_residuals(residuals):
    """Visualiza diagnóstico de residuos."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Residuos vs Tiempo',
            'Histograma de Residuos',
            'Q-Q Plot',
            'ACF de Residuos'
        )
    )
    
    # 1. Residuos vs Tiempo
    fig.add_trace(
        go.Scatter(x=list(range(len(residuals))), y=residuals, mode='lines+markers', name='Residuos'),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # 2. Histograma
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=20, name='Histograma'),
        row=1, col=2
    )
    
    # 3. Q-Q Plot
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    fig.add_trace(
        go.Scatter(x=osm, y=osr, mode='markers', name='Q-Q'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=osm, y=slope*osm + intercept, mode='lines', name='Línea teórica', 
                   line=dict(color='red', dash='dash')),
        row=2, col=1
    )
    
    # 4. ACF de Residuos
    acf_vals = acf(residuals, nlags=min(20, len(residuals)//4), fft=False)
    fig.add_trace(
        go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name='ACF'),
        row=2, col=2
    )
    conf_interval = 1.96 / np.sqrt(len(residuals))
    fig.add_hline(y=conf_interval, line_dash="dash", line_color="red", row=2, col=2)
    fig.add_hline(y=-conf_interval, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=False)
    
    return fig

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    """Función principal de la aplicación Streamlit."""
    
    # Header
    st.markdown('<p class="main-header">🤖 Agente RL-ARIMA para Forecasting de Series Temporales</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=RL-ARIMA", use_container_width=True)
        st.title("⚙️ Configuración")
        
        # Cargar datos
        data_path = st.text_input("Ruta de datos", "data/germany_monthly_power.csv")
        
        if os.path.exists(data_path):
            st.success("✅ Datos encontrados")
            processor = load_data(data_path)
        else:
            st.error("❌ Archivo de datos no encontrado")
            st.info("💡 Ejecute: `python data/download_data.py`")
            st.stop()
        
        st.markdown("---")
        
        # Info del dataset
        st.subheader("📊 Información del Dataset")
        st.metric("Total de meses", len(processor.df))
        st.metric("Período", f"{processor.df.index[0].strftime('%Y-%m')} - {processor.df.index[-1].strftime('%Y-%m')}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Train", f"{len(processor.train)}")
        with col2:
            st.metric("Val", f"{len(processor.val)}")
        with col3:
            st.metric("Test", f"{len(processor.test)}")
        
        st.markdown("---")
        
        # Verificar agente RL
        model_path = "models/arima_dqn_agent.zip"
        if os.path.exists(model_path):
            st.success("✅ Modelo RL disponible")
            rl_agent = load_rl_agent(data_path, model_path)
        else:
            st.warning("⚠️ Modelo RL no entrenado")
            st.info("Entrene el agente para usar Modo Automático")
            rl_agent = None
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Exploración de Datos",
        "🤖 Agente RL / Manual",
        "📈 Comparación de Modelos",
        "🔍 Diagnósticos"
    ])
    
    # ========================================================================
    # TAB 1: EXPLORACIÓN DE DATOS
    # ========================================================================
    
    with tab1:
        st.markdown('<p class="sub-header">📊 Exploración de Datos</p>', unsafe_allow_html=True)
        
        # Visualización de serie temporal
        st.plotly_chart(plot_time_series(processor), use_container_width=True)
        
        # Estadísticas descriptivas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Estadísticas Descriptivas")
            stats_df = pd.DataFrame({
                'Métrica': ['Media', 'Desv. Estándar', 'Mínimo', 'Máximo', 'Mediana'],
                'Valor (GWh)': [
                    f"{processor.df['value'].mean():.2f}",
                    f"{processor.df['value'].std():.2f}",
                    f"{processor.df['value'].min():.2f}",
                    f"{processor.df['value'].max():.2f}",
                    f"{processor.df['value'].median():.2f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("🔍 Test de Estacionariedad (ADF)")
            adf_result = processor.test_stationarity(processor.train['value'], name='Training Set')
            
            adf_df = pd.DataFrame({
                'Métrica': ['Estadístico ADF', 'P-value', 'Estacionaria'],
                'Valor': [
                    f"{adf_result['test_statistic']:.4f}",
                    f"{adf_result['p_value']:.4f}",
                    "✅ Sí" if adf_result['is_stationary'] else "❌ No"
                ]
            })
            st.dataframe(adf_df, use_container_width=True, hide_index=True)
        
        # ACF y PACF
        if st.checkbox("📊 Mostrar funciones ACF/PACF"):
            nlags = st.slider("Número de lags", 6, 24, 12)
            st.plotly_chart(plot_acf_pacf(processor.train['value'], nlags=nlags), use_container_width=True)
    
    # ========================================================================
    # TAB 2: AGENTE RL / MANUAL
    # ========================================================================
    
    with tab2:
        st.markdown('<p class="sub-header">🤖 Predicción con Agente RL / Modo Manual</p>', unsafe_allow_html=True)
        
        # Selector de modo
        mode = st.radio("Seleccione modo:", ["🤖 Modo Automático (Agente RL)", "🎛️ Modo Manual (Sliders)"], horizontal=True)
        
        if mode == "🤖 Modo Automático (Agente RL)":
            st.markdown("### 🤖 Modo Automático con Agente RL")
            
            if rl_agent is None:
                st.error("❌ Agente RL no disponible. Entrene el agente primero.")
                st.code("python -m src.rl_agent --train --timesteps 50000", language="bash")
                st.stop()
            
            if st.button("🎯 Predecir Mejor Configuración", type="primary"):
                with st.spinner("🤖 Agente RL analizando..."):
                    p_pred, d_pred, q_pred = rl_agent.predict_best_config()
                    
                    st.session_state['rl_config'] = (p_pred, d_pred, q_pred)
                    
                    st.success(f"✅ Agente RL recomienda: ARIMA({p_pred}, {d_pred}, {q_pred})")
            
            if 'rl_config' in st.session_state:
                p_rl, d_rl, q_rl = st.session_state['rl_config']
                
                st.info(f"📋 Configuración propuesta: **ARIMA({p_rl}, {d_rl}, {q_rl})**")
                
                if st.button("▶️ Entrenar Modelo ARIMA Propuesto"):
                    with st.spinner("⏳ Entrenando modelo..."):
                        try:
                            model = ARIMAModel(order=(p_rl, d_rl, q_rl))
                            model.fit(processor.train['value'].values)
                            
                            eval_metrics = model.evaluate(processor.val['value'].values)
                            forecast, lower, upper = model.forecast(len(processor.val), return_conf_int=True)
                            
                            st.session_state['rl_model'] = model
                            st.session_state['rl_metrics'] = eval_metrics
                            st.session_state['rl_forecast'] = (forecast, lower, upper)
                            
                            # Métricas
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("AIC", f"{eval_metrics['aic']:.2f}")
                            with col2:
                                st.metric("BIC", f"{eval_metrics['bic']:.2f}")
                            with col3:
                                st.metric("RMSE", f"{eval_metrics['rmse']:.2f}")
                            with col4:
                                st.metric("MAE", f"{eval_metrics['mae']:.2f}")
                            
                            # Gráfica
                            st.plotly_chart(
                                plot_forecast(processor.train, processor.val, forecast, lower, upper,
                                            title=f'Pronóstico ARIMA({p_rl}, {d_rl}, {q_rl})'),
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error al entrenar modelo: {e}")
        
        else:  # Modo Manual
            st.markdown("### 🎛️ Modo Manual con Sliders")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                p_manual = st.slider("p (Orden AR)", 0, 5, 1)
            with col2:
                d_manual = st.slider("d (Diferenciación)", 0, 2, 1)
            with col3:
                q_manual = st.slider("q (Orden MA)", 0, 4, 1)
            
            st.info(f"📋 Configuración seleccionada: **ARIMA({p_manual}, {d_manual}, {q_manual})**")
            
            if st.button("🚀 Entrenar y Evaluar Modelo", type="primary"):
                with st.spinner("⏳ Entrenando modelo..."):
                    try:
                        model = ARIMAModel(order=(p_manual, d_manual, q_manual))
                        model.fit(processor.train['value'].values)
                        
                        eval_metrics = model.evaluate(processor.val['value'].values)
                        forecast, lower, upper = model.forecast(len(processor.val), return_conf_int=True)
                        
                        st.session_state['manual_model'] = model
                        st.session_state['manual_metrics'] = eval_metrics
                        st.session_state['manual_forecast'] = (forecast, lower, upper)
                        
                        # Métricas
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("AIC", f"{eval_metrics['aic']:.2f}")
                        with col2:
                            st.metric("BIC", f"{eval_metrics['bic']:.2f}")
                        with col3:
                            st.metric("RMSE", f"{eval_metrics['rmse']:.2f}")
                        with col4:
                            st.metric("MAE", f"{eval_metrics['mae']:.2f}")
                        
                        # Gráfica
                        st.plotly_chart(
                            plot_forecast(processor.train, processor.val, forecast, lower, upper,
                                        title=f'Pronóstico ARIMA({p_manual}, {d_manual}, {q_manual})'),
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error al entrenar modelo: {e}")
    
    # ========================================================================
    # TAB 3: COMPARACIÓN DE MODELOS
    # ========================================================================
    
    with tab3:
        st.markdown('<p class="sub-header">📈 Comparación de Modelos ARIMA</p>', unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Configurar Modelos a Comparar")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Modelo 1**")
            p1 = st.number_input("p", 0, 5, 1, key='p1')
            d1 = st.number_input("d", 0, 2, 1, key='d1')
            q1 = st.number_input("q", 0, 4, 1, key='q1')
        
        with col2:
            st.markdown("**Modelo 2**")
            p2 = st.number_input("p", 0, 5, 2, key='p2')
            d2 = st.number_input("d", 0, 2, 1, key='d2')
            q2 = st.number_input("q", 0, 4, 1, key='q2')
        
        with col3:
            st.markdown("**Modelo 3**")
            p3 = st.number_input("p", 0, 5, 1, key='p3')
            d3 = st.number_input("d", 0, 2, 1, key='d3')
            q3 = st.number_input("q", 0, 4, 2, key='q3')
        
        if st.button("📊 Comparar Modelos", type="primary"):
            configs = [(p1, d1, q1), (p2, d2, q2), (p3, d3, q3)]
            
            with st.spinner("⏳ Comparando modelos..."):
                results = compare_models(
                    processor.train['value'].values,
                    processor.val['value'].values,
                    configs,
                    return_best=False
                )
                
                # Tabla comparativa
                comparison_df = create_comparison_table(results)
                
                st.markdown("### 📊 Tabla Comparativa")
                
                # Destacar mejor modelo
                def highlight_best(row):
                    if row.name == 0:  # Primera fila (mejor AIC)
                        return ['background-color: #d4edda'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    comparison_df.style.apply(highlight_best, axis=1).format({
                        'AIC': '{:.2f}',
                        'BIC': '{:.2f}',
                        'RMSE': '{:.2f}',
                        'MAE': '{:.2f}',
                        'MAPE': '{:.2f}%',
                        'R²': '{:.4f}'
                    }),
                    use_container_width=True
                )
                
                st.success(f"🏆 Mejor modelo: ARIMA({comparison_df.iloc[0]['p']}, {comparison_df.iloc[0]['d']}, {comparison_df.iloc[0]['q']}) con AIC = {comparison_df.iloc[0]['AIC']:.2f}")
                
                # Gráfica comparativa
                fig_comparison = go.Figure()
                
                metrics = ['AIC', 'BIC', 'RMSE', 'MAE']
                for i, row in comparison_df.iterrows():
                    config_name = f"ARIMA({int(row['p'])},{int(row['d'])},{int(row['q'])})"
                    fig_comparison.add_trace(go.Bar(
                        name=config_name,
                        x=metrics,
                        y=[row['AIC'], row['BIC'], row['RMSE'], row['MAE']]
                    ))
                
                fig_comparison.update_layout(
                    title='Comparación de Métricas por Modelo',
                    xaxis_title='Métrica',
                    yaxis_title='Valor',
                    barmode='group',
                    height=500
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Opción de exportar
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Tabla como CSV",
                    data=csv,
                    file_name="comparacion_modelos_arima.csv",
                    mime="text/csv"
                )
    
    # ========================================================================
    # TAB 4: DIAGNÓSTICOS
    # ========================================================================
    
    with tab4:
        st.markdown('<p class="sub-header">🔍 Diagnóstico de Residuos</p>', unsafe_allow_html=True)
        
        # Seleccionar modelo para diagnosticar
        model_option = st.selectbox(
            "Seleccione modelo para diagnosticar:",
            ["Modelo RL (si disponible)", "Modelo Manual (si disponible)"]
        )
        
        model_to_diagnose = None
        config_name = ""
        
        if model_option == "Modelo RL (si disponible)" and 'rl_model' in st.session_state:
            model_to_diagnose = st.session_state['rl_model']
            config_name = f"RL: ARIMA{model_to_diagnose.order}"
        elif model_option == "Modelo Manual (si disponible)" and 'manual_model' in st.session_state:
            model_to_diagnose = st.session_state['manual_model']
            config_name = f"Manual: ARIMA{model_to_diagnose.order}"
        
        if model_to_diagnose is not None:
            st.markdown(f"### 📋 Modelo: {config_name}")
            
            # Obtener diagnósticos
            diagnostics = model_to_diagnose.diagnose_residuals()
            residuals = model_to_diagnose.metrics['residuals']
            
            # Estadísticas de residuos
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Media", f"{diagnostics['residuals_mean']:.4f}")
            with col2:
                st.metric("Desv. Estándar", f"{diagnostics['residuals_std']:.4f}")
            with col3:
                normal_status = "✅" if diagnostics['is_normal'] else "❌"
                st.metric("Normalidad (JB)", f"{normal_status} (p={diagnostics['jb_pvalue']:.4f})")
            with col4:
                autocorr_status = "✅" if diagnostics['no_autocorrelation'] else "❌"
                st.metric("Sin Autocorr (LB)", f"{autocorr_status} (p={diagnostics['lb_pvalue']:.4f})")
            
            # Interpretación
            st.markdown("### 📊 Interpretación")
            
            checks = []
            checks.append(("✅" if abs(diagnostics['residuals_mean']) < 0.1 else "❌", 
                          "Media de residuos cercana a 0", 
                          f"Media = {diagnostics['residuals_mean']:.4f}"))
            checks.append(("✅" if diagnostics['is_normal'] else "⚠️", 
                          "Residuos siguen distribución normal", 
                          f"Test JB p-value = {diagnostics['jb_pvalue']:.4f}"))
            checks.append(("✅" if diagnostics['no_autocorrelation'] else "⚠️", 
                          "No hay autocorrelación en residuos", 
                          f"Test LB p-value = {diagnostics['lb_pvalue']:.4f}"))
            checks.append(("✅" if diagnostics['variance_stable'] else "⚠️", 
                          "Varianza de residuos estable", 
                          f"Ratio = {diagnostics['variance_ratio']:.2f}"))
            
            for status, check, detail in checks:
                st.markdown(f"{status} **{check}** - {detail}")
            
            # Gráficas de diagnóstico
            st.markdown("### 📈 Gráficas de Diagnóstico")
            st.plotly_chart(plot_residuals(residuals), use_container_width=True)
            
        else:
            st.info("ℹ️ Entrene un modelo primero en la pestaña 'Agente RL / Manual'")


if __name__ == "__main__":
    main()
