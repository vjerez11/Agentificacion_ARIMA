#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: arima_utils.py
# Descripción: Utilidades para entrenamiento, evaluación y comparación de modelos ARIMA
# ============================================================================

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import time

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class ARIMAModel:
    """
    Wrapper para modelo ARIMA con funcionalidades completas de entrenamiento,
    pronóstico, evaluación y diagnóstico.
    """
    
    def __init__(self, order=(1, 1, 1)):
        """
        Inicializa modelo ARIMA.
        
        Args:
            order: Tupla (p, d, q) con hiperparámetros
        """
        self.order = order
        self.p, self.d, self.q = order
        self.model = None
        self.fitted_model = None
        self.metrics = {}
        
    def fit(self, train_data):
        """
        Entrena modelo ARIMA con datos de entrenamiento.
        
        Args:
            train_data: Array o Series con datos de entrenamiento
            
        Returns:
            self: Modelo entrenado
        """
        start_time = time.time()
        
        try:
            # Crear y ajustar modelo
            self.model = ARIMA(train_data, order=self.order)
            self.fitted_model = self.model.fit()
            
            # Guardar tiempo de entrenamiento
            self.metrics['training_time'] = time.time() - start_time
            
            # Guardar métricas de complejidad
            self.metrics['aic'] = self.fitted_model.aic
            self.metrics['bic'] = self.fitted_model.bic
            self.metrics['aicc'] = self.fitted_model.aicc
            self.metrics['n_params'] = self.p + self.q + 1
            
            # Residuos
            self.metrics['residuals'] = self.fitted_model.resid
            self.metrics['residuals_mean'] = np.mean(self.fitted_model.resid)
            self.metrics['residuals_std'] = np.std(self.fitted_model.resid)
            
        except Exception as e:
            raise RuntimeError(f"Error al entrenar ARIMA{self.order}: {e}")
        
        return self
    
    def forecast(self, steps, return_conf_int=True, alpha=0.05):
        """
        Genera pronósticos fuera de muestra.
        
        Args:
            steps: Número de pasos adelante a pronosticar
            return_conf_int: Si retornar intervalos de confianza
            alpha: Nivel de significancia (default: 0.05 para IC 95%)
            
        Returns:
            tuple: (forecast, lower_conf_int, upper_conf_int) si return_conf_int=True
                   forecast si return_conf_int=False
        """
        if self.fitted_model is None:
            raise ValueError("Modelo no entrenado. Ejecute fit() primero.")
        
        # Generar pronóstico
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        
        if return_conf_int:
            conf_int = forecast_result.conf_int(alpha=alpha)
            lower = conf_int.iloc[:, 0].values
            upper = conf_int.iloc[:, 1].values
            return forecast.values, lower, upper
        else:
            return forecast.values
    
    def evaluate(self, val_data):
        """
        Evalúa modelo en datos de validación/test.
        
        Args:
            val_data: Datos de validación
            
        Returns:
            dict: Métricas de evaluación
        """
        if self.fitted_model is None:
            raise ValueError("Modelo no entrenado.")
        
        # Generar pronóstico
        forecast = self.forecast(steps=len(val_data), return_conf_int=False)
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(val_data, forecast))
        mae = mean_absolute_error(val_data, forecast)
        mape = np.mean(np.abs((val_data - forecast) / val_data)) * 100
        
        # R² (coeficiente de determinación)
        ss_res = np.sum((val_data - forecast) ** 2)
        ss_tot = np.sum((val_data - np.mean(val_data)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        eval_metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'aic': self.metrics['aic'],
            'bic': self.metrics['bic'],
            'aicc': self.metrics['aicc']
        }
        
        return eval_metrics
    
    def diagnose_residuals(self):
        """
        Realiza diagnóstico completo de residuos.
        
        Returns:
            dict: Resultados de diagnóstico
        """
        if self.fitted_model is None:
            raise ValueError("Modelo no entrenado.")
        
        residuals = self.fitted_model.resid
        
        # Test de normalidad (Jarque-Bera)
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        
        # Test de autocorrelación (Ljung-Box)
        lb_test = acorr_ljungbox(residuals, lags=min(10, len(residuals)//5), return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].iloc[-1]
        
        # Homogeneidad de varianza (simple check)
        # Dividir residuos en dos mitades y comparar varianzas
        mid = len(residuals) // 2
        var_first_half = np.var(residuals[:mid])
        var_second_half = np.var(residuals[mid:])
        variance_ratio = max(var_first_half, var_second_half) / min(var_first_half, var_second_half)
        
        diagnostics = {
            'residuals_mean': self.metrics['residuals_mean'],
            'residuals_std': self.metrics['residuals_std'],
            'jb_statistic': jb_stat,
            'jb_pvalue': jb_pvalue,
            'is_normal': jb_pvalue > 0.05,
            'lb_pvalue': lb_pvalue,
            'no_autocorrelation': lb_pvalue > 0.05,
            'variance_ratio': variance_ratio,
            'variance_stable': variance_ratio < 2.0
        }
        
        return diagnostics
    
    def summary(self):
        """
        Genera resumen completo del modelo.
        
        Returns:
            str: Resumen formateado
        """
        if self.fitted_model is None:
            return "Modelo no entrenado."
        
        return str(self.fitted_model.summary())


# ============================================================================
# FUNCIONES DE COMPARACIÓN DE MODELOS
# ============================================================================

def compare_models(train_data, val_data, configs, return_best=True):
    """
    Compara múltiples configuraciones ARIMA y retorna el mejor.
    
    Args:
        train_data: Datos de entrenamiento
        val_data: Datos de validación
        configs: Lista de tuplas (p, d, q)
        return_best: Si retornar solo el mejor modelo
        
    Returns:
        dict o list: Mejor modelo o lista de todos los modelos con métricas
    """
    print(f"\n🔍 Comparando {len(configs)} configuraciones ARIMA...")
    
    results = []
    
    for i, config in enumerate(configs):
        p, d, q = config
        print(f"   [{i+1}/{len(configs)}] Evaluando ARIMA{config}...", end=' ')
        
        try:
            # Entrenar modelo
            model = ARIMAModel(order=config)
            model.fit(train_data)
            
            # Evaluar
            eval_metrics = model.evaluate(val_data)
            
            # Diagnóstico de residuos
            diagnostics = model.diagnose_residuals()
            
            result = {
                'config': config,
                'model': model,
                'metrics': eval_metrics,
                'diagnostics': diagnostics,
                'success': True
            }
            
            print(f"✅ AIC: {eval_metrics['aic']:.2f}, RMSE: {eval_metrics['rmse']:.2f}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            result = {
                'config': config,
                'model': None,
                'metrics': {'aic': np.inf, 'rmse': np.inf},
                'diagnostics': {},
                'success': False,
                'error': str(e)
            }
        
        results.append(result)
    
    # Ordenar por AIC
    results_sorted = sorted(results, key=lambda x: x['metrics'].get('aic', np.inf))
    
    if return_best:
        best = results_sorted[0]
        print(f"\n🏆 Mejor modelo: ARIMA{best['config']}")
        print(f"   AIC: {best['metrics']['aic']:.2f}")
        print(f"   RMSE: {best['metrics']['rmse']:.2f}")
        return best
    else:
        return results_sorted


def grid_search_arima(train_data, val_data, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
    """
    Búsqueda exhaustiva de hiperparámetros ARIMA.
    
    Args:
        train_data: Datos de entrenamiento
        val_data: Datos de validación
        p_range: Rango de valores p (min, max)
        d_range: Rango de valores d (min, max)
        q_range: Rango de valores q (min, max)
        
    Returns:
        dict: Mejor modelo encontrado
    """
    # Generar todas las configuraciones
    configs = []
    for p in range(p_range[0], p_range[1] + 1):
        for d in range(d_range[0], d_range[1] + 1):
            for q in range(q_range[0], q_range[1] + 1):
                configs.append((p, d, q))
    
    print(f"🔎 Grid Search: {len(configs)} configuraciones totales")
    
    # Comparar todos los modelos
    best = compare_models(train_data, val_data, configs, return_best=True)
    
    return best


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN Y EXPORTACIÓN
# ============================================================================

def create_comparison_table(results):
    """
    Crea tabla comparativa de modelos.
    
    Args:
        results: Lista de resultados de compare_models
        
    Returns:
        pd.DataFrame: Tabla comparativa
    """
    rows = []
    
    for result in results:
        if result['success']:
            p, d, q = result['config']
            metrics = result['metrics']
            
            row = {
                'p': p,
                'd': d,
                'q': q,
                'AIC': metrics['aic'],
                'BIC': metrics['bic'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'MAPE': metrics['mape'],
                'R²': metrics['r2']
            }
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('AIC').reset_index(drop=True)
    
    return df


def forecast_with_intervals(model, steps, alpha=0.05):
    """
    Genera pronóstico con intervalos de confianza en formato DataFrame.
    
    Args:
        model: Modelo ARIMA entrenado
        steps: Número de pasos adelante
        alpha: Nivel de significancia
        
    Returns:
        pd.DataFrame: Pronóstico con intervalos
    """
    forecast, lower, upper = model.forecast(steps, return_conf_int=True, alpha=alpha)
    
    df_forecast = pd.DataFrame({
        'forecast': forecast,
        'lower_bound': lower,
        'upper_bound': upper
    })
    
    return df_forecast


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def auto_select_d(series, max_d=2):
    """
    Selecciona automáticamente el orden de diferenciación mediante prueba ADF.
    
    Args:
        series: Serie temporal
        max_d: Máximo orden de diferenciación a probar
        
    Returns:
        int: Orden de diferenciación óptimo
    """
    from statsmodels.tsa.stattools import adfuller
    
    for d in range(max_d + 1):
        # Aplicar diferenciación
        if d == 0:
            series_diff = series
        else:
            series_diff = series.copy()
            for _ in range(d):
                series_diff = series_diff.diff().dropna()
        
        # Prueba ADF
        adf_result = adfuller(series_diff, autolag='AIC')
        pvalue = adf_result[1]
        
        # Si es estacionaria, retornar d
        if pvalue < 0.05:
            return d
    
    # Si no se encontró estacionariedad, retornar max_d
    return max_d


def suggest_arima_order_from_acf_pacf(acf_values, pacf_values, threshold=0.2):
    """
    Sugiere orden ARIMA basándose en gráficas ACF/PACF.
    
    Args:
        acf_values: Valores de ACF
        pacf_values: Valores de PACF
        threshold: Umbral de significancia
        
    Returns:
        tuple: (p_suggested, q_suggested)
    """
    # Encontrar primer lag donde PACF cae por debajo del umbral
    p = 0
    for i in range(1, len(pacf_values)):
        if abs(pacf_values[i]) > threshold:
            p = i
        else:
            break
    
    # Encontrar primer lag donde ACF cae por debajo del umbral
    q = 0
    for i in range(1, len(acf_values)):
        if abs(acf_values[i]) > threshold:
            q = i
        else:
            break
    
    return (p, q)


if __name__ == "__main__":
    # Ejemplo de uso
    print("🧪 Probando utilidades ARIMA...")
    
    # Generar datos sintéticos
    np.random.seed(42)
    train_data = np.random.randn(48) * 5 + 50
    val_data = np.random.randn(6) * 5 + 50
    
    # Probar modelo individual
    print("\n1️⃣  Probando modelo individual...")
    model = ARIMAModel(order=(1, 1, 1))
    model.fit(train_data)
    metrics = model.evaluate(val_data)
    print(f"   Métricas: {metrics}")
    
    # Probar comparación de modelos
    print("\n2️⃣  Probando comparación de modelos...")
    configs = [(1,1,1), (2,1,1), (1,1,2)]
    best = compare_models(train_data, val_data, configs)
    print(f"   Mejor: ARIMA{best['config']}")
    
    # Probar pronóstico
    print("\n3️⃣  Probando pronóstico...")
    forecast_df = forecast_with_intervals(best['model'], steps=6)
    print(forecast_df.head())
    
    print("\n✅ Pruebas completadas!")
