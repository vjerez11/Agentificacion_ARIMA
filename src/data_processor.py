#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: data_processor.py
# Descripción: Procesamiento y análisis de series temporales
# ============================================================================

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesProcessor:
    """
    Procesador de series temporales con análisis de estacionariedad,
    extracción de características ACF/PACF y preparación de datos.
    """
    
    def __init__(self, data_path='data/germany_monthly_power.csv'):
        """
        Inicializa el procesador con ruta a los datos.
        
        Args:
            data_path: Ruta al archivo CSV con datos mensuales
        """
        self.data_path = data_path
        self.df = None
        self.train = None
        self.val = None
        self.test = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Carga datos desde CSV y configura índice temporal."""
        print(f"📂 Cargando datos desde {self.data_path}...")
        
        self.df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        
        # Renombrar columna si es necesario
        if 'load_gwh' in self.df.columns:
            self.df.rename(columns={'load_gwh': 'value'}, inplace=True)
        
        print(f"✅ {len(self.df)} observaciones cargadas")
        print(f"   Período: {self.df.index[0].strftime('%Y-%m')} a {self.df.index[-1].strftime('%Y-%m')}")
        
        return self.df
    
    def split_data(self, train_size=30, val_size=3):
        """
        Divide datos en train/validation/test.
        
        Args:
            train_size: Número de meses para entrenamiento (default: 48)
            val_size: Número de meses para validación (default: 6)
        """
        if self.df is None:
            self.load_data()
        
        self.train = self.df.iloc[:train_size]
        self.val = self.df.iloc[train_size:train_size+val_size]
        self.test = self.df.iloc[train_size+val_size:]
        
        print(f"\n✂️  División de datos:")
        print(f"   Train: {len(self.train)} meses")
        print(f"   Val:   {len(self.val)} meses")
        print(f"   Test:  {len(self.test)} meses")
        
        return self.train, self.val, self.test
    
    def test_stationarity(self, series, name='Series'):
        """
        Realiza prueba de estacionariedad Augmented Dickey-Fuller (ADF).
        
        Args:
            series: Serie temporal a analizar
            name: Nombre de la serie para el reporte
            
        Returns:
            dict: Resultados de la prueba ADF
        """
        # Ejecutar prueba ADF
        result = adfuller(series.dropna(), autolag='AIC')
        
        adf_results = {
            'test_statistic': result[0],
            'p_value': result[1],
            'n_lags': result[2],
            'n_obs': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
        
        # Imprimir resultados
        print(f"\n📊 Prueba ADF para {name}:")
        print(f"   Estadístico ADF: {adf_results['test_statistic']:.4f}")
        print(f"   P-value: {adf_results['p_value']:.4f}")
        print(f"   Valores críticos:")
        for key, value in adf_results['critical_values'].items():
            print(f"      {key}: {value:.4f}")
        
        if adf_results['is_stationary']:
            print(f"   ✅ La serie ES ESTACIONARIA (p-value < 0.05)")
        else:
            print(f"   ⚠️  La serie NO es estacionaria (p-value >= 0.05)")
            print(f"      → Recomendación: Aplicar diferenciación (d=1 o d=2)")
        
        return adf_results
    
    def compute_acf_pacf(self, series, nlags=12):
        """
        Calcula funciones de autocorrelación (ACF) y autocorrelación parcial (PACF).
        
        Args:
            series: Serie temporal
            nlags: Número de lags a calcular (default: 12)
            
        Returns:
            tuple: (acf_values, pacf_values)
        """
        series_clean = series.dropna()
        
        # Calcular ACF y PACF
        acf_values = acf(series_clean, nlags=nlags, fft=False)
        pacf_values = pacf(series_clean, nlags=nlags, method='ywm')
        
        return acf_values, pacf_values
    
    def get_statistics(self, series):
        """
        Calcula estadísticas descriptivas de la serie.
        
        Args:
            series: Serie temporal
            
        Returns:
            dict: Estadísticas descriptivas
        """
        stats = {
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'median': series.median(),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75),
            'cv': series.std() / series.mean() if series.mean() != 0 else 0,
            'skewness': series.skew(),
            'kurtosis': series.kurtosis()
        }
        
        return stats
    
    def extract_features(self, series, nlags=12):
        """
        Extrae características completas para el estado del agente RL.
        
        Args:
            series: Serie temporal
            nlags: Número de lags para ACF/PACF
            
        Returns:
            dict: Diccionario con todas las características
        """
        # Estadísticas básicas
        stats = self.get_statistics(series)
        
        # Estacionariedad
        adf_result = adfuller(series.dropna(), autolag='AIC')
        
        # ACF y PACF
        acf_vals, pacf_vals = self.compute_acf_pacf(series, nlags=nlags)
        
        # Tendencia (coeficiente de regresión lineal)
        x = np.arange(len(series))
        z = np.polyfit(x, series.values, 1)
        trend = z[0]
        
        features = {
            'mean': stats['mean'],
            'std': stats['std'],
            'cv': stats['cv'],
            'trend': trend,
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'is_stationary': adf_result[1] < 0.05,
            'acf_lag1': acf_vals[1] if len(acf_vals) > 1 else 0,
            'acf_lag2': acf_vals[2] if len(acf_vals) > 2 else 0,
            'pacf_lag1': pacf_vals[1] if len(pacf_vals) > 1 else 0,
            'pacf_lag2': pacf_vals[2] if len(pacf_vals) > 2 else 0,
            'skewness': stats['skewness'],
            'kurtosis': stats['kurtosis']
        }
        
        return features
    
    def difference_series(self, series, d=1):
        """
        Aplica diferenciación de orden d a la serie.
        
        Args:
            series: Serie temporal
            d: Orden de diferenciación
            
        Returns:
            Series diferenciada
        """
        diff_series = series.copy()
        
        for i in range(d):
            diff_series = diff_series.diff().dropna()
        
        return diff_series
    
    def decompose_series(self, series, model='additive', period=12):
        """
        Descomposición de serie temporal en tendencia, estacionalidad y residuos.
        
        Args:
            series: Serie temporal
            model: 'additive' o 'multiplicative'
            period: Período de estacionalidad (12 para datos mensuales anuales)
            
        Returns:
            Objeto de descomposición de statsmodels
        """
        if len(series) < 2 * period:
            print(f"⚠️  Serie muy corta para descomposición (necesita >= {2*period} obs)")
            return None
        
        try:
            decomposition = seasonal_decompose(series, model=model, period=period, extrapolate_trend='freq')
            return decomposition
        except Exception as e:
            print(f"⚠️  Error en descomposición: {e}")
            return None
    
    def get_summary(self):
        """
        Genera resumen completo de los datos cargados.
        
        Returns:
            dict: Resumen con estadísticas y características
        """
        if self.df is None:
            self.load_data()
        
        if self.train is None:
            self.split_data()
        
        summary = {
            'total_months': len(self.df),
            'train_months': len(self.train),
            'val_months': len(self.val),
            'test_months': len(self.test),
            'date_range': {
                'start': self.df.index[0].strftime('%Y-%m'),
                'end': self.df.index[-1].strftime('%Y-%m')
            },
            'train_stats': self.get_statistics(self.train['value']),
            'full_stats': self.get_statistics(self.df['value']),
            'train_features': self.extract_features(self.train['value'])
        }
        
        return summary
    
    def prepare_for_rl(self):
        """
        Prepara datos en formato requerido por el agente RL.
        
        Returns:
            tuple: (train_array, val_array, test_array, features_dict)
        """
        if self.train is None:
            self.split_data()
        
        # Convertir a arrays numpy
        train_array = self.train['value'].values
        val_array = self.val['value'].values
        test_array = self.test['value'].values
        
        # Extraer características
        features = self.extract_features(self.train['value'])
        
        return train_array, val_array, test_array, features


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_and_prepare_data(data_path='data/germany_monthly_power.csv'):
    """
    Función conveniente para cargar y preparar datos en un solo paso.
    
    Args:
        data_path: Ruta al archivo CSV
        
    Returns:
        TimeSeriesProcessor configurado
    """
    processor = TimeSeriesProcessor(data_path)
    processor.load_data()
    processor.split_data()
    
    return processor


def print_data_summary(processor):
    """
    Imprime resumen completo de los datos.
    
    Args:
        processor: Instancia de TimeSeriesProcessor
    """
    summary = processor.get_summary()
    
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE DATOS")
    print("=" * 80)
    
    print(f"\n📅 Información temporal:")
    print(f"   Total de meses: {summary['total_months']}")
    print(f"   Rango: {summary['date_range']['start']} a {summary['date_range']['end']}")
    
    print(f"\n✂️  División:")
    print(f"   Train: {summary['train_months']} meses")
    print(f"   Val:   {summary['val_months']} meses")
    print(f"   Test:  {summary['test_months']} meses")
    
    print(f"\n📈 Estadísticas (conjunto completo):")
    stats = summary['full_stats']
    print(f"   Media:          {stats['mean']:.2f} GWh")
    print(f"   Desv. Estándar: {stats['std']:.2f} GWh")
    print(f"   Mínimo:         {stats['min']:.2f} GWh")
    print(f"   Máximo:         {stats['max']:.2f} GWh")
    print(f"   Coef. Variación: {stats['cv']:.4f}")
    
    print(f"\n🔍 Características (training set):")
    features = summary['train_features']
    print(f"   ADF p-value: {features['adf_pvalue']:.4f}")
    print(f"   Estacionaria: {'✅ Sí' if features['is_stationary'] else '⚠️  No'}")
    print(f"   Tendencia: {features['trend']:.4f}")
    print(f"   ACF lag-1: {features['acf_lag1']:.4f}")
    print(f"   PACF lag-1: {features['pacf_lag1']:.4f}")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Ejemplo de uso
    processor = load_and_prepare_data()
    print_data_summary(processor)
    
    # Prueba de estacionariedad
    processor.test_stationarity(processor.train['value'], name='Training Set')
