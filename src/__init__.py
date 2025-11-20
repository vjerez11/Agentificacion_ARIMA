# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: __init__.py
# Descripción: Inicialización del paquete src
# ============================================================================

__version__ = "1.0.0"
__author__ = "ARIMA-RL Project"
__description__ = "Optimización de hiperparámetros ARIMA mediante Aprendizaje Reforzado"

from .data_processor import TimeSeriesProcessor, load_and_prepare_data
from .arima_env import ARIMAHyperparamEnv, make_arima_env
from .rl_agent import ARIMAAgent
from .arima_utils import ARIMAModel, compare_models, grid_search_arima

__all__ = [
    'TimeSeriesProcessor',
    'load_and_prepare_data',
    'ARIMAHyperparamEnv',
    'make_arima_env',
    'ARIMAAgent',
    'ARIMAModel',
    'compare_models',
    'grid_search_arima'
]
