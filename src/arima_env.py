#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: arima_env.py
# Descripción: Entorno Gymnasium personalizado para optimización de hiperparámetros ARIMA
# ============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import time

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class ARIMAHyperparamEnv(gym.Env):
    """
    Entorno Gymnasium para optimización de hiperparámetros ARIMA mediante RL.
    
    Espacio de Estados:
        - Características de la serie temporal (ACF, PACF, tendencia, estacionariedad)
        - Configuración actual de hiperparámetros (p, d, q)
        - Métricas de desempeño (RMSE, AIC, tiempo)
        
    Espacio de Acciones:
        - Selección directa de tupla (p, d, q)
        - p: [0, p_max], d: [0, d_max], q: [0, q_max]
        
    Función de Recompensa:
        - Multiobjetivo: balancea precisión, complejidad y eficiencia
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, train_data, val_data, 
                 p_max=5, d_max=2, q_max=4,
                 max_steps=50,
                 reward_weights=None):
        """
        Inicializa el entorno ARIMA.
        
        Args:
            train_data: Array numpy con datos de entrenamiento
            val_data: Array numpy con datos de validación
            p_max: Máximo orden autorregresivo
            d_max: Máximo orden de diferenciación
            q_max: Máximo orden de media móvil
            max_steps: Máximo de pasos por episodio
            reward_weights: Dict con pesos para componentes de recompensa
        """
        super().__init__()
        
        self.train_data = train_data
        self.val_data = val_data
        self.p_max = p_max
        self.d_max = d_max
        self.q_max = q_max
        self.max_steps = max_steps
        
        # Pesos de la función de recompensa
        if reward_weights is None:
            self.reward_weights = {
                'accuracy': 1.0,
                'aic': 0.3,
                'time': 0.1,
                'diagnostics': 0.2
            }
        else:
            self.reward_weights = reward_weights
        
        # Espacios de acción y observación
        self.action_space = spaces.MultiDiscrete([p_max+1, d_max+1, q_max+1])
        
        # Espacio de observación (8 características)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(8,), 
            dtype=np.float32
        )
        
        # Variables de estado
        self.current_step = 0
        self.current_config = None
        self.best_aic = np.inf
        self.best_rmse = np.inf
        self.history = []
        
        # Normalización
        self.data_mean = np.mean(train_data)
        self.data_std = np.std(train_data)
        
    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno al estado inicial.
        
        Returns:
            observation: Estado inicial
            info: Información adicional
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_config = [1, 1, 1]  # Configuración inicial por defecto
        self.best_aic = np.inf
        self.best_rmse = np.inf
        self.history = []
        
        # Estado inicial
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def _discrete_to_p_d_q(self, action):
        """
        Convierte un índice de acción discreto a un triplete (p, d, q).
        """
        action = int(action)
        p = action // ((self.d_max + 1) * (self.q_max + 1))
        d = (action % ((self.d_max + 1) * (self.q_max + 1))) // (self.q_max + 1)
        q = action % (self.q_max + 1)
        return p, d, q

    def step(self, action):
        """
        Ejecuta una acción (entrenar ARIMA con configuración específica).
        
        Args:
            action: Array [p, d, q] o entero discreto
            
        Returns:
            observation: Nuevo estado
            reward: Recompensa obtenida
            terminated: Si el episodio terminó
            truncated: Si el episodio fue truncado
            info: Información adicional
        """
        self.current_step += 1
        
        # Extraer configuración de la acción
        if np.isscalar(action) or (isinstance(action, np.ndarray) and action.ndim == 0):
            # Acción es discreta (escalar o array 0-dim), convertir a (p, d, q)
            p, d, q = self._discrete_to_p_d_q(action)
        else:
            # Acción ya es (p, d, q)
            p, d, q = int(action[0]), int(action[1]), int(action[2])

        self.current_config = [p, d, q]
        
        # Entrenar modelo ARIMA y obtener métricas
        try:
            metrics = self._train_and_evaluate_arima(p, d, q)
            
            # Calcular recompensa
            reward = self._compute_reward(metrics)
            
            # Actualizar mejor configuración
            if metrics['aic'] < self.best_aic:
                self.best_aic = metrics['aic']
            if metrics['rmse'] < self.best_rmse:
                self.best_rmse = metrics['rmse']
            
            # Guardar en historial
            self.history.append({
                'step': self.current_step,
                'config': (p, d, q),
                'metrics': metrics,
                'reward': reward
            })
            
            success = True
            
        except Exception as e:
            # Si el modelo falla, penalizar fuertemente
            metrics = {
                'rmse': 1e6,
                'mae': 1e6,
                'aic': 1e6,
                'bic': 1e6,
                'training_time': 0,
                'failed': True
            }
            reward = -10.0
            success = False
        
        # Nuevo estado
        observation = self._get_observation()
        
        # Condiciones de terminación
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Información adicional
        info = {
            'step': self.current_step,
            'config': (p, d, q),
            'metrics': metrics,
            'reward': reward,
            'success': success,
            'best_aic': self.best_aic,
            'best_rmse': self.best_rmse
        }
        
        return observation, reward, terminated, truncated, info
    
    def _train_and_evaluate_arima(self, p, d, q):
        """
        Entrena modelo ARIMA y calcula métricas de evaluación.
        
        Args:
            p, d, q: Hiperparámetros ARIMA
            
        Returns:
            dict: Métricas de desempeño
        """
        start_time = time.time()
        
        # Entrenar modelo
        model = ARIMA(self.train_data, order=(p, d, q))
        fitted_model = model.fit()
        
        training_time = time.time() - start_time
        
        # Predicciones en validación
        forecast = fitted_model.forecast(steps=len(self.val_data))
        
        # Métricas de precisión
        rmse = np.sqrt(mean_squared_error(self.val_data, forecast))
        mae = mean_absolute_error(self.val_data, forecast)
        mape = np.mean(np.abs((self.val_data - forecast) / self.val_data)) * 100
        
        # Métricas de complejidad
        aic = fitted_model.aic
        bic = fitted_model.bic
        
        # Diagnóstico de residuos
        residuals = fitted_model.resid
        residuals_mean = np.mean(residuals)
        residuals_std = np.std(residuals)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'aic': aic,
            'bic': bic,
            'training_time': training_time,
            'residuals_mean': residuals_mean,
            'residuals_std': residuals_std,
            'n_params': p + q + 1,
            'failed': False
        }
        
        return metrics
    
    def _compute_reward(self, metrics):
        """
        Calcula recompensa multiobjetivo.
        
        Args:
            metrics: Dict con métricas del modelo
            
        Returns:
            float: Recompensa total
        """
        if metrics['failed']:
            return -10.0
        
        # Componente 1: Precisión de predicción (normalizada)
        rmse_norm = metrics['rmse'] / self.data_std
        mae_norm = metrics['mae'] / self.data_mean
        mape_norm = metrics['mape'] / 100.0
        
        r_accuracy = -(0.5 * rmse_norm + 0.3 * mae_norm + 0.2 * mape_norm)
        
        # Componente 2: Complejidad del modelo (AIC normalizado)
        aic_norm = metrics['aic'] / 1000.0
        r_aic = -0.1 * aic_norm
        
        # Componente 3: Eficiencia computacional
        time_norm = metrics['training_time'] / 10.0  # Asumir 10s como máximo
        r_time = -0.01 * time_norm
        
        # Componente 4: Calidad de residuos
        residuals_good = (abs(metrics['residuals_mean']) < 0.1)
        r_diagnostics = 0.5 if residuals_good else 0.0
        
        # Recompensa total ponderada
        reward = (
            self.reward_weights['accuracy'] * r_accuracy +
            self.reward_weights['aic'] * r_aic +
            self.reward_weights['time'] * r_time +
            self.reward_weights['diagnostics'] * r_diagnostics
        )
        
        # Bonus por mejorar el mejor AIC
        if metrics['aic'] < self.best_aic:
            reward += 1.0
        
        # Bonus por mejorar el mejor RMSE
        if metrics['rmse'] < self.best_rmse:
            reward += 0.5
        
        return reward
    
    def _get_observation(self):
        """
        Construye el vector de observación (estado).
        
        Returns:
            np.array: Vector de estado (8 características)
        """
        p, d, q = self.current_config
        
        observation = np.array([
            self.best_rmse / self.data_std,      # RMSE normalizado
            self.best_aic / 1000.0,              # AIC normalizado
            p / self.p_max,                       # p normalizado
            d / self.d_max,                       # d normalizado
            q / self.q_max,                       # q normalizado
            self.current_step / self.max_steps,   # Progreso del episodio
            len([h for h in self.history if h['metrics']['aic'] < self.best_aic + 10]) / max(1, len(self.history)),  # Tasa de mejora
            np.mean([h['reward'] for h in self.history[-10:]]) if len(self.history) > 0 else 0  # Recompensa promedio reciente
        ], dtype=np.float32)
        
        return observation
    
    def render(self, mode='human'):
        """
        Renderiza el estado actual del entorno (imprime información).
        """
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Current Config: p={self.current_config[0]}, d={self.current_config[1]}, q={self.current_config[2]}")
            print(f"Best AIC: {self.best_aic:.2f}")
            print(f"Best RMSE: {self.best_rmse:.2f}")
            if len(self.history) > 0:
                last = self.history[-1]
                print(f"Last Reward: {last['reward']:.4f}")
            print(f"{'='*60}")
    
    def get_best_config(self):
        """
        Retorna la mejor configuración encontrada hasta ahora.
        
        Returns:
            tuple: (p, d, q, aic, rmse)
        """
        if len(self.history) == 0:
            return None
        
        # Encontrar configuración con mejor AIC
        best_entry = min(self.history, key=lambda x: x['metrics']['aic'])
        
        return {
            'config': best_entry['config'],
            'aic': best_entry['metrics']['aic'],
            'rmse': best_entry['metrics']['rmse'],
            'mae': best_entry['metrics']['mae'],
            'step': best_entry['step']
        }


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def make_arima_env(train_data, val_data, **kwargs):
    """
    Función factory para crear el entorno ARIMA.
    
    Args:
        train_data: Datos de entrenamiento
        val_data: Datos de validación
        **kwargs: Argumentos adicionales para el entorno
        
    Returns:
        ARIMAHyperparamEnv: Entorno configurado
    """
    return ARIMAHyperparamEnv(train_data, val_data, **kwargs)


def test_env():
    """
    Función de prueba del entorno con configuraciones aleatorias.
    """
    print("🧪 Probando entorno ARIMA...")
    
    # Datos sintéticos
    np.random.seed(42)
    train_data = np.random.randn(48) * 5 + 50
    val_data = np.random.randn(6) * 5 + 50
    
    # Crear entorno
    env = ARIMAHyperparamEnv(train_data, val_data)
    
    print(f"✅ Entorno creado")
    print(f"   Espacio de acciones: {env.action_space}")
    print(f"   Espacio de observación: {env.observation_space}")
    
    # Ejecutar episodio de prueba
    observation, info = env.reset()
    print(f"\n✅ Estado inicial: {observation}")
    
    for step in range(5):
        # Acción aleatoria
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"\n   Step {step+1}:")
        print(f"   Acción: p={action[0]}, d={action[1]}, q={action[2]}")
        print(f"   Recompensa: {reward:.4f}")
        print(f"   AIC: {info['metrics']['aic']:.2f}")
        
        if terminated or truncated:
            break
    
    # Mejor configuración
    best = env.get_best_config()
    print(f"\n✅ Mejor configuración encontrada:")
    print(f"   (p, d, q) = {best['config']}")
    print(f"   AIC: {best['aic']:.2f}")
    print(f"   RMSE: {best['rmse']:.2f}")
    
    print("\n✅ Prueba completada!")


if __name__ == "__main__":
    test_env()
