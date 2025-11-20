#!/usr/bin/env python3
# ============================================================================
# Proyecto: Agentificación de Modelos ARIMA con Aprendizaje Reforzado
# Archivo: rl_agent.py
# Descripción: Agente DQN con Stable-Baselines3 para optimización ARIMA
# ============================================================================

import os
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import torch

from src.arima_env import ARIMAHyperparamEnv
from src.data_processor import TimeSeriesProcessor


class ARIMAAgent:
    """
    Agente de Aprendizaje Reforzado para optimización de hiperparámetros ARIMA.
    Usa algoritmo DQN (Deep Q-Network) de Stable-Baselines3.
    """
    
    def __init__(self, train_data, val_data, config=None):
        """
        Inicializa el agente RL.
        
        Args:
            train_data: Datos de entrenamiento (numpy array)
            val_data: Datos de validación (numpy array)
            config: Dict con configuración del agente y entorno
        """
        self.train_data = train_data
        self.val_data = val_data
        
        # Configuración por defecto
        if config is None:
            self.config = {
                'p_max': 5,
                'd_max': 2,
                'q_max': 4,
                'max_steps': 50,
                'learning_rate': 1e-4,
                'buffer_size': 10000,
                'learning_starts': 100,
                'batch_size': 32,
                'tau': 1.0,
                'gamma': 0.99,
                'train_freq': 4,
                'gradient_steps': 1,
                'target_update_interval': 1000,
                'exploration_fraction': 0.3,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.05,
                'policy_kwargs': {
                    'net_arch': [128, 128]
                }
            }
        else:
            self.config = config
        
        # Crear entorno
        self.env = self._create_env()
        
        # Modelo (será inicializado en train o load)
        self.model = None
        
    def _create_env(self):
        """
        Crea y configura el entorno ARIMA.
        
        Returns:
            Monitor: Entorno wrapeado con Monitor
        """
        env = ARIMAHyperparamEnv(
            train_data=self.train_data,
            val_data=self.val_data,
            p_max=self.config['p_max'],
            d_max=self.config['d_max'],
            q_max=self.config['q_max'],
            max_steps=self.config['max_steps']
        )
        
        # Verificar entorno
        try:
            check_env(env, warn=True)
            print("✅ Entorno verificado correctamente")
        except Exception as e:
            print(f"⚠️  Advertencia en verificación de entorno: {e}")
        
        # Wrapear con Monitor para tracking
        env = Monitor(env)
        
        return env
    
    def train(self, total_timesteps=50000, save_path='models/arima_dqn_agent', 
              tensorboard_log='models/tensorboard_logs', save_freq=5000):
        """
        Entrena el agente DQN.
        
        Args:
            total_timesteps: Número total de timesteps de entrenamiento
            save_path: Ruta para guardar el modelo entrenado
            tensorboard_log: Directorio para logs de TensorBoard
            save_freq: Frecuencia de guardado de checkpoints
        """
        print("\n" + "=" * 80)
        print("🚀 INICIANDO ENTRENAMIENTO DEL AGENTE RL")
        print("=" * 80)
        
        print(f"\n⚙️  Configuración:")
        print(f"   Total timesteps: {total_timesteps}")
        print(f"   Learning rate: {self.config['learning_rate']}")
        print(f"   Buffer size: {self.config['buffer_size']}")
        print(f"   Exploration fraction: {self.config['exploration_fraction']}")
        print(f"   Network architecture: {self.config['policy_kwargs']['net_arch']}")
        
        # Crear directorios
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(tensorboard_log, exist_ok=True)
        
        # Crear modelo DQN
        self.model = DQN(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=self.config['learning_rate'],
            buffer_size=self.config['buffer_size'],
            learning_starts=self.config['learning_starts'],
            batch_size=self.config['batch_size'],
            tau=self.config['tau'],
            gamma=self.config['gamma'],
            train_freq=self.config['train_freq'],
            gradient_steps=self.config['gradient_steps'],
            target_update_interval=self.config['target_update_interval'],
            exploration_fraction=self.config['exploration_fraction'],
            exploration_initial_eps=self.config['exploration_initial_eps'],
            exploration_final_eps=self.config['exploration_final_eps'],
            policy_kwargs=self.config['policy_kwargs'],
            tensorboard_log=tensorboard_log,
            verbose=1,
            device='auto'
        )
        
        print(f"\n✅ Modelo DQN creado")
        print(f"   Device: {self.model.device}")
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.dirname(save_path),
            name_prefix='arima_dqn_checkpoint'
        )
        
        callbacks = [checkpoint_callback]
        
        # Entrenar
        print(f"\n🎓 Entrenando agente...")
        print(f"   (Progreso visible en TensorBoard: tensorboard --logdir {tensorboard_log})")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100,
            progress_bar=True
        )
        
        # Guardar modelo final
        self.model.save(save_path)
        print(f"\n✅ Modelo guardado en: {save_path}.zip")
        
        print("\n" + "=" * 80)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("=" * 80)
        
        return self.model
    
    def load(self, model_path='models/arima_dqn_agent.zip'):
        """
        Carga un modelo previamente entrenado.
        
        Args:
            model_path: Ruta al modelo guardado
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        print(f"📂 Cargando modelo desde: {model_path}")
        
        self.model = DQN.load(model_path, env=self.env)
        
        print("✅ Modelo cargado exitosamente")
        
        return self.model
    
    def predict_best_config(self, deterministic=True):
        """
        Predice la mejor configuración ARIMA usando el agente entrenado.
        
        Args:
            deterministic: Si usar política determinística (sin exploración)
            
        Returns:
            tuple: (p, d, q) predicho
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado/cargado. Ejecute train() o load() primero.")
        
        # Reset del entorno
        obs, _ = self.env.reset()
        
        # Predecir acción
        action, _states = self.model.predict(obs, deterministic=deterministic)
        
        p, d, q = int(action[0]), int(action[1]), int(action[2])
        
        print(f"\n🤖 Agente RL predice configuración óptima:")
        print(f"   (p, d, q) = ({p}, {d}, {q})")
        
        return (p, d, q)
    
    def evaluate(self, n_episodes=10):
        """
        Evalúa el agente entrenado.
        
        Args:
            n_episodes: Número de episodios de evaluación
            
        Returns:
            dict: Estadísticas de evaluación
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado/cargado.")
        
        print(f"\n📊 Evaluando agente en {n_episodes} episodios...")
        
        episode_rewards = []
        episode_aic = []
        episode_configs = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            # Guardar estadísticas del episodio
            best = self.env.get_best_config()
            episode_rewards.append(episode_reward)
            episode_aic.append(best['aic'])
            episode_configs.append(best['config'])
        
        # Calcular estadísticas
        stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_aic': np.mean(episode_aic),
            'std_aic': np.std(episode_aic),
            'best_aic': min(episode_aic),
            'best_config': episode_configs[np.argmin(episode_aic)]
        }
        
        print(f"\n📈 Resultados de evaluación:")
        print(f"   Recompensa promedio: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"   AIC promedio: {stats['mean_aic']:.2f} ± {stats['std_aic']:.2f}")
        print(f"   Mejor AIC: {stats['best_aic']:.2f}")
        print(f"   Mejor configuración: {stats['best_config']}")
        
        return stats


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def train_agent_from_file(data_path='data/germany_monthly_power.csv',
                          timesteps=50000,
                          output_dir='models'):
    """
    Entrena agente RL desde archivo CSV.
    
    Args:
        data_path: Ruta al archivo de datos
        timesteps: Número de timesteps de entrenamiento
        output_dir: Directorio de salida para modelos
    """
    # Cargar y preparar datos
    print("📂 Cargando datos...")
    processor = TimeSeriesProcessor(data_path)
    processor.load_data()
    processor.split_data()
    
    train_data = processor.train['value'].values
    val_data = processor.val['value'].values
    
    print(f"✅ Datos cargados: {len(train_data)} train, {len(val_data)} val")
    
    # Crear y entrenar agente
    agent = ARIMAAgent(train_data, val_data)
    
    save_path = os.path.join(output_dir, 'arima_dqn_agent')
    tensorboard_log = os.path.join(output_dir, 'tensorboard_logs')
    
    agent.train(
        total_timesteps=timesteps,
        save_path=save_path,
        tensorboard_log=tensorboard_log
    )
    
    # Evaluar agente
    print("\n🧪 Evaluando agente entrenado...")
    stats = agent.evaluate(n_episodes=5)
    
    # Guardar estadísticas
    stats_file = os.path.join(output_dir, 'training_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("Estadísticas de Entrenamiento del Agente RL\n")
        f.write("=" * 60 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"✅ Estadísticas guardadas en: {stats_file}")
    
    return agent


def main():
    """
    Función principal para entrenamiento desde línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description='Entrenar agente RL para optimización de hiperparámetros ARIMA'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Entrenar un nuevo agente'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/germany_monthly_power.csv',
        help='Ruta al archivo de datos CSV'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=50000,
        help='Número de timesteps de entrenamiento'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directorio de salida para modelos'
    )
    
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Evaluar agente existente'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/arima_dqn_agent.zip',
        help='Ruta al modelo para evaluación'
    )
    
    args = parser.parse_args()
    
    if args.train:
        # Entrenar nuevo agente
        agent = train_agent_from_file(
            data_path=args.data,
            timesteps=args.timesteps,
            output_dir=args.output_dir
        )
        
        # Predecir mejor configuración
        best_config = agent.predict_best_config()
        print(f"\n🎯 Configuración recomendada: (p, d, q) = {best_config}")
        
    elif args.eval:
        # Evaluar agente existente
        print("📂 Cargando datos para evaluación...")
        processor = TimeSeriesProcessor(args.data)
        processor.load_data()
        processor.split_data()
        
        train_data = processor.train['value'].values
        val_data = processor.val['value'].values
        
        # Crear agente y cargar modelo
        agent = ARIMAAgent(train_data, val_data)
        agent.load(args.model_path)
        
        # Evaluar
        stats = agent.evaluate(n_episodes=10)
        
        # Predecir mejor configuración
        best_config = agent.predict_best_config()
        print(f"\n🎯 Configuración recomendada: (p, d, q) = {best_config}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
