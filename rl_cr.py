import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor  # ← ДОБАВЬТЕ ЭТО
import matplotlib.pyplot as plt
from rl_env import TradingEnv
# Загрузка данных
data_path = "archive/M15/BTCUSDT_M15.csv"
df = pd.read_csv(data_path)

# Разделение на тренировочную и тестовую выборки
split_ratio = 0.8
split_index = int(len(df) * split_ratio)

train_df = df[:split_index].reset_index(drop=True)
test_df = df[split_index:].reset_index(drop=True)

print(f"Тренировочных данных: {len(train_df)}")
print(f"Тестовых данных: {len(test_df)}")

# Создание среды для обучения
train_env = DummyVecEnv([lambda: TradingEnv(train_df)])

# Создание среды для валидации с Monitor wrapper
eval_env = DummyVecEnv([lambda: Monitor(TradingEnv(test_df))])  # ← ИЗМЕНИТЕ ЭТУ СТРОКУ

# ============================================================================
# ОБУЧЕНИЕ PPO
# ============================================================================

print("=== Обучение PPO ===")

# Создание модели PPO
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./tensorboard_logs/",
    verbose=1,
    device='auto'
)

# Callback для валидации во время обучения
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    render=False,
    n_eval_episodes=3
)

# Обучение модели
print("Начинаем обучение...")
model.learn(
    total_timesteps=100000,
    callback=eval_callback,
    tb_log_name="PPO_training",
    progress_bar=True
)

# Сохранение модели
model.save("trading_ppo_model")
print("Модель сохранена как 'trading_ppo_model'")

# ============================================================================
# ОЦЕНКА МОДЕЛИ
# ============================================================================

print("=== Оценка модели ===")

# Загрузка лучшей модели (если использовался callback)
try:
    model = PPO.load("./best_model/best_model")
    print("Загружена лучшая модель из валидации")
except:
    print("Используется последняя обученная модель")

# Создание тестовой среды с Monitor wrapper
test_env = DummyVecEnv([lambda: Monitor(TradingEnv(test_df))])  # ← ИЗМЕНИТЕ ЭТУ СТРОКУ

# Оценка модели
mean_reward, std_reward = evaluate_policy(
    model, 
    test_env, 
    n_eval_episodes=5,
    deterministic=True
)

print(f"Средняя награда: {mean_reward:.2f} +/- {std_reward:.2f}")

# ============================================================================
# ТЕСТИРОВАНИЕ НА ТЕСТОВЫХ ДАННЫХ
# ============================================================================

def test_model(model, test_df, num_episodes=3):
    """Тестирование модели на нескольких эпизодах"""
    
    env = TradingEnv(test_df)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
        final_balance = info['balance']
        profit_loss = ((final_balance - env.initial_balance) / env.initial_balance) * 100
        
        print(f"Эпизод {episode + 1}:")
        print(f"  Шагов: {steps}")
        print(f"  Общая награда: {total_reward:.2f}")
        print(f"  Финальный баланс: {final_balance:.2f}")
        print(f"  Прибыль/убыток: {profit_loss:.2f}%")
        print("-" * 50)

print("=== Тестирование модели ===")
test_model(model, test_df)

# Закрытие сред
train_env.close()
eval_env.close()
test_env.close()
