import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import pandas as pd
from rl_env import TradingEnv

# Быстрая загрузка данных
data_path = "archive/M15/BTCUSDT_M15.csv"
df = pd.read_csv(data_path)
split_index = int(len(df) * 0.8)
train_df = df[:split_index].reset_index(drop=True)
test_df = df[split_index:].reset_index(drop=True)

# Создание сред
train_env = DummyVecEnv([lambda: TradingEnv(train_df)])
eval_env = DummyVecEnv([lambda: Monitor(TradingEnv(test_df))])

# Автоматический поиск последней модели
model_path = None
for path in ["./best_model/best_model.zip", "trading_ppo_model.zip"]:
    if os.path.exists(path):
        model_path = path
        break

if model_path:
    print(f"🔄 Продолжаем обучение модели: {model_path}")
    
    model = PPO.load(model_path, env=train_env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_continued/",
        eval_freq=5000,
        n_eval_episodes=3
    )
    
    model.learn(
        total_timesteps=50000,
        callback=eval_callback,
        reset_num_timesteps=False,
        progress_bar=True
    )
    
    model.save("trading_ppo_model_continued")
    print("✅ Обучение продолжено!")
else:
    print("❌ Не найдена модель для продолжения")
