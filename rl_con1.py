import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import pandas as pd
from rl_env import TradingEnv

# ===============================
# 1. Загрузка данных
# ===============================
data_path = "archive/M15/BTCUSDT_M15.csv"
df = pd.read_csv(data_path)
split_index = int(len(df) * 0.8)

train_df = df[:split_index].reset_index(drop=True)
test_df = df[split_index:].reset_index(drop=True)

# ===============================
# 2. Создание сред
# ===============================
train_env = DummyVecEnv([lambda: TradingEnv(train_df)])
eval_env = DummyVecEnv([lambda: Monitor(TradingEnv(test_df))])

# ===============================
# 3. Ищем ТОЛЬКО лучшую модель
# ===============================
best_model_path = "./best_model/best_model.zip"

if os.path.exists(best_model_path):
    print(f"🔄 Продолжаем обучение ЛУЧШЕЙ модели: {best_model_path}")

    # Загружаем лучшую модель
    model = PPO.load(best_model_path, env=train_env)

    # Новый EvalCallback для продолжения
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_continued/",
        eval_freq=5000,
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )

    # Продолжаем обучение ЛУЧШЕЙ модели
    model.learn(
        total_timesteps=50000,
        callback=eval_callback,
        reset_num_timesteps=False,   # <── ВАЖНО!
        progress_bar=True
    )

    # Сохраняем последнюю (не лучшую)
    model.save("trading_ppo_model_continued")
    print("✅ Дообучение лучшей модели завершено!")

else:
    print("❌ Лучшая модель не найдена. Запусти первоначальное обучение.")
