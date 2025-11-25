import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from rl_env import TradingEnv

# ----------------------------
# Настройки (меняй по необходимости)
# ----------------------------
BEST_MODEL_PATH = "./best_model/best_model.zip"      # <-- должна существовать заранее
TMP_MODEL_PATH  = "./best_model/_candidate_model.zip"
BACKUP_MODEL_PATH = "./best_model/_best_backup.zip"
TRAIN_CSV = "archive/M15/BTCUSDT_M15.csv"
TRAIN_SPLIT = 0.8
SEED = 45
N_EVAL_EPISODES = 3     # сколько эпизодов для оценки (больше — более точная оценка)
TOTAL_TIMESTEPS = 100_000
N_ITERATIONS = 100        # сколько раз повторить цикл "train->eval->compare"
VERBOSE = 0

# ----------------------------
# Вспомогательные функции
# ----------------------------
def make_vec_env_from_df(df, seed=0):
    """Возвращает DummyVecEnv обёрнутый Monitor'ом."""
    def _init():
        return Monitor(TradingEnv(df))
    env = DummyVecEnv([_init])
    env.seed(seed)
    return env

def evaluate(model, eval_env, n_eval_episodes=N_EVAL_EPISODES):
    """
    Оценка политики на eval_env.
    Возвращает mean_reward, std_reward.
    """
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True, render=False)
    return mean_reward, std_reward

def atomic_replace(src_path, dst_path):
    """
    Атомарная замена файла: записываем временно, затем os.replace.
    """
    os.replace(src_path, dst_path)

# ----------------------------
# Главная логика
# ----------------------------
def main(
    best_model_path=BEST_MODEL_PATH,
    train_csv=TRAIN_CSV,
    train_split=TRAIN_SPLIT,
    total_timesteps=TOTAL_TIMESTEPS,
    n_iterations=N_ITERATIONS,
    n_eval_episodes=N_EVAL_EPISODES,
    seed=SEED
):
    # Проверки
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Лучшей модели не найдено по пути '{best_model_path}'. Поместите файл и повторите запуск.")

    # Загрузка данных и разбиение
    df = pd.read_csv(train_csv)
    split_index = int(len(df) * train_split)
    train_df = df[:split_index].reset_index(drop=True)
    val_df   = df[split_index:].reset_index(drop=True)

    # Создаём env'ы
    train_env = make_vec_env_from_df(train_df, seed=seed)
    eval_env  = make_vec_env_from_df(val_df, seed=seed+1)  # отдельный сид для валидации

    # Загружаем одну лучшую модель — **один раз** (но мы будем подменять её в памяти при rollback/save)
    print("🔁 Загружаем существующую лучшую модель один раз...")
    model = PPO.load(best_model_path, env=train_env)   # сразу привязываем к train_env
    model.set_random_seed(seed)

    # Сохраним копию лучшей модели на случай отката
    shutil.copy2(best_model_path, BACKUP_MODEL_PATH)

    # 1) Оценим baseline-reward **только один раз**:
    print("📊 Оцениваем baseline (initial) reward на валидации (один запуск)...")
    mean_reward_best, std_reward_best = evaluate(model, eval_env, n_eval_episodes=n_eval_episodes)
    print(f"Baseline mean reward: {mean_reward_best:.6f} ± {std_reward_best:.6f}")

    # Математическое правило сравнения:
    # Считаем улучшение Δ = R_new - R_best.
    # Если Δ > 0  => новое лучше => сохраняем.
    # (Можно поменять критерий на статистический тест, но сейчас простой детерминированный критерий.)
    # Формула процента улучшения: 100 * Δ / |R_best|  (если R_best != 0).

    for it in range(1, n_iterations + 1):
        print("\n" + "="*60)
        print(f"Итерация {it}/{n_iterations}: train {total_timesteps} шагов")
        print("="*60)

        # 2) Train: создаём candidate копию модели, чтобы не потерять текущую модель в памяти при ошибке обучения
        #    (мы используем .save/.load для атомарности)
        # Сохраним текущую модель во временный файл (чтобы иметь candidate стартовую точку)
        candidate_start_path = TMP_MODEL_PATH
        print("💾 Сохраняем текущую модель в качестве стартовой точки для дообучения...")
        model.save(candidate_start_path)

        # Загружаем candidate (новый объект) и привязываем train_env
        candidate = PPO.load(candidate_start_path, env=train_env,verbose = 0)
        candidate.set_random_seed(seed + it)  # немного изменить сид для вариативности обучения

        # 3) Обучаем candidate
        print(f"⚙️  Обучение candidate модели: {total_timesteps} шагов...")
        candidate.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, progress_bar=True)

        # 4) Оцениваем candidate на валидации — **единственная оценка** в этой итерации
        print("📈 Оцениваем candidate на валидации...")
        mean_reward_candidate, std_reward_candidate = evaluate(candidate, eval_env, n_eval_episodes=n_eval_episodes)
        print(f"Candidate mean reward: {mean_reward_candidate:.6f} ± {std_reward_candidate:.6f}")

        # 5) Сравнение
        delta = mean_reward_candidate - mean_reward_best
        # процент улучшения (защита от деления на ноль)
        pct = (100.0 * delta / abs(mean_reward_best)) if mean_reward_best != 0 else float('inf') if delta>0 else -float('inf')

        print(f"Δ = R_candidate - R_best = {delta:.6f} ( {pct:.4f}% )")

        if delta > 0:
            # candidate лучше — сохраняем как новый лучший (атомарно заменяем файл)
            print("🏆 Candidate лучше. Сохраняем его как новую лучшую модель...")
            # Сохраняем временно в файл, затем атомарно replace
            candidate.save(TMP_MODEL_PATH)
            atomic_replace(TMP_MODEL_PATH, best_model_path)
            # обновим in-memory модель и best reward
            model = PPO.load(best_model_path, env=train_env)
            mean_reward_best = mean_reward_candidate
            std_reward_best = std_reward_candidate
            print(f"✅ Новая лучшая модель сохранена: {best_model_path}")
        else:
            # candidate не лучше — не сохраняем; восстанавливаем старую лучшую модель в памяти
            print("❌ Candidate хуже или равен. Откатываемся к лучшей модели в памяти (без перезаписи файла).")
            model = PPO.load(best_model_path, env=train_env)
            # mean_reward_best остаётся прежним

        # краткая сводка по итерации
        print(f"Итерация {it} завершена. Текущий best mean reward = {mean_reward_best:.6f}")

    # Итог
    print("\n" + "="*40)
    print("Готово. Финальная лучшая модель:")
    print(f" -> path: {best_model_path}")
    print(f" -> mean reward: {mean_reward_best:.6f} ± {std_reward_best:.6f}")
    print("="*40)

if __name__ == "__main__":
    main()
