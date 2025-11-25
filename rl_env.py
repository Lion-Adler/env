import numpy as np
import pandas as pd
from gymnasium import spaces
import gymnasium as gym








class TradingEnv(gym.Env):
    """
    Торговая среда с логарифмической нормализацией.
    
    Observation: (window_size, 5) - скользящее окно OHLCV с нормализацией
    Action: 0=Hold, 1=Long, 2=Short
    Reward: процент прибыли с учётом комиссий
    """
    
    metadata = {"render_modes": ["console"]}
    
    # Действия
    HOLD = 0
    LONG = 1
    SHORT = 2
    
    def __init__(self, df, window_size=200, commission=0.001, initial_balance=10000.0):
        super(TradingEnv, self).__init__()
        
        # Параметры
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.commission = commission
        self.initial_balance = initial_balance
        self.epsilon = 1e-10
        
        # Пространства
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size* 5,), 
            dtype=np.float32
        )
        
        # Предвычисляем returns для всего датасета
        self._precompute_returns()
        
        # Состояние среды
        self.current_step = None
        self.position = None  # None, 'long', 'short'
        self.entry_price = None
        self.balance = None
        
    def _precompute_returns(self):
        """Вычисляем относительные изменения: r_t = (p_t - p_{t-1}) / p_{t-1}"""
        ohlcv = self.df[['open', 'high', 'low', 'close', 'volume']].values
        
        self.returns = np.zeros_like(ohlcv, dtype=np.float32)
        self.returns[1:] = (ohlcv[1:] - ohlcv[:-1]) / (ohlcv[:-1] + self.epsilon)
        self.returns[0] = 0
        
    def _get_observation(self, step):
        """
        Получаем нормализованное окно для текущего шага.
        
        Алгоритм:
        1. Берём окно returns[start:step+1]
        2. Находим max(|returns|) для каждого столбца
        3. Нормируем: normalized = returns / (max + ε)
        """
        if step < self.window_size:
            window_start = 0
        else:
            window_start = step - self.window_size + 1
            
        window_returns = self.returns[window_start:step + 1]
        
        # Максимальный модуль для каждого столбца (5 значений)
        max_abs = np.max(np.abs(window_returns), axis=0)
        
        # Нормализуем окно
        normalized_window = window_returns / (max_abs + self.epsilon)
        
        # Если окно меньше window_size, дополняем нулями сверху
        if normalized_window.shape[0] < self.window_size:
            padding = np.zeros((self.window_size - normalized_window.shape[0], 5), dtype=np.float32)
            normalized_window = np.vstack([padding, normalized_window])
            
        return normalized_window.flatten().astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        self.current_step = self.window_size
        self.position = None
        self.entry_price = None
        self.balance = self.initial_balance
        
        return self._get_observation(self.current_step), {}
    
    def step(self, action):
        """
    Выполняет один шаг в торговой среде.
    
    Логика:
    - Позиция удерживается только при явном повторении действия (LONG->LONG, SHORT->SHORT)
    - HOLD означает отсутствие позиции (позиция = None)
    - Reward начисляется только при закрытии позиции (переход между состояниями)
    - Баланс изменяется только при закрытии позиции
    
    Параметры:
    -----------
    action : int
        0 = HOLD (нет позиции)
        1 = LONG (длинная позиция)
        2 = SHORT (короткая позиция)
    
    Возвращает:
    -----------
    obs : np.ndarray
        Наблюдение (нормализованное окно OHLCV)
    reward : float
        Награда (PnL с учётом комиссий, начисляется только при закрытии)
    terminated : bool
        Достигнут ли конец эпизода
    truncated : bool
        Эпизод прерван досрочно (не используется здесь)
    info : dict
        Дополнительная информация о состоянии
        """
    
    # Получаем текущую цену закрытия
        current_price = self.df.loc[self.current_step, 'close']
    
    # Инициализируем награду (по умолчанию 0)
        reward = 0.0
    
    # ========================================================================
    # БЛОК 1: ОБРАБОТКА ДЕЙСТВИЯ LONG (action == 1)
    # ========================================================================
        if action == self.LONG:
        
            if self.position == 'long':
            # Случай 1.1: Уже в LONG и снова выбрали LONG
            # → Просто удерживаем позицию, никаких изменений
            # → Reward = 0, баланс не меняется
                pass
        
            elif self.position == 'short':
            # Случай 1.2: Были в SHORT, теперь выбрали LONG
            # → Закрываем SHORT-позицию
            # → Рассчитываем PnL для SHORT: (entry_price - exit_price) / entry_price
            # → Вычитаем комиссию за закрытие SHORT и открытие LONG (2 × commission)
            # → Обновляем баланс
            # → Открываем новую LONG-позицию
            
                pnl = (self.entry_price - current_price) / self.entry_price
                reward = pnl - 2 * self.commission  # Закрытие SHORT + открытие LONG
            
            # Обновляем баланс (только здесь!)
                self.balance += ( reward)
            
            # Открываем LONG
                self.position = 'long'
                self.entry_price = current_price
        
            elif self.position is None:
            # Случай 1.3: Не было позиции, открываем LONG
            # → Вычитаем комиссию за открытие
            # → Баланс не меняется (позиция только открылась)
            
                reward = -self.commission
                self.balance += (reward)
            
                self.position = 'long'
                self.entry_price = current_price
    
    # ========================================================================
    # БЛОК 2: ОБРАБОТКА ДЕЙСТВИЯ SHORT (action == 2)
    # ========================================================================
        elif action == self.SHORT:
        
            if self.position == 'short':
            # Случай 2.1: Уже в SHORT и снова выбрали SHORT
            # → Просто удерживаем позицию, никаких изменений
            # → Reward = 0, баланс не меняется
                pass
        
            elif self.position == 'long':
            # Случай 2.2: Были в LONG, теперь выбрали SHORT
            # → Закрываем LONG-позицию
            # → Рассчитываем PnL для LONG: (exit_price - entry_price) / entry_price
            # → Вычитаем комиссию за закрытие LONG и открытие SHORT (2 × commission)
            # → Обновляем баланс
            # → Открываем новую SHORT-позицию
            
                pnl = (current_price - self.entry_price) / self.entry_price
                reward = pnl - 2 * self.commission  # Закрытие LONG + открытие SHORT
            
            # Обновляем баланс (только здесь!)
                self.balance += (reward)
            
            # Открываем SHORT
                self.position = 'short'
                self.entry_price = current_price
        
            elif self.position is None:
            # Случай 2.3: Не было позиции, открываем SHORT
            # → Вычитаем комиссию за открытие
            # → Баланс не меняется (позиция только открылась)
            
                reward = -self.commission
                self.balance += ( reward)
            
                self.position = 'short'
                self.entry_price = current_price
    
    # ========================================================================
    # БЛОК 3: ОБРАБОТКА ДЕЙСТВИЯ HOLD (action == 0)
    # ========================================================================
        elif action == self.HOLD:
        
            if self.position == 'long':
            # Случай 3.1: Были в LONG, выбрали HOLD
            # → Закрываем LONG-позицию
            # → Рассчитываем PnL для LONG
            # → Вычитаем комиссию за закрытие (1 × commission)
            # → Обновляем баланс
            # → Переходим в состояние без позиции (position = None)
            
                pnl = (current_price - self.entry_price) / self.entry_price
                reward = pnl - self.commission  # Только закрытие LONG
            
            # Обновляем баланс
                self.balance += (reward)
            
            # Закрываем позицию
                self.position = None
                self.entry_price = None
        
            elif self.position == 'short':
            # Случай 3.2: Были в SHORT, выбрали HOLD
            # → Закрываем SHORT-позицию
            # → Рассчитываем PnL для SHORT
            # → Вычитаем комиссию за закрытие (1 × commission)
            # → Обновляем баланс
            # → Переходим в состояние без позиции (position = None)
            
                pnl = (self.entry_price - current_price) / self.entry_price
                reward = pnl - self.commission  # Только закрытие SHORT
            
            # Обновляем баланс
                self.balance += (reward)
            
            # Закрываем позицию
                self.position = None
                self.entry_price = None
        
            elif self.position is None:
            # Случай 3.3: Не было позиции и выбрали HOLD
            # → Ничего не делаем
            # → Reward = 0, баланс не меняется
                pass
    
    # ========================================================================
    # БЛОК 4: ПЕРЕХОД К СЛЕДУЮЩЕМУ ШАГУ
    # ========================================================================
    
    # Двигаемся на следующий временной шаг
        self.current_step += 1
    
    # Проверяем, достигли ли конца данных
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
    
    # Получаем новое наблюдение
        if terminated:
        # Если эпизод завершён, возвращаем нулевое наблюдение
            obs = np.zeros((self.window_size* 5), dtype=np.float32)
        else:
        # Иначе получаем нормализованное окно для нового шага
            obs = self._get_observation(self.current_step)
    
    # Формируем дополнительную информацию
        info = {
        'balance': self.balance,
        'position': self.position,
        'price': current_price,
        'step': self.current_step
        }
    
    # ========================================================================
    # ВОЗВРАТ
    # ========================================================================
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "console":
            print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}")
    
    def close(self):
        pass





from stable_baselines3.common.env_checker import check_env

data_path = "archive/M15/BTCUSDT_M15.csv"
df = pd.read_csv(data_path)
env = TradingEnv(df)
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)
