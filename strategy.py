from abc import ABC, abstractmethod
import numpy as np
class Strategy(ABC):
    def __init__(self, exchange, random_seed=42):
        self.exchange = exchange
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
    
    @abstractmethod
    def generate_signals(self, date):
        """生成交易信号"""
        pass

# 随机策略
class RandomStrategy(Strategy):
    def __init__(self, exchange, random_seed=42):
        super().__init__(exchange, random_seed)
    
    def generate_signals(self, code, date):
        buyorsell = np.random.randint(0, 3)
        volume = np.random.randint(1, 3000)
        if buyorsell == 0:
            return 'buy', volume
        elif buyorsell == 1:
            return 'sell', volume
        else:
            return None

# 均线交叉策略
class SimpleMAStrategy(Strategy):
    def __init__(self, exchange, short_window=5, long_window=20, volume=3000):
        super().__init__(exchange)
        self.short_window = short_window
        self.long_window = long_window
        self.volume = volume
    
    def generate_signals(self, code, date):
        data = self.exchange.historical_data[code]
        

        current_idx = data[data['date'] == date].index[0]
        if current_idx < self.long_window + 1:
            return None
            
        # 只使用到当前日期之前的数据
        data_until_yesterday = data.iloc[:current_idx]
        short_ma = data_until_yesterday['close'].rolling(self.short_window).mean()
        long_ma = data_until_yesterday['close'].rolling(self.long_window).mean()
        
        current_short_ma = short_ma.iloc[-1]
        current_long_ma = long_ma.iloc[-1]
        prev_short_ma = short_ma.iloc[-2]
        prev_long_ma = long_ma.iloc[-2]
        
        if current_short_ma > current_long_ma and prev_short_ma <= prev_long_ma:
            return 'buy', self.volume
        elif current_short_ma < current_long_ma and prev_short_ma >= prev_long_ma:
            return 'sell', self.volume
            
        return None

# 震荡行情套利策略
class OscillationStrategy(Strategy):
    def __init__(self, exchange, window=20, std_dev_threshold=0.7, volume=3000, random_seed=42):
        super().__init__(exchange, random_seed)
        self.window = window
        self.std_dev_threshold = std_dev_threshold
        self.volume = volume
        
    def generate_signals(self, code, date):
        data = self.exchange.historical_data[code]
        current_idx = data[data['date'] == date].index[0]
        
        if current_idx < self.window + 1:
            return None
            
        hist_data = data.iloc[:current_idx]
        prices = hist_data['close'].values
        
        rolling_mean = hist_data['close'].rolling(window=self.window).mean()
        rolling_std = hist_data['close'].rolling(window=self.window).std()
        
        current_price = prices[-1]
        current_mean = rolling_mean.iloc[-1]
        current_std = rolling_std.iloc[-1]
        
        z_score = (current_price - current_mean) / current_std if current_std > 0 else 0
        
        current_position = self.exchange.get_position(code)
        
        if z_score > self.std_dev_threshold:  # 价格显著高于均值
            if current_position > 0:  # 有持仓就卖出
                return 'sell', min(self.volume, current_position)
            return None
            
        elif z_score < -self.std_dev_threshold:  # 价格显著低于均值
            return 'buy', self.volume
            
        return None
    
    def _calculate_bollinger_bands(self, prices, window=20):
        """计算布林带"""
        rolling_mean = pd.Series(prices).rolling(window=window).mean()
        rolling_std = pd.Series(prices).rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        return rolling_mean, upper_band, lower_band

# 强化学习策略
class RLStrategy(Strategy):
    def __init__(self, exchange, learning_rate=0.01, gamma=0.95, epsilon=0.1, 
                 state_size=6, random_seed=42):
        """
        参数:
            learning_rate: Q学习的学习率
            gamma: 折扣因子
            epsilon: epsilon-greedy策略的探索率
            state_size: 状态空间的特征数量
        """
        super().__init__(exchange, random_seed)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_size = state_size
        self.q_table = {}  # 状态-动作价值表
        self.actions = ['buy', 'sell', 'hold']
        
    def _get_state(self, code, date):
        """构建状态特征"""
        data = self.exchange.historical_data[code]
        current_idx = data[data['date'] == date].index[0]
        if current_idx < 5:  # 需要足够的历史数据
            return None
            
        # 提取特征
        prices = data['close'].values[current_idx-5:current_idx+1]
        volumes = data['volume'].values[current_idx-5:current_idx+1]
        
        # 计算技术指标
        price_change = (prices[-1] - prices[-2]) / prices[-2]  # 价格变化率
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2]  # 成交量变化率
        ma5 = prices.mean()  # 5日均价
        price_ma_ratio = prices[-1] / ma5  # 价格与均价的比率
        
        # 获取当前持仓情况
        position = self.exchange.get_position(code)
        total_assets = self.exchange.get_total_assets(date)
        position_ratio = position * prices[-1] / total_assets if total_assets > 0 else 0
        
        # 将特征离散化
        state = (
            self._discretize(price_change, 10),
            self._discretize(volume_change, 10),
            self._discretize(price_ma_ratio - 1, 10),
            self._discretize(position_ratio, 5)
        )
        return state
        
    def _discretize(self, value, bins):
        """将连续值离散化"""
        if np.isnan(value):
            return 0
        return int(np.clip(np.floor(value * bins), -bins + 1, bins - 1))
    
    def _get_q_value(self, state, action):
        """获取Q值"""
        return self.q_table.get((state, action), 0.0)
    
    def _update_q_value(self, state, action, reward, next_state):
        """更新Q值"""
        old_q = self._get_q_value(state, action)
        next_max_q = max(self._get_q_value(next_state, a) for a in self.actions)
        new_q = old_q + self.learning_rate * (reward + self.gamma * next_max_q - old_q)
        self.q_table[(state, action)] = new_q
    
    def generate_signals(self, code, date):
        """生成交易信号"""
        state = self._get_state(code, date)
        if state is None:
            return None
            
        # epsilon-greedy策略
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            q_values = [self._get_q_value(state, a) for a in self.actions]
            action = self.actions[np.argmax(q_values)]
        
        # 设定交易量
        if action in ['buy', 'sell']:
            volume = 100  # 可以根据实际情况调整
            return action, volume
        return None
    
    def update(self, code, date, action, next_date, reward):
        """更新策略（训练）"""
        state = self._get_state(code, date)
        next_state = self._get_state(code, next_date)
        
        if state is not None and next_state is not None:
            self._update_q_value(state, action, reward, next_state)