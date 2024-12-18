from data_generator import DataGenerator
from exchange import Exchange
from strategy import SimpleMAStrategy, RandomStrategy, RLStrategy, OscillationStrategy
from logger import BacktestLogger
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

class Backtest:
    # start_date: 开始日期
    # end_date: 结束日期
    # stock_codes: 股票代码
    # strategy_type: 策略类型
    def __init__(self, start_date, end_date, stock_codes, strategy_type='random'):
        self.data_generator = DataGenerator(start_date, end_date, stock_codes)
        self.historical_data = self.data_generator.generate_historical_data()
        self.exchange = Exchange(self.historical_data)
        if strategy_type == 'random':
            self.strategy = RandomStrategy(self.exchange)
        elif strategy_type == 'simple_ma':
            self.strategy = SimpleMAStrategy(self.exchange)
        elif strategy_type == 'rl':
            self.strategy = RLStrategy(self.exchange)
        elif strategy_type == 'oscillation':
            self.strategy = OscillationStrategy(self.exchange)
        else:
            raise ValueError(f"Invalid strategy type: {strategy_type}")
        self.logger = BacktestLogger()
        self.portfolio_values = []  # 存储每日组合价值
        self.dates = []            # 存储对应的日期
        
    def run(self):
        """运行回测"""
        for date in self.historical_data[list(self.historical_data.keys())[0]]['date']:
            for code in self.historical_data.keys():
                # 使用策略生成交易信号
                signal = self.strategy.generate_signals(code, date)
                
                if signal is not None:
                    direction, volume = signal
                    if self.exchange.execute_order(code, date, direction, volume):
                        self.logger.log_trade(
                            date, code, direction, volume,
                            self.exchange.get_price(code, date)
                        )
            
            # 计算每日组合价值
            portfolio_value = self.exchange.cash
            for code, position in self.exchange.positions.items():
                portfolio_value += position * self.exchange.get_price(code, date)
            
            # 记录日期和组合价值
            self.dates.append(date)
            self.portfolio_values.append(portfolio_value)
            self.logger.log_portfolio(date, portfolio_value)
    
    def plot_results(self, save_path=None):
        """绘制回测结果"""
        plt.figure(figsize=(12, 6))
        
        # 转换日期格式
        dates = [pd.to_datetime(date) for date in self.dates]
        
        # 绘制投资组合价值曲线
        plt.plot(dates, self.portfolio_values, label='Portfolio Value', color='blue')
        
        # 计算基准线（初始资金）
        plt.axhline(y=self.exchange.initial_cash, color='r', linestyle='--', 
                   label='Initial Capital')
        
        # 设置图表格式
        plt.title('Backtest Results', fontsize=12)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Portfolio Value', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 旋转x轴日期标签
        plt.xticks(rotation=45)
        
        # 计算收益统计
        total_return = ((self.portfolio_values[-1] - self.exchange.initial_cash) 
                       / self.exchange.initial_cash * 100)
        max_value = max(self.portfolio_values)
        min_value = min(self.portfolio_values)
        
        # 添加统计信息文本框
        stats_text = f'Total Return: {total_return:.2f}%\n'
        stats_text += f'Max Value: {max_value:,.2f}\n'
        stats_text += f'Min Value: {min_value:,.2f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()  # 自动调整布局
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def get_statistics(self):
        """计算回测统计指标"""
        initial_value = self.exchange.initial_cash
        final_value = self.portfolio_values[-1]
        
        # 计算收益率
        total_return = (final_value - initial_value) / initial_value
        
        # 计算每日收益率
        daily_returns = [(self.portfolio_values[i] - self.portfolio_values[i-1]) 
                        / self.portfolio_values[i-1] 
                        for i in range(1, len(self.portfolio_values))]
        
        # 计算统计指标
        stats = {
            'Total Return (%)': total_return * 100,
            'Annual Return (%)': (total_return * 365 / len(self.dates)) * 100,
            'Max Drawdown (%)': self._calculate_max_drawdown() * 100,
            'Volatility (%)': np.std(daily_returns) * np.sqrt(252) * 100,
            'Sharpe Ratio': np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252),
        }
        
        return stats
    
    def _calculate_max_drawdown(self):
        """计算最大回撤"""
        peak = self.portfolio_values[0]
        max_drawdown = 0
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown