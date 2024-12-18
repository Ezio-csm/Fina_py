import logging
from datetime import datetime

class BacktestLogger:
    def __init__(self, log_file='backtest.log'):
        self.logger = logging.getLogger('backtest')
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_trade(self, date, code, direction, volume, price):
        """记录交易信息"""
        message = f"交易执行: 日期={date}, 股票={code}, " \
                 f"方向={direction}, 数量={volume}, 价格={price:.2f}"
        self.logger.info(message)
    
    def log_portfolio(self, date, portfolio_value):
        """记录组合价值"""
        self.logger.info(f"组合价值: 日期={date}, 价值={portfolio_value:.2f}")