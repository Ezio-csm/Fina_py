import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class DataGenerator:
    def __init__(self, start_date, end_date, stock_codes, data_dir='data', random_seed=42):
        self.start_date = start_date
        self.end_date = end_date
        self.stock_codes = stock_codes
        self.data_dir = data_dir
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 创建数据存储目录
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def generate_historical_data(self):
        """生成模拟历史数据"""
        date_range = pd.date_range(self.start_date, self.end_date, freq='D')
        data = {}
        
        for code in self.stock_codes:
            # 生成随机价格数据
            prices = np.random.random(len(date_range)) * 10 + 30
            # 生成成交量数据
            volumes = np.random.random(len(date_range)) * 1000000
            
            df = pd.DataFrame({
                'date': date_range,
                'open': prices,
                'high': prices * 1.02,
                'low': prices * 0.98,
                'close': prices * 1.01,
                'volume': volumes
            })
            data[code] = df
            
            # 保存数据到CSV文件
            self.save_data(code, df)
            
        return data
    
    def save_data(self, code, df):
        """将数据保存为CSV文件"""
        file_path = os.path.join(self.data_dir, f"{code}.csv")
        df.to_csv(file_path, index=False)
        
    def load_data(self, code):
        """从CSV文件加载数据"""
        file_path = os.path.join(self.data_dir, f"{code}.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path, parse_dates=['date'])
        return None
    
    def load_all_data(self):
        """加载所有股票的历史数据"""
        data = {}
        for code in self.stock_codes:
            df = self.load_data(code)
            if df is not None:
                data[code] = df
        return data