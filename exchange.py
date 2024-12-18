import pandas as pd

class Exchange:
    """交易系统"""
    # historical_data: 历史数据
    # initial_cash: 初始现金
    # max_hold_position: 最大持仓量
    # rebalance_days: 调仓周期
    # max_position_ratio: 最大持仓比例
    # commission_rate: 手续费率
    def __init__(self, historical_data, initial_cash=1000000, max_hold_position=300000,
                 rebalance_days=1, max_position_ratio=0.3, commission_rate=0.001):
        self.historical_data = historical_data
        self.positions = {}
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.max_hold_position = max_hold_position
        self.rebalance_days = rebalance_days
        self.last_trade_dates = {}
        self.max_position_ratio = max_position_ratio
        self.commission_rate = commission_rate
        
    def get_price(self, code, date, price_type='close'):
        """获取某个股票在特定日期的价格"""
        return self.historical_data[code].loc[
            self.historical_data[code]['date'] == date, price_type
        ].values[0]
    
    def get_total_assets(self, date):
        """计算当前总资产"""
        total = self.cash
        for code, volume in self.positions.items():
            price = self.get_price(code, date)
            total += price * volume
        return total
    
    def get_position(self, code):
        """获取某个股票的持仓量"""
        return self.positions.get(code, 0)
    
    def execute_order(self, code, date, direction, volume):
        """执行交易订单"""
        # 检查是否满足调仓周期要求
        if code in self.last_trade_dates:
            last_date = self.last_trade_dates[code]
            days_since_last_trade = (pd.to_datetime(date) - pd.to_datetime(last_date)).days
            if days_since_last_trade < self.rebalance_days:
                return False
        
        # 价格与手续费计算
        price = self.get_price(code, date)
        amount = price * volume
        commission = amount * self.commission_rate
        
        if direction == 'buy':
            # 检查是否超过最大持仓比例
            total_assets = self.get_total_assets(date)
            current_position_value = price * (self.get_position(code) + volume)
            if current_position_value > total_assets * self.max_position_ratio:
                return False
            
            total_cost = amount + commission
            if self.cash >= total_cost:
                if self.get_position(code) + volume > self.max_hold_position:
                    return False
                self.cash -= total_cost
                self.positions[code] = self.get_position(code) + volume
                self.last_trade_dates[code] = date
                return True
        else:  # sell
            if self.get_position(code) >= volume:
                self.cash += amount - commission
                self.positions[code] -= volume
                self.last_trade_dates[code] = date
                return True
        return False