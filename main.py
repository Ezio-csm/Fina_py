from datetime import datetime
from backtest import Backtest

if __name__ == "__main__":
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    stock_codes = ['000001.SZ', '600000.SH']
    
    backtest = Backtest(start_date, end_date, stock_codes, strategy_type='oscillation')
    backtest.run()
    # 回测结果可视化
    backtest.plot_results(save_path='backtest_results.png')

    # 统计指标
    stats = backtest.get_statistics()
    for metric, value in stats.items():
        print(f"{metric}: {value:.2f}")