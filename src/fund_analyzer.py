import akshare as ak
import pdb 
from datetime import datetime
import akshare_proxy_patch
import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from talib import RSI, BBANDS, MACD

akshare_proxy_patch.install_patch("101.201.173.125", "", 30)

DATA_DIR = 'data/'

def fund_hist(code, latest_date=datetime.now().strftime('%Y-%m-%d')):
    if os.path.exists(os.path.join(DATA_DIR, code + '_' + str(latest_date) + '.csv')):
        print('load from local csv, latest date: ' + str(latest_date))
        fund_hist = pd.read_csv(os.path.join(DATA_DIR, code + '_' + str(latest_date) + '.csv'))
        fund_hist['净值日期'] = pd.to_datetime(fund_hist['净值日期'])
        if fund_hist['净值日期'].iloc[-1] >= datetime.strptime(latest_date, '%Y-%m-%d'):
            return fund_hist
    print('fetch from akshare, latest date: ' + str(latest_date))
    fund_hist = ak.fund_open_fund_info_em(symbol= code, indicator = "单位净值走势")
    fund_hist = fund_hist.sort_values(by='净值日期', ascending=True)
    latest_date = fund_hist['净值日期'].iloc[-1].strftime('%Y-%m-%d')
    fund_hist.to_csv(os.path.join(DATA_DIR, code + '_' + str(latest_date) + '.csv'), index=0)
    return fund_hist

def calculate_rsi(data, period=14):
    """
    计算RSI (Relative Strength Index)
    默认参数: 14周期 (最常用的股票技术分析参数)
    """
    rsi = RSI(data.values, timeperiod=period)
    return rsi

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """
    计算布林带 (Bollinger Bands)
    默认参数: 20周期, 2个标准差 (最常用的股票技术分析参数)
    """
    upper, middle, lower = BBANDS(data.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    计算MACD (Moving Average Convergence Divergence)
    默认参数: fast=12, slow=26, signal=9 (最常用的股票技术分析参数)
    """
    macd_line, signal_line, histogram = MACD(data.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def analyze_fund_indicators(code, latest_date=datetime.now().strftime('%Y-%m-%d'), 
                           rsi_period=14, boll_period=20, boll_std=2, 
                           macd_fast=12, macd_slow=26, macd_signal=9):
    """
    计算基金的技术指标 (RSI, BOLL, MACD)
    
    参数:
        code: 基金代码
        latest_date: 最新日期
        rsi_period: RSI周期 (默认14)
        boll_period: 布林带周期 (默认20)
        boll_std: 布林带标准差倍数 (默认2)
        macd_fast: MACD快线周期 (默认12)
        macd_slow: MACD慢线周期 (默认26)
        macd_signal: MACD信号线周期 (默认9)
    
    返回: 包含原始数据和所有指标的DataFrame
    """
    # 获取基金历史数据
    hist_data = fund_hist(code, latest_date)
    
    # 确保净值列是数值类型
    hist_data['单位净值'] = pd.to_numeric(hist_data['单位净值'])
    
    # 计算RSI
    hist_data['RSI'] = calculate_rsi(hist_data['单位净值'], rsi_period)
    
    # 计算布林带
    boll_result = calculate_bollinger_bands(hist_data['单位净值'], boll_period, boll_std)
    hist_data['BOLL_upper'] = boll_result['upper']
    hist_data['BOLL_middle'] = boll_result['middle']
    hist_data['BOLL_lower'] = boll_result['lower']
    
    # 计算MACD
    macd_result = calculate_macd(hist_data['单位净值'], macd_fast, macd_slow, macd_signal)
    hist_data['MACD'] = macd_result['macd']
    hist_data['MACD_signal'] = macd_result['signal']
    hist_data['MACD_histogram'] = macd_result['histogram']
    
    return hist_data

if __name__ == "__main__":
    stock_code = '004815'
    DATA_DIR = '../data/'
    
    # 使用默认参数计算指标
    result = analyze_fund_indicators(stock_code)
    print("\n基金指标分析结果 (使用默认参数):")
    print(result[['净值日期', '单位净值', 'RSI', 'BOLL_upper', 'BOLL_middle', 'BOLL_lower', 'MACD', 'MACD_signal']].tail(10))
    
    # 可选: 自定义参数计算指标
    # result_custom = analyze_fund_indicators(
    #     stock_code,
    #     rsi_period=14,
    #     boll_period=20,
    #     boll_std=2,
    #     macd_fast=12,
    #     macd_slow=26,
    #     macd_signal=9
    # )
    # print("\n基金指标分析结果 (自定义参数):")
    # print(result_custom[['净值日期', '单位净值', 'RSI']].tail(10))