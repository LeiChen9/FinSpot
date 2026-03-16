import akshare as ak
import pdb 
from datetime import datetime, timedelta
import akshare_proxy_patch
import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from talib import RSI, BBANDS, MACD

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

def plot_fund_analysis(code, days=90, latest_date=datetime.now().strftime('%Y-%m-%d'),
                       rsi_period=14, boll_period=20, boll_std=2,
                       macd_fast=12, macd_slow=26, macd_signal=9):
    """
    绘制基金分析图表 (净值走势 + 技术指标)
    
    参数:
        code: 基金代码
        days: 显示过去多少天的数据 (默认90天/3个月)
        latest_date: 最新日期
        其他参数: 技术指标参数
    """
    # 获取指标分析数据
    data = analyze_fund_indicators(code, latest_date, rsi_period, boll_period, boll_std,
                                   macd_fast, macd_slow, macd_signal)
    
    # 按日期筛选数据 (过去N天)
    end_date = pd.to_datetime(latest_date)
    start_date = end_date - timedelta(days=days)
    data = data[(data['净值日期'] >= start_date) & (data['净值日期'] <= end_date)].reset_index(drop=True)
    
    # 创建图表 (4个子图: 净值+布林带, RSI, MACD, 柱状图)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'基金 {code} 技术分析 (过去{days}天)', fontsize=16, fontweight='bold')
    
    # ===== 第一行: 净值走势 + 布林带 =====
    ax1 = axes[0]
    ax1.plot(data.index, data['单位净值'], label='净值', color='black', linewidth=2)
    ax1.fill_between(data.index, data['BOLL_lower'], data['BOLL_upper'], 
                      alpha=0.2, color='blue', label='布林带')
    ax1.plot(data.index, data['BOLL_middle'], label='中轨', color='blue', linestyle='--', linewidth=1)
    ax1.set_ylabel('净值', fontsize=11, fontweight='bold')
    ax1.set_title('基金净值走势 (布林带)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ===== 第二行: RSI =====
    ax2 = axes[1]
    ax2.plot(data.index, data['RSI'], label='RSI', color='red', linewidth=1.5)
    ax2.axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='超买(70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='超卖(30)')
    ax2.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
    ax2.set_ylabel('RSI', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.set_title(f'相对强弱指数 (RSI, 周期={rsi_period})', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ===== 第三行: MACD =====
    ax3 = axes[2]
    # 绘制柱状图 (MACD_histogram)
    colors = ['green' if x >= 0 else 'red' for x in data['MACD_histogram']]
    ax3.bar(data.index, data['MACD_histogram'], label='MACD柱', color=colors, alpha=0.3, width=0.8)
    # 绘制MACD线和信号线
    ax3.plot(data.index, data['MACD'], label='MACD', color='blue', linewidth=1.5)
    ax3.plot(data.index, data['MACD_signal'], label='信号线', color='red', linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax3.set_ylabel('MACD', fontsize=11, fontweight='bold')
    ax3.set_xlabel('日期', fontsize=11, fontweight='bold')
    ax3.set_title(f'MACD指标 (快={macd_fast}, 慢={macd_slow}, 信号={macd_signal})', fontsize=12)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 调整x轴标签 (每10个数据点显示一个标签)
    step = max(1, len(data) // 10)
    x_ticks = range(0, len(data), step)
    x_labels = [data['净值日期'].iloc[i].strftime('%m-%d') if i < len(data) else '' 
                for i in x_ticks]
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_labels, rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(os.path.dirname(__file__), f'../output/{code}_analysis.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_path}")
    
    plt.show()
    
    return data

if __name__ == "__main__":
    stock_code = '015716'
    DATA_DIR = '../data/'
    # 绘制过去3个月的分析图表 (默认90天)
    result = plot_fund_analysis(stock_code, days=90)
    print("\n显示数据行数:", len(result))
    print("最后5行数据:")
    print(result[['净值日期', '单位净值', 'RSI', 'BOLL_middle', 'MACD']].tail(5))