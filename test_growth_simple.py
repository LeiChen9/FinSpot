#!/usr/bin/env python
"""成长股策略简化测试 - 使用本地数据"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests

print('='*60)
print('成长股策略简化测试（本地数据版）')
print('='*60)

# 数据路径
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# 测试QQ数据源
print('\n[1] 测试QQ数据源...')

def qq_fetch(code, end_date='2024-01-31'):
    """QQ财经API获取行情数据"""
    url = 'https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get'
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://gu.qq.com/'}
    
    ticker = f'sh{code}' if code.startswith(('6', '5', '9')) else f'sz{code}'
    params = {
        '_var': 'kline_dayqfq',
        'param': f'{ticker},day,2020-01-01,{end_date},800,qfq',
        'r': '0.1'
    }
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        text = resp.text
        json_str = text[text.find('{'):text.rfind('}') + 1]
        data = json.loads(json_str)
        d = data.get('data', {}).get(ticker, {})
        rows = d.get('qfqday') or d.get('day') or []
        if rows:
            df = pd.DataFrame(rows, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
            print(f'[√] {code} QQ数据: {len(df)} 条')
            return True
    except Exception as e:
        print(f'[x] {code} QQ数据失败: {e}')
    return False

# 测试几只股票
test_codes = ['000001', '600000', '000002']
for code in test_codes:
    qq_fetch(code)
    import time
    time.sleep(0.5)

# 测试本地数据
print('\n[2] 测试本地数据文件...')
local_files = [
    os.path.join(DATA_DIR, '000001_qfq.csv'),
    os.path.join(DATA_DIR, '600000_qfq.csv'),
    os.path.join(DATA_DIR, 'meta', 'graham_universe.csv')
]

for f in local_files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(f'[√] {os.path.basename(f)}: {len(df)} 行')
    else:
        print(f'[x] {os.path.basename(f)}: 不存在')

# 测试策略逻辑
print('\n[3] 测试策略逻辑...')

# 简化的选股逻辑
def simple_growth_filter(code, as_of_date):
    """简化的成长股筛选"""
    # 条件1: 股票代码在合理范围
    code_num = int(code)
    if code_num < 1000:
        return False  # 排除银行等大盘股
    
    # 条件2: 小盘股（代码>300000）
    if code_num < 300000:
        return False
    
    return True

# 测试筛选
test_pool = ['000001', '000002', '000063', '000333', '000651', '000858', '002027', '300015', '600000', '600036']
candidates = [code for code in test_pool if simple_growth_filter(code, '2024-01-01')]
print(f'[√] 筛选出 {len(candidates)} 只候选股: {candidates}')

# 测试资金管理
print('\n[4] 测试资金管理逻辑...')

class SimpleBacktest:
    def __init__(self):
        self.cash = 1_000_000
        self.positions = {}
        self.position_size = 100_000
        
    def buy(self, code, price):
        if code in self.positions:
            return False
        if len(self.positions) >= 10:
            return False
        
        shares = (self.position_size / price) // 100 * 100
        if shares <= 0:
            return False
        
        cost = shares * price * 1.0003
        if cost > self.cash:
            return False
        
        self.positions[code] = {'shares': shares, 'price': price}
        self.cash -= cost
        return True

bt = SimpleBacktest()
for code in candidates[:5]:
    bt.buy(code, 10.0)  # 假设价格10元

print(f'[√] 持仓数量: {len(bt.positions)}')
print(f'[√] 剩余现金: {bt.cash:,.2f} 元')

print('\n' + '='*60)
print('简化测试完成!')
print('='*60)
