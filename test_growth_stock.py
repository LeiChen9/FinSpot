#!/usr/bin/env python
"""成长股策略简化测试"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import time

print('='*60)
print('成长股策略简化测试')
print('='*60)

# 测试数据获取
print('\n[1] 测试数据获取...')

# 获取股票列表
try:
    df_stocks = ak.stock_info_a_code_name()
    print(f'[√] 获取A股列表: {len(df_stocks)} 只')
    # 取前10只测试
    test_stocks = df_stocks['code'].head(10).tolist()
    print(f'  测试股票: {test_stocks}')
except Exception as e:
    print(f'[x] 获取A股列表失败: {e}')
    test_stocks = ['000001', '000002', '600000', '600036', '601318']

# 测试财务数据
print('\n[2] 测试财务数据获取...')
for code in test_stocks[:3]:
    try:
        df = ak.stock_financial_abstract_ths(symbol=code, indicator='按报告期')
        if df is not None and not df.empty:
            print(f'[√] {code} 财务数据: {len(df)} 条')
            print(f'  列名: {df.columns.tolist()[:5]}...')
        else:
            print(f'[x] {code} 无财务数据')
    except Exception as e:
        print(f'[x] {code} 获取财务数据失败: {e}')
    time.sleep(0.5)

# 测试行情数据
print('\n[3] 测试行情数据获取...')
for code in test_stocks[:3]:
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, 
            period='daily',
            start_date='20240101',
            end_date='20240131',
            adjust='qfq'
        )
        if df is not None and not df.empty:
            print(f'[√] {code} 行情数据: {len(df)} 条')
        else:
            print(f'[x] {code} 无行情数据')
    except Exception as e:
        print(f'[x] {code} 获取行情数据失败: {e}')
    time.sleep(0.5)

# 测试机构持股数据
print('\n[4] 测试机构持股数据...')
try:
    df = ak.stock_report_fund_hold(symbol='基金持仓', date='20240331')
    if df is not None and not df.empty:
        print(f'[√] 机构持股数据: {len(df)} 条')
        print(f'  列名: {df.columns.tolist()[:5]}...')
    else:
        print('[x] 无机构持股数据')
except Exception as e:
    print(f'[x] 获取机构持股数据失败: {e}')

# 测试行业板块数据
print('\n[5] 测试行业板块数据...')
try:
    df = ak.stock_board_industry_hist_em(
        symbol='银行',
        start_date='20240101',
        end_date='20240131',
        period='日k'
    )
    if df is not None and not df.empty:
        print(f'[√] 银行板块数据: {len(df)} 条')
    else:
        print('[x] 无行业板块数据')
except Exception as e:
    print(f'[x] 获取行业板块数据失败: {e}')

print('\n' + '='*60)
print('简化测试完成!')
print('='*60)
