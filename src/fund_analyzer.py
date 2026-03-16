import akshare as ak
import pdb 
from datetime import datetime
import akshare_proxy_patch
import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    stock_code = '004815'
    DATA_DIR = '../data/'
    fund_hist(stock_code)