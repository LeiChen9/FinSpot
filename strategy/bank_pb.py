"""银行板块低 PB 轮动策略 (point-in-time)。

规则:
  - 池: 33 家 A 股银行, 每日取市值 Top10 中 PB < 0.6 者
  - 首次建仓: 新合格股按 初始资金×0.4/合格数 等额买入
  - 补仓: 较持仓最高价回撤 ≥10% → 加仓 初始资金×0.2/合格数
  - 换股: 现金 < 初始 10% 且有新合格股 → 卖出涨幅最大的非合格持仓
  - 财务口径: THS 财务摘要宽表, 年报按次年 4-30 披露可见
"""
from typing import Dict, Tuple

import pandas as pd

from dataload.readers import load_fin_summary, load_qfq

BANK_STOCKS = {
    '601398': '工商银行', '601288': '农业银行', '601988': '中国银行',
    '601939': '建设银行', '601328': '交通银行', '601658': '邮储银行',
    '600036': '招商银行', '601166': '兴业银行', '600016': '民生银行',
    '600000': '浦发银行', '601818': '光大银行', '601998': '中信银行',
    '000001': '平安银行', '002142': '宁波银行', '601009': '南京银行',
    '601838': '成都银行', '600926': '杭州银行', '601077': '渝农商行',
    '601577': '长沙银行', '601169': '北京银行', '600919': '江苏银行',
    '601229': '上海银行', '601528': '瑞丰银行', '600908': '无锡银行',
    '601128': '常熟银行', '601860': '紫金银行', '601665': '苏农银行',
    '603323': '苏州银行', '002839': '张家港行', '002948': '郑州银行',
    '601963': '重庆银行', '600928': '西安银行', '001227': '兰州银行',
}


def load_financial_data(code: str, cache: Dict) -> Dict:
    """解析 THS 财务摘要宽表 → {bvps, shares, total_equity} 按报告期。"""
    if code in cache:
        return cache[code]
    result = {'bvps': {}, 'shares': {}, 'total_equity': {}}
    df = load_fin_summary(code)
    if df is not None:
        bvps_row = df[df.iloc[:, 1] == '每股净资产']
        if not bvps_row.empty:
            row = bvps_row.iloc[0]
            for col in df.columns[2:]:
                try:
                    rd = pd.to_datetime(str(col), format='%Y%m%d')
                    v = float(row[col]) if pd.notna(row[col]) else None
                    if v and v > 0:
                        result['bvps'][rd] = v
                except Exception:
                    pass
        equity_row = df[df.iloc[:, 1] == '股东权益合计(净资产)']
        if not equity_row.empty:
            row = equity_row.iloc[0]
            for col in df.columns[2:]:
                try:
                    rd = pd.to_datetime(str(col), format='%Y%m%d')
                    v = float(row[col]) if pd.notna(row[col]) else None
                    if v and v > 0:
                        result['total_equity'][rd] = v
                except Exception:
                    pass
    for rd in result['bvps']:
        if rd in result['total_equity']:
            b, e = result['bvps'][rd], result['total_equity'][rd]
            if b > 0 and e > 0:
                result['shares'][rd] = e / b
    cache[code] = result
    return result


def load_bank_data(cache: Dict = None, min_history: int = 100) -> Tuple[Dict, Dict, Dict]:
    """预加载全部银行财务与前复权行情。

    返回 (financial_cache, market_data, valid_banks)。
    """
    if cache is None:
        cache = {}
    market_data: Dict[str, pd.DataFrame] = {}
    valid_banks: Dict[str, str] = {}
    for code, name in BANK_STOCKS.items():
        load_financial_data(code, cache)
        df = load_qfq(code)
        if df is not None and len(df) > min_history:
            market_data[code] = df
            fd = cache.get(code, {})
            if fd.get('bvps') and fd.get('shares'):
                valid_banks[code] = name
    return cache, market_data, valid_banks


def _pit_value(data_dict: Dict, as_of) -> float:
    """年报视为次年 4-30 披露, 季报按惯例披露日, 取 as_of 可见最新值。"""
    for rd in sorted(data_dict.keys(), reverse=True):
        m, y = rd.month, rd.year
        if m == 12:
            pub = pd.Timestamp(y + 1, 4, 30)
        elif m == 3:
            pub = pd.Timestamp(y, 4, 30)
        elif m == 6:
            pub = pd.Timestamp(y, 8, 31)
        elif m == 9:
            pub = pd.Timestamp(y, 10, 31)
        else:
            continue
        if pub <= as_of:
            return data_dict[rd]
    return None


def screen_top10_banks(as_of, market_data: Dict, valid_banks: Dict,
                       financial_cache: Dict) -> list:
    """市值 Top10 中 PB < 0.6 的银行。"""
    candidates = []
    for code, name in valid_banks.items():
        if code not in market_data:
            continue
        df = market_data[code]
        if as_of not in df.index:
            continue
        price = df.loc[as_of, 'close']
        if price <= 0:
            continue
        fd = financial_cache[code]
        bvps = _pit_value(fd['bvps'], as_of)
        shares = _pit_value(fd['shares'], as_of)
        if bvps and bvps > 0 and shares and shares > 0:
            pb = price / bvps
            mcap = shares * price
            candidates.append({'code': code, 'name': name, 'price': price, 'pb': pb, 'market_cap': mcap})
    candidates.sort(key=lambda x: x['market_cap'], reverse=True)
    top10 = candidates[:10]
    return [c for c in top10 if c['pb'] < 0.6]


class BankPBBacktest:
    """每日收盘触发: 换股/补仓/首次建仓。"""

    def __init__(self, market_data: Dict, valid_banks: Dict, financial_cache: Dict,
                 initial_capital=1_000_000, commission=0.0003, stamp_tax=0.0005):
        self.market_data = market_data
        self.valid_banks = valid_banks
        self.financial_cache = financial_cache
        self.initial_capital = initial_capital
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.cash = initial_capital
        self.positions = {}
        self.nav_history = []
        self.trade_history = []

    def get_nav(self, as_of):
        total = self.cash
        for code, pos in self.positions.items():
            if code in self.market_data and as_of in self.market_data[code].index:
                total += pos['shares'] * self.market_data[code].loc[as_of, 'close']
        return total

    def buy(self, code, name, price, amount, as_of, reason):
        if amount <= 0 or amount > self.cash:
            return
        shares = int(amount / price / 100) * 100
        if shares <= 0:
            return
        actual_amount = shares * price
        self.cash -= (actual_amount + actual_amount * self.commission)
        if code in self.positions:
            pos = self.positions[code]
            ts = pos['shares'] + shares
            pos['avg_cost'] = (pos['avg_cost'] * pos['shares'] + price * shares) / ts
            pos['shares'] = ts
        else:
            self.positions[code] = {'shares': shares, 'avg_cost': price, 'highest_price': price, 'name': name}
        self.trade_history.append({'date': as_of, 'code': code, 'name': name, 'action': 'BUY',
                                   'price': price, 'shares': shares, 'amount': actual_amount, 'reason': reason})

    def sell(self, code, price, as_of, reason):
        if code not in self.positions:
            return
        pos = self.positions[code]
        amt = pos['shares'] * price
        self.cash += (amt - amt * self.commission - amt * self.stamp_tax)
        self.trade_history.append({'date': as_of, 'code': code, 'name': pos['name'], 'action': 'SELL',
                                   'price': price, 'shares': pos['shares'], 'amount': amt, 'reason': reason})
        del self.positions[code]

    def run(self, start_date, end_date):
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        trading_days = self.market_data[list(self.market_data.keys())[0]].index
        trading_days = trading_days[(trading_days >= start) & (trading_days <= end)]
        print(f'回测期间: {start_date} ~ {end_date}, 共 {len(trading_days)} 个交易日')

        for i, today in enumerate(trading_days):
            # Update current prices and highest_price for all positions
            for code in list(self.positions.keys()):
                pos = self.positions[code]
                if code in self.market_data and today in self.market_data[code].index:
                    pos['current_price'] = self.market_data[code].loc[today, 'close']
                    if pos['current_price'] > pos.get('highest_price', 0):
                        pos['highest_price'] = pos['current_price']

            # Screen qualified stocks
            qualified = screen_top10_banks(today, self.market_data, self.valid_banks, self.financial_cache)
            qualified_codes = {s['code'] for s in qualified}

            # Sell biggest gainer if cash low and new stocks to buy
            if qualified and self.cash < self.initial_capital * 0.1:
                max_gain, sell_code = -1, None
                for code, pos in self.positions.items():
                    if code not in qualified_codes and 'current_price' in pos:
                        g = (pos['current_price'] - pos['avg_cost']) / pos['avg_cost']
                        if g > max_gain:
                            max_gain, sell_code = g, code
                if sell_code and sell_code in self.market_data and today in self.market_data[sell_code].index:
                    self.sell(sell_code, self.market_data[sell_code].loc[today, 'close'], today, '换股')

            # Check for add-position (drop 10% from highest_price)
            for code in list(self.positions.keys()):
                pos = self.positions[code]
                hp = pos.get('highest_price', pos['avg_cost'])
                if 'current_price' in pos and hp > 0:
                    drop = (hp - pos['current_price']) / hp
                    if drop >= 0.1:
                        n = max(len(qualified), 1)
                        add_amt = self.initial_capital * 0.2 / n
                        if add_amt <= self.cash:
                            self.buy(code, pos['name'], pos['current_price'], add_amt, today, f'补仓跌{drop*100:.0f}%')

            # Buy new qualifying stocks
            if qualified:
                new = [s for s in qualified if s['code'] not in self.positions]
                if new and self.cash > 0:
                    per = self.initial_capital * 0.4 / len(qualified)
                    for s in new:
                        if s['code'] not in self.positions and per <= self.cash:
                            self.buy(s['code'], s['name'], s['price'], per, today, '首次建仓')

            nav = self.get_nav(today)
            self.nav_history.append({'date': today, 'nav': nav, 'cash': self.cash, 'n_positions': len(self.positions)})

            if (i + 1) % 100 == 0:
                print(f'  进度: {i+1}/{len(trading_days)}, 净值: {nav:,.0f}, 持仓: {len(self.positions)}')

        print(f'回测完成! 最终净值: {self.nav_history[-1]["nav"]:,.0f}, 总交易: {len(self.trade_history)} 笔')
