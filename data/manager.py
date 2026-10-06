"""数据管理模块 — 统一管理指数 / 个股 / 基金的数据获取、缓存、增量更新"""
import os
import pandas as pd
from datetime import datetime, timedelta

from data.sources import qq, baostock, akshare, financial


class DataManager:
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

    # A 股行情源优先级（按稳定性从高到低）
    A_SOURCE_LIST = [qq.fetch, baostock.fetch, akshare.fetch]

    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)

    # =========================================================================
    # 公共接口 — 指数 / 个股 行情
    # =========================================================================

    def get_stock_market_data(self, stock_code: str, days: int = 365 * 3,
                              force_download: bool = False) -> pd.DataFrame:
        """获取个股行情数据，优先本地缓存，过期则增量更新"""
        return self._load_or_fetch(stock_code, days, force_download)

    def get_index_market_data(self, index_code: str, days: int = 365 * 3,
                              force_download: bool = False) -> pd.DataFrame:
        """获取指数行情数据，优先本地缓存，过期则增量更新"""
        return self._load_or_fetch(index_code, days, force_download)

    def get_market_data(self, code: str, days: int = 365 * 3,
                        force_download: bool = False) -> pd.DataFrame:
        """通用行情接口 — 自动识别市场并路由（供 scripts 使用）"""
        if self._is_hk(code):
            return self._fetch_hk(code, days)
        return self._load_or_fetch(code, days, force_download)

    # =========================================================================
    # 公共接口 — 基金净值
    # =========================================================================

    def get_fund_nav_data(self, fund_code: str,
                          force_download: bool = False) -> pd.DataFrame:
        """获取基金净值数据"""
        path = self._fund_nav_path(fund_code)
        end = self._nearest_trading_day()

        if not force_download and os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            if df.index.max() >= end:
                print(f"   [√] 数据已最新: {path} ({len(df)} 条记录)")
                return df
            # 增量：下载最新之后的数据
            start = (df.index.max() + timedelta(days=1)).strftime('%Y%m%d')

        print(f"   [...] 下载基金净值: {fund_code}")
        df = akshare.fetch_nav(fund_code)
        if df is not None and not df.empty:
            if os.path.exists(path):
                df_old = pd.read_csv(path, index_col='date', parse_dates=True)
                df = pd.concat([df_old, df])
                df = df[~df.index.duplicated(keep='last')].sort_index()
            df.to_csv(path)
            print(f"   [√] 基金净值已保存: {path} ({len(df)} 条记录)")
            return df
        raise RuntimeError(f"无法获取基金净值: {fund_code}")

    # =========================================================================
    # 公共接口 — 估值
    # =========================================================================

    def get_index_valuation_data(self, index_code: str,
                                 force_download: bool = False) -> pd.DataFrame:
        """获取指数估值数据（历史 PE / PB / 股息率）"""
        path = self._valuation_path(index_code)

        if not force_download and os.path.exists(path):
            return pd.read_csv(path, index_col='date', parse_dates=True)

        print(f"   [...] 下载估值数据: {index_code}")
        df = akshare.fetch_valuation(index_code)
        if df is not None and not df.empty:
            df.to_csv(path)
            print(f"   [√] 估值数据已保存: {path} ({len(df)} 条记录)")
            return df
        raise RuntimeError(f"无法从任何源获取估值数据: {index_code}")

    def get_valuation_data(self, index_code: str,
                           force_download: bool = False) -> pd.DataFrame:
        """兼容别名"""
        return self.get_index_valuation_data(index_code, force_download)

    # =========================================================================
    # 公共接口 — 财务数据
    # =========================================================================

    def get_financial_data(self, code: str,
                            force_download: bool = False) -> pd.DataFrame:
        """获取个股现金流量表年报数据（含经营现金流、CAPEX、净利润）"""
        path = self._financial_path(code)
        if not force_download and os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=['report_date', 'publish_date'])
            print(f"   [√] 财务数据已缓存: {path} ({len(df)} 条记录)")
            return df

        print(f"   [...] 下载财务数据: {code}")
        df = financial.fetch_cash_flow(code)
        if df is not None and not df.empty:
            df.to_csv(path)
            print(f"   [√] 财务数据已保存: {path} ({len(df)} 条记录)")
            return df
        raise RuntimeError(f"无法获取财务数据: {code}")

    def get_batch_financial_data(self, codes: list,
                                  force_download: bool = False) -> dict:
        """批量获取多只股票财务数据，返回 {code: DataFrame}"""
        result = {}
        for code in codes:
            try:
                result[code] = self.get_financial_data(code, force_download)
            except Exception as e:
                print(f"   [x] {code}: {e}")
        return result

    # =========================================================================
    # 内部 — 缓存策略 + 多源降级
    # =========================================================================

    def _load_or_fetch(self, code: str, days: int,
                       force: bool = False) -> pd.DataFrame:
        """
        统一缓存策略：优先读本地 CSV，存在且最新则直接返回；
        存在但过期则增量追加；不存在则全量下载。
        """
        path = self._market_path(code)
        end = self._nearest_trading_day()
        start = end - timedelta(days=int(days * 1.5))

        # 本地缓存已存在且最新
        if not force and os.path.exists(path):
            df_local = pd.read_csv(path, index_col='date', parse_dates=True)
            if df_local.index.max() >= end:
                print(f"   [√] 数据已最新: {path} ({len(df_local)} 条记录)")
                return df_local.tail(days) if len(df_local) > days else df_local
            # 增量更新：只下载最新日期之后的数据
            start_inc = df_local.index.max() + timedelta(days=1)
            df_new = self._try_sources(code, start_inc, end)
            if df_new is not None and not df_new.empty:
                combined = pd.concat([df_local, df_new])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.sort_index().to_csv(path)
                print(f"   [√] 数据已增量更新: {path} ({len(combined)} 条)")
                return combined.tail(days) if len(combined) > days else combined
            return df_local.tail(days) if len(df_local) > days else df_local

        # 全量下载
        print(f"   [...] 下载行情数据: {code}")
        df = self._try_sources(code, start, end)
        if df is not None and not df.empty:
            df.sort_index().to_csv(path)
            print(f"   [√] 行情已保存: {path} ({len(df)} 条记录)")
            return df.tail(days) if len(df) > days else df

        # 本地有数据兜底
        if os.path.exists(path):
            df_local = pd.read_csv(path, index_col='date', parse_dates=True)
            print(f"   [√] 使用本地数据: {path} ({len(df_local)} 条记录)")
            return df_local.tail(days) if len(df_local) > days else df_local

        raise RuntimeError(
            f"无法从任何源获取行情数据: {code}。\n"
            f"请在 {self.DATA_DIR} 下放置 {code}_market.csv\n"
            f"CSV 格式: date,open,high,low,close,volume"
        )

    def _try_sources(self, code: str, start: datetime, end: datetime,
                     source_list: list = None) -> pd.DataFrame | None:
        """按优先级依次尝试数据源，返回第一个成功的结果"""
        if source_list is None:
            source_list = self.A_SOURCE_LIST
        for source in source_list:
            try:
                df = source(code, start, end)
                if df is not None and not df.empty:
                    return df
            except Exception:
                continue
        return None

    # =========================================================================
    # 内部 — 港股
    # =========================================================================

    def _is_hk(self, code: str) -> bool:
        return (code.startswith('hk') or code.endswith('.HK')
                or (code.startswith('8') and len(code) == 6)
                or len(code) == 5)

    def _fetch_hk(self, code: str, days: int) -> pd.DataFrame:
        end = self._nearest_trading_day()
        start = end - timedelta(days=int(days * 1.5))
        print(f"   [...] 下载港股数据: {code}")

        df = akshare.fetch(code, start, end)
        if df is not None and not df.empty:
            return df.tail(days) if len(df) > days else df

        df = baostock.fetch_hk(code, start, end)
        if df is not None and not df.empty:
            return df.tail(days) if len(df) > days else df

        raise RuntimeError(f"无法获取港股数据: {code}")

    # =========================================================================
    # 内部 — 工具
    # =========================================================================

    @staticmethod
    def _nearest_trading_day() -> datetime:
        """方案 A：当前时间 > 15:30 则最近交易日为今天，否则为上一个交易日"""
        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if now.hour > 15 or (now.hour == 15 and now.minute >= 30):
            if today.weekday() < 5:
                return today
        d = today - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d

    def _market_path(self, code: str) -> str:
        return os.path.join(self.DATA_DIR, f'{code}_market.csv')

    def _valuation_path(self, code: str) -> str:
        return os.path.join(self.DATA_DIR, f'{code}_valuation.csv')

    def _fund_nav_path(self, code: str) -> str:
        return os.path.join(self.DATA_DIR, f'{code}_fund_nav.csv')

    def _financial_path(self, code: str) -> str:
        fin_dir = os.path.join(self.DATA_DIR, 'financial')
        os.makedirs(fin_dir, exist_ok=True)
        return os.path.join(fin_dir, f'{code}_cashflow.csv')
