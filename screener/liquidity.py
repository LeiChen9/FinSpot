"""流动性筛选器 - 筛选成交额和成交量排名靠前的股票"""
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
from screener.base import Screener, FilterResult

# 简单的股票名称映射（部分常用股票）
STOCK_NAMES = {
    '300308': '中际旭创', '300502': '新易盛', '300476': '胜宏科技',
    '300750': '宁德时代', '601138': '工业富联', '300274': '阳光电源',
    '002384': '东山精密', '002475': '立讯精密', '601899': '紫金矿业',
    '300059': '东方财富', '600519': '贵州茅台', '000858': '五粮液',
    '601318': '中国平安', '600036': '招商银行', '000001': '平安银行',
    '600900': '长江电力', '601012': '隆基绿能', '300015': '爱尔眼科',
    '002594': '比亚迪', '600887': '伊利股份', '000333': '美的集团',
    '601888': '中国中免', '600809': '山西汾酒', '002304': '洋河股份',
    '000568': '泸州老窖', '600585': '海螺水泥', '601668': '中国建筑',
    '601390': '中国中铁', '601186': '中国铁建', '600031': '三一重工',
    '000002': '万科A', '600048': '保利发展', '601088': '中国神华',
    '600028': '中国石化', '601857': '中国石油', '600941': '中国移动',
    '601728': '中国电信', '600050': '中国联通', '601398': '工商银行',
    '601288': '农业银行', '601988': '中国银行', '601939': '建设银行',
    '601328': '交通银行', '601658': '邮储银行', '601166': '兴业银行',
    '600016': '民生银行', '600000': '浦发银行', '601818': '光大银行',
    '601998': '中信银行', '002142': '宁波银行', '601009': '南京银行',
    '601838': '成都银行', '600926': '杭州银行', '601077': '渝农商行',
    '601577': '长沙银行', '601169': '北京银行', '600919': '江苏银行',
    '601229': '上海银行', '601528': '瑞丰银行', '600908': '无锡银行',
    '601128': '常熟银行', '601860': '紫金银行', '601665': '苏农银行',
    '603323': '苏州银行', '002839': '张家港行', '002948': '郑州银行',
    '601963': '重庆银行', '600928': '西安银行', '001227': '兰州银行',
    '002460': '赣锋锂业', '300394': '天孚通信', '000988': '华工科技',
    '600487': '亨通光电', '600584': '长电科技', '600111': '北方稀土',
    '002050': '三花智控', '000725': '京东方A', '000063': '中兴通讯',
    '002463': '沪电股份', '301308': '凌云光', '300058': '蓝色光标',
    '002156': '通富微电', '002281': '光迅科技', '600522': '中天科技',
    '300475': '聚隆科技', '300136': '信维通信', '603019': '中科曙光',
    '002371': '北方华创', '603259': '药明康德', '600089': '特变电工',
    '000977': '浪潮信息', '600183': '生益科技', '300604': '长城科技',
    '600105': '永鼎股份', '601318': '中国平安', '600118': '中国卫星',
    '600030': '中信证券', '000938': '紫光股份', '300014': '亿纬锂能',
    '002202': '金风科技', '000547': '航天发展', '002709': '天赐材料',
    '300223': '北京君正', '002407': '多氟多', '000021': '深科技',
    '000657': '中钨高新', '600498': '烽火通信',
}

def get_stock_name(code: str) -> str:
    """获取股票名称"""
    return STOCK_NAMES.get(code, f'股票{code}')


class LiquidityScreener(Screener):
    """流动性筛选器

    筛选标准：
    1. 过去一年成交额排名前 N
    2. 过去一年每日成交额都在阈值以上
    3. 过去一年每日成交量都在阈值以上（可选）
    """

    def __init__(
        self,
        data_dir: str = '/Users/riceball/Documents/Projs/fund/data',
        top_n: int = 100,
        min_daily_amount: float = 1e7,  # 最低日成交额（元）
        min_daily_volume: float = 1e6,  # 最低日成交量（股）
        lookback_days: int = 252,  # 回看天数（约一年）
        min_history_days: int = 100,  # 数据刚开始时允许的最少历史长度
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.top_n = top_n
        self.min_daily_amount = min_daily_amount
        self.min_daily_volume = min_daily_volume
        self.lookback_days = lookback_days
        self.min_history_days = min_history_days

        if self.lookback_days <= 0:
            raise ValueError("lookback_days 必须为正数")
        if self.min_history_days <= 0 or self.min_history_days > self.lookback_days:
            raise ValueError("min_history_days 必须在 1 和 lookback_days 之间")

    def _load_market_data(self, code: str) -> Optional[pd.DataFrame]:
        """加载股票的前复权数据"""
        file_path = self.data_dir / f'{code}_qfq.csv'
        if not file_path.exists():
            return None

        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            # 计算成交额
            df['amount'] = df['close'] * df['volume']

            return df
        except Exception as e:
            print(f"加载 {code} 数据失败: {e}")
            return None

    def _calculate_liquidity_metrics(
        self, code: str, df: pd.DataFrame, as_of_date=None
    ) -> dict:
        """计算流动性指标"""
        # 只使用 as_of_date 当天及之前的数据，避免把未来成交量带入历史回测。
        if as_of_date is not None:
            recent_df = df.loc[:pd.Timestamp(as_of_date)].tail(self.lookback_days)
        else:
            recent_df = df.tail(self.lookback_days)

        if len(recent_df) < self.min_history_days:  # 数据不足
            return None

        # 计算日均成交额
        avg_daily_amount = recent_df['amount'].mean()

        # 计算日均成交量
        avg_daily_volume = recent_df['volume'].mean()

        # 计算成交额排名（总成交额）
        total_amount = recent_df['amount'].sum()

        # 检查是否满足最低阈值
        min_amount_check = (recent_df['amount'] >= self.min_daily_amount).all()
        min_volume_check = (recent_df['volume'] >= self.min_daily_volume).all()

        return {
            'code': code,
            'avg_daily_amount': avg_daily_amount,
            'avg_daily_volume': avg_daily_volume,
            'total_amount': total_amount,
            'min_amount_check': min_amount_check,
            'min_volume_check': min_volume_check,
            'data_points': len(recent_df),
            'as_of_date': recent_df.index[-1],
        }

    def run(self, as_of_date=None, pool: Optional[List[str]] = None) -> FilterResult:
        """运行流动性筛选"""
        # 获取所有可用的股票代码
        all_codes = []
        for file_path in self.data_dir.glob('*_qfq.csv'):
            code = file_path.stem.replace('_qfq', '')
            all_codes.append(code)
        all_codes = sorted(set(all_codes))

        if not all_codes:
            return FilterResult(codes=[], names={}, info={})

        # 如果提供了池子，只筛选池子中的股票
        if pool:
            all_codes = [c for c in all_codes if c in pool]

        # 计算每只股票的流动性指标
        liquidity_metrics = []
        for code in all_codes:
            df = self._load_market_data(code)
            if df is None:
                continue

            metrics = self._calculate_liquidity_metrics(code, df, as_of_date=as_of_date)
            if metrics:
                liquidity_metrics.append(metrics)

        if not liquidity_metrics:
            return FilterResult(codes=[], names={}, info={})

        # 转换为 DataFrame 并排序
        metrics_df = pd.DataFrame(liquidity_metrics)
        effective_as_of = metrics_df['as_of_date'].max()

        # 筛选满足最低阈值的股票
        qualified = metrics_df[
            metrics_df['min_amount_check'] &
            metrics_df['min_volume_check']
        ]

        # 按总成交额降序排列，取前 top_n
        qualified_sorted = qualified.sort_values(
            ['total_amount', 'code'], ascending=[False, True]
        )
        top_codes = qualified_sorted.head(self.top_n)['code'].tolist()

        # 创建股票名称映射
        names = {code: get_stock_name(code) for code in top_codes}

        # 准备结果信息
        info = {
            'total_stocks': len(all_codes),
            'qualified_stocks': len(qualified),
            'selected_stocks': len(top_codes),
            'top_stocks': qualified_sorted.head(10).to_dict('records'),
            'as_of_date': effective_as_of,
            'lookback_days': self.lookback_days,
            'min_history_days': self.min_history_days,
        }

        return FilterResult(codes=top_codes, names=names, info=info)
