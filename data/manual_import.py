#!/usr/bin/env python
"""
手动数据导入工具

如果无法通过API自动下载，可以手动导入CSV文件

支持的数据来源：
1. 同花顺（通达信）- 导出为 CSV
2. Yahoo Finance - 网页下载为 CSV
3. 新浪财经 - 复制粘贴为 Excel
4. 腾讯财经 - 复制粘贴为 Excel
"""

import pandas as pd
import os
import sys
from datetime import datetime


def validate_and_save_market_data(csv_file: str, index_code: str):
    """验证和保存行情数据"""

    # 读取 CSV
    df = pd.read_csv(csv_file)

    print(f"读取文件: {csv_file}")
    print(f"列名: {df.columns.tolist()}")
    print(f"行数: {len(df)}")

    # 标准化列名
    rename_map = {
        '日期': 'date', 'Date': 'date', 'date': 'date',
        '开盘': 'open', 'Open': 'open', 'open': 'open',
        '高': 'high', 'High': 'high', 'high': 'high', '最高': 'high',
        '低': 'low', 'Low': 'low', 'low': 'low', '最低': 'low',
        '收盘': 'close', 'Close': 'close', 'close': 'close',
        '成交量': 'volume', 'Volume': 'volume', 'volume': 'volume',
    }

    df = df.rename(columns=rename_map)

    # 检查必要的列
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"✗ 缺少必要列: {missing}")
        print(f"可用列: {df.columns.tolist()}")
        return False

    # 保留必要的列
    df = df[required_cols]

    # 转换日期和数值
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 移除无效行
    df = df.dropna()
    df = df.sort_values('date')

    print(f"\n处理后:")
    print(f"  有效行数: {len(df)}")
    print(f"  时间范围: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")

    # 保存
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_file = os.path.join(output_dir, f'{index_code}_market.csv')

    df.set_index('date', inplace=True)
    df.to_csv(output_file)

    print(f"\n✓ 数据已保存: {output_file}")
    return True


def create_sample_csv(index_code: str):
    """创建示例 CSV 文件"""
    from datetime import datetime, timedelta

    data = {
        'date': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(5, 0, -1)],
        'open': [100.0, 101.5, 102.0, 101.8, 103.0],
        'high': [101.5, 102.5, 103.0, 102.5, 104.0],
        'low': [99.5, 101.0, 101.5, 101.0, 102.5],
        'close': [101.0, 101.8, 102.5, 101.5, 103.5],
        'volume': [1000000, 1100000, 1050000, 950000, 1200000],
    }

    df = pd.DataFrame(data)
    sample_file = f'sample_{index_code}_market.csv'
    df.to_csv(sample_file, index=False)

    print(f"✓ 示例文件已创建: {sample_file}")
    print("\n示例格式:")
    print(df.to_string())


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  导入数据: python manual_data_import.py <csv_file> <index_code>")
        print("  创建示例: python manual_data_import.py --sample <index_code>")
        print("\n例如:")
        print("  python manual_data_import.py ~/Downloads/800700.csv 800700")
        print("  python manual_data_import.py --sample 800700")
        sys.exit(1)

    if sys.argv[1] == '--sample':
        index_code = sys.argv[2] if len(sys.argv) > 2 else '800700'
        create_sample_csv(index_code)
    else:
        csv_file = sys.argv[1]
        index_code = sys.argv[2] if len(sys.argv) > 2 else '800700'

        if not os.path.exists(csv_file):
            print(f"✗ 文件不存在: {csv_file}")
            sys.exit(1)

        validate_and_save_market_data(csv_file, index_code)


if __name__ == '__main__':
    main()
