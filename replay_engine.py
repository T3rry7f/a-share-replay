"""
A股历史复盘系统 - 核心引擎
功能: 秒级分时数据回放、实时排行榜计算展示
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from config import SECTOR_MAPPING_CONFIG

logging.basicConfig(level=logging.INFO)


class ReplayEngine:
    """复盘引擎 - 负责数据回放和计算"""
    
    def __init__(self, data_dir: str):
        """
        初始化复盘引擎
        
        Args:
            data_dir: 数据目录路径(如 data/tick_20251216)
        """
        self.data_dir = Path(data_dir)
        self.all_data = {}  # {stock_code: DataFrame}
        self.current_time = None  # 当前回放时间
        self.start_time = time(9, 30)  # 开盘时间
        self.end_time = time(15, 0)   # 收盘时间
        
        # 多维度板块映射
        self.industry_map = {}  # {stock_code: [industry_list]}
        self.concept_map = {}   # {stock_code: [concept_list]}
        self.region_map = {}    # {stock_code: [region_list]}
        self.stock_name_map = {}  # {stock_code: stock_name}
        self.pre_close_map = {}  # {stock_code: pre_close} 真实昨收价
        
        # 实时缓存
        self.fast_data_cache = {} # {code: (times, prices, vols, pre_close)} 纯NumPy极速缓存
        self.stock_cache = {}  # 股票实时数据缓存
        self.sector_cache = {}  # 板块实时数据缓存
        
        # 快照缓存（LRU缓存，最多保存100个时间点的快照）
        self.snapshot_cache = {}  # {time_key: snapshot_data}
        self.snapshot_cache_size = 100
        self.snapshot_cache_order = []  # LRU 顺序记录
        
        # 加载股票信息
        self.load_stock_names()
        self.load_sector_mappings()
        self.load_pre_close_prices()
        
    def load_all_data(self, progress_callback=None):
        """
        加载所有股票数据到内存
        
        注意: 这会占用大量内存,建议只在内存充足时使用
        或者采用按需加载策略
        """
        logging.info("开始加载数据...")
        
        parquet_files = list(self.data_dir.glob("*.parquet"))
        total = len(parquet_files)
        
        for idx, file_path in enumerate(parquet_files):
            stock_code = file_path.stem
            try:
                df = pd.read_parquet(file_path)
                
                # 数据预处理
                df = self._preprocess_tick_data(df)
                
                self.all_data[stock_code] = df
                
                if progress_callback and idx % 100 == 0:
                    progress_callback(idx, total)
                    
            except Exception as e:
                logging.warning(f"加载 {stock_code} 失败: {e}")
        
        logging.info(f"数据加载完成,共 {len(self.all_data)} 只股票")
    
    def detect_data_time_range(self):
        """
        检测已加载数据的实际时间范围
        """
        if not self.all_data:
            return
        
        min_time = None
        max_time = None
        
        # 采样检查（避免遍历所有股票）
        sample_size = min(100, len(self.all_data))
        sample_codes = list(self.all_data.keys())[:sample_size]
        
        for code in sample_codes:
            df = self.all_data[code]
            if not df.empty and 'datetime' in df.columns:
                stock_min = df['datetime'].min()
                stock_max = df['datetime'].max()
                
                if min_time is None or stock_min < min_time:
                    min_time = stock_min
                if max_time is None or stock_max > max_time:
                    max_time = stock_max
        
        
        # 限制结束时间为15:00（A股收盘时间）
        # 数据可能包含尾盘集合竞价，但显示时截断到15:00
        if max_time is not None:
            market_close = max_time.replace(hour=15, minute=0, second=0, microsecond=0)
            if max_time.time() > time(15, 0):
                max_time = market_close
                logging.info(f"检测到数据超过15:00，自动截断到15:00")
        
        self.data_start_time = min_time
        self.data_end_time = max_time
        
        logging.info(f"数据时间范围: {min_time.strftime('%H:%M:%S') if min_time else 'N/A'} - {max_time.strftime('%H:%M:%S') if max_time else 'N/A'}")
    
    def _preprocess_tick_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理tick数据
        
        Args:
            df: 原始tick数据
            
        Returns:
            处理后的DataFrame
        """
        if df.empty:
            return df
        
        # 解析时间字段
        if 'time' in df.columns and 'date' in df.columns:
            # 组合date和time创建完整的datetime
            date_col = df['date'].astype(str)
            time_col = df['time'].astype(str)
            df['datetime'] = pd.to_datetime(date_col + ' ' + time_col)
        elif 'time' in df.columns:
            # 如果只有time,假设是今天
            from datetime import date as dt_date
            today = dt_date.today().strftime('%Y%m%d')
            df['datetime'] = pd.to_datetime(today + ' ' + df['time'].astype(str))
        else:
            logging.warning("数据中没有时间字段")
            return df
        
        # 排序
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # --- 分秒平滑处理 (Plan A) ---
        # 如果数据只有分钟精度，将同一分钟内的多笔成交均匀分布在 60 秒内
        if len(df) > 1:
            df['_min_group'] = df['datetime'].dt.floor('min')
            df['_cum_count'] = df.groupby('_min_group').cumcount()
            df['_total_in_min'] = df.groupby('_min_group')['datetime'].transform('count')
            
            # 只有当秒数为 0 时才尝试平滑（避免破坏原本就有秒数的数据）
            # 检查第一笔是否有秒数
            if df['datetime'].iloc[0].second == 0:
                df['datetime'] = df['_min_group'] + pd.to_timedelta(
                    (df['_cum_count'] * 60 / df['_total_in_min']).astype(int), unit='s'
                )
            
            df.drop(columns=['_min_group', '_cum_count', '_total_in_min'], inplace=True)
            # 平滑后重新排序以防万一
            df = df.sort_values('datetime').reset_index(drop=True)
        
        # 计算累计成交量
        if 'vol' in df.columns:
            df['cum_volume'] = df['vol'].cumsum()
        
        # 设置昨收价
        if 'price' in df.columns and len(df) > 0:
            if 'pre_close' not in df.columns:
                # 获取股票代码
                if 'stock_code' in df.columns:
                    stock_code = str(df['stock_code'].iloc[0]).zfill(6)
                    # 优先使用真实昨收价
                    if stock_code in self.pre_close_map:
                        df['pre_close'] = self.pre_close_map[stock_code]
                    else:
                        # 降级方案：使用第一笔价格
                        df['pre_close'] = df['price'].iloc[0]
                        logging.warning(f"{stock_code}: 未找到昨收价，使用第一笔价格 {df['price'].iloc[0]:.2f}")
                else:
                    # 如果没有 stock_code 字段，使用第一笔价格
                    df['pre_close'] = df['price'].iloc[0]
        
        # 预先缓存需要的高速列 (NumPy arrays) - 性能关键优化
        df['_datetime_values'] = df['datetime'].values
        df['_price_values'] = df['price'].values
        df['_vol_values'] = df['vol'].values if 'vol' in df.columns else np.zeros(len(df))
        df['_cum_vol_values'] = df['cum_volume'].values if 'cum_volume' in df.columns else np.zeros(len(df))
        
        return df
    
    def lazy_load_stock(self, stock_code: str) -> pd.DataFrame:
        """
        按需加载单只股票数据(节省内存)
        
        Args:
            stock_code: 股票代码
            
        Returns:
            股票数据DataFrame
        """
        if stock_code not in self.all_data:
            file_path = self.data_dir / f"{stock_code}.parquet"
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    df = self._preprocess_tick_data(df)
                    self.all_data[stock_code] = df
                except Exception as e:
                    logging.warning(f"加载 {stock_code} 失败: {e}")
        
        return self.all_data.get(stock_code)
    
    def _load_single_stock(self, file_path: Path) -> tuple:
        """
        加载单只股票数据（用于多线程）
        
        Args:
            file_path: parquet文件路径
            
        Returns:
            (stock_code, dataframe) 或 (stock_code, None) 如果失败
        """
        stock_code = file_path.stem
        try:
            df = pd.read_parquet(file_path)
            df = self._preprocess_tick_data(df)
            return (stock_code, df)
        except Exception as e:
            logging.warning(f"加载 {stock_code} 失败: {e}")
            return (stock_code, None)
    
    def load_all_stocks_parallel(self, max_workers: int = 8, progress_callback=None) -> int:
        """
        多线程并行加载所有股票数据
        
        Args:
            max_workers: 最大线程数
            progress_callback: 进度回调函数 callback(current, total)
            
        Returns:
            成功加载的股票数量
        """
        # ========================================
        # ✅ 性能优化：优先使用单文件快速加载
        # ========================================
        
        # 检查是否存在合并的单文件格式 
        # 情况1: self.data_dir 是 tick/ 文件夹，合并文件在父目录
        # 情况2: self.data_dir 就是日期目录，合并文件就在此处
        tick_data_file = self.data_dir / "tick_data.parquet" if not self.data_dir.name == 'tick' else self.data_dir.parent / "tick_data.parquet"
        
        if tick_data_file.exists():
            logging.info(f"⚡ 检测到优化格式，使用快速加载: {tick_data_file}")
            return self._load_from_single_file(tick_data_file, progress_callback)
        
        # 否则使用传统的多文件加载 (单个股票一个文件)
        parquet_files = list(self.data_dir.glob("*.parquet"))
        
        # 排除掉合并文件，以防万一遍历到了 (虽然概率极低)
        parquet_files = [f for f in parquet_files if f.name != "tick_data.parquet"]
        
        total = len(parquet_files)
        loaded_count = 0
        
        if total == 0:
            logging.warning(f"目录 {self.data_dir} 中未找到有效的parquet文件")
            return 0
        
        logging.info(f"开始多线程加载 {total} 只股票数据，线程数: {max_workers}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_file = {executor.submit(self._load_single_stock, f): f for f in parquet_files}
            
            # 处理完成的任务
            for idx, future in enumerate(as_completed(future_to_file), 1):
                stock_code, df = future.result()
                
                if df is not None:
                    self.all_data[stock_code] = df
                    loaded_count += 1
                
                # 调用进度回调
                if progress_callback:
                    progress_callback(idx, total)
        
        logging.info(f"数据加载完成: {loaded_count}/{total}")
        return loaded_count
    
    def _load_from_single_file(self, tick_data_file: Path, progress_callback=None) -> int:
        """
        从合并的单个parquet文件快速加载所有股票数据
        (极致优化版：向量化预处理 + 极速拆分)
        """
        import time
        start_time = time.time()
        
        logging.info(f"⚡ 快速加载模式：正在读取数据...")
        
        # 1. 极速读取
        df = pd.read_parquet(tick_data_file)
        
        read_time = time.time() - start_time
        logging.info(f"   读取完成: {read_time:.2f}秒 (行数: {len(df):,})")
        
        process_start = time.time()
        logging.info(f"   正在进行全量向量化预处理...")

        # 2. 全量预处理 (Vectorized Preprocessing) - 在循环外一次性完成
        
        # A. 确保时间列 
        if 'datetime' not in df.columns:
            if 'date' in df.columns and 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
            elif 'time' in df.columns:
                from datetime import date as dt_date
                today = dt_date.today().strftime('%Y%m%d')
                df['datetime'] = pd.to_datetime(today + ' ' + df['time'].astype(str))
        
        # B. 向量化计算累计成交量
        if 'vol' in df.columns:
            # GroupBy + CumSum 速度非常快
            df['cum_volume'] = df.groupby('stock_code')['vol'].cumsum()
        
        # C. 处理昨收价
        if 'pre_close' in df.columns:
            # 如果下载的数据中已经包含了补丁后的昨收价，直接使用
            # 确保类型正确
            df['pre_close'] = df['pre_close'].astype('float32')
        elif self.pre_close_map and 'stock_code' in df.columns:
            # 确保代码格式一致
            df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
            # 映射昨收价
            df['pre_close'] = df['stock_code'].map(self.pre_close_map).astype('float32')
            
            # 对未匹配到的填充每组的第一笔价格
            if df['pre_close'].isnull().any():
                if 'price' in df.columns:
                    first_prices = df.groupby('stock_code')['price'].transform('first')
                    df['pre_close'] = df['pre_close'].fillna(first_prices)
        elif 'price' in df.columns:
             # 如果没有昨收价表，且数据中也没有昨收价列，全部使用第一笔价格
             df['pre_close'] = df.groupby('stock_code')['price'].transform('first')
        
        process_time = time.time() - process_start
        logging.info(f"   预处理完成: {process_time:.2f}秒 (向量化)")
        
        # D. 分秒平滑处理 (Plan A)
        # 如果数据只有分钟精度，将同一分钟内的多笔成交均匀分布在 60 秒内
        logging.info(f"   正在执行分秒平滑处理 (Plan A)...")
        # 确保数据有序
        df = df.sort_values(['stock_code', 'datetime']).reset_index(drop=True)
        
        df['_cum_count'] = df.groupby(['stock_code', 'datetime']).cumcount()
        df['_total_in_min'] = df.groupby(['stock_code', 'datetime'])['price'].transform('count')
        
        # 只有在检测到是分钟级数据（秒数为0）时才平滑
        if not df.empty and df['datetime'].iloc[0].second == 0:
            # 性能关键：使用向量化加法
            df['datetime'] = df['datetime'] + pd.to_timedelta(
                (df['_cum_count'] * 60 / df['_total_in_min']).astype(int), unit='s'
            )
            
        df.drop(columns=['_cum_count', '_total_in_min'], inplace=True)
        
        # 3. 极速拆分
        split_start = time.time()
        total_stocks = df['stock_code'].nunique()
        logging.info(f"   正在拆分为 {total_stocks} 只股票...")
        
        loaded_count = 0
        has_vol = 'vol' in df.columns
        has_cum_vol = 'cum_volume' in df.columns
        
        # 使用 groupby 迭代拆分
        for stock_code, group_df in df.groupby('stock_code'):
            # 关键修复: 必须 reset_index
            stock_df = group_df.reset_index(drop=True)
            
            self.all_data[stock_code] = stock_df
            
            # --- 构建极速缓存 (Pure NumPy) ---
            # 提取 float32 数组以节省内存并加速
            t_values = stock_df['datetime'].values
            p_values = stock_df['price'].values
            v_values = stock_df['cum_volume'].values if has_cum_vol else None
            
            # 提取昨收价 (标量)
            pre_close = float(stock_df['pre_close'].iloc[0]) if 'pre_close' in stock_df.columns else float(p_values[0])
            
            self.fast_data_cache[stock_code] = (t_values, p_values, v_values, pre_close)
            
            loaded_count += 1
            
            if progress_callback and loaded_count % 1000 == 0:
                 progress_callback(loaded_count, total_stocks)
                 
        if progress_callback:
            progress_callback(total_stocks, total_stocks)
            
        split_time = time.time() - split_start
        total_time = time.time() - start_time
        
        logging.info(f"   拆分与缓存: {split_time:.2f}秒")
        logging.info(f"✅ 极速加载完成: {total_time:.2f}秒!")
        
        return loaded_count
    
    def get_snapshot_at_time(self, target_time: datetime) -> Dict:
        """
        获取指定时间点的市场快照 - 极速版 (Pure NumPy)
        
        完全绕过 Pandas DataFrame，直接操作预缓存的 NumPy 数组。
        性能提升目标：比原有逻辑快 10-50 倍。
        """
        # 生成缓存键（精确到秒）
        time_key = target_time.strftime('%Y%m%d_%H%M%S')
        
        # 检查快照缓存 (LRU)
        if time_key in self.snapshot_cache:
            # 更新LRU顺序
            if time_key in self.snapshot_cache_order:
                self.snapshot_cache_order.remove(time_key)
            self.snapshot_cache_order.append(time_key)
            
            self.current_time = target_time
            return self.snapshot_cache[time_key]
        
        self.current_time = target_time
        
        # 初始化索引缓存
        if not hasattr(self, 'index_cache'):
            self.index_cache = {code: 0 for code in self.fast_data_cache.keys()}
        
        # 检测时间回退，重置索引缓存
        if hasattr(self, 'last_snapshot_time') and target_time < self.last_snapshot_time:
            self.index_cache = {code: 0 for code in self.fast_data_cache.keys()}
        
        self.last_snapshot_time = target_time
        
        snapshot = {
            'time': target_time,
            'stocks': {},
            'stats': {
                'total_stocks': 0,
                'up_count': 0,
                'down_count': 0,
                'flat_count': 0,
                'limit_up_count': 0,
                'limit_down_count': 0,
            }
        }
        
        # 转换为 numpy.datetime64[ns] 以匹配 Pandas 的默认精度
        target_np = np.array(target_time, dtype='datetime64[ns]')
        
        # 遍历极速缓存 (Pure NumPy Loop)
        # 这里的 items() 迭代速度远快于 DataFrame 的 items 或 iterrows
        for stock_code, (times, price_vals, vol_vals, pre_close) in self.fast_data_cache.items():
            if len(times) == 0:
                continue

            # 获取上次查找的索引位置
            last_idx = self.index_cache.get(stock_code, 0)
            
            # --- 极速索引查找 ---
            # 优化逻辑：智能切换线性扫描和二分查找
            # 1. 正常回放（时间微增）：只能线性扫描（非常快）
            # 2. 拖动滑块（大幅跳转）：强制二分查找（避免数千次循环）
            
            should_scan_linearly = False
            
            if last_idx < len(times) and target_np >= times[last_idx]:
                # 只有当目标时间在当前位置的"附近"时，才使用线性扫描
                # 设定阈值：例如检查往后20个点的位置
                lookahead = 20
                if last_idx + lookahead >= len(times):
                    # 剩余数据不足20个，直接线性扫完
                    should_scan_linearly = True
                elif times[last_idx + lookahead] >= target_np:
                    # 如果往后20个点的时间已经超过目标时间，说明目标就在这20个点之内
                    # 此时线性扫描比二分查找更快
                    should_scan_linearly = True
                # else: 目标在20个点之外，意味着发生了较大跨度跳转 -> 使用二分查找
            
            if should_scan_linearly:
                # 向前线性扫描
                idx = last_idx
                # 使用 numpy 的逐元素比较通常比 python 循环快，但在小范围内 Python 循环 overhead 也不大
                # 为了极致性能，保持原逻辑但有范围限制
                while idx + 1 < len(times) and times[idx + 1] <= target_np:
                    idx += 1
            else:
                # 时间回退或大幅度跳跃，使用二分查找
                idx = np.searchsorted(times, target_np, side='right') - 1
            
            # 更新索引缓存
            self.index_cache[stock_code] = max(0, idx)
            
            if idx >= 0:
                # 直接访问 NumPy 数组 (极快)
                current_price = price_vals[idx]
                
                # 如果没有vol数据，设为0
                cum_volume = vol_vals[idx] if vol_vals is not None else 0
                
                # 计算涨跌幅
                if pre_close > 0:
                    pct_change = (current_price - pre_close) / pre_close * 100
                    # 计算开盘涨跌幅 (使用当日前几笔作为开盘价)
                    open_price = price_vals[0]
                    open_pct_change = (open_price - pre_close) / pre_close * 100
                else:
                    pct_change = 0.0
                    open_pct_change = 0.0
                
                snapshot['stocks'][stock_code] = {
                    'price': float(current_price),
                    'open_price': float(open_price),
                    'volume': float(cum_volume),
                    'pct_change': float(pct_change),
                    'open_pct_change': float(open_pct_change),
                }
                
                # 统计涨跌
                if pct_change > 0.001:
                    snapshot['stats']['up_count'] += 1
                elif pct_change < -0.001:
                    snapshot['stats']['down_count'] += 1
                else:
                    snapshot['stats']['flat_count'] += 1
                
                # 统计涨跌停
                # 1. 判定涨跌幅比例
                if stock_code.startswith(('688', '300', '689')):
                    ratio = 0.2
                elif stock_code.startswith(('8', '4', '92')):
                    ratio = 0.3
                else:
                    ratio = 0.1
                    # 主板 ST 股 5%
                    if "ST" in self.stock_name_map.get(stock_code, ""):
                        ratio = 0.05
                
                # 2. 计算涨跌停价格 (同 detect_limit_movements 逻辑)
                limit_up = round(pre_close * (1 + ratio) + 0.0001, 2)
                limit_down = round(pre_close * (1 - ratio) + 0.0001, 2)
                
                if current_price >= limit_up:
                    snapshot['stats']['limit_up_count'] += 1
                elif current_price <= limit_down:
                    snapshot['stats']['limit_down_count'] += 1
        
        snapshot['stats']['total_stocks'] = len(snapshot['stocks'])
        
        # 存入缓存
        self.snapshot_cache[time_key] = snapshot
        self.snapshot_cache_order.append(time_key)
        
        # 维护缓存大小（LRU淘汰）
        if len(self.snapshot_cache_order) > self.snapshot_cache_size:
            oldest_key = self.snapshot_cache_order.pop(0)
            if oldest_key in self.snapshot_cache:
                del self.snapshot_cache[oldest_key]
                logging.debug(f"✂️ 移除过期快照: {oldest_key}")
        
        return snapshot
    
    def calculate_stock_rankings(self, snapshot: Dict, top_n: int = 50) -> pd.DataFrame:
        """
        计算个股涨幅排行 - 优化版 (利用预处理好的数组)
        """
        if not snapshot.get('stocks'):
            return pd.DataFrame(columns=['stock_code', 'stock_name', 'price', 'pct_change', 'volume'])
        
        # 提取数据
        stocks_data = snapshot['stocks']
        codes = list(stocks_data.keys())
        
        # 向量化构建 DataFrame (比循环快)
        df = pd.DataFrame({
            'stock_code': codes,
            'price': [d['price'] for d in stocks_data.values()],
            'pct_change': [d['pct_change'] for d in stocks_data.values()],
            'volume': [d['volume'] for d in stocks_data.values()]
        })
        
        # 映射名称
        df['stock_name'] = df['stock_code'].map(self.stock_name_map)
        
        # 排序并截断
        df = df.sort_values('pct_change', ascending=False).head(top_n)
        df = df.reset_index(drop=True)
        df.index += 1
        
        return df
    
    def calculate_sector_rankings(self, snapshot: Dict, sector_type: str = 'industry', top_n: int = 20) -> pd.DataFrame:
        """
        计算板块涨幅排行 - 优化版
        """
        # 选择映射表
        if sector_type == 'industry':
            sector_map = self.industry_map
        elif sector_type == 'concept':
            sector_map = self.concept_map
        elif sector_type == 'region':
            sector_map = self.region_map
        else:
            sector_map = self.industry_map
        
        if not snapshot.get('stocks'):
            return pd.DataFrame(columns=['sector', 'avg_pct_change', 'stock_count', 'total_volume', 'sector_type'])

        sector_stats = defaultdict(lambda: {'total_pct': 0.0, 'count': 0, 'volume': 0.0})
        
        # 聚合板块数据 (优化循环)
        for code, data in snapshot['stocks'].items():
            sectors = sector_map.get(code)
            if not sectors:
                continue
            
            # 修正异常涨幅贡献
            stock_pct = data.get('pct_change', 0)
            if abs(stock_pct) > 30:
                open_price = data.get('open_price', 0)
                current_price = data.get('price', 0)
                stock_pct = ((current_price - open_price) / open_price * 100) if open_price > 0 else 0
            
            vol = data.get('volume', 0)
            
            for sector in sectors:
                stat = sector_stats[sector]
                stat['total_pct'] += stock_pct
                stat['count'] += 1
                stat['volume'] += vol
        
        # 转化为 DataFrame
        if not sector_stats:
            return pd.DataFrame(columns=['sector', 'avg_pct_change', 'stock_count', 'total_volume', 'sector_type'])
            
        res = []
        for sector, stat in sector_stats.items():
            if stat['count'] > 0:
                res.append({
                    'sector': sector,
                    'avg_pct_change': stat['total_pct'] / stat['count'],
                    'stock_count': stat['count'],
                    'total_volume': stat['volume'],
                    'sector_type': sector_type
                })
        
        df = pd.DataFrame(res)
        df = df.sort_values('avg_pct_change', ascending=False).head(top_n)
        df = df.reset_index(drop=True)
        df.index += 1
        
        return df
    
    def detect_rapid_rise(self, time_window_minutes: int = 5, 
                          pct_threshold: float = 3.0) -> List[Dict]:
        """
        检测快速拉升个股（向后兼容方法）
        
        Args:
            time_window_minutes: 时间窗口(分钟)
            pct_threshold: 涨幅阈值(%)
            
        Returns:
            拉升股票列表
        """
        return self.detect_abnormal_movement(
            time_window_minutes=time_window_minutes,
            rise_threshold=pct_threshold,
            fall_threshold=None,  # 只检测涨幅
            volume_threshold=None
        )
    
    def detect_abnormal_movement(self, time_window_minutes: int = 5, 
                                 rise_threshold: float = 3.0,
                                 fall_threshold: float = -3.0,
                                 volume_threshold: float = None) -> List[Dict]:
        """
        检测异动个股 - 超高速版 (O(N_stocks * log N) + 缓存优化)
        
        使用NumPy数组缓存和二分查找，性能提升10倍以上
        
        Args:
            time_window_minutes: 时间窗口(分钟)
            rise_threshold: 涨幅阈值(%)，None表示不监控涨幅
            fall_threshold: 跌幅阈值(%)，应为负数，None表示不监控跌幅
            volume_threshold: 成交额阈值(万元)，None表示不限制
            
        Returns:
            异动股票列表
        """
        abnormal_stocks = []
        
        if self.current_time is None:
            return abnormal_stocks
        
        time_window_start = self.current_time - pd.Timedelta(minutes=time_window_minutes)
        # 关键修正：确保与 times 数组的 datetime64[ns] 精度一致
        start_np = np.array(time_window_start, dtype='datetime64[ns]')
        end_np = np.array(self.current_time, dtype='datetime64[ns]')
        
        # 遍历极速缓存 (Pure NumPy Loop)
        # items() 迭代比 DataFrame items 极快
        for stock_code, (times, price_vals, cum_vol_vals, _) in self.fast_data_cache.items():
            if len(times) == 0:
                continue
            
            # 使用二分查找定位窗口边界
            start_idx = np.searchsorted(times, start_np, side='left')
            end_idx = np.searchsorted(times, end_np, side='right') - 1
            
            if end_idx > start_idx and start_idx >= 0 and end_idx < len(times):
                start_price = price_vals[start_idx]
                end_price = price_vals[end_idx]
                
                if start_price > 0:
                    pct_change = (end_price - start_price) / start_price * 100
                    
                    # 检查是否满足条件
                    is_abnormal = False
                    movement_type = None
                    
                    # 检查涨幅
                    if rise_threshold is not None and pct_change >= rise_threshold:
                        is_abnormal = True
                        movement_type = 'rise'
                    
                    # 检查跌幅
                    if fall_threshold is not None and pct_change <= fall_threshold:
                        is_abnormal = True
                        movement_type = 'fall'
                    
                    if is_abnormal:
                        # 计算成交额（万元）- 使用累计成交量差值 (O(1)复杂度)
                        if cum_vol_vals is not None:
                            # 累计量差值 = 结束时刻累计 - 开始前时刻累计
                            vol_end = cum_vol_vals[end_idx]
                            vol_start = cum_vol_vals[start_idx - 1] if start_idx > 0 else 0
                            window_vol = vol_end - vol_start
                        else:
                            window_vol = 0
                            
                        # 成交量单位是手(100股)，价格单位是元
                        volume_amount = window_vol * 100 * end_price / 10000
                        
                        # 成交额过滤
                        if volume_threshold is not None and volume_amount < volume_threshold:
                            continue
                        
                        abnormal_stocks.append({
                            'stock_code': stock_code,
                            'movement_type': movement_type,
                            'start_price': start_price,
                            'end_price': end_price,
                            'pct_change': pct_change,
                            'volume_amount': volume_amount,
                            'start_time': pd.Timestamp(times[start_idx]),
                            'end_time': pd.Timestamp(times[end_idx]),
                        })
        
        # 按涨跌幅绝对值排序（异动幅度大的在前）
        abnormal_stocks.sort(key=lambda x: abs(x['pct_change']), reverse=True)
        
        return abnormal_stocks

    def detect_limit_movements(self) -> List[Dict]:
        """
        检测涨跌停异动（封板/炸板）
        
        Returns:
            涨跌停异动列表
        """
        limit_events = []
        if self.current_time is None:
            return limit_events

        end_np = np.array(self.current_time, dtype='datetime64[ns]')

        for stock_code, (times, price_vals, _, pre_close) in self.fast_data_cache.items():
            if len(times) == 0:
                continue

            # 寻找当前时间点对应的最新 Tick
            end_idx = np.searchsorted(times, end_np, side='right') - 1
            if end_idx < 0:
                continue

            # 只有在 Tick 刚刚发生变化时才触发异动（避免重复触发）
            # 注意：这里的逻辑假设调用方会根据时间推移持续调用
            # 为简单起见，我们检测 end_idx 对应的 Tick 时间是否就是当前“模拟秒”或者是最近几秒内
            tick_time = pd.Timestamp(times[end_idx])
            if (self.current_time - tick_time).total_seconds() >= 3:
                # 如果这个 Tick 已经是 3 秒前的了，说明是老数据，不视作“新异动”
                #（除非是刚开盘或者数据断流，这里权衡一下）
                continue

            current_price = price_vals[end_idx]
            
            # 计算该股的涨跌停价格
            if stock_code.startswith(('688', '300', '689')):
                ratio = 0.2
            elif stock_code.startswith(('8', '4', '92')):
                ratio = 0.3
            else:
                ratio = 0.1
                # 只有主板的 ST 股才是 5% 限制，创业板和科创板 ST 仍是 20%
                stock_name = self.stock_name_map.get(stock_code, "")
                if "ST" in stock_name:
                    ratio = 0.05
                
            # A股涨跌停计算通常是四舍五入到分，但为了稳健，我们使用 0.005 的偏移
            limit_up = round(pre_close * (1 + ratio) + 0.0001, 2)
            limit_down = round(pre_close * (1 - ratio) + 0.0001, 2)
            
            event_type = None
            desc = ""
            
            if end_idx > 0:
                prev_price = price_vals[end_idx - 1]
                was_at_limit_up = prev_price >= limit_up
                was_at_limit_down = prev_price <= limit_down
                
                is_at_limit_up = current_price >= limit_up
                is_at_limit_down = current_price <= limit_down
                
                if not was_at_limit_up and is_at_limit_up:
                    event_type = "hit_limit_up"
                    desc = "🚀 封涨停"
                elif was_at_limit_up and not is_at_limit_up:
                    event_type = "break_limit_up"
                    desc = "💥 炸涨停"
                elif not was_at_limit_down and is_at_limit_down:
                    event_type = "hit_limit_down"
                    desc = "📉 封跌停"
                elif was_at_limit_down and not is_at_limit_down:
                    event_type = "break_limit_down"
                    desc = "♻️ 炸跌停"
            else:
                # 开盘第一笔
                if current_price >= limit_up:
                    event_type = "hit_limit_up"
                    desc = "🚀 涨停开盘"
                elif current_price <= limit_down:
                    event_type = "hit_limit_down"
                    desc = "📉 跌停开盘"
            
            if event_type:
                limit_events.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'event_type': event_type,
                    'desc': desc,
                    'price': current_price,
                    'time': tick_time.strftime('%H:%M:%S'),
                    'pct_change': (current_price - pre_close) / pre_close * 100
                })
        
        return limit_events
    
    def load_sector_mappings(self):
        """加载行业、概念、地区映射"""
        import os
        
        source = SECTOR_MAPPING_CONFIG.get('source', 'iwencai')
        logging.info(f"正在从 {source} 加载板块映射...")
        
        if source == 'iwencai':
            files = SECTOR_MAPPING_CONFIG['iwencai_files']
            self._load_iwencai_mapping(files['industry'], self.industry_map, "行业")
            self._load_iwencai_mapping(files['concept'], self.concept_map, "概念")
            self._load_iwencai_mapping(files['region'], self.region_map, "地区")
        else:
            files = SECTOR_MAPPING_CONFIG['eastmoney_files']
            self._load_eastmoney_mapping(files['industry'], self.industry_map, "行业")
            self._load_eastmoney_mapping(files['concept'], self.concept_map, "概念")
            self._load_eastmoney_mapping(files['region'], self.region_map, "地区")

    def _load_iwencai_mapping(self, file_path: str, target_map: Dict, label: str):
        """加载 iwencai 格式的映射文件"""
        import os
        if not os.path.exists(file_path):
            logging.warning(f"未找到 iWencai {label}文件: {file_path}")
            return
            
        try:
            # 尝试不同的编码
            for encoding in ['utf-8-sig', 'gbk', 'utf-8']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    # 清理列名
                    df.columns = [c.strip() for c in df.columns]
                    break
                except Exception:
                    continue
            else:
                logging.error(f"无法读取文件 {file_path}，请检查编码")
                return

            # 寻找代码列
            code_col = None
            for col in ['stock_code', '代码', '证券代码']:
                if col in df.columns:
                    code_col = col
                    break
            
            # 寻找名称列
            name_col = None
            for col in ['classification_name', '名称', '板块名称', 'concept_name']:
                if col in df.columns:
                    name_col = col
                    break
            
            if not code_col or not name_col:
                logging.warning(f"iWencai {label}文件格式不正确 (缺失代码或名称列): {file_path}")
                return

            count = 0
            for _, row in df.iterrows():
                val = row[code_col]
                if pd.isna(val): continue
                
                code = str(val).split('.')[0].zfill(6)
                name = str(row[name_col]).strip()
                
                if name == 'nan' or not name: continue
                
                if code not in target_map:
                    target_map[code] = []
                if name not in target_map[code]:
                    target_map[code].append(name)
                    count += 1
            
            logging.info(f"✅ iWencai {label}映射加载完成, 共加载 {count} 条映射, 覆盖 {len(target_map)} 只股票")
        except Exception as e:
            logging.warning(f"加载 iWencai {label}映射失败: {e}")

    def _load_eastmoney_mapping(self, file_path: str, target_map: Dict, label: str):
        """加载 eastmoney 格式的映射文件"""
        import os
        if not os.path.exists(file_path):
            logging.warning(f"未找到 Eastmoney {label}文件: {file_path}")
            return
            
        try:
            # 尝试不同的编码
            for encoding in ['utf-8-sig', 'gbk', 'utf-8']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    df.columns = [c.strip() for c in df.columns]
                    break
                except Exception:
                    continue
            else:
                return

            # 寻找代码列
            code_col = None
            for col in ['stock_code', '代码', '证券代码']:
                if col in df.columns:
                    code_col = col
                    break
            
            # 寻找名称列 (Eastmoney 特有的可能是 board_name 或 concept_name)
            name_col = None
            for col in ['board_name', 'concept_name', '名称', '板块名称']:
                if col in df.columns:
                    name_col = col
                    break
            
            if not code_col or not name_col:
                logging.warning(f"Eastmoney {label}文件格式不正确: {file_path}")
                return

            count = 0
            for _, row in df.iterrows():
                val = row[code_col]
                if pd.isna(val): continue
                
                code = str(val).zfill(6)
                name = str(row[name_col]).strip()
                
                if name == 'nan' or not name: continue
                
                if code not in target_map:
                    target_map[code] = []
                if name not in target_map[code]:
                    target_map[code].append(name)
                    count += 1
            logging.info(f"✅ Eastmoney {label}映射加载完成, 共加载 {count} 条映射, 覆盖 {len(target_map)} 只股票")
        except Exception as e:
            logging.warning(f"加载 Eastmoney {label}映射失败: {e}")
    
    def load_industry_mapping(self, mapping_file: str):
        """
        加载行业映射关系(兼容旧接口)
        
        Args:
            mapping_file: 映射文件路径(CSV格式,包含stock_code和industry列)
        """
        try:
            df = pd.read_csv(mapping_file)
            for _, row in df.iterrows():
                code = str(row['stock_code']).zfill(6)
                industry = row['industry']
                if code not in self.industry_map:
                    self.industry_map[code] = []
                if industry not in self.industry_map[code]:
                    self.industry_map[code].append(industry)
            logging.info(f"行业映射加载完成,共 {len(self.industry_map)} 条记录")
        except Exception as e:
            logging.error(f"加载行业映射失败: {e}")
    
    def load_stock_names(self):
        """加载股票名称映射"""
        try:
            # 从parquet文件中提取股票名称
            for stock_code, df in self.all_data.items():
                if 'stock_name' in df.columns and len(df) > 0:
                    self.stock_name_map[stock_code] = df['stock_name'].iloc[0]
            
            # 如果有CSV文件，也从CSV加载
            import os
            csv_path = 'data/eastmoney_all_stocks.csv'
            if os.path.exists(csv_path):
                import pandas as pd
                df = pd.read_csv(csv_path)
                df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
                name_dict = dict(zip(df['stock_code'], df['stock_name']))
                self.stock_name_map.update(name_dict)
                
            logging.info(f"股票名称映射加载完成，共 {len(self.stock_name_map)} 条记录")
        except Exception as e:
            logging.warning(f"加载股票名称映射失败: {e}")
    
    def get_stock_name(self, stock_code: str) -> str:
        """获取股票名称"""
        return self.stock_name_map.get(stock_code, stock_code)
    
    def load_pre_close_prices(self):
        """加载股票昨收价"""
        try:
            import os
            # 支持两种数据结构:
            # 1. data/20251222/tick/*.parquet + data/20251222/stock_pre_close_20251222.csv
            # 2. data/tick_20251222/*.parquet + data/stock_pre_close_20251222.csv
            
            # 从数据目录名提取日期
            dir_name = self.data_dir.name
            
            # 尝试新结构 (data/20251222/tick)
            if dir_name == 'tick':
                date_str = self.data_dir.parent.name
                pre_close_file = self.data_dir.parent / f'stock_pre_close_{date_str}.csv'
            # 尝试旧结构 (data/tick_20251222)
            elif dir_name.startswith('tick_'):
                date_str = dir_name.replace("tick_", "")
                pre_close_file = Path(f'data/stock_pre_close_{date_str}.csv')
            else:
                # 直接使用目录名作为日期 (data/20251222)
                date_str = dir_name
                pre_close_file = self.data_dir / f'stock_pre_close_{date_str}.csv'
            
            if os.path.exists(pre_close_file):
                df = pd.read_csv(pre_close_file)
                df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
                # 昨收价单位可能是分，需要转换为元
                df['pre_close'] = df['pre_close'] / 100
                self.pre_close_map = dict(zip(df['stock_code'], df['pre_close']))
                logging.info(f"昨收价加载完成，共 {len(self.pre_close_map)} 条记录 (文件: {pre_close_file})")
            else:
                logging.warning(f"未找到昨收价文件: {pre_close_file}")
        except Exception as e:
            logging.warning(f"加载昨收价失败: {e}")
    
    def replay_iterator(self, start_time: str = "09:30:00", 
                       end_time: str = "15:00:00",
                       speed_seconds: int = 1):
        """
        生成器: 按时间顺序回放数据
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            speed_seconds: 回放速度(秒/次)
            
        Yields:
            (current_time, snapshot)
        """
        # 解析时间
        start_dt = pd.to_datetime(f"2025-01-01 {start_time}")
        end_dt = pd.to_datetime(f"2025-01-01 {end_time}")
        
        current = start_dt
        
        while current <= end_dt:
            self.current_time = current
            
            # 获取快照
            snapshot = self.get_snapshot_at_time(current)
            
            yield current, snapshot
            
            # 增加时间步长
            current += pd.Timedelta(seconds=speed_seconds)


if __name__ == "__main__":
    # 使用示例
    engine = ReplayEngine("data/tick_20251216")
    
    # 测试加载
    logging.info("正在加载数据...")
    # engine.load_all_data()
    
    logging.info("复盘引擎初始化完成")
