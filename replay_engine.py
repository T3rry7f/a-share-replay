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
        tick_data_file = self.data_dir.parent / "tick_data.parquet" if self.data_dir.name == 'tick' else self.data_dir.parent / "tick_data.parquet"
        
        if tick_data_file.exists():
            logging.info(f"⚡ 检测到优化格式，使用快速加载: {tick_data_file.name}")
            return self._load_from_single_file(tick_data_file, progress_callback)
        
        # 否则使用传统的多文件加载
        parquet_files = list(self.data_dir.glob("*.parquet"))
        total = len(parquet_files)
        loaded_count = 0
        
        if total == 0:
            logging.warning(f"目录 {self.data_dir} 中未找到parquet文件")
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
        
        # C. 向量化匹配昨收价
        if self.pre_close_map and 'stock_code' in df.columns:
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
             # 如果没有昨收价表，全部使用第一笔价格
             df['pre_close'] = df.groupby('stock_code')['price'].transform('first')
        
        process_time = time.time() - process_start
        logging.info(f"   预处理完成: {process_time:.2f}秒 (向量化)")
        
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
            # 智能查找：如果时间递增，从上次位置开始线性查找（大多数情况）
            if last_idx < len(times) and target_np >= times[last_idx]:
                # 向前线性扫描
                idx = last_idx
                # 只有当差距较小时才线性扫描，否则还是二分快
                while idx + 1 < len(times) and times[idx + 1] <= target_np:
                    idx += 1
            else:
                # 时间回退或跳跃，使用二分查找
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
                else:
                    pct_change = 0.0
                
                snapshot['stocks'][stock_code] = {
                    'price': float(current_price),
                    'volume': float(cum_volume),
                    'pct_change': float(pct_change),
                }
                
                # 统计涨跌
                if pct_change > 0.001:
                    snapshot['stats']['up_count'] += 1
                elif pct_change < -0.001:
                    snapshot['stats']['down_count'] += 1
                else:
                    snapshot['stats']['flat_count'] += 1
        
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
        计算个股涨幅排行
        
        Args:
            snapshot: 市场快照
            top_n: 返回前N名
            
        Returns:
            排行榜DataFrame
        """
        stocks_list = []
        
        for code, data in snapshot['stocks'].items():
            stocks_list.append({
                'stock_code': code,
                'stock_name': self.get_stock_name(code),
                'price': data.get('price', 0),
                'pct_change': data.get('pct_change', 0),
                'volume': data.get('volume', 0),
            })
        
        # 如果没有数据,返回空DataFrame但包含所需列
        if not stocks_list:
            return pd.DataFrame(columns=['stock_code', 'stock_name', 'price', 'pct_change', 'volume'])
        
        df = pd.DataFrame(stocks_list)
        
        # 按涨跌幅排序
        df = df.sort_values('pct_change', ascending=False).head(top_n)
        df = df.reset_index(drop=True)
        df.index += 1  # 排名从1开始
        
        return df
    
    def calculate_sector_rankings(self, snapshot: Dict, sector_type: str = 'industry', top_n: int = 20) -> pd.DataFrame:
        """
        计算板块涨幅排行
        
        Args:
            snapshot: 市场快照
            sector_type: 板块类型 ('industry', 'concept', 'region')
            top_n: 返回前N名
            
        Returns:
            板块排行榜DataFrame
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
        
        sector_data = defaultdict(lambda: {'total_pct': 0, 'count': 0, 'volume': 0, 'stocks': []})
        
        # 聚合板块数据
        for code, data in snapshot['stocks'].items():
            sectors = sector_map.get(code, ['未分类'])
            if not isinstance(sectors, list):
                sectors = [sectors]
            
            # 一只股票可能属于多个板块
            for sector in sectors:
                sector_data[sector]['total_pct'] += data.get('pct_change', 0)
                sector_data[sector]['count'] += 1
                sector_data[sector]['volume'] += data.get('volume', 0)
                sector_data[sector]['stocks'].append(code)
        
        # 计算板块平均涨幅
        sectors_list = []
        for sector, data in sector_data.items():
            # 过滤掉"未分类"板块
            if sector == '未分类':
                continue
                
            if data['count'] > 0:
                avg_pct = data['total_pct'] / data['count']
                sectors_list.append({
                    'sector': sector,
                    'avg_pct_change': avg_pct,
                    'stock_count': data['count'],
                    'total_volume': data['volume'],
                    'sector_type': sector_type,
                })
        
        # 如果没有数据,返回空DataFrame但包含所需列
        if not sectors_list:
            return pd.DataFrame(columns=['sector', 'avg_pct_change', 'stock_count', 'total_volume', 'sector_type'])
        
        df = pd.DataFrame(sectors_list)
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
