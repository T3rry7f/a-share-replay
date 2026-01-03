"""
A股历史复盘系统 - Web可视化界面
基于Streamlit构建
"""

import streamlit as st
import pandas as pd
from datetime import datetime, time
import time as time_module
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from replay_engine import ReplayEngine
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sector_analysis import render_sector_analysis, render_sector_heatmap, render_rapid_rise_sectors, render_sector_detail_view
from downloader import StockDataDownloader
# from download_pre_close import download_pre_close_parallel, get_stock_pre_close_single
from config import SECTOR_MAPPING_CONFIG

# 页面配置
st.set_page_config(
    page_title="A股历史复盘系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    div[data-testid="stMetric"], .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .gain {
        color: #ff4444;
    }
    .loss {
        color: #00aa00;
    }
</style>
""", unsafe_allow_html=True)


def format_pct_change(value):
    """格式化涨跌幅"""
    if value > 0:
        return f'<span class="gain">+{value:.2f}%</span>'
    elif value < 0:
        return f'<span class="loss">{value:.2f}%</span>'
    else:
        return f'{value:.2f}%'


@st.fragment(run_every="0.5s")
def auto_refresh_display(engine, current_date, start_time, end_time, 
                         replay_speed_multiplier, top_n_stocks, top_n_sectors,
                         rapid_rise_window, rapid_rise_threshold):
    """
    自动刷新的数据展示区域（使用 fragment 实现局部刷新）
    """
    # 如果引擎正在加载或未就绪，跳过执行，避免 Fragment ID 冲突
    if 'engine' not in st.session_state or st.session_state.get('current_dir') is None:
        return
        
    # 初始化或获取回放时间
    if 'replay_time' not in st.session_state:
        if hasattr(engine, 'data_start_time') and engine.data_start_time:
            st.session_state.replay_time = engine.data_start_time
        else:
            st.session_state.replay_time = datetime.combine(current_date, start_time)
    
    end_datetime = datetime.combine(current_date, end_time)
    
    # 只有在自动刷新开启时才推进时间
    if st.session_state.get('auto_refresh', False):
        if st.session_state.replay_time < end_datetime:
            # 改进：由于刷新率提高到了 0.5s，每次推进的时间应该是 (倍速 * 0.5) 秒
            # 使用 milliseconds 避免浮点数精度问题
            increment_ms = int(replay_speed_multiplier * 500)
            new_time = st.session_state.replay_time + pd.Timedelta(milliseconds=increment_ms)
            
            # 跳过午休时间 (11:30-13:00)
            if new_time.time() >= time(11, 30) and new_time.time() < time(13, 0):
                # 如果推进后进入午休时间，直接跳到 13:00
                if st.session_state.replay_time.time() < time(11, 30):
                    new_time = datetime.combine(current_date, time(13, 0))
            
            # 修正：确保不超过结束时间 (15:00)，如果到达结束时间则停止自动刷新
            if new_time >= end_datetime:
                new_time = end_datetime
                st.session_state.auto_refresh = False
            
            st.session_state.replay_time = new_time
    
    current_time = st.session_state.replay_time
    
    # 显示当前时间
    st.markdown(
        f"<h2 style='text-align: center;'>⏰ {current_time.strftime('%H:%M:%S')}</h2>",
        unsafe_allow_html=True
    )
    
    # 时间轴滑块（手动定位）
    start_datetime = datetime.combine(current_date, start_time)
    if hasattr(engine, 'data_start_time') and engine.data_start_time:
        start_datetime = engine.data_start_time
    
    # 确保当前时间在范围内 (鲁棒性检查)
    slider_value = current_time
    if slider_value < start_datetime: slider_value = start_datetime
    if slider_value > end_datetime: slider_value = end_datetime

    # 创建滑块和控制按钮
    col_slider, col_btn_play, col_btn_reset, col_time = st.columns([6, 0.8, 0.8, 1.2])
    
    # 确保时间类型统一为 python datetime (避免 pandas Timestamp 导致的 streamlit 错误)
    def to_pydatetime(dt):
        if hasattr(dt, 'to_pydatetime'):
            return dt.to_pydatetime()
        return dt

    start_datetime = to_pydatetime(start_datetime)
    end_datetime = to_pydatetime(end_datetime)
    slider_value = to_pydatetime(slider_value)

    with col_slider:
        # 使用 datetime 对象作为滑块，支持 format 显示时间预览
        new_replay_time = st.slider(
            "🕐 时间轴",
            min_value=start_datetime,
            max_value=end_datetime,
            value=slider_value,
            step=pd.Timedelta(seconds=1).to_pytimedelta(), # step 也要统一
            format="HH:mm:ss",
            label_visibility="collapsed"
        )
        
    with col_btn_play:
        # 根据状态显示不同的按钮
        is_playing = st.session_state.get('auto_refresh', False)
        if is_playing:
            if st.button("⏸️", help="暂停", width="stretch"):
                st.session_state.auto_refresh = False
        else:
            if st.button("▶️", help="播放", width="stretch"):
                st.session_state.auto_refresh = True

    with col_btn_reset:
        if st.button("🔄", help="重置", width="stretch"):
            if 'replay_time' in st.session_state:
                del st.session_state.replay_time
            st.session_state.auto_refresh = False

    with col_time:
        # 垂直居中对齐时间
        st.markdown(f"<div style='line-height: 2.2;'>⏱️ {new_replay_time.strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)
    
    # 如果用户拖动了滑块，更新时间
    if new_replay_time != slider_value:
        st.session_state.replay_time = new_replay_time
        st.session_state.auto_refresh = False  # 拖动时自动暂停
    
    # 获取快照
    snapshot = engine.get_snapshot_at_time(current_time)
    
    # --- 检查是否处于板块详情模式 (Drill-down) ---
    if st.session_state.get('active_sector'):
        render_sector_detail_view(engine, snapshot)
        return
    # ----------------------------------------
    
    # 显示市场统计 - 使用自定义样式确保文字清晰
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);">
                <p style="color: rgba(255, 255, 255, 0.9); font-size: 14px; margin: 0 0 8px 0; font-weight: 500;">总股票数</p>
                <p style="color: #FFFFFF; font-size: 32px; margin: 0; font-weight: bold; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);">{}</p>
            </div>
            """.format(snapshot['stats']['total_stocks']),
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);">
                <p style="color: rgba(255, 255, 255, 0.9); font-size: 14px; margin: 0 0 8px 0; font-weight: 500;">上涨</p>
                <p style="color: #FFFFFF; font-size: 32px; margin: 0; font-weight: bold; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);">{}</p>
            </div>
            """.format(snapshot['stats']['up_count']),
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #ff4b2b 0%, #ff416c 100%); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);">
                <p style="color: rgba(255, 255, 255, 0.9); font-size: 14px; margin: 0 0 8px 0; font-weight: 500;">涨停</p>
                <p style="color: #FFFFFF; font-size: 32px; margin: 0; font-weight: bold; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);">{}</p>
            </div>
            """.format(snapshot['stats']['limit_up_count']),
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);">
                <p style="color: rgba(255, 255, 255, 0.9); font-size: 14px; margin: 0 0 8px 0; font-weight: 500;">下跌</p>
                <p style="color: #FFFFFF; font-size: 32px; margin: 0; font-weight: bold; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);">{}</p>
            </div>
            """.format(snapshot['stats']['down_count']),
            unsafe_allow_html=True
        )
    with col5:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #1d976c 0%, #34a853 100%); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);">
                <p style="color: rgba(255, 255, 255, 0.9); font-size: 14px; margin: 0 0 8px 0; font-weight: 500;">跌停</p>
                <p style="color: #FFFFFF; font-size: 32px; margin: 0; font-weight: bold; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);">{}</p>
            </div>
            """.format(snapshot['stats']['limit_down_count']),
            unsafe_allow_html=True
        )
    with col6:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);">
                <p style="color: rgba(255, 255, 255, 0.9); font-size: 14px; margin: 0 0 8px 0; font-weight: 500;">平盘</p>
                <p style="color: #FFFFFF; font-size: 32px; margin: 0; font-weight: bold; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);">{}</p>
            </div>
            """.format(snapshot['stats']['flat_count']),
            unsafe_allow_html=True
        )
    
    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 个股排行", 
        "🏢 板块排行", 
        "⚡ 异动监控", 
        "📈 分时图",
        "🔥 板块热度"
    ])
    
    with tab1:
        st.subheader("📊 个股涨幅排行")
        stock_rankings = engine.calculate_stock_rankings(snapshot, top_n=6000)
        
        if not stock_rankings.empty:
            # 布局：在表格上方添加排序控制
            col_ctrl, col_info = st.columns([1, 3])
            with col_ctrl:
                # 使用单选框来控制排序，这样刷新时能保持状态
                sort_mode = st.radio(
                    "排序模式", 
                    ["涨幅榜 🔴", "跌幅榜 🟢"], 
                    horizontal=True,
                    label_visibility="collapsed"
                )
            with col_info:
                st.info(f"💡 当前查看：{sort_mode}  | 数据实时刷新中...")

            # 根据选择的模式对数据进行排序
            if "跌幅榜" in sort_mode:
                # 升序排列（跌幅大的在前）
                stock_rankings = stock_rankings.sort_values(by='pct_change', ascending=True)
            else:
                # 降序排列（涨幅大的在前，默认已经排好，但为了保险再排一次）
                stock_rankings = stock_rankings.sort_values(by='pct_change', ascending=False)

            # 构造用于显示的 DataFrame
            # 必须 reset_index，否则 on_select 返回的 row index 可能会与 iloc 不匹配（如果原始 index 不连续）
            df_display = stock_rankings.copy().reset_index(drop=True)
            
            # 添加排名列（注意：这只是初始排名，用户排序后排名列数字不会变）
            df_display['排名'] = df_display.index + 1
            
            # 重命名列为中文
            df_show = df_display.rename(columns={
                'stock_name': '名称',
                'stock_code': '代码',
                'price': '价格',
                'pct_change': '涨跌幅',
                'volume': '成交量'
            })
            
            # 只保留需要的列
            df_show = df_show[['排名', '代码', '名称', '价格', '涨跌幅', '成交量']]
            
            # 定义样式函数：红涨绿跌
            def color_change(val):
                if val > 0:
                    return 'color: #ff4444'  # 红色
                elif val < 0:
                    return 'color: #00dd88'  # 绿色
                return 'color: #e0e0e0'      # 默认灰白
            
            # 应用样式
            # 注意：Styler 对象传给 dataframe 后，on_select 返回的索引依然对应原始 DataFrame 的索引
            styled_df = df_show.style.map(color_change, subset=['涨跌幅'])
            
            # 配置列显示格式
            column_config = {
                "排名": st.column_config.NumberColumn("排名", width="small", format="%d"),
                "代码": st.column_config.TextColumn("代码", width="medium"),
                "名称": st.column_config.TextColumn("名称", width="medium"),
                # 价格去掉 ¥ 符号
                "价格": st.column_config.NumberColumn("价格", width="medium", format="%.2f"),
                "涨跌幅": st.column_config.NumberColumn("涨跌幅", width="medium", format="%.2f%%"),
                "成交量": st.column_config.NumberColumn("成交量", width="medium", format="%d")
            }
            
            # 显示表格（简化版，不用on_select）
            st.dataframe(
                styled_df,
                column_config=column_config,
                width="stretch",
                hide_index=True,
                key="stock_ranking_display"
            )
            
        else:
            st.info("暂无数据")
    
    with tab2:
        st.subheader("🏢 板块涨幅排行")
        sector_rankings = engine.calculate_sector_rankings(snapshot, top_n=top_n_sectors)
        
        if not sector_rankings.empty:
            display_df = sector_rankings.copy()
            display_df['平均涨跌幅'] = display_df['avg_pct_change'].apply(
                lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%"
            )
            
            st.dataframe(
                display_df[['sector', '平均涨跌幅', 'stock_count']],
                column_config={
                    'sector': '板块',
                    'stock_count': '成分股数量',
                },
                hide_index=False,
                width='stretch'
            )
        else:
            st.info("暂无数据(请确保已加载行业映射文件)")
    
    with tab3:
        st.subheader("⚡ 异动监控")
        
        # 添加异动监控条件设置
        col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
        
        with col_filter1:
            monitor_rise = st.checkbox("监控涨幅", value=True, help="监控快速拉升", key="cb_monitor_rise")
            if monitor_rise:
                rise_threshold = st.slider("涨幅阈值(%)", 1.0, 10.0, rapid_rise_threshold, 0.5, key="rise_thresh")
            else:
                rise_threshold = None
        
        with col_filter2:
            monitor_fall = st.checkbox("监控跌幅", value=True, help="监控快速下跌", key="cb_monitor_fall")
            if monitor_fall:
                fall_threshold = st.slider("跌幅阈值(%)", -10.0, -1.0, -rapid_rise_threshold, 0.5, key="fall_thresh")
            else:
                fall_threshold = None
        
        with col_filter3:
            monitor_limit = st.checkbox("监控涨跌停", value=True, help="监控封板与炸板异动", key="cb_monitor_limit")
            enable_volume_filter = st.checkbox("成交额过滤", value=False, help="只显示成交额达到一定金额的异动", key="cb_vol_filter")
            if enable_volume_filter:
                volume_threshold = st.number_input("最小成交额(万元)", min_value=0, value=100, step=50, key="vol_thresh")
            else:
                volume_threshold = None
        
        # 异动监控布局：增加控制按钮
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 8])
        with col_ctrl1:
            if st.button("🧹 清除异动日志", width="stretch"):
                st.session_state.abnormal_log = []
                st.session_state.processed_events = set()
        with col_ctrl2:
            monitor_paused = st.toggle("⏸️ 暂停监控更新", value=False)
            
        # 调用后端检测逻辑
        if not monitor_paused:
            # 1. 监测涨跌停（封板/炸板）
            limit_events = engine.detect_limit_movements() if monitor_limit else []
            for event in limit_events:
                # 生成指纹：股票代码 + 事件类型 + 时间(分钟级别去重)
                event_fingerprint = f"{event['stock_code']}_{event['event_type']}_{event['time'][:5]}"
                if event_fingerprint not in st.session_state.processed_events:
                    st.session_state.processed_events.add(event_fingerprint)
                    # 添加到日志开头
                    st.session_state.abnormal_log.insert(0, {
                        'time': event['time'],
                        'stock_code': event['stock_code'],
                        'stock_name': event['stock_name'],
                        'type': event['desc'],
                        'detail': f"价格: ¥{event['price']:.2f} ({event['pct_change']:+.2f}%)",
                        'color': 'red' if 'up' in event['event_type'] else 'green',
                        'timestamp': time_module.time()
                    })

            # 2. 监测快速涨跌
            abnormal_stocks = engine.detect_abnormal_movement(
                time_window_minutes=rapid_rise_window,
                rise_threshold=rise_threshold,
                fall_threshold=fall_threshold,
                volume_threshold=volume_threshold
            )
            for stock in abnormal_stocks:
                # 快速涨跌去重逻辑更严格一些：每个股票每 5 分钟只记录一次同类型异动
                # 这里使用当前回放时间的小时:分钟作为指纹的一部分
                replay_time_str = st.session_state.replay_time.strftime('%H:%M')
                event_fingerprint = f"{stock['stock_code']}_{stock['movement_type']}_{replay_time_str}"
                
                if event_fingerprint not in st.session_state.processed_events:
                    st.session_state.processed_events.add(event_fingerprint)
                    m_type_name = "🚀 快速拉升" if stock['movement_type'] == 'rise' else "📉 快速下挫"
                    st.session_state.abnormal_log.insert(0, {
                        'time': st.session_state.replay_time.strftime('%H:%M:%S'),
                        'stock_code': stock['stock_code'],
                        'stock_name': engine.get_stock_name(stock['stock_code']),
                        'type': m_type_name,
                        'detail': f"{rapid_rise_window}分钟内变动 {stock['pct_change']:+.2f}% (¥{stock['start_price']:.2f} -> ¥{stock['end_price']:.2f})",
                        'color': 'red' if stock['movement_type'] == 'rise' else 'green',
                        'timestamp': time_module.time()
                    })

        # 日志限制：只保留最近 100 条
        if len(st.session_state.abnormal_log) > 100:
            st.session_state.abnormal_log = st.session_state.abnormal_log[:100]

        # 展示异动日志
        if st.session_state.abnormal_log:
            log_df = pd.DataFrame(st.session_state.abnormal_log)
            
            # 使用原生 dataframe 渲染，增加一些样式指示
            st.dataframe(
                log_df[['time', 'stock_code', 'stock_name', 'type', 'detail']],
                column_config={
                    'time': '时间',
                    'stock_code': '代码',
                    'stock_name': '名称',
                    'type': '异动类型',
                    'detail': '详情'
                },
                hide_index=True,
                width="stretch",
                height=600
            )
            st.caption(f"💡 当前日志共 {len(st.session_state.abnormal_log)} 条记录。已自动过滤重复项。")
        # 显示当前监控条件
        conditions = []
        if monitor_limit: conditions.append("封板/炸板")
        if rise_threshold: conditions.append(f"涨幅>{rise_threshold}%")
        if fall_threshold: conditions.append(f"跌幅<{fall_threshold}%")
        condition_text = " + ".join(conditions) if conditions else "无"
        st.caption(f"⚙️ 当前监控中（窗口:{rapid_rise_window}分）: {condition_text}")
    
    with tab4:
        st.subheader("🔍 个股分时查看器")
        
        if len(engine.all_data) > 0:
            # 获取个股排行以供默认选择（如果没有搜索的话）
            stock_rankings = engine.calculate_stock_rankings(snapshot, top_n=50)
            
            # 统一的说明
            st.caption("💡 可在左侧搜索框输入代码或名称，从右侧选择个股查看全天数据")
            
            # 搜索框和选择器（同一水平线）
            col_search, col_select = st.columns([2, 3])
            
            with col_search:
                search_text = st.text_input(
                    "搜索股票",
                    placeholder="输入代码或名称...",
                    help="支持模糊搜索",
                    label_visibility="collapsed",
                    key="stock_search_input"
                )
            
            # 创建完整的股票列表（所有已加载的股票）
            all_stock_options = []
            for code in sorted(engine.all_data.keys()):
                name = engine.get_stock_name(code)
                all_stock_options.append(f"{code} {name}")
            
            # 根据搜索文本过滤
            if search_text:
                filtered_options = [
                    opt for opt in all_stock_options 
                    if search_text.upper() in opt.upper()
                ]
            else:
                # 默认显示排行榜前50只
                filtered_options = [f"{row['stock_code']} {row['stock_name']}" 
                                  for _, row in stock_rankings.head(50).iterrows()]
            
            with col_select:
                if filtered_options:
                    selected_option = st.selectbox(
                        "选择股票" if not search_text else f"搜索结果 ({len(filtered_options)} 只)",
                        options=["不选择"] + filtered_options,
                        label_visibility="collapsed",
                        key="stock_detail_selector"
                    )
                else:
                    st.info("未找到匹配的股票")
                    selected_option = "不选择"
            
            # 显示股票详情
            if selected_option != "不选择":
                stock_code = selected_option.split()[0]
                stock_name = selected_option.split()[1]
                
                # 获取股票数据
                stock_data = engine.all_data.get(stock_code)
                
                if stock_data is not None and not stock_data.empty:
                    # ✅ 修改：直接使用全天数据，不再跟随回放时间筛选
                    display_data = stock_data
                    
                    # 创建标签页：分时图 和 逐笔交易
                    detail_tab1, detail_tab2 = st.tabs(["📈 分时图", "📋 逐笔交易"])
                    
                    with detail_tab1:
                        # 分时图
                        fig = make_subplots(
                            rows=2, cols=1,
                            row_heights=[0.7, 0.3],
                            vertical_spacing=0.05,
                            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
                        )
                        
                        # 价格线
                        fig.add_trace(
                            go.Scatter(
                                x=display_data['datetime'],
                                y=display_data['price'],
                                mode='lines',
                                name='价格',
                                line=dict(color='#1f77b4', width=1.5),
                                fill='tozeroy',
                                fillcolor='rgba(31, 119, 180, 0.1)'
                            ),
                            row=1, col=1
                        )
                        
                        # 昨收价参考线
                        if 'pre_close' in display_data.columns:
                            pre_close = display_data['pre_close'].iloc[0]
                            fig.add_hline(
                                y=pre_close,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text=f"昨收: {pre_close:.2f}",
                                row=1, col=1
                            )
                        
                        # 成交量柱状图
                        fig.add_trace(
                            go.Bar(
                                x=display_data['datetime'],
                                y=display_data['vol'],
                                name='成交量',
                                marker_color='rgba(100, 100, 255, 0.5)'
                            ),
                            row=2, col=1
                        )
                        
                        # 更新布局
                        current_price = display_data['price'].iloc[-1]
                        
                        # 确定昨收价和价格范围
                        if 'pre_close' in display_data.columns:
                            real_pre_close = display_data['pre_close'].iloc[0]
                            pct_change = (current_price - real_pre_close) / real_pre_close * 100
                        else:
                            real_pre_close = display_data['price'].iloc[0] # Fallback
                            pct_change = 0

                        # 计算基础涨跌停范围
                        base_limit = 0.2 if (stock_code.startswith('688') or stock_code.startswith('300') or stock_code.startswith('689')) else 0.1
                        if stock_code.startswith(('8', '4', '92')): base_limit = 0.3
                        
                        # 检查实际价格波动是否超过限制
                        max_price = display_data['price'].max()
                        min_price = display_data['price'].min()
                        max_dev = max(abs(max_price - real_pre_close), abs(min_price - real_pre_close)) / real_pre_close
                        limit_ratio = max(base_limit, max_dev * 1.1)
                        
                        y_min = real_pre_close * (1 - limit_ratio)
                        y_max = real_pre_close * (1 + limit_ratio)

                        fig.update_layout(
                            title=f"{stock_code} {stock_name} - 当前: ¥{current_price:.2f} ({pct_change:+.2f}%)",
                            height=450,
                            showlegend=False,
                            hovermode='x unified',
                            margin=dict(l=0, r=0, t=40, b=0),
                            yaxis=dict(
                                title="价格",
                                range=[y_min, y_max],
                                tickformat=".2f",
                                gridcolor='rgba(128,128,128,0.2)'
                            ),
                            yaxis2=dict(
                                title="涨跌幅",
                                range=[-limit_ratio*100, limit_ratio*100],
                                tickformat=".1f",
                                ticksuffix="%",
                                showgrid=False
                            )
                        )
                        
                        fig.update_xaxes(tickformat="%H:%M")
                        st.plotly_chart(fig, width="stretch")
                        
                        # 显示统计信息
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
                        with stat_col1:
                            st.markdown(
                                """
                                <div style="
                                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    padding: 15px;
                                    border-radius: 8px;
                                    text-align: center;
                                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                                ">
                                    <p style="
                                        color: rgba(255, 255, 255, 0.9);
                                        font-size: 12px;
                                        margin: 0 0 5px 0;
                                        font-weight: 500;
                                    ">当前价</p>
                                    <p style="
                                        color: #FFFFFF;
                                        font-size: 24px;
                                        margin: 0;
                                        font-weight: bold;
                                        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
                                    ">¥{:.2f}</p>
                                </div>
                                """.format(current_price),
                                unsafe_allow_html=True
                            )
                        
                        with stat_col2:
                            pct_color = "#f093fb" if pct_change >= 0 else "#4facfe"
                            pct_color2 = "#f5576c" if pct_change >= 0 else "#00f2fe"
                            st.markdown(
                                """
                                <div style="
                                    background: linear-gradient(135deg, {} 0%, {} 100%);
                                    padding: 15px;
                                    border-radius: 8px;
                                    text-align: center;
                                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                                ">
                                    <p style="
                                        color: rgba(255, 255, 255, 0.9);
                                        font-size: 12px;
                                        margin: 0 0 5px 0;
                                        font-weight: 500;
                                    ">涨跌幅</p>
                                    <p style="
                                        color: #FFFFFF;
                                        font-size: 24px;
                                        margin: 0;
                                        font-weight: bold;
                                        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
                                    ">{:+.2f}%</p>
                                </div>
                                """.format(pct_color, pct_color2, pct_change),
                                unsafe_allow_html=True
                            )
                        
                        with stat_col3:
                            st.markdown(
                                """
                                <div style="
                                    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                                    padding: 15px;
                                    border-radius: 8px;
                                    text-align: center;
                                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                                ">
                                    <p style="
                                        color: rgba(255, 255, 255, 0.9);
                                        font-size: 12px;
                                        margin: 0 0 5px 0;
                                        font-weight: 500;
                                    ">成交量</p>
                                    <p style="
                                        color: #FFFFFF;
                                        font-size: 24px;
                                        margin: 0;
                                        font-weight: bold;
                                        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
                                    ">{:.0f}手</p>
                                </div>
                                """.format(display_data['vol'].sum()),
                                unsafe_allow_html=True
                            )
                        
                        with stat_col4:
                            if 'cum_volume' in display_data.columns:
                                total_amount = display_data['cum_volume'].iloc[-1] * current_price / 10000
                                st.markdown(
                                    """
                                    <div style="
                                        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                                        padding: 15px;
                                        border-radius: 8px;
                                        text-align: center;
                                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                                    ">
                                        <p style="
                                            color: rgba(255, 255, 255, 0.9);
                                            font-size: 12px;
                                            margin: 0 0 5px 0;
                                            font-weight: 500;
                                        ">成交额</p>
                                        <p style="
                                            color: #FFFFFF;
                                            font-size: 24px;
                                            margin: 0;
                                            font-weight: bold;
                                            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
                                        ">{:.1f}万</p>
                                    </div>
                                    """.format(total_amount),
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown("-")
                    
                    with detail_tab2:
                        # 逐笔交易明细
                        st.markdown(f"**全天交易明细（共 {len(display_data)} 笔）**")
                        
                        # 添加涨跌标识
                        tick_display = display_data.copy()
                        if len(tick_display) > 1:
                            tick_display['变化'] = tick_display['price'].diff()
                            tick_display['方向'] = tick_display['变化'].apply(
                                lambda x: '🔴 ↑' if x > 0 else ('🟢 ↓' if x < 0 else '⚪ ─')
                            )
                        else:
                            tick_display['方向'] = '⚪ ─'
                        
                        # 反向排序（最新的在前）
                        tick_display = tick_display.sort_values('datetime', ascending=False)
                        
                        # 格式化列
                        tick_display['时间'] = tick_display['datetime'].dt.strftime('%H:%M:%S')
                        tick_display['价格'] = tick_display['price'].apply(lambda x: f"¥{x:.2f}")
                        tick_display['成交量'] = tick_display['vol'].apply(lambda x: f"{int(x)}")
                        
                        # 选择显示的列
                        display_cols = ['时间', '价格', '成交量', '方向']
                        
                        # 显示数据表格（带滚动）
                        st.dataframe(
                            tick_display[display_cols],
                            column_config={
                                "时间": st.column_config.TextColumn("时间", width="medium"),
                                "价格": st.column_config.TextColumn("价格", width="medium"),
                                "成交量": st.column_config.TextColumn("成交量(手)", width="medium"),
                                "方向": st.column_config.TextColumn("方向", width="small"),
                            },
                            hide_index=True,
                            height=500,
                            width="stretch"
                        )
                        
                        # 添加下载按钮
                        st.divider()
                        csv = tick_display[display_cols].to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 下载逐笔数据 (CSV)",
                            data=csv,
                            file_name=f"{stock_code}_{stock_name}_全天逐笔.csv",
                            mime="text/csv",
                            width="stretch"
                        )
                else:
                    st.warning(f"❌ 未加载股票 {stock_code} 的数据")
        else:
            st.info("暂无数据")
        
    with tab5:
        heat_tab1, heat_tab2, heat_tab3 = st.tabs(["📊 热度卡片", "🗺️ 热力图", "🚀 拉升板块"])
        
        with heat_tab1:
            render_sector_analysis(engine, snapshot, top_n=10)
        
        with heat_tab2:
            render_sector_heatmap(engine, snapshot)
        
        with heat_tab3:
            render_rapid_rise_sectors(
                engine, 
                snapshot, 
                time_window=rapid_rise_window,
                threshold=rapid_rise_threshold,
                top_n=10
            )


def render_replay_page():
    st.title("📈 A股历史复盘系统")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 复盘配置")
        
        # 数据目录选择 - 支持两种结构
        # 1. 新结构: data/20251222/tick/
        # 2. 旧结构: data/tick_20251222/
        
        data_dirs = []
        
        # 查找包含数据的目录
        for date_dir in Path("data").glob("*"):
            if date_dir.is_dir() and date_dir.name.isdigit() and len(date_dir.name) == 8:
                # 检查是否包含合并后的数据文件（优先）或传统的 tick 文件夹
                merged_file = date_dir / "tick_data.parquet"
                tick_dir = date_dir / "tick"
                if merged_file.exists():
                    data_dirs.append(date_dir)
                elif tick_dir.exists() and tick_dir.is_dir():
                    data_dirs.append(tick_dir)
        
        # 查找旧结构 (tick_YYYYMMDD)
        old_dirs = list(Path("data").glob("tick_*"))
        for d in old_dirs:
            if d.is_dir():
                data_dirs.append(d)
        
        if not data_dirs:
            st.error("未找到数据目录,请先下载数据!")
            st.stop()
        
        # 自定义排序键
        def get_date_key(path):
            name = path.name
            if name.startswith('tick_'): return name.replace('tick_', '')
            if name == 'tick': return path.parent.name
            return name # 目录本身就是日期名 (如 20240103)
            
        data_dirs.sort(reverse=True, key=get_date_key)
        
        selected_dir = st.selectbox(
            "选择交易日期",
            options=data_dirs,
            format_func=get_date_key
        )
        
        st.divider()
        
        # 提取数据日期
        if selected_dir.name == 'tick':
            date_str = selected_dir.parent.name
        elif selected_dir.name.startswith('tick_'):
            date_str = selected_dir.name.replace("tick_", "")
        else:
            date_str = selected_dir.name
        try:
            current_date = datetime.strptime(date_str, "%Y%m%d").date()
        except:
            current_date = datetime.today().date()
            
        start_time = st.time_input("开始时间", value=time(9, 30))
        end_time = st.time_input("结束时间", value=time(15, 0))
        
        replay_speed_multiplier = st.select_slider(
            "回放速度",
            options=[1, 5, 10, 30, 60, 120, 300, 600],
            value=60,
            format_func=lambda x: f"{x}x"
        )
        
        st.caption(f"💡 每秒推进 {replay_speed_multiplier} 秒真实时间")
        
        st.divider()
        
        # 显示设置
        st.subheader("显示设置")
        
        # 板块映射源选择
        current_source = SECTOR_MAPPING_CONFIG.get('source', 'iwencai')
        sector_source = st.selectbox(
            "板块映射源",
            options=["iwencai", "eastmoney"],
            index=0 if current_source == 'iwencai' else 1,
            help="选择不同的板块映射格式：iwencai（更全面）或 eastmoney（传统分类）"
        )
        
        # 如果设置改变，更新 config 并重新加载
        if sector_source != current_source:
            SECTOR_MAPPING_CONFIG['source'] = sector_source
            if 'engine' in st.session_state:
                # 清空旧映射
                st.session_state.engine.industry_map = {}
                st.session_state.engine.concept_map = {}
                st.session_state.engine.region_map = {}
                # 重新加载
                st.session_state.engine.load_sector_mappings()
                st.toast(f"✅ 板块映射源已切换至 {sector_source}")
                st.rerun()
        
        top_n_stocks = st.number_input("个股排行显示数量", min_value=10, max_value=100, value=30)
        top_n_sectors = st.number_input("板块排行显示数量", min_value=5, max_value=50, value=15)
        
        rapid_rise_window = st.slider("异动检测时间窗口(分钟)", 1, 30, 5, help="检测股票在此时间窗口内的涨跌幅变化")
        rapid_rise_threshold = st.slider("异动幅度阈值(%)", 1.0, 10.0, 3.0, 0.5, help="默认涨跌幅阈值，可在异动监控页面单独调整")
    
    # 初始化引擎
    if 'engine' not in st.session_state or st.session_state.get('current_dir') != str(selected_dir):
        st.session_state.initialized = False
        with st.spinner("正在初始化复盘引擎..."):
            tick_data_file = selected_dir.parent / "tick_data.parquet"
            if tick_data_file.exists():
                st.session_state.engine = ReplayEngine(str(tick_data_file.parent / "tick"))
                #logging.info(f"使用优化格式: {tick_data_file}")
            else:
                st.session_state.engine = ReplayEngine(str(selected_dir))
            
            st.session_state.current_dir = str(selected_dir)
            st.session_state.loaded_stocks = set()
            st.session_state.data_date = current_date
            # 初始化异动日志记录
            st.session_state.abnormal_log = []
            st.session_state.processed_events = set() # 用于去重的事件指纹
        
        # 数据加载 - 使用多线程并行加载
        with st.spinner(f"正在加载 {current_date} 的全量数据..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 定义进度回调函数
            def update_progress(current, total):
                progress_bar.progress(current / total)
                status_text.text(f"⚡ 多线程加载中: {current}/{total} 只股票 ({current/total*100:.1f}%)")
            
            # 使用多线程加载（8个线程并行）
            loaded_count = st.session_state.engine.load_all_stocks_parallel(
                max_workers=8, 
                progress_callback=update_progress
            )
            
            # 更新已加载股票集合
            st.session_state.loaded_stocks = set(st.session_state.engine.all_data.keys())
            
            progress_bar.empty()
            status_text.empty()
            
            # 刷新一次股票名称映射（从parquet文件中提取）
            st.session_state.engine.load_stock_names()
            
            # 智能检测数据实际起始时间
            st.session_state.engine.detect_data_time_range()
        
        st.success(f"✅ 全量数据加载完成！已加载 {loaded_count} 只股票")
        
        # 显示数据时间范围
        if hasattr(st.session_state.engine, 'data_start_time') and st.session_state.engine.data_start_time:
            data_start = st.session_state.engine.data_start_time.strftime('%H:%M:%S')
            data_end = st.session_state.engine.data_end_time.strftime('%H:%M:%S')
            st.info(f"📊 数据时间范围: {data_start} - {data_end}")
        
        if 'replay_time' in st.session_state:
            del st.session_state.replay_time
            
        st.session_state.initialized = True
    
    # 检查回放时间逻辑
    engine = st.session_state.engine
    if 'replay_time' not in st.session_state or st.session_state.replay_time.date() != current_date:
        if hasattr(engine, 'data_start_time') and engine.data_start_time:
            st.session_state.replay_time = engine.data_start_time
        else:
            st.session_state.replay_time = datetime.combine(current_date, start_time)
    
    # 只有在引擎完全初始化后才渲染回放片段，防止初始化期间的 Fragment ID 错误
    if st.session_state.get('initialized', False):
        auto_refresh_display(
            engine=engine,
            current_date=current_date,
            start_time=start_time,
            end_time=end_time,
            replay_speed_multiplier=replay_speed_multiplier,
            top_n_stocks=top_n_stocks,
            top_n_sectors=top_n_sectors,
            rapid_rise_window=rapid_rise_window,
            rapid_rise_threshold=rapid_rise_threshold
        )
    else:
        st.info("⌛ 正在准备回放环境...")


def render_download_page():
    """渲染数据下载页面"""
    st.title("⬇️ A股历史数据下载")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📅 日期选择")
        # 使用 date_input 实现日历选择，支持范围
        selected_date = st.date_input(
            "选择下载日期 (支持选择时间段)",
            value=datetime.now().date(),
            min_value=datetime(2020, 1, 1).date(),
            max_value=datetime.now().date(),
            help="点击日历选择日期。如果是单日直接点击；如果是范围，先点开始日期再点结束日期。"
        )
        
        # 数据类型选择已合并
        
        st.subheader("⚙️ 下载配置")
        tick_workers = st.slider("并发下载线程数", 1, 100, 50, help="建议设置在 30-60 之间以平衡速度与稳定性")
    
    with col2:
        st.info("ℹ️ 说明：\n\n1. 系统会自动跳过周末。\n2. 下载的数据将保存在 `data/YYYYMMDD` 目录下。\n3. **重要改进**：由于底层的 `pytdx` 库已打补丁，现在下载分时成交的同时会自动提取准确的**昨日收盘价**，下载速度提升了 200%，且涨跌幅计算完全准确。")
        
        # 预览选择
        if isinstance(selected_date, tuple):
            if len(selected_date) == 2:
                start_date, end_date = selected_date
                days = (end_date - start_date).days + 1
                st.write(f"已选择范围: **{start_date}** 至 **{end_date}** (共 {days} 天)")
                date_range_mode = True
            else:
                st.write(f"已选择: **{selected_date[0]}**")
                date_range_mode = False
                start_date = end_date = selected_date[0]
        else:
            st.write(f"已选择: **{selected_date}**")
            date_range_mode = False
            start_date = end_date = selected_date

    st.markdown("---")
    
    # 下载按钮
    if st.button("🚀 开始下载全量行情数据", type="primary", width="stretch"):
        status_container = st.status("正在进行大规模行情同步...", expanded=True)
        
        try:
            # 准备日期列表
            from datetime import timedelta
            current = start_date
            dates_to_process = []
            
            while current <= end_date:
                if current.weekday() < 5:  # 跳过周末
                    dates_to_process.append(current)
                current += timedelta(days=1)
            
            if not dates_to_process:
                status_container.warning("所选范围内没有交易日（全是周末）")
                return
                
            progress_bar = status_container.progress(0, text="总进度")
            total_steps = len(dates_to_process)
            
            for idx, date_obj in enumerate(dates_to_process):
                date_str = date_obj.strftime('%Y%m%d')
                date_display = date_obj.strftime('%Y-%m-%d')
                
                status_container.write(f"👉 **正在同步: {date_display}**")
                
                # 下载分时数据 (已通过补丁集成昨收价)
                output_dir = f"data/{date_str}/tick"
                downloader = StockDataDownloader()
                
                # 创建子进度条
                task_progress = status_container.progress(0, text=f"正在采集数据...")
                
                # 定义回调函数
                def update_tick_progress(curr, total):
                    percent = min(curr / total, 1.0)
                    task_progress.progress(percent, text=f"📥 行情同步中: {curr}/{total} ({percent:.1%})")
                    
                # 开始下载
                downloader.download_all_stocks(int(date_str), max_workers=tick_workers, output_dir=output_dir, progress_callback=update_tick_progress)
                
                # 下载完成，清空或标记子进度条
                task_progress.empty()
                status_container.write(f"   - ✅ {date_display} 数据同步完成 (含高精度基准价)")
                
                # 更新总进度
                progress_bar.progress((idx + 1) / total_steps, text=f"总进度: {idx + 1}/{total_steps}")
            
            status_container.update(label="🎉 所有下载任务已完成！", state="complete", expanded=False)
            st.success("✅ 数据下载成功！请前往「历史复盘」页面选择对应日期进行回放。")
            st.balloons()
            
        except Exception as e:
            status_container.update(label="❌ 下载过程中发生错误", state="error")
            st.error(f"错误详情: {str(e)}")


def main():
    # 侧边栏导航
    st.sidebar.title("🧭 系统导航")
    page = st.sidebar.radio(
        "选择功能模块", 
        ["📺 历史复盘", "⬇️ 数据下载"],
        captions=["回放分时行情与热度", "获取最新的市场数据"]
    )
    
    st.sidebar.divider()
    
    if page == "📺 历史复盘":
        render_replay_page()
    else:
        render_download_page()


if __name__ == "__main__":
    main()
