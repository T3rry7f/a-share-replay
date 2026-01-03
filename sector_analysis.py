"""
板块分析页面组件
提供行业、概念、地区三维度的板块热度分析
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots




def set_active_sector(name, sec_type):
    """设置当前活跃板块（用于钻取详情）"""
    st.session_state.active_sector = {'name': name, 'type': sec_type}

def close_sector_detail():
    """关闭板块详情（返回列表）"""
    st.session_state.active_sector = None



def render_stat_card(label, value, sub_text, bg_gradient):
    """渲染带渐变背景的统计卡片"""
    st.markdown(f"""
    <div style="
        background: {bg_gradient}; 
        padding: 15px; 
        border-radius: 10px; 
        text-align: center; 
        color: white; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">{label}</div>
        <div style="font-size: 26px; font-weight: bold; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">{value}</div>
        <div style="font-size: 12px; opacity: 0.8;">{sub_text}</div>
    </div>
    """, unsafe_allow_html=True)

def render_sector_detail_view(engine, snapshot):
    """
    渲染板块详情视图（全屏模式，替换原有仪表盘内容）
    支持实时刷新
    """
    active = st.session_state.get('active_sector')
    if not active:
        st.error("未选择板块")
        if st.button("返回"):
            close_sector_detail()
        return

    sector_name = active['name']
    sector_type = active['type']
    
    # 顶部导航区域
    # 使用 container 包裹以增加 padding
    with st.container():
        c1, c2 = st.columns([1, 10])
        with c1:
            # 增加一些上边距以对齐标题
            st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
            st.button("🔙 返回", on_click=close_sector_detail, use_container_width=True, type="secondary")
        with c2:
            st.markdown(f"""
            <h2 style='margin: 0; padding: 0;'>
                <span style='font-size: 24px;'>📊</span> {sector_name} 
                <span style='font-size: 16px; color: #888; font-weight: normal; margin-left: 10px;'>实时成份股行情</span>
            </h2>
            """, unsafe_allow_html=True)
            
    st.markdown("---")

    # 筛选该板块的股票
    sector_stocks = []
    
    # 获取板块内的股票列表
    if sector_type == 'industry':
        mapping = engine.industry_map
    elif sector_type == 'concept':
        mapping = engine.concept_map
    else:
        mapping = engine.region_map
        
    # 遍历映射找出属于该板块的股票
    for code, sectors in mapping.items():
        if sector_name in sectors:
            if code in snapshot['stocks']:
                stock_data = snapshot['stocks'][code]
                sector_stocks.append({
                    '代码': code,
                    '名称': engine.get_stock_name(code),
                    '最新价': stock_data['price'],
                    '涨跌幅': stock_data['pct_change'],
                    '成交量': stock_data['volume'],
                    '成交额': stock_data.get('amount', 0), # 若有
                    'raw_pct': stock_data['pct_change'] # 用于排序
                })
            
    if not sector_stocks:
        st.warning(f"暂无 {sector_name} 的成分股数据")
        return

    # 创建DataFrame
    df = pd.DataFrame(sector_stocks)
    
    # 汇总数据计算
    avg_change = df['raw_pct'].mean()
    up_count = len(df[df['raw_pct'] > 0])
    down_count = len(df[df['raw_pct'] < 0])
    flat_count = len(df[df['raw_pct'] == 0])
    total_vol = df['成交量'].sum()
    
    # 指标卡片区域
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        # 平均涨跌幅：红/绿渐变
        if avg_change > 0:
            bg = "linear-gradient(135deg, #FF6B6B 0%, #d63031 100%)"
        elif avg_change < 0:
            bg = "linear-gradient(135deg, #2ecc71 0%, #27ae60 100%)"
        else:
            bg = "linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%)"
        render_stat_card("板块平均涨跌", f"{avg_change:+.2f}%", "加权平均", bg)
        
    with m2:
        # 上涨家数：红色系
        bg = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
        render_stat_card("上涨家数", f"{up_count} 只", f"占比 {up_count/len(df):.0%}", bg)
        
    with m3:
        # 下跌家数：蓝色/绿色系
        bg = "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)" 
        render_stat_card("下跌家数", f"{down_count} 只", f"占比 {down_count/len(df):.0%}", bg)
        
    with m4:
        # 成交量：橙色/黄色系
        bg = "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
        render_stat_card("总成交量", f"{total_vol/10000:.0f} 万手", "实时累计", bg)
    
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # 排序控制栏
    # 使用 columns 来紧凑排列
    c_sort, c_space = st.columns([2, 5])
    with c_sort:
        sort_by = st.selectbox(
            "📋 列表排序", 
            ["涨幅从高到低 ⬇️", "涨幅从低到高 ⬆️", "成交量从高到低 ⬇️"],
            label_visibility="visible"
        )
    
    if "涨幅从高到低" in sort_by:
        df = df.sort_values('raw_pct', ascending=False)
    elif "涨幅从低到高" in sort_by:
        df = df.sort_values('raw_pct', ascending=True)
    elif "成交量" in sort_by:
        df = df.sort_values('成交量', ascending=False)
    
    # 格式化
    df['涨跌幅'] = df['raw_pct'].apply(lambda x: f"{x:+.2f}%")
    df['最新价'] = df['最新价'].apply(lambda x: f"¥{x:.2f}")
    df['成交量'] = df['成交量'].apply(lambda x: f"{int(x):,}")
    
    # 定义高亮样式
    def highlight_change(val):
        if '+' in val:
            return 'color: #ff4444'
        elif '-' in val and '0.00' not in val:
            return 'color: #00dd88'
        return ''

    # 显示可交互表格
    st.dataframe(
        df[['代码', '名称', '最新价', '涨跌幅', '成交量']].style.map(highlight_change, subset=['涨跌幅']),
        column_config={
            "代码": st.column_config.TextColumn("代码", width="small"),
            "名称": st.column_config.TextColumn("名称", width="medium"),
            "最新价": st.column_config.TextColumn("最新价", width="medium"),
            "涨跌幅": st.column_config.TextColumn("涨跌幅", width="medium"),
            "成交量": st.column_config.TextColumn("成交量(手)", width="medium"),
        },
        use_container_width=True,
        hide_index=True,
        height=600
    )


def render_sector_card(sector_name, avg_pct, stock_count, total_volume, rank, sector_type='industry', engine=None, snapshot=None):
    """
    渲染单个板块卡片
    Args:
        ...
        engine: 复盘引擎实例 (用于交互)
        snapshot: 市场快照 (此处主要占位，实际切换视图不需要snapshot)
    """
    # 确定颜色
    if avg_pct > 0:
        color = "#ff4444"  # 红色
        icon = "🔥"
    elif avg_pct < 0:
        color = "#00aa00"  # 绿色
        icon = "❄️"
    else:
        color = "#666666"  # 灰色
        icon = "⚪"
    
    # 创建卡片容器
    with st.container():
        # 自定义CSS样式卡片
        card_html = f"""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 5px;
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 1;">
                    <div style="font-size: 12px; opacity: 0.7;">#{rank}</div>
                    <div style="font-size: 16px; font-weight: bold; margin: 4px 0;">{icon} {sector_name}</div>
                    <div style="font-size: 12px; opacity: 0.7;">{stock_count} 只成分股</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 24px; font-weight: bold; color: {color}; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                        {avg_pct:+.2f}%
                    </div>
                    <div style="font-size: 12px; opacity: 0.7;">
                        {total_volume/10000:.0f}万手
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        
        # 添加透明按钮覆盖或在下方
        btn_key = f"btn_detail_{sector_type}_{rank}_{sector_name}"
        # 使用 on_click 回调更新 Session State
        st.button(
            "🔍 详情", 
            key=btn_key, 
            use_container_width=True,
            on_click=set_active_sector,
            args=(sector_name, sector_type)
        )


def render_sector_analysis(engine, snapshot, top_n=10):
    """
    渲染板块分析页面
    """
    st.header("📊 板块热度分析")
    
    # 创建三列布局
    col1, col2, col3 = st.columns(3)
    
    # 行业板块
    with col1:
        st.subheader("🏭 行业板块")
        industry_rankings = engine.calculate_sector_rankings(
            snapshot, 
            sector_type='industry', 
            top_n=top_n
        )
        
        if not industry_rankings.empty:
            for idx, row in industry_rankings.iterrows():
                render_sector_card(
                    row['sector'],
                    row['avg_pct_change'],
                    row['stock_count'],
                    row['total_volume'],
                    idx + 1,
                    'industry',
                    engine,
                    snapshot
                )
        else:
            st.info("暂无行业数据")
    
    # 概念板块
    with col2:
        st.subheader("💡 概念板块")
        concept_rankings = engine.calculate_sector_rankings(
            snapshot,
            sector_type='concept',
            top_n=top_n
        )
        
        if not concept_rankings.empty:
            for idx, row in concept_rankings.iterrows():
                render_sector_card(
                    row['sector'],
                    row['avg_pct_change'],
                    row['stock_count'],
                    row['total_volume'],
                    idx + 1,
                    'concept',
                    engine,
                    snapshot
                )
        else:
            st.info("暂无概念数据")
    
    # 地区板块
    with col3:
        st.subheader("🌏 地区板块")
        region_rankings = engine.calculate_sector_rankings(
            snapshot,
            sector_type='region',
            top_n=top_n
        )
        
        if not region_rankings.empty:
            for idx, row in region_rankings.iterrows():
                render_sector_card(
                    row['sector'],
                    row['avg_pct_change'],
                    row['stock_count'],
                    row['total_volume'],
                    idx + 1,
                    'region',
                    engine,
                    snapshot
                )
        else:
            st.info("暂无地区数据")


def render_sector_heatmap(engine, snapshot):
    """
    渲染板块热力图
    """
    st.subheader("🗺️ 板块热力图")
    
    # 选择维度
    dimension = st.radio(
        "选择维度",
        ["行业", "概念", "地区"],
        horizontal=True
    )
    
    # 映射维度类型
    sector_type_map = {
        "行业": "industry",
        "概念": "concept",
        "地区": "region"
    }
    
    sector_type = sector_type_map[dimension]
    
    # 获取数据
    rankings = engine.calculate_sector_rankings(
        snapshot,
        sector_type=sector_type,
        top_n=30
    )
    
    if not rankings.empty:
        # 创建热力图
        fig = go.Figure(data=go.Bar(
            x=rankings['avg_pct_change'],
            y=rankings['sector'],
            orientation='h',
            marker=dict(
                color=rankings['avg_pct_change'],
                colorscale='RdYlGn',
                colorbar=dict(title="涨跌幅%"),
                cmin=-5,
                cmax=5
            ),
            text=rankings['avg_pct_change'].apply(lambda x: f"{x:+.2f}%"),
            textposition='auto',
        ))
        
        fig.update_layout(
            title=f"{dimension}板块涨跌幅分布",
            xaxis_title="平均涨跌幅(%)",
            yaxis_title=dimension,
            height=800,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"暂无{dimension}数据")


def render_rapid_rise_sectors(engine, snapshot, time_window=5, threshold=3.0, top_n=10):
    """
    渲染快速拉升板块
    """
    st.subheader("🚀 快速拉升板块")
    
    # 获取快速拉升股票
    rapid_stocks = engine.detect_rapid_rise(
        time_window_minutes=time_window,
        pct_threshold=threshold
    )
    
    if not rapid_stocks:
        st.info(f"暂无{time_window}分钟内涨幅超过{threshold}%的板块")
        return
    
    # 统计各板块的拉升股票数
    sector_rapid_count = {
        'industry': {},
        'concept': {},
        'region': {}
    }
    
    for stock in rapid_stocks:
        code = stock['stock_code']
        
        # 行业
        industries = engine.industry_map.get(code, ['未知'])
        for industry in industries:
            sector_rapid_count['industry'][industry] = sector_rapid_count['industry'].get(industry, 0) + 1
        
        # 概念
        concepts = engine.concept_map.get(code, ['未知'])
        for concept in concepts:
            sector_rapid_count['concept'][concept] = sector_rapid_count['concept'].get(concept, 0) + 1
        
        # 地区
        regions = engine.region_map.get(code, ['未知'])
        for region in regions:
            sector_rapid_count['region'][region] = sector_rapid_count['region'].get(region, 0) + 1
    
    # 展示三维度拉升板块
    col1, col2, col3 = st.columns(3)
    
    def render_rapid_column(title, data, sector_type, col):
        with col:
            st.markdown(f"### {title}")
            if data:
                sorted_items = sorted(
                    data.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]
                
                for i, (sector, count) in enumerate(sorted_items):
                    # 使用 container 包裹 metric 和 button
                    with st.container():
                        st.metric(
                            label=sector,
                            value=f"{count}只",
                            delta="拉升中"
                        )
                        st.button(
                            "查看个股", 
                            key=f"rapid_btn_{sector_type}_{i}_{sector}",
                            on_click=set_active_sector,
                            args=(sector, sector_type)
                        )
            else:
                 st.info("无拉升")

    render_rapid_column("🏭 行业拉升", sector_rapid_count['industry'], 'industry', col1)
    render_rapid_column("💡 概念拉升", sector_rapid_count['concept'], 'concept', col2)
    render_rapid_column("🌏 地区拉升", sector_rapid_count['region'], 'region', col3)
