import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
import os
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN (BRANDING STYLE)
# ==========================================
st.set_page_config(page_title="Mộc Phát Analytics", layout="wide", page_icon="🌲")

PRIMARY = "#066839"    # Xanh Mộc Phát
ACCENT  = "#1B7D4F"
BG_COLOR = "#F4F6F9"
WARNING = "#FF8C00"
DANGER = "#D32F2F"
SUCCESS = "#2E7D32"

# Hàm load logo
def get_base64_logo(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

def fmt_num(n):
    return f"{n:,.0f}"

# CSS Custom - Giữ nguyên phong cách bạn thích
st.markdown(f"""
<style>
    .main {{ background-color: {BG_COLOR}; }}
    h1, h2, h3 {{ font-family: 'Segoe UI', sans-serif; color: #333; }}
    
    /* Sticky Header */
    .header-sticky {{
        position: sticky; top: 0; z-index: 999;
        background: white; border-bottom: 3px solid {PRIMARY};
        padding: 12px 20px; margin: -60px -50px 20px -50px;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}
    .app-title {{ font-size: 24px; font-weight: 800; color: {PRIMARY}; margin: 0; }}
    
    /* KPI Cards - Phong cách sạch sẽ */
    .kpi-card {{
        background: white; border-radius: 12px; padding: 20px;
        border-left: 5px solid {PRIMARY};
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }}
    .kpi-card:hover {{ transform: translateY(-3px); }}
    .kpi-val {{ font-size: 28px; font-weight: 800; color: #2C3E50; }}
    .kpi-lbl {{ font-size: 13px; text-transform: uppercase; color: #7F8C8D; font-weight: 600; }}
    .kpi-sub {{ font-size: 13px; font-weight: 600; margin-top: 5px; }}
    .pos {{ color: {SUCCESS}; }} 
    .neg {{ color: {DANGER}; }}
    
    /* Insight Box - Khung phân tích thông minh */
    .insight-box {{
        background-color: #E8F5E9; border-left: 4px solid {PRIMARY};
        padding: 15px; border-radius: 5px; margin-bottom: 20px;
    }}
    .insight-title {{ color: {PRIMARY}; font-weight: bold; margin-bottom: 5px; }}
</style>
""", unsafe_allow_html=True)

# Render Header
logo_b64 = get_base64_logo("mocphat_logo.png")
logo_img = f'<img src="data:image/png;base64,{logo_b64}" height="45">' if logo_b64 else "🌲"
st.markdown(f"""
<div class="header-sticky">
    <div style="display:flex; gap:15px; align-items:center;">
        {logo_img}
        <div>
            <div class="app-title">MỘC PHÁT INTELLIGENCE</div>
            <div style="font-size:14px; color:#666;">Hệ thống Báo cáo Quản trị Toàn diện</div>
        </div>
    </div>
    <div style="text-align:right;">
        <span style="font-weight:bold; color:{PRIMARY};">Dữ liệu Master 2023-2025</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (LOGIC PHÂN TÍCH SÂU)
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    FILE_NAME = "Master_2023_2025_PRO_clean.xlsx"
    if not os.path.exists(FILE_NAME): return None, f"⚠️ Không tìm thấy file {FILE_NAME}"

    try:
        df = pd.read_excel(FILE_NAME, engine='openpyxl')
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Xử lý thời gian
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
        df = df[(df['year'] > 2020) & (df['month'].between(1, 12))]
        df['ym'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])
        
        # Map Mùa vụ
        season_map = {12:'Đông', 1:'Đông', 2:'Đông', 3:'Xuân', 4:'Xuân', 5:'Xuân', 
                      6:'Hè', 7:'Hè', 8:'Hè', 9:'Thu', 10:'Thu', 11:'Thu'}
        df['mua'] = df['month'].map(season_map)
        
        # Điền dữ liệu thiếu
        cols_text = ['khach_hang', 'ma_hang', 'mau_son', 'khu_vuc', 'dim', 'mo_ta']
        for c in cols_text:
            if c not in df.columns: df[c] = "Unknown"
            else: df[c] = df[c].fillna("Unknown").astype(str).str.upper()
        df['sl'] = pd.to_numeric(df['sl'], errors='coerce').fillna(0)

        # Logic Nhóm Màu (Smart Bucketing)
        def bucket_color(v):
            if any(x in v for x in ["BROWN","COCOA","BRONZE","UMBER","NAU"]): return "NÂU/GỖ"
            if any(x in v for x in ["WHITE","CREAM","IVORY","TRANG"]): return "TRẮNG/KEM"
            if "BLACK" in v or "DEN" in v: return "ĐEN"
            if "GREY" in v or "GRAY" in v or "XAM" in v: return "XÁM"
            if any(x in v for x in ["NAT","OAK","WALNUT","HONEY"]): return "TỰ NHIÊN"
            return "KHÁC"
        
        if 'nhom_mau' not in df.columns:
            df['nhom_mau'] = df['mau_son'].apply(bucket_color)

        # Xử lý USB (cho phần Trend)
        if 'is_usb' in df.columns:
            df['is_usb_clean'] = df['is_usb'].astype(str).apply(lambda x: 'Có USB' if 'true' in x.lower() else 'Không USB')
        else:
            df['is_usb_clean'] = 'N/A'

        return df, None
    except Exception as e:
        return None, str(e)

df_raw, error = load_data()
if error: st.error(error); st.stop()

# ==========================================
# 3. SIDEBAR (KHÔNG CÒN BỘ LỌC XƯỞNG)
# ==========================================
st.sidebar.markdown("### 🎯 BỘ LỌC")
years = sorted(df_raw['year'].unique(), reverse=True)
sel_years = st.sidebar.multiselect("Năm", years, default=years)
sel_cust = st.sidebar.multiselect("Khách Hàng", sorted(df_raw['khach_hang'].unique()))

df = df_raw.copy()
if sel_years: df = df[df['year'].isin(sel_years)]
if sel_cust: df = df[df['khach_hang'].isin(sel_cust)]

if df.empty: st.warning("Không có dữ liệu!"); st.stop()

# ==========================================
# 4. KPI CARDS (TÍCH HỢP YoY)
# ==========================================
# Tính toán tăng trưởng
vol_by_year = df.groupby('year')['sl'].sum()
v24 = vol_by_year.get(2024, 0)
v23 = vol_by_year.get(2023, 0)
g24 = ((v24 - v23) / v23 * 100) if v23 > 0 else 0

top_cust = df.groupby('khach_hang')['sl'].sum().idxmax()
total_sku = df['ma_hang'].nunique()

c1, c2, c3, c4 = st.columns(4)

def kpi_card(col, val, label, sub_val, sub_label, is_growth=True):
    color = "pos" if (sub_val >= 0 if is_growth else sub_val) else "neg"
    icon = "▲" if (sub_val >= 0 if is_growth else False) else "▼"
    icon = "" if not is_growth else icon
    
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-val">{val}</div>
        <div class="kpi-lbl">{label}</div>
        <div class="kpi-sub {color}">
            {icon} {sub_val} <span style="color:#888; font-weight:normal;">{sub_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

kpi_card(c1, fmt_num(df['sl'].sum()), "TỔNG SẢN LƯỢNG", g24 if len(sel_years)>1 else 0, "% Tăng trưởng (2024 vs 2023)" if len(sel_years)>1 else "Trong giai đoạn này")
kpi_card(c2, fmt_num(total_sku), "MÃ HÀNG (SKU)", "Active", "Đang sản xuất", is_growth=False)
kpi_card(c3, top_cust, "KHÁCH HÀNG TOP 1", "VIP", "Đối tác chiến lược", is_growth=False)
# KPI thứ 4: Tỷ lệ USB (Trend công nghệ)
usb_pct = (df[df['is_usb_clean']=='Có USB']['sl'].sum() / df['sl'].sum() * 100)
kpi_card(c4, f"{usb_pct:.1f}%", "TỶ LỆ SP CÓ USB", "Tech", "Xu hướng công nghệ", is_growth=False)

st.markdown("---")

# ==========================================
# 5. TABS PHÂN TÍCH
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 TỔNG QUAN & XU HƯỚNG", 
    "🎨 SỨC KHỎE SẢN PHẨM", 
    "🌡️ MÙA VỤ (HEATMAP)", 
    "⚖️ KHÁCH HÀNG (PARETO)",
    "📋 DỮ LIỆU CHI TIẾT"
])

# --- TAB 1: TỔNG QUAN (Kết hợp Chart Đẹp + Anomaly + Insight Text) ---
with tab1:
    c1_left, c1_right = st.columns([2, 1])
    
    with c1_left:
        st.subheader("Diễn biến sản xuất & Dự báo xu hướng")
        # Chuẩn bị dữ liệu
        ts_data = df.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
        ts_data['ma3'] = ts_data['sl'].rolling(3).mean() # Moving Average
        
        # Vẽ biểu đồ kết hợp (Area + Line)
        fig = go.Figure()
        # Vùng thực tế (Style đẹp của bạn)
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['sl'], fill='tozeroy', 
                                 mode='lines+markers', name='Thực tế', 
                                 line=dict(color=PRIMARY, width=2),
                                 fillcolor="rgba(6, 104, 57, 0.1)"))
        # Đường xu hướng (Logic Copilot)
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['ma3'], mode='lines', 
                                 name='Xu hướng (TB 3 tháng)', 
                                 line=dict(color=WARNING, dash='dot', width=2)))
        
        fig.update_layout(height=400, xaxis_title="Thời gian", yaxis_title="Sản lượng", 
                          template="plotly_white", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with c1_right:
        st.subheader("Cơ cấu Màu sắc")
        # Pie Chart (Style bạn thích)
        pie_data = df.groupby('nhom_mau')['sl'].sum().reset_index()
        fig_pie = px.pie(pie_data, values='sl', names='nhom_mau', 
                         color_discrete_sequence=px.colors.qualitative.Prism, hole=0.5)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=False, height=300, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # AI Insights Box (Logic Copilot)
        last_m = ts_data.iloc[-1]
        prev_m = ts_data.iloc[-2] if len(ts_data)>1 else last_m
        mom = ((last_m['sl'] - prev_m['sl'])/prev_m['sl']*100) if prev_m['sl']>0 else 0
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">🤖 AI Phân tích nhanh:</div>
            <ul style="margin:0; padding-left:20px; font-size:14px;">
                <li>Tháng <b>{last_m['ym'].strftime('%m/%Y')}</b> đạt <b>{fmt_num(last_m['sl'])}</b> SP.</li>
                <li>Biến động tháng: <b style="color:{'green' if mom>0 else 'red'}">{mom:+.1f}%</b> so với tháng trước.</li>
                <li>Màu sắc chủ đạo: <b>{pie_data.sort_values('sl', ascending=False).iloc[0]['nhom_mau']}</b> vẫn chiếm ưu thế.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2: SẢN PHẨM (SKU) ---
with tab2:
    c2_1, c2_2 = st.columns(2)
    with c2_1:
        st.subheader("Top 10 SKU Chủ lực")
        top_sku = df.groupby('ma_hang')['sl'].sum().nlargest(10).sort_values(ascending=True).reset_index()
        fig_bar = px.bar(top_sku, x='sl', y='ma_hang', orientation='h', text_auto='.2s',
                         color='sl', color_continuous_scale='Greens', title="")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2_2:
        st.subheader("Xu hướng USB (Tech Trend)")
        # Stacked Bar Chart cho USB
        usb_trend = df.groupby(['year', 'is_usb_clean'])['sl'].sum().reset_index()
        fig_usb = px.bar(usb_trend, x='year', y='sl', color='is_usb_clean', barmode='group',
                         color_discrete_map={'Có USB': WARNING, 'Không USB': '#E0E0E0'})
        st.plotly_chart(fig_usb, use_container_width=True)

# --- TAB 3: MÙA VỤ (HEATMAP - Tính năng mới từ Copilot) ---
with tab3:
    st.subheader("Bản đồ nhiệt: Mùa vụ & Màu sắc")
    st.caption("Biểu đồ này giúp Kế hoạch biết mùa nào nên chuẩn bị sơn màu gì.")
    
    # Chuẩn bị dữ liệu Heatmap
    heat_data = df.groupby(['mua', 'nhom_mau'])['sl'].sum().reset_index()
    # Tính tỷ trọng % trong từng mùa
    heat_data['pct'] = heat_data['sl'] / heat_data.groupby('mua')['sl'].transform('sum')
    
    pivot = heat_data.pivot(index='mua', columns='nhom_mau', values='pct').fillna(0)
    pivot = pivot.reindex(['Xuân', 'Hè', 'Thu', 'Đông']) # Sắp xếp mùa
    
    fig_heat = px.imshow(pivot, text_auto='.0%', aspect="auto", 
                         color_continuous_scale='Greens', origin='upper')
    st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 4: KHÁCH HÀNG (PARETO) ---
with tab4:
    c4_1, c4_2 = st.columns([2, 1])
    
    with c4_1:
        st.subheader("Biểu đồ Pareto (80/20)")
        pareto = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
        pareto['cum'] = pareto['sl'].cumsum()
        pareto['perc'] = pareto['cum'] / pareto['sl'].sum() * 100
        
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=pareto['khach_hang'], y=pareto['sl'], name='Sản lượng', marker_color=PRIMARY))
        fig_p.add_trace(go.Scatter(x=pareto['khach_hang'], y=pareto['perc'], name='% Tích lũy', yaxis='y2', line=dict(color=DANGER, width=2)))
        fig_p.add_hline(y=80, line_dash="dash", annotation_text="Ngưỡng 80%")
        fig_p.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110]), showlegend=False, height=450)
        st.plotly_chart(fig_p, use_container_width=True)
        
    with c4_2:
        st.subheader("Top Khách Hàng Tăng Trưởng")
        # Tính tăng trưởng YoY cho từng khách
        curr_y = df['year'].max()
        prev_y = curr_y - 1
        v_curr = df[df['year']==curr_y].groupby('khach_hang')['sl'].sum()
        v_prev = df[df['year']==prev_y].groupby('khach_hang')['sl'].sum()
        growth = ((v_curr - v_prev)/v_prev*100).fillna(0).sort_values(ascending=False)
        
        st.dataframe(growth.head(10).rename("% Tăng trưởng"), height=400)

# --- TAB 5: DỮ LIỆU CHI TIẾT (AG-GRID) ---
with tab5:
    st.subheader("Tra cứu dữ liệu chi tiết")
    # Group data (Không còn cột Xưởng)
    grid_df = df.groupby(['ma_hang', 'khach_hang', 'nhom_mau', 'mua', 'year']).agg(
        Tong_SL=('sl', 'sum'),
        So_Don=('ym', 'count')
    ).reset_index().sort_values('Tong_SL', ascending=False)
    
    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('multiple', use_checkbox=True)
    gb.configure_column("Tong_SL", type=["numericColumn", "numberColumnFilter"], precision=0)
    gb.configure_column("ma_hang", pinned=True)
    
    AgGrid(grid_df, gridOptions=gb.build(), height=600, fit_columns_on_grid_load=False, theme='streamlit')

# Footer
st.markdown("---")
st.caption(f"© 2026 Mộc Phát Furniture | Dashboard Version 4.0 Final | Generated: {datetime.now().strftime('%d/%m/%Y')}")
