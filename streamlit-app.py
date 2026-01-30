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
# 1. CẤU HÌNH & GIAO DIỆN (BRANDING)
# ==========================================
st.set_page_config(page_title="Báo Cáo Mộc Phát", layout="wide", page_icon="🌲")

PRIMARY = "#066839"    # Xanh Mộc Phát
BG_COLOR = "#F0F2F6"

# Hàm load logo
def get_base64_logo(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except: return None

# CSS Tùy chỉnh (Header Sticky & KPI)
st.markdown(f"""
<style>
    /* Header Sticky */
    .header-sticky {{
        position: sticky; top: 0; z-index: 999;
        background: white; border-bottom: 3px solid {PRIMARY};
        padding: 15px 20px; margin: -60px -50px 20px -50px;
        display: flex; align-items: center; gap: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }}
    .header-text h1 {{ margin: 0; color: {PRIMARY}; font-size: 26px; font-weight: 900; }}
    .header-text p {{ margin: 0; color: #555; font-size: 15px; font-weight: 500; }}
    
    /* KPI Cards */
    .kpi-card {{
        background: white; border-radius: 8px; padding: 20px;
        border-left: 5px solid {PRIMARY};
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center;
    }}
    .kpi-val {{ font-size: 28px; font-weight: 800; color: #333; }}
    .kpi-lbl {{ font-size: 14px; text-transform: uppercase; color: #666; font-weight: bold; margin-top: 5px; }}
</style>
""", unsafe_allow_html=True)

# Hiển thị Header
logo_b64 = get_base64_logo("mocphat_logo.png")
logo_html = f'<img src="data:image/png;base64,{logo_b64}" height="50">' if logo_b64 else "🌲"

st.markdown(f"""
<div class="header-sticky">
    <div>{logo_html}</div>
    <div class="header-text">
        <h1>BÁO CÁO HIỆU QUẢ SẢN XUẤT & KINH DOANH</h1>
        <p>Dữ liệu tổng hợp 2023 - 2025 | Mộc Phát Furniture</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DỮ LIỆU TỰ ĐỘNG
# ==========================================
@st.cache_data(ttl=3600)
def load_report_data():
    # TÊN FILE CỐ ĐỊNH - KHÔNG CẦN UPLOAD
    FILE_PATH = "Master_3Y_Clean.csv"
    
    if not os.path.exists(FILE_PATH):
        return None, f"⚠️ Không tìm thấy file dữ liệu '{FILE_PATH}'. Vui lòng copy file vào cùng thư mục với app.py."

    try:
        df = pd.read_csv(FILE_PATH)
        
        # 1. Chuẩn hóa cột
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # 2. Xử lý ngày tháng
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
        df = df[(df['year'] > 2020) & (df['month'].between(1, 12))]
        df['ym'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])
        
        # 3. Xử lý text & số
        cols_str = ['khach_hang', 'ma_hang', 'mau_son', 'xuong', 'khu_vuc', 'dim']
        for c in cols_str:
            if c not in df.columns: df[c] = "N/A"
            else: df[c] = df[c].fillna("N/A").astype(str)
            
        df['sl'] = pd.to_numeric(df['sl'], errors='coerce').fillna(0)
        
        # 4. Logic Nhóm Màu (Tự động map nếu thiếu file map)
        if 'nhom_mau' not in df.columns:
            def quick_color_map(c):
                c = c.upper()
                if any(x in c for x in ['WHITE', 'CREAM', 'IVORY']): return 'WHITE'
                if any(x in c for x in ['BLACK', 'CHARCOAL']): return 'BLACK'
                if any(x in c for x in ['BROWN', 'WALNUT', 'ESPRESSO']): return 'BROWN'
                if any(x in c for x in ['GREY', 'GRAY']): return 'GREY'
                if any(x in c for x in ['NATURAL', 'OAK']): return 'NATURAL'
                return 'KHÁC'
            df['nhom_mau'] = df['mau_son'].apply(quick_color_map)

        # 5. Xử lý USB
        if 'is_usb' in df.columns:
            df['is_usb'] = df['is_usb'].astype(str).replace({'True': 'Có USB', 'False': 'Không', 'nan': 'Không'})

        return df, None
    except Exception as e:
        return None, str(e)

# Load dữ liệu ngay khi vào App
df_raw, error = load_report_data()

if error:
    st.error(error)
    st.stop()

# ==========================================
# 3. SIDEBAR (CHỈ LỌC - KHÔNG UPLOAD)
# ==========================================
st.sidebar.markdown("### 🎯 BỘ LỌC BÁO CÁO")

# Lọc Năm
all_years = sorted(df_raw['year'].unique(), reverse=True)
sel_years = st.sidebar.multiselect("Chọn Năm", all_years, default=all_years[:1]) # Mặc định chọn năm mới nhất

# Lọc Xưởng (Tách Xưởng 1, Xưởng 2)
all_xuong = sorted(df_raw['xuong'].unique())
sel_xuong = st.sidebar.multiselect("Chọn Xưởng", all_xuong, default=all_xuong)

# Lọc Khách
all_cust = sorted(df_raw['khach_hang'].unique())
sel_cust = st.sidebar.multiselect("Chọn Khách Hàng", all_cust)

# Lọc SKU
search_sku = st.sidebar.text_input("Tìm Mã Hàng (SKU)", placeholder="VD: MP-102...")

# Áp dụng lọc
df = df_raw.copy()
if sel_years: df = df[df['year'].isin(sel_years)]
if sel_xuong: df = df[df['xuong'].isin(sel_xuong)]
if sel_cust: df = df[df['khach_hang'].isin(sel_cust)]
if search_sku: df = df[df['ma_hang'].str.contains(search_sku, case=False)]

# ==========================================
# 4. DASHBOARD CONTENT
# ==========================================

# --- KPI Highlight ---
k1, k2, k3, k4 = st.columns(4)
total_sl = df['sl'].sum()
total_sku = df['ma_hang'].nunique()
top_cust_val = df.groupby('khach_hang')['sl'].sum().max()
top_cust_name = df.groupby('khach_hang')['sl'].sum().idxmax()
growth_label = "So với cùng kỳ" # Placeholder

k1.markdown(f'<div class="kpi-card"><div class="kpi-val">{total_sl:,.0f}</div><div class="kpi-lbl">Tổng Sản Lượng</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi-card"><div class="kpi-val">{total_sku}</div><div class="kpi-lbl">Mã Hàng (SKU)</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi-card"><div class="kpi-val" style="font-size:20px">{top_cust_name}</div><div class="kpi-lbl">Khách Hàng Top 1</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi-card"><div class="kpi-val" style="font-size:20px">{df["xuong"].nunique()}</div><div class="kpi-lbl">Xưởng Tham Gia</div></div>', unsafe_allow_html=True)

st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 TỔNG QUAN & XU HƯỚNG", "📋 CHI TIẾT SẢN PHẨM", "🌍 THỊ TRƯỜNG"])

# TAB 1: BIỂU ĐỒ
with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Diễn biến sản xuất theo tháng")
        trend = df.groupby('ym')['sl'].sum().reset_index()
        fig_trend = px.area(trend, x='ym', y='sl', color_discrete_sequence=[PRIMARY])
        fig_trend.update_layout(xaxis_title="Thời gian", yaxis_title="Sản lượng", height=350)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with c2:
        st.subheader("Tỷ trọng theo Nhóm Màu")
        pie_data = df.groupby('nhom_mau')['sl'].sum().reset_index()
        fig_pie = px.pie(pie_data, values='sl', names='nhom_mau', 
                         color_discrete_sequence=px.colors.sequential.Greens_r, hole=0.4)
        fig_pie.update_layout(height=350, showlegend=False)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Top 10 Mã Hàng (SKU) Chủ lực")
    top_sku = df.groupby('ma_hang')['sl'].sum().nlargest(10).sort_values(ascending=True).reset_index()
    fig_bar = px.bar(top_sku, x='sl', y='ma_hang', orientation='h', text_auto='.2s',
                     color='sl', color_continuous_scale='Greens')
    st.plotly_chart(fig_bar, use_container_width=True)

# TAB 2: AG-GRID (Interactive Report)
with tab2:
    st.subheader("Bảng dữ liệu chi tiết")
    st.caption("Dùng chuột kéo thả cột, lọc hoặc tìm kiếm trực tiếp trên bảng bên dưới.")
    
    # Group data cho gọn
    grid_df = df.groupby(['ma_hang', 'khach_hang', 'xuong', 'nhom_mau', 'dim', 'is_usb']).agg(
        Tong_SL=('sl', 'sum'),
        Don_Hang_Cuoi=('ym', 'max')
    ).reset_index().sort_values('Tong_SL', ascending=False)
    
    grid_df['Don_Hang_Cuoi'] = grid_df['Don_Hang_Cuoi'].dt.strftime('%Y-%m')

    # Config AgGrid
    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)
    gb.configure_column("Tong_SL", type=["numericColumn", "numberColumnFilter"], precision=0)
    gb.configure_column("ma_hang", pinned=True)
    gridOptions = gb.build()

    AgGrid(grid_df, gridOptions=gridOptions, height=500, fit_columns_on_grid_load=False)

# TAB 3: KHÁCH HÀNG (PARETO)
with tab3:
    c3, c4 = st.columns([2, 1])
    with c3:
        st.subheader("Phân tích Pareto (80/20)")
        pareto = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
        pareto['cum_sl'] = pareto['sl'].cumsum()
        pareto['cum_perc'] = pareto['cum_sl'] / pareto['sl'].sum() * 100
        
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=pareto['khach_hang'], y=pareto['sl'], name='Sản lượng', marker_color=PRIMARY))
        fig_p.add_trace(go.Scatter(x=pareto['khach_hang'], y=pareto['cum_perc'], name='Tích lũy %', yaxis='y2', line=dict(color='red')))
        fig_p.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110]), showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True)
        
    with c4:
        st.subheader("Chi tiết theo Xưởng")
        xuong_stat = df.groupby('xuong')['sl'].sum().reset_index()
        st.dataframe(xuong_stat.style.format({"sl": "{:,.0f}"}), use_container_width=True)

# Footer
st.markdown("---")
st.caption(f"Báo cáo được trích xuất tự động từ hệ thống dữ liệu Mộc Phát | Ngày: {datetime.now().strftime('%d/%m/%Y')}")
