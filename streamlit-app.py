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
# 1. CẤU HÌNH & GIAO DIỆN
# ==========================================
st.set_page_config(page_title="Báo Cáo Mộc Phát", layout="wide", page_icon="🌲")

PRIMARY = "#066839"
BG_COLOR = "#F0F2F6"

# Load logo
def get_base64_logo(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# CSS
st.markdown(f"""
<style>
    .header-sticky {{
        position: sticky; top: 0; z-index: 999;
        background: white; border-bottom: 3px solid {PRIMARY};
        padding: 15px 20px; margin: -60px -50px 20px -50px;
        display: flex; align-items: center; gap: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }}
    .kpi-card {{
        background: white; border-radius: 8px; padding: 20px;
        border-left: 5px solid {PRIMARY};
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center;
    }}
    .kpi-val {{ font-size: 28px; font-weight: 800; color: #333; }}
    .kpi-lbl {{ font-size: 14px; text-transform: uppercase; color: #666; font-weight: bold; margin-top: 5px; }}
</style>
""", unsafe_allow_html=True)

# Header
logo_b64 = get_base64_logo("mocphat_logo.png")
logo_html = f'<img src="data:image/png;base64,{logo_b64}" height="50">' if logo_b64 else "🌲"
st.markdown(f"""
<div class="header-sticky">
    <div>{logo_html}</div>
    <div class="header-text">
        <h1 style="margin:0; color:{PRIMARY}; font-size:24px;">BÁO CÁO HIỆU QUẢ SẢN XUẤT & KINH DOANH</h1>
        <p style="margin:0; color:#666;">Dữ liệu 2023 - 2025 | Mộc Phát Furniture</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DỮ LIỆU TỪ EXCEL (.XLSX)
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    # Tên file EXCEL chính xác
    EXCEL_FILE = "Master_2023_2025_PRO_clean.xlsx"
    
    # Kiểm tra file tồn tại
    if not os.path.exists(EXCEL_FILE):
        return None, f"⚠️ Không tìm thấy file: {EXCEL_FILE}. Hãy để file này cùng thư mục với app.py"

    try:
        # Đọc file Excel (Giả định dữ liệu nằm ở Sheet đầu tiên hoặc Sheet tên 'Master_3Y_Clean')
        # Nếu muốn chỉ định sheet cụ thể, thêm tham số: sheet_name='Tên_Sheet'
        df = pd.read_excel(EXCEL_FILE, engine='openpyxl')
        
        # 1. Chuẩn hóa tên cột
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # 2. Xử lý ngày tháng (year, month)
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
        df = df[(df['year'] > 2020) & (df['month'].between(1, 12))]
        df['ym'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])
        
        # 3. Điền giá trị trống
        cols_str = ['khach_hang', 'ma_hang', 'mau_son', 'xuong', 'khu_vuc', 'dim', 'is_usb']
        for c in cols_str:
            if c not in df.columns: df[c] = "N/A"
            else: df[c] = df[c].fillna("N/A").astype(str)
            
        df['sl'] = pd.to_numeric(df['sl'], errors='coerce').fillna(0)
        
        # 4. Map Nhóm Màu (Nếu cột nhom_mau chưa có)
        if 'nhom_mau' not in df.columns:
            def map_color(c):
                c = c.upper()
                if 'WHITE' in c or 'CREAM' in c: return 'WHITE'
                if 'BLACK' in c: return 'BLACK'
                if 'BROWN' in c or 'WALNUT' in c: return 'BROWN'
                if 'GREY' in c or 'GRAY' in c: return 'GREY'
                if 'NATURAL' in c or 'OAK' in c: return 'NATURAL'
                return 'KHÁC'
            df['nhom_mau'] = df['mau_son'].apply(map_color)

        return df, None
    except Exception as e:
        return None, f"Lỗi đọc file Excel: {str(e)}"

# Load Data
df_raw, error = load_data()

if error:
    st.error(error)
    st.stop()

# ==========================================
# 3. SIDEBAR LỌC
# ==========================================
st.sidebar.markdown("### 🎯 BỘ LỌC")

# Năm
years = sorted(df_raw['year'].unique(), reverse=True)
sel_years = st.sidebar.multiselect("Năm", years, default=years)

# Xưởng
xuongs = sorted(df_raw['xuong'].unique())
sel_xuong = st.sidebar.multiselect("Xưởng SX", xuongs, default=xuongs)

# Khách
custs = sorted(df_raw['khach_hang'].unique())
sel_cust = st.sidebar.multiselect("Khách Hàng", custs)

# SKU Search
search = st.sidebar.text_input("Tìm SKU", "")

# Apply Filter
df = df_raw.copy()
if sel_years: df = df[df['year'].isin(sel_years)]
if sel_xuong: df = df[df['xuong'].isin(sel_xuong)]
if sel_cust: df = df[df['khach_hang'].isin(sel_cust)]
if search: df = df[df['ma_hang'].str.contains(search, case=False)]

# ==========================================
# 4. DASHBOARD
# ==========================================

# --- KPI ---
k1, k2, k3, k4 = st.columns(4)
total_sl = df['sl'].sum()
total_sku = df['ma_hang'].nunique()
top_cust = df.groupby('khach_hang')['sl'].sum().idxmax() if not df.empty else "N/A"
active_xuong = df['xuong'].nunique()

k1.markdown(f'<div class="kpi-card"><div class="kpi-val">{total_sl:,.0f}</div><div class="kpi-lbl">Tổng Sản Lượng</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi-card"><div class="kpi-val">{total_sku}</div><div class="kpi-lbl">Mã Hàng (SKU)</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi-card"><div class="kpi-val" style="font-size:18px">{top_cust}</div><div class="kpi-lbl">Khách Hàng #1</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi-card"><div class="kpi-val">{active_xuong}</div><div class="kpi-lbl">Xưởng Hoạt Động</div></div>', unsafe_allow_html=True)

st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 TỔNG QUAN", "📋 DATA CHI TIẾT", "🌍 KHÁCH HÀNG"])

# TAB 1: Chart
with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Biến động sản lượng")
        trend = df.groupby('ym')['sl'].sum().reset_index()
        fig = px.area(trend, x='ym', y='sl', color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Tỷ trọng Màu")
        pie = df.groupby('nhom_mau')['sl'].sum().reset_index()
        fig2 = px.pie(pie, values='sl', names='nhom_mau', color_discrete_sequence=px.colors.sequential.Greens_r, hole=0.4)
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

# TAB 2: AgGrid
with tab2:
    st.subheader("Bảng dữ liệu SKU")
    
    # Group lại cho gọn bảng
    grid_df = df.groupby(['ma_hang', 'khach_hang', 'xuong', 'nhom_mau', 'dim', 'is_usb']).agg(
        Tong_SL=('sl', 'sum'),
        Don_Hang_Cuoi=('ym', 'max')
    ).reset_index().sort_values('Tong_SL', ascending=False)
    
    grid_df['Don_Hang_Cuoi'] = grid_df['Don_Hang_Cuoi'].dt.strftime('%Y-%m')

    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)
    gb.configure_column("Tong_SL", type=["numericColumn", "numberColumnFilter"], precision=0)
    gb.configure_column("ma_hang", pinned=True)
    
    AgGrid(grid_df, gridOptions=gb.build(), height=500, fit_columns_on_grid_load=False)

# TAB 3: Pareto
with tab3:
    st.subheader("Pareto Khách Hàng (80/20)")
    pareto = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
    pareto['cum'] = pareto['sl'].cumsum()
    pareto['perc'] = pareto['cum'] / pareto['sl'].sum() * 100
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=pareto['khach_hang'], y=pareto['sl'], name='Sản lượng', marker_color=PRIMARY))
    fig3.add_trace(go.Scatter(x=pareto['khach_hang'], y=pareto['perc'], name='% Tích lũy', yaxis='y2', line=dict(color='red')))
    fig3.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110]), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.caption(f"Cập nhật: {datetime.now().strftime('%d/%m/%Y')}")
