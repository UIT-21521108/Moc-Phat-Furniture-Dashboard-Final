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
# 1. CẤU HÌNH & GIAO DIỆN (GIỮ NGUYÊN STYLE BẠN THÍCH)
# ==========================================
st.set_page_config(page_title="Mộc Phát Analytics Pro", layout="wide", page_icon="🌲")

PRIMARY = "#066839"    # Xanh Mộc Phát
ACCENT  = "#1B7D4F"
BG_COLOR = "#F4F6F9"
WARNING = "#FF8C00"    # Cam (Cảnh báo)
DANGER = "#D32F2F"     # Đỏ (Giảm/Xấu)
SUCCESS = "#2E7D32"    # Xanh (Tăng/Tốt)

# Hàm load logo
def get_base64_logo(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

def fmt_num(n):
    return f"{n:,.0f}"

# CSS Custom
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
    
    /* KPI Cards Advanced */
    .kpi-card {{
        background: white; border-radius: 12px; padding: 15px;
        border-left: 5px solid {PRIMARY};
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }}
    .kpi-card:hover {{ transform: translateY(-3px); }}
    .kpi-val {{ font-size: 26px; font-weight: 800; color: #2C3E50; }}
    .kpi-lbl {{ font-size: 13px; text-transform: uppercase; color: #7F8C8D; font-weight: 600; }}
    .kpi-sub {{ font-size: 13px; font-weight: 600; margin-top: 5px; }}
    .pos {{ color: {SUCCESS}; }} 
    .neg {{ color: {DANGER}; }}
    
    /* Insight Box (Hộp phân tích thông minh) */
    .insight-box {{
        background-color: #E8F5E9; border-left: 4px solid {PRIMARY};
        padding: 15px; border-radius: 5px; margin-bottom: 20px;
    }}
    .insight-title {{ color: {PRIMARY}; font-weight: bold; margin-bottom: 5px; font-size: 16px; }}
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
            <div style="font-size:14px; color:#666;">Báo cáo Sản xuất & Kinh doanh (Phiên bản Deep Dive)</div>
        </div>
    </div>
    <div style="text-align:right;">
        <span style="font-weight:bold; color:{PRIMARY};">Dữ liệu Master 2023-2025</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU NÂNG CAO (LOGIC PHÂN TÍCH)
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    FILE_NAME = "Master_2023_2025_PRO_clean.xlsx"
    if not os.path.exists(FILE_NAME): return None, f"⚠️ Không tìm thấy file {FILE_NAME}"

    try:
        df = pd.read_excel(FILE_NAME, engine='openpyxl')
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # 1. Xử lý thời gian
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
        df = df[(df['year'] > 2020) & (df['month'].between(1, 12))]
        df['ym'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])
        
        # 2. Tạo cột Mùa (Seasonality)
        season_map = {12:'Đông', 1:'Đông', 2:'Đông', 3:'Xuân', 4:'Xuân', 5:'Xuân', 
                      6:'Hè', 7:'Hè', 8:'Hè', 9:'Thu', 10:'Thu', 11:'Thu'}
        df['mua'] = df['month'].map(season_map)
        
        # 3. Chuẩn hóa dữ liệu Text
        cols_text = ['khach_hang', 'ma_hang', 'mau_son', 'khu_vuc', 'dim', 'mo_ta']
        for c in cols_text:
            if c not in df.columns: df[c] = "Unknown"
            else: df[c] = df[c].fillna("Unknown").astype(str).str.upper()
        df['sl'] = pd.to_numeric(df['sl'], errors='coerce').fillna(0)

        # 4. Logic Nhóm Màu (Smart Bucketing)
        def bucket_color(v):
            if any(x in v for x in ["BROWN","COCOA","BRONZE","UMBER","NAU"]): return "NÂU/GỖ"
            if any(x in v for x in ["WHITE","CREAM","IVORY","TRANG"]): return "TRẮNG/KEM"
            if "BLACK" in v or "DEN" in v: return "ĐEN"
            if "GREY" in v or "GRAY" in v or "XAM" in v: return "XÁM"
            if any(x in v for x in ["NAT","OAK","WALNUT","HONEY"]): return "TỰ NHIÊN"
            return "KHÁC"
        
        if 'nhom_mau' not in df.columns:
            df['nhom_mau'] = df['mau_son'].apply(bucket_color)

        # 5. Xử lý USB Trend
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
# 3. SIDEBAR (KHÔNG CHIA XƯỞNG)
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
# 4. KPI CARDS (CHI TIẾT & SÂU SẮC HƠN)
# ==========================================
st.subheader("🚀 HIỆU QUẢ KINH DOANH (YoY)")

# Tính toán tổng hợp theo năm
vol_by_year = df.groupby('year')['sl'].sum()
v23 = vol_by_year.get(2023, 0)
v24 = vol_by_year.get(2024, 0)
v25 = vol_by_year.get(2025, 0)

# Tính % Tăng trưởng
g24 = ((v24 - v23) / v23 * 100) if v23 > 0 else 0
g25 = ((v25 - v24) / v24 * 100) if v24 > 0 else 0

c1, c2, c3, c4 = st.columns(4)

def kpi_card(col, year_label, val, growth_val, compare_label="so với năm trước"):
    color_class = "pos" if growth_val >= 0 else "neg"
    icon = "▲" if growth_val >= 0 else "▼"
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-lbl">{year_label}</div>
        <div class="kpi-val">{fmt_num(val)}</div>
        <div class="kpi-sub {color_class}">
            {icon} {abs(growth_val):.1f}% <span style="color:#888; font-weight:normal;">{compare_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Hiển thị 3 năm để thấy bức tranh toàn cảnh
kpi_card(c1, "SẢN LƯỢNG 2023 (NỀN)", v23, 0, "(Năm gốc)")
kpi_card(c2, "SẢN LƯỢNG 2024", v24, g24, "vs 2023")
kpi_card(c3, "SẢN LƯỢNG 2025", v25, g25, "vs 2024")

# Card 4: Tổng quan Khách hàng (Sức khỏe thị trường)
total_cust = df['khach_hang'].nunique()
kpi_card(c4, "KHÁCH HÀNG ACTIVE", total_cust, 0, "Đối tác",)

st.markdown("---")

# ==========================================
# 5. TABS PHÂN TÍCH CHUYÊN SÂU
# ==========================================
t1, t2, t3, t4, t5 = st.tabs([
    "📊 TỔNG QUAN & DỰ BÁO", 
    "🎨 SỨC KHỎE SẢN PHẨM", 
    "🌡️ MÙA VỤ & NHIỆT KẾ", 
    "⚖️ KHÁCH HÀNG (PARETO)",
    "📋 DỮ LIỆU GỐC"
])

# --- TAB 1: TỔNG QUAN & DỰ BÁO (ANOMALY & FORECAST) ---
with t1:
    col_chart, col_text = st.columns([3, 1])
    
    with col_chart:
        st.subheader("📈 Xu hướng & Phát hiện Bất thường")
        
        # Chuẩn bị dữ liệu chuỗi thời gian
        ts_data = df.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
        
        # 1. Vẽ đường thực tế
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['sl'], mode='lines+markers', 
                                 name='Thực tế', line=dict(color=PRIMARY, width=3)))
        
        # 2. Tính toán Moving Average (Dự báo xu hướng)
        ts_data['ma3'] = ts_data['sl'].rolling(window=3).mean()
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['ma3'], mode='lines', 
                                 name='Trung bình 3 tháng', line=dict(color='orange', dash='dot')))
        
        # 3. Anomaly Detection (Bollinger Bands đơn giản - Logic Copilot)
        std = ts_data['sl'].rolling(window=3).std()
        upper = ts_data['ma3'] + (1.8 * std) # 1.8 độ lệch chuẩn
        anomalies = ts_data[ts_data['sl'] > upper]
        
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies['ym'], y=anomalies['sl'], mode='markers', 
                                     name='Đột biến (Anomaly)', marker=dict(color=DANGER, size=12, symbol='star')))

        fig.update_layout(height=400, xaxis_title="Thời gian", yaxis_title="Sản lượng", 
                          template="plotly_white", margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_text:
        # Automated Insight Text - Phần này bạn rất cần!
        if not ts_data.empty:
            last_month = ts_data.iloc[-1]
            prev_month = ts_data.iloc[-2] if len(ts_data) > 1 else last_month
            mom_growth = ((last_month['sl'] - prev_month['sl']) / prev_month['sl'] * 100) if prev_month['sl']>0 else 0
            
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">🤖 AI Insights:</div>
                <ul style="padding-left: 20px; font-size: 14px;">
                    <li><b>Tháng gần nhất ({last_month['ym'].strftime('%m/%Y')}):</b> Đạt <b>{fmt_num(last_month['sl'])}</b> sản phẩm.</li>
                    <li><b>Biến động MoM:</b> <span style="color:{'green' if mom_growth>0 else 'red'}">{mom_growth:+.1f}%</span> so với tháng trước.</li>
                    <li><b>Xu hướng MA3:</b> Đường trung bình 3 tháng đang {'đi lên' if ts_data['ma3'].iloc[-1] > ts_data['ma3'].iloc[-2] else 'đi xuống'}, báo hiệu nhu cầu ngắn hạn.</li>
                    <li><b>Cảnh báo:</b> Phát hiện <b>{len(anomalies)}</b> điểm đột biến bất thường (Dấu sao đỏ). Cần kiểm tra năng lực sản xuất tại các điểm này.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 2: SỨC KHỎE SẢN PHẨM (SKU HEALTH) ---
with t2:
    c_h1, c_h2 = st.columns(2)
    
    with c_h1:
        st.subheader("🎨 Phân tích Nhóm Màu (Pie Chart)")
        # Pie Chart phong cách bạn thích
        pie_data = df.groupby('nhom_mau')['sl'].sum().reset_index()
        fig_pie = px.pie(pie_data, values='sl', names='nhom_mau', 
                         title="Cơ cấu nhóm màu tổng thể",
                         color_discrete_sequence=px.colors.qualitative.Prism, hole=0.5)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c_h2:
        st.subheader("⚡ Tỷ lệ Sản phẩm Công nghệ (USB)")
        # Stacked Bar Chart theo năm để thấy xu hướng công nghệ
        usb_trend = df.groupby(['year', 'is_usb_clean'])['sl'].sum().reset_index()
        fig_usb = px.bar(usb_trend, x='year', y='sl', color='is_usb_clean', barmode='group',
                         title="Sự chuyển dịch sang sản phẩm có USB",
                         color_discrete_map={'Có USB': WARNING, 'Không USB': '#E0E0E0'})
        st.plotly_chart(fig_usb, use_container_width=True)
        
    st.info("💡 **Góc nhìn chiến lược:** Nếu tỷ lệ USB tăng trưởng mạnh qua các năm (cột Cam cao lên), Mộc Phát cần đàm phán hợp đồng dài hạn với nhà cung cấp linh kiện điện tử để tối ưu giá vốn.")

# --- TAB 3: MÙA VỤ (HEATMAP - TÍNH NĂNG MỚI) ---
with t3:
    st.subheader("🌡️ Heatmap Mùa vụ: Khi nào bán Màu gì?")
    st.caption("Biểu đồ nhiệt giúp Kế hoạch sản xuất chuẩn bị nguyên liệu (Sơn, Veneer) đúng thời điểm.")
    
    # Pivot Data cho Heatmap
    heat_data = df.groupby(['mua', 'nhom_mau'])['sl'].sum().reset_index()
    # Chuẩn hóa % theo mùa (Trong Mùa Xuân, màu nào bán chạy nhất?)
    heat_data['share'] = heat_data['sl'] / heat_data.groupby('mua')['sl'].transform('sum')
    
    heatmap_matrix = heat_data.pivot(index='mua', columns='nhom_mau', values='share').fillna(0)
    # Sắp xếp lại thứ tự mùa
    heatmap_matrix = heatmap_matrix.reindex(['Xuân', 'Hè', 'Thu', 'Đông'])
    
    fig_heat = px.imshow(heatmap_matrix, text_auto='.0%', aspect="auto",
                         color_continuous_scale='Greens', origin='upper')
    st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 4: PARETO 80/20 ---
with t4:
    col_p1, col_p2 = st.columns([2, 1])
    
    with col_p1:
        st.subheader("⚖️ Pareto Khách Hàng")
        pareto = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
        pareto['cum'] = pareto['sl'].cumsum()
        pareto['perc'] = pareto['cum'] / pareto['sl'].sum() * 100
        
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=pareto['khach_hang'], y=pareto['sl'], name='Sản lượng', marker_color=PRIMARY))
        fig_p.add_trace(go.Scatter(x=pareto['khach_hang'], y=pareto['perc'], name='% Tích lũy', yaxis='y2', line=dict(color=DANGER, width=2)))
        fig_p.add_hline(y=80, line_dash="dash", annotation_text="Ngưỡng 80%")
        fig_p.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110]), showlegend=False, height=450)
        st.plotly_chart(fig_p, use_container_width=True)
        
    with col_p2:
        st.subheader("Top Movers (Tăng trưởng)")
        # So sánh 2024 vs 2023 (hoặc năm gần nhất)
        curr_year = df['year'].max()
        prev_year = curr_year - 1
        
        vol_curr = df[df['year'] == curr_year].groupby('khach_hang')['sl'].sum()
        vol_prev = df[df['year'] == prev_year].groupby('khach_hang')['sl'].sum()
        
        growth = ((vol_curr - vol_prev) / vol_prev * 100).fillna(0).sort_values(ascending=False)
        
        st.dataframe(growth.head(10).rename("% Tăng trưởng"), height=400)

# --- TAB 5: DỮ LIỆU GỐC (AG-GRID) ---
with t5:
    st.subheader("📋 Tra cứu dữ liệu chi tiết")
    
    # Aggregation for Grid
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
st.caption(f"© 2026 Mộc Phát Furniture | Dashboard Version 5.0 Ultimate | Generated: {datetime.now().strftime('%d/%m/%Y')}")
