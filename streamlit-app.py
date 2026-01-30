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
# 1. CẤU HÌNH GIAO DIỆN & BRANDING
# ==========================================
st.set_page_config(page_title="Mộc Phát Intelligence", layout="wide", page_icon="🌲")

# Bảng màu chuẩn Mộc Phát & Phân tích
PRIMARY = "#066839"    # Xanh thương hiệu
SECONDARY = "#1B7D4F"  # Xanh phụ trợ
BG_COLOR = "#F4F6F9"   # Màu nền xám nhẹ
WARNING = "#FF8C00"    # Màu cảnh báo
DANGER = "#D32F2F"     # Màu nguy hiểm

# Load Logo
def get_base64_logo(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# Custom CSS cho giao diện "DA Pro"
st.markdown(f"""
<style>
    /* Tổng thể */
    .main {{ background-color: {BG_COLOR}; }}
    h1, h2, h3 {{ font-family: 'Segoe UI', sans-serif; color: #333; }}
    
    /* Header Sticky chuyên nghiệp */
    .header-sticky {{
        position: sticky; top: 0; z-index: 999;
        background: white; border-bottom: 3px solid {PRIMARY};
        padding: 12px 20px; margin: -60px -50px 20px -50px;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}
    .header-left {{ display: flex; align-items: center; gap: 15px; }}
    .app-title {{ font-size: 24px; font-weight: 800; color: {PRIMARY}; margin: 0; }}
    .app-subtitle {{ font-size: 14px; color: #666; margin: 0; }}
    
    /* KPI Cards phong cách Dashboard hiện đại */
    .kpi-card {{
        background: white; border-radius: 12px; padding: 20px;
        border-left: 5px solid {PRIMARY};
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }}
    .kpi-card:hover {{ transform: translateY(-5px); }}
    .kpi-val {{ font-size: 32px; font-weight: 800; color: #2C3E50; line-height: 1.2; }}
    .kpi-lbl {{ font-size: 13px; text-transform: uppercase; color: #7F8C8D; font-weight: 600; letter-spacing: 0.5px; }}
    .kpi-trend {{ font-size: 14px; font-weight: bold; margin-top: 5px; }}
    .trend-up {{ color: {PRIMARY}; }}
    .trend-down {{ color: {DANGER}; }}

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px; white-space: pre-wrap; background-color: white; border-radius: 5px 5px 0 0;
        box-shadow: 0 -1px 3px rgba(0,0,0,0.05);
    }}
    .stTabs [aria-selected="true"] {{ background-color: {PRIMARY}; color: white; }}
</style>
""", unsafe_allow_html=True)

# Header Render
logo_b64 = get_base64_logo("mocphat_logo.png")
logo_img = f'<img src="data:image/png;base64,{logo_b64}" height="45">' if logo_b64 else "🌲"

st.markdown(f"""
<div class="header-sticky">
    <div class="header-left">
        {logo_img}
        <div>
            <div class="app-title">MỘC PHÁT INTELLIGENCE</div>
            <div class="app-subtitle">Hệ thống Phân tích Dữ liệu Sản xuất & Kinh doanh</div>
        </div>
    </div>
    <div style="font-size: 14px; font-weight: 600; color: {PRIMARY};">
        Cập nhật: {datetime.now().strftime('%d/%m/%Y')}
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU THÔNG MINH
# ==========================================
@st.cache_data(ttl=3600)
def load_smart_data():
    FILE_NAME = "Master_2023_2025_PRO_clean.xlsx"
    
    if not os.path.exists(FILE_NAME):
        return None, f"⚠️ Lỗi: Không tìm thấy file '{FILE_NAME}'. Vui lòng đặt file vào cùng thư mục chạy app."

    try:
        df = pd.read_excel(FILE_NAME, engine='openpyxl')
        
        # 1. Chuẩn hóa tên cột
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # 2. Xử lý thời gian
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
        df = df[(df['year'] > 2020) & (df['month'].between(1, 12))]
        df['ym'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])
        df['quarter'] = df['month'].apply(lambda x: f"Q{(x-1)//3 + 1}") # Thêm cột Quý
        
        # 3. Điền dữ liệu thiếu
        cols_text = ['khach_hang', 'ma_hang', 'mau_son', 'xuong', 'khu_vuc', 'dim']
        for c in cols_text:
            if c not in df.columns: df[c] = "N/A"
            else: df[c] = df[c].fillna("N/A").astype(str).str.upper() # Chuyển chữ hoa để đồng nhất
            
        df['sl'] = pd.to_numeric(df['sl'], errors='coerce').fillna(0)
        
        # 4. Phân nhóm Màu (Color Logic) - Insight quan trọng
        def categorize_color(c):
            if any(x in c for x in ['WHITE', 'CREAM', 'IVORY', 'WASH', 'TRANG']): return 'NHÓM TRẮNG/SÁNG'
            if any(x in c for x in ['BLACK', 'CHARCOAL', 'EBONY', 'DEN']): return 'NHÓM ĐEN/TỐI'
            if any(x in c for x in ['BROWN', 'WALNUT', 'ESPRESSO', 'COCOA', 'NAU']): return 'NHÓM NÂU/GỖ'
            if any(x in c for x in ['GREY', 'GRAY', 'SLATE', 'XAM']): return 'NHÓM XÁM'
            if any(x in c for x in ['NATURAL', 'OAK', 'PINE', 'TU NHIEN']): return 'NHÓM TỰ NHIÊN'
            return 'MÀU KHÁC'
        
        if 'nhom_mau' not in df.columns:
            df['nhom_mau'] = df['mau_son'].apply(categorize_color)
            
        # 5. Xử lý USB (Feature Trend)
        if 'is_usb' in df.columns:
            # Chuẩn hóa về Boolean hoặc Text rõ ràng
            df['is_usb_clean'] = df['is_usb'].astype(str).apply(lambda x: 'Có USB' if x.lower() == 'true' else 'Không USB')
        else:
            df['is_usb_clean'] = 'N/A'

        return df, None
    except Exception as e:
        return None, f"Lỗi xử lý dữ liệu: {str(e)}"

# Load Data
df_raw, error = load_smart_data()
if error:
    st.error(error)
    st.stop()

# ==========================================
# 3. SIDEBAR & BỘ LỌC GLOBAL
# ==========================================
st.sidebar.markdown("### 🎯 BỘ LỌC DỮ LIỆU")

# Năm (Mặc định chọn năm gần nhất)
years = sorted(df_raw['year'].unique(), reverse=True)
sel_years = st.sidebar.multiselect("Chọn Năm", years, default=years)

# Xưởng
xuongs = sorted(df_raw['xuong'].unique())
sel_xuong = st.sidebar.multiselect("Chọn Xưởng", xuongs, default=xuongs)

# Khách Hàng
custs = sorted(df_raw['khach_hang'].unique())
sel_cust = st.sidebar.multiselect("Khách Hàng", custs)

# Filter Logic
df = df_raw.copy()
if sel_years: df = df[df['year'].isin(sel_years)]
if sel_xuong: df = df[df['xuong'].isin(sel_xuong)]
if sel_cust: df = df[df['khach_hang'].isin(sel_cust)]

if df.empty:
    st.warning("Không có dữ liệu phù hợp với bộ lọc!")
    st.stop()

# ==========================================
# 4. KPI HIGHLIGHTS
# ==========================================
c1, c2, c3, c4 = st.columns(4)

# Tính toán KPI
total_sl = df['sl'].sum()
total_sku = df['ma_hang'].nunique()
top_cust_name = df.groupby('khach_hang')['sl'].sum().idxmax()
usb_ratio = (df[df['is_usb_clean'] == 'Có USB']['sl'].sum() / total_sl * 100) if 'is_usb_clean' in df.columns else 0

# Render KPI Cards
def kpi(col, val, label, sub=""):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-val">{val}</div>
        <div class="kpi-lbl">{label}</div>
        <div class="kpi-trend trend-up">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

kpi(c1, f"{total_sl:,.0f}", "Tổng Sản Lượng (Chiếc)", "📦 Sản xuất")
kpi(c2, f"{total_sku}", "Mã Hàng (SKU) Active", "🧩 Độ phức tạp")
kpi(c3, top_cust_name, "Khách Hàng Top 1", "👑 Strategic Partner")
kpi(c4, f"{usb_ratio:.1f}%", "Tỷ lệ SP có USB", "⚡ Tech Trend")

st.markdown("---")

# ==========================================
# 5. PHÂN TÍCH CHUYÊN SÂU (INSIGHTS)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 XU HƯỚNG & MÙA VỤ", 
    "🎨 PHÂN TÍCH SẢN PHẨM", 
    "⚖️ NGUYÊN TẮC 80/20 (PARETO)", 
    "🔍 DỮ LIỆU CHI TIẾT"
])

# --- TAB 1: SEASONALITY & TREND ---
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Diễn biến đơn hàng theo Tháng")
        trend = df.groupby(['ym', 'year'])['sl'].sum().reset_index().sort_values('ym')
        fig = px.area(trend, x='ym', y='sl', title="Biểu đồ vùng sản lượng",
                      color_discrete_sequence=[PRIMARY])
        fig.update_traces(line_color=PRIMARY, fillcolor="rgba(6, 104, 57, 0.1)")
        fig.update_layout(xaxis_title="", yaxis_title="Sản lượng", height=350)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Phân bổ theo Quý")
        quarter_data = df.groupby('quarter')['sl'].sum().reset_index()
        fig_q = px.bar(quarter_data, x='quarter', y='sl', color='sl',
                       color_continuous_scale='Greens', title="Sản lượng theo Quý")
        fig_q.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_q, use_container_width=True)
    
    st.info("💡 **Insight:** Quan sát biểu đồ Quý để chuẩn bị kế hoạch nhập vật tư. Nếu Q3 cao nhất, cần order vật tư từ Q2.")

# --- TAB 2: PRODUCT INTELLIGENCE ---
with tab2:
    c_p1, c_p2 = st.columns(2)
    
    with c_p1:
        st.subheader("Xu hướng Màu sắc (Color Trend)")
        color_trend = df.groupby('nhom_mau')['sl'].sum().reset_index().sort_values('sl', ascending=False)
        fig_c = px.pie(color_trend, values='sl', names='nhom_mau', 
                       color_discrete_sequence=px.colors.qualitative.Prism, hole=0.5)
        fig_c.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_c, use_container_width=True)
        
    with c_p2:
        st.subheader("Sản phẩm Công nghệ (USB Trend)")
        usb_trend = df.groupby(['year', 'is_usb_clean'])['sl'].sum().reset_index()
        fig_u = px.bar(usb_trend, x='year', y='sl', color='is_usb_clean', barmode='group',
                       color_discrete_map={'Có USB': WARNING, 'Không USB': 'lightgrey'},
                       title="Tăng trưởng sản phẩm có USB qua các năm")
        st.plotly_chart(fig_u, use_container_width=True)

    st.markdown("#### Top 10 SKU Chủ lực")
    top_sku = df.groupby('ma_hang')['sl'].sum().nlargest(10).sort_values(ascending=True).reset_index()
    fig_sku = px.bar(top_sku, x='sl', y='ma_hang', orientation='h', text_auto='.2s',
                     color='sl', color_continuous_scale='Greens')
    st.plotly_chart(fig_sku, use_container_width=True)

# --- TAB 3: PARETO 80/20 ---
with tab3:
    st.markdown("### ⚖️ Phân tích Khách hàng trọng yếu")
    st.caption("Nguyên tắc Pareto: 80% sản lượng thường đến từ 20% khách hàng (Nhóm Vital Few).")
    
    pareto_df = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
    pareto_df['cum_sl'] = pareto_df['sl'].cumsum()
    pareto_df['cum_perc'] = (pareto_df['cum_sl'] / pareto_df['sl'].sum() * 100)
    
    # Xác định điểm cắt 80%
    vital_few = pareto_df[pareto_df['cum_perc'] <= 80]
    
    p1, p2 = st.columns([3, 1])
    
    with p1:
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=pareto_df['khach_hang'], y=pareto_df['sl'], name='Sản lượng', marker_color=PRIMARY))
        fig_p.add_trace(go.Scatter(x=pareto_df['khach_hang'], y=pareto_df['cum_perc'], name='% Tích lũy', 
                                   yaxis='y2', line=dict(color=DANGER, width=2)))
        
        fig_p.add_hline(y=80, line_dash="dash", line_color="gray", annotation_text="Ngưỡng 80%")
        
        fig_p.update_layout(
            yaxis2=dict(overlaying='y', side='right', range=[0, 110], title="% Tích lũy"),
            yaxis=dict(title="Sản lượng"),
            showlegend=True, height=500
        )
        st.plotly_chart(fig_p, use_container_width=True)
    
    with p2:
        st.success(f"**VITAL FEW (Nhóm Cốt lõi):**\n\nCó **{len(vital_few)}** khách hàng đóng góp 80% sản lượng.")
        st.dataframe(vital_few[['khach_hang', 'sl']], height=400)

# --- TAB 4: AG-GRID DETAILED DATA ---
with tab4:
    st.subheader("Dữ liệu chi tiết & Tải xuống")
    
    grid_df = df.groupby(['ma_hang', 'khach_hang', 'xuong', 'nhom_mau', 'is_usb_clean', 'year']).agg(
        Tong_SL=('sl', 'sum'),
        Lan_Cuoi_SX=('ym', 'max')
    ).reset_index().sort_values('Tong_SL', ascending=False)
    
    grid_df['Lan_Cuoi_SX'] = grid_df['Lan_Cuoi_SX'].dt.strftime('%Y-%m')

    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)
    gb.configure_column("Tong_SL", type=["numericColumn", "numberColumnFilter"], precision=0)
    gb.configure_column("ma_hang", pinned=True)
    
    AgGrid(grid_df, gridOptions=gb.build(), height=600, fit_columns_on_grid_load=False)

# Footer
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #888;'>Powered by <b>Ly (IT Dept)</b> | Moc Phat Data System 2.0</div>", unsafe_allow_html=True)
