# app.py — Moc Phat Dashboard v2.0 (Strategic Edition)
# Kế thừa: Brand xanh, Sticky header, Insight text
# Phát triển: Ma trận SKU (Runner/Stranger), Mô phỏng kịch bản (What-if), Cross-filtering

import os, base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime

# =========================
# 1. CONFIG & CSS (Giữ nguyên Brand Identity)
# =========================
st.set_page_config(page_title="Mộc Phát Strategic Hub", layout="wide", page_icon="🌲")

PRIMARY = "#066839"
ACCENT  = "#1B7D4F"
COLOR_PALETTE = {
    "BROWN": "#8B5A2B", "WHITE": "#F2F2F2", "BLACK": "#2E2E2E",
    "GREY": "#9E9E9E", "GREEN": "#2E7D32", "BLUE": "#1565C0",
    "NATURAL": "#C4A484", "PINK": "#E57373", "YELLOW": "#FBC02D",
    "RED": "#D32F2F", "OTHER": "#BDBDBD"
}
PLOT_TEMPLATE = 'plotly_white' # Dùng nền trắng cho sạch, chuyên nghiệp

st.markdown(f"""
<style>
:root {{ --brand:{PRIMARY}; --brand2:{ACCENT}; }}
html {{ scroll-behavior:smooth; }}
h1,h2,h3,h4 {{ font-weight:800 !important; letter-spacing: -0.5px; }}
.stDataFrame thead tr th {{ font-weight:800 !important; background: #f0f2f6; }}

/* Card KPI thiết kế lại: Clean & Modern */
.kpi-box {{
    background: #fff; border: 1px solid #e0e0e0; border-radius: 10px;
    padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    border-left: 5px solid {PRIMARY}; transition: transform 0.2s;
}}
.kpi-box:hover {{ transform: translateY(-3px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
.kpi-label {{ color: #666; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; }}
.kpi-value {{ color: #222; font-size: 1.8rem; font-weight: 800; margin: 5px 0; }}
.kpi-delta {{ font-size: 0.9rem; font-weight: 700; }}
.delta-pos {{ color: {PRIMARY}; }} .delta-neg {{ color: #D32F2F; }}

/* Sticky Header */
.header-sticky {{
    position: sticky; top: 0; z-index: 999; 
    background: rgba(255,255,255,0.95); backdrop-filter: blur(10px);
    border-bottom: 1px solid #eee; padding: 10px 0; margin-bottom: 20px;
}}
.header-content {{ display: flex; align-items: center; gap: 15px; }}
.header-title {{ font-size: 1.8rem; font-weight: 900; color: #111; margin: 0; }}
.header-badge {{ 
    background: {PRIMARY}; color: white; padding: 4px 10px; 
    border-radius: 20px; font-size: 0.8rem; font-weight: 700; 
}}

/* Insight Box Nâng cao */
.strategy-box {{
    background: #f8fcf9; border: 1px solid {ACCENT}; border-radius: 8px;
    padding: 15px; margin: 15px 0; position: relative;
}}
.strategy-icon {{ position: absolute; top: -12px; left: 15px; background: {PRIMARY}; color: white; padding: 2px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }}
</style>
""", unsafe_allow_html=True)

# =========================
# 2. LOGIC XỬ LÝ DỮ LIỆU (Nâng cấp)
# =========================
@st.cache_data(show_spinner=False)
def load_and_process(file):
    if not file: return None
    try:
        df = pd.read_excel(file, engine='openpyxl') if file.name.endswith('.xlsx') else pd.read_csv(file)
    except: return None
    
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Ép kiểu số & thời gian
    for c in ['sl','sl_container','month','year']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df = df[df['year'] > 0] # Lọc rác
    df['ym'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-' + df['month'].astype(int).astype(str) + '-01')
    
    # --- LOGIC NÂNG CAO 1: Phân loại USB & Phụ kiện ---
    df['is_usb'] = df['ma_hang'].fillna('').str.contains('USB', case=False) | \
                   df['mo_ta'].fillna('').str.contains('USB', case=False)
    
    # --- LOGIC NÂNG CAO 2: Phân loại Màu ---
    def categorize_color(v):
        v = str(v).upper()
        for k, hex_val in COLOR_PALETTE.items():
            if k in v: return k
            # Mapping thêm các từ khóa phụ
            if k=="BROWN" and any(x in v for x in ["COCOA","BRONZE","WALNUT"]): return "BROWN"
            if k=="WHITE" and any(x in v for x in ["CREAM","IVORY","WASH"]): return "WHITE"
            if k=="NATURAL" and any(x in v for x in ["OAK","HONEY"]): return "NATURAL"
        return "OTHER"
    
    df['nhom_mau'] = df['mau_son'].apply(categorize_color)
    
    # --- LOGIC NÂNG CAO 3: Ma trận SKU (Supply Chain Matrix) ---
    # Tính Frequency (Số tháng xuất hiện đơn hàng) & Volume
    sku_stats = df.groupby('ma_hang').agg(
        total_vol=('sl', 'sum'),
        freq=('ym', 'nunique') # Số tháng có đơn
    ).reset_index()
    
    # Phân loại Runner (Chạy đều, SL lớn) / Repeater (Lặp lại) / Stranger (Lạ, ít)
    # Ngưỡng: freq >= 6 tháng/năm là đều (giả định dữ liệu > 1 năm)
    # Vol: Top 20% sản lượng
    vol_80 = sku_stats['total_vol'].quantile(0.8)
    freq_med = sku_stats['freq'].median()
    
    def classify_sku(row):
        if row['total_vol'] >= vol_80 and row['freq'] >= 4: return "RUNNER (Trụ cột)"
        if row['total_vol'] < vol_80 and row['freq'] >= 4: return "REPEATER (Ổn định)"
        return "STRANGER (Thời vụ/Mẫu mới)"
    
    sku_stats['sku_class'] = sku_stats.apply(classify_sku, axis=1)
    df = df.merge(sku_stats[['ma_hang', 'sku_class']], on='ma_hang', how='left')

    return df

# =========================
# 3. VISUALIZATION FUNCTIONS (Nâng cấp Interactivity)
# =========================

def plot_kpi_modern(df):
    """Hiển thị KPI dạng custom HTML/CSS thay vì st.metric chuẩn"""
    now_year = df['year'].max()
    prev_year = now_year - 1
    
    v_now = df[df['year']==now_year]['sl'].sum()
    v_prev = df[df['year']==prev_year]['sl'].sum()
    delta = (v_now - v_prev) / v_prev * 100 if v_prev else 0
    
    # Tính thêm: Số container (giả định trung bình nếu thiếu data)
    cont_now = df[df['year']==now_year]['sl_container'].sum()
    
    cols = st.columns(4)
    
    # KPI 1: Sản lượng Năm nay
    with cols[0]:
        cls = "delta-pos" if delta >= 0 else "delta-neg"
        icon = "▲" if delta >= 0 else "▼"
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Sản lượng {now_year}</div>
            <div class="kpi-value">{v_now:,.0f}</div>
            <div class="kpi-delta {cls}">{icon} {abs(delta):.1f}% <span style="color:#999;font-weight:400">vs {prev_year}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    # KPI 2: Số Container
    with cols[1]:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Tổng Container (Est)</div>
            <div class="kpi-value">{cont_now:,.1f}</div>
            <div class="kpi-delta" style="color:{PRIMARY}">📦 Vận chuyển</div>
        </div>
        """, unsafe_allow_html=True)
        
    # KPI 3: Tỷ lệ USB
    usb_rate = df[df['year']==now_year]['is_usb'].mean() * 100
    with cols[2]:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Tỷ lệ có USB</div>
            <div class="kpi-value">{usb_rate:.1f}%</div>
            <div class="kpi-delta" style="color:#E65100">⚡ Xu hướng công nghệ</div>
        </div>
        """, unsafe_allow_html=True)

    # KPI 4: SKU Active
    sku_act = df[df['year']==now_year]['ma_hang'].nunique()
    with cols[3]:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">SKU Hoạt động</div>
            <div class="kpi-value">{sku_act:,}</div>
            <div class="kpi-delta" style="color:#1565C0">🏷️ Độ phủ danh mục</div>
        </div>
        """, unsafe_allow_html=True)

def plot_sku_matrix(df):
    """Vẽ ma trận SKU: Volume vs Frequency (Scatter plot)"""
    # Lấy dữ liệu 2024-2025 (gần nhất)
    recent = df[df['year'] >= df['year'].max()-1]
    stats = recent.groupby(['ma_hang', 'sku_class', 'nhom_mau']).agg(
        vol=('sl', 'sum'),
        freq=('ym', 'nunique')
    ).reset_index()
    
    fig = px.scatter(stats, x='freq', y='vol', color='sku_class',
                     size='vol', hover_name='ma_hang',
                     color_discrete_map={
                         "RUNNER (Trụ cột)": PRIMARY,
                         "REPEATER (Ổn định)": "#FFA726",
                         "STRANGER (Thời vụ/Mẫu mới)": "#9E9E9E"
                     },
                     log_y=True, # Dùng log scale vì sản lượng chênh lệch lớn
                     title="Ma trận Sản phẩm (Product Matrix)")
    
    fig.add_vline(x=4, line_dash="dash", line_color="grey", annotation_text="Ngưỡng ổn định")
    fig.update_layout(xaxis_title="Số tháng có đơn hàng (Frequency)", yaxis_title="Tổng sản lượng (Log Scale)")
    return fig

# =========================
# 4. MAIN APP
# =========================

# --- HEADER STICKY ---
def render_header():
    logo_path = "mocphat_logo.png"
    logo_html = ""
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{b64}" style="height:45px; margin-right:15px;">'
    
    st.markdown(f"""
    <div class="header-sticky">
        <div class="header-content">
            {logo_html}
            <div>
                <h1 class="header-title">MỘC PHÁT ANALYTICS</h1>
                <span class="header-badge">Strategic Edition v2.0</span>
            </div>
            <div style="flex-grow:1; text-align:right; font-weight:600; color:{PRIMARY}">
                Dữ liệu cập nhật: {datetime.now().strftime('%d/%m/%Y')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

render_header()

# --- SIDEBAR & DATA LOAD ---
with st.sidebar:
    st.header("🎛️ Bảng điều khiển")
    uploaded_file = st.file_uploader("Nạp dữ liệu (Excel/CSV)", type=['xlsx', 'csv'])
    
    # Fallback file mặc định
    if not uploaded_file and os.path.exists('Master_2023_2025_PRO_clean.xlsx'):
        uploaded_file = open('Master_2023_2025_PRO_clean.xlsx', 'rb')
        st.caption("ℹ️ Đang dùng dữ liệu mẫu hệ thống")

    if not uploaded_file:
        st.warning("Vui lòng tải file dữ liệu.")
        st.stop()

df = load_and_process(uploaded_file)
if df is None: st.error("Lỗi đọc file!"); st.stop()

# Filter nhanh
years = sorted(df['year'].unique(), reverse=True)
sel_years = st.multiselect("Chọn Năm phân tích", years, default=years[:2])
df_filtered = df[df['year'].isin(sel_years)]

# --- DASHBOARD BODY ---

# 1. KPI SECTION
plot_kpi_modern(df_filtered)

# 2. TABS PHÂN TÍCH CHIẾN LƯỢC
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tổng quan & Xu hướng", 
    "🎯 Ma trận Sản phẩm (SKU)", 
    "🌍 Khách hàng & Thị trường", 
    "🎨 Màu sắc & Xu hướng",
    "🧪 Mô phỏng (What-if)"  # NEW FEATURE
])

# --- TAB 1: OVERVIEW ---
with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Diễn biến sản lượng theo tháng")
        monthly = df_filtered.groupby('ym')['sl'].sum().reset_index()
        fig_trend = px.line(monthly, x='ym', y='sl', markers=True, line_shape='spline')
        fig_trend.update_traces(line_color=PRIMARY, line_width=3)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Insight Text tự động
        peak_month = monthly.loc[monthly['sl'].idxmax()]
        st.markdown(f"""
        <div class="strategy-box">
            <span class="strategy-icon">INSIGHT</span>
            <b>Nhịp độ sản xuất:</b> Đỉnh điểm sản lượng rơi vào tháng <b>{peak_month['ym'].strftime('%m/%Y')}</b> 
            với <b>{peak_month['sl']:,.0f}</b> sản phẩm. Đây là mốc cần chuẩn bị vật tư trước 2 tháng.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.subheader("Tỷ trọng theo Khu vực")
        # Logic heuristic khu vực (giả định dựa trên tên KH)
        fig_region = px.pie(df_filtered, values='sl', names='khu_vuc', hole=0.6, 
                            color_discrete_sequence=[PRIMARY, ACCENT, "#9EA7AD"])
        st.plotly_chart(fig_region, use_container_width=True)

# --- TAB 2: SKU MATRIX (Advanced) ---
with tab2:
    st.markdown("### 🎯 Phân loại chiến lược SKU (Runner - Repeater - Stranger)")
    st.caption("Biểu đồ giúp quyết định chiến lược tồn kho: **Runner** (Sản xuất stock), **Stranger** (Sản xuất theo đơn).")
    
    col_mat1, col_mat2 = st.columns([3, 1])
    with col_mat1:
        fig_matrix = plot_sku_matrix(df_filtered)
        st.plotly_chart(fig_matrix, use_container_width=True)
    
    with col_mat2:
        st.markdown("**Thống kê nhóm:**")
        sku_counts = df_filtered.drop_duplicates('ma_hang')['sku_class'].value_counts()
        for cls, count in sku_counts.items():
            color = PRIMARY if "RUNNER" in cls else ("#FFA726" if "REPEATER" in cls else "#9E9E9E")
            st.markdown(f"""
            <div style="padding:10px; border-radius:5px; background:{color}20; border-left:4px solid {color}; margin-bottom:10px;">
                <div style="font-weight:bold; color:{color}">{cls}</div>
                <div style="font-size:1.2rem">{count} SKU</div>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 3: KHÁCH HÀNG (Interactive Drill-down) ---
with tab3:
    c3_1, c3_2 = st.columns([1, 2])
    
    with c3_1:
        st.subheader("Top Khách Hàng")
        top_cust = df_filtered.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
        
        # Tương tác: Chọn khách hàng từ bảng để lọc biểu đồ bên cạnh
        selection = st.dataframe(
            top_cust.style.background_gradient(cmap="Greens"), 
            use_container_width=True, 
            height=400,
            on_select="rerun", # Yêu cầu Streamlit mới nhất
            selection_mode="single-row"
        )
        
        selected_cust = None
        if len(selection.selection.rows):
            idx = selection.selection.rows[0]
            selected_cust = top_cust.iloc[idx]['khach_hang']

    with c3_2:
        if selected_cust:
            st.subheader(f"Chi tiết: {selected_cust}")
            cust_df = df_filtered[df_filtered['khach_hang'] == selected_cust]
            
            # Biểu đồ con 1: Trend của KH này
            fig_c1 = px.bar(cust_df.groupby('ym')['sl'].sum().reset_index(), x='ym', y='sl', title="Sản lượng theo tháng")
            fig_c1.update_traces(marker_color=ACCENT)
            st.plotly_chart(fig_c1, use_container_width=True)
            
            # Biểu đồ con 2: Top SKU của KH này
            top_sku_cust = cust_df.groupby('ma_hang')['sl'].sum().nlargest(5).reset_index()
            fig_c2 = px.bar(top_sku_cust, x='sl', y='ma_hang', orientation='h', title="Top 5 SKU mua nhiều nhất")
            st.plotly_chart(fig_c2, use_container_width=True)
        else:
            st.info("👈 Chọn một khách hàng bên trái để xem chi tiết drill-down.")
            # Hiển thị mặc định Pareto chung
            st.subheader("Phân bổ Pareto (80/20)")
            top_cust['cumulative'] = top_cust['sl'].cumsum() / top_cust['sl'].sum()
            fig_pareto = px.line(top_cust.reset_index(), x='index', y='cumulative', markers=True)
            fig_pareto.add_hline(y=0.8, line_dash="dash", line_color="red")
            st.plotly_chart(fig_pareto, use_container_width=True)

# --- TAB 4: MÀU SẮC ---
with tab4:
    st.subheader("Xu hướng Nhóm màu (Color Trend)")
    color_trend = df_filtered.groupby(['year', 'nhom_mau'])['sl'].sum().reset_index()
    # Tính %
    color_trend['share'] = color_trend['sl'] / color_trend.groupby('year')['sl'].transform('sum')
    
    fig_color = px.bar(color_trend, x='year', y='share', color='nhom_mau', 
                       barmode='stack', color_discrete_map=COLOR_PALETTE)
    st.plotly_chart(fig_color, use_container_width=True)
    
    st.markdown("""
    <div class="strategy-box">
        <span class="strategy-icon">HÀNH ĐỘNG</span>
        <b>Quản lý vật tư sơn:</b> Nếu nhóm màu <b>WHITE/CREAM</b> đang tăng tỷ trọng, cần lưu ý quy trình phòng sơn sạch (chống bụi) 
        kỹ hơn so với màu tối. Đặt hàng trước các loại sơn hệ nước/dầu tương ứng.
    </div>
    """, unsafe_allow_html=True)

# --- TAB 5: WHAT-IF SIMULATION (New) ---
with tab5:
    st.markdown("### 🧪 Mô phỏng Kế hoạch 2026")
    st.caption("Công cụ tính toán nhu cầu nguồn lực dựa trên giả định tăng trưởng.")
    
    # Lấy dữ liệu năm gần nhất làm base
    base_year = df['year'].max()
    base_data = df[df['year'] == base_year]
    base_vol = base_data['sl'].sum()
    base_cont = base_data['sl_container'].sum()
    
    c5_1, c5_2, c5_3 = st.columns([1,1,2])
    
    with c5_1:
        growth_rate = st.slider("Dự báo Tăng trưởng (%)", -20, 50, 15)
        usb_penetration = st.slider("Tỷ lệ hàng có USB dự kiến (%)", 0, 100, int(base_data['is_usb'].mean()*100))
        
    target_vol = base_vol * (1 + growth_rate/100)
    target_cont = base_cont * (1 + growth_rate/100)
    target_usb_units = target_vol * (usb_penetration/100)
    
    with c5_2:
        st.metric("Sản lượng Mục tiêu", f"{target_vol:,.0f}", f"{growth_rate}%")
        st.metric("Số Container cần book", f"{target_cont:,.0f}")
        st.metric("Bộ phụ kiện USB cần nhập", f"{target_usb_units:,.0f}")
        
    with c5_3:
        st.markdown(f"""
        <div class="kpi-box" style="background:#fff3e0; border-left-color:#ff9800">
            <h4>📦 Kế hoạch Supply Chain</h4>
            <ul>
                <li>Cần chuẩn bị kho bãi cho khoảng <b>{target_cont/12:,.0f}</b> cont/tháng.</li>
                <li>Đàm phán giá phụ kiện USB cho lô <b>{target_usb_units:,.0f}</b> bộ ngay từ bây giờ.</li>
                <li>Nếu tỷ lệ tăng trưởng > 20%, cần kích hoạt thuê ngoài (outsourcing) phần phôi thô.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown(f"<div style='text-align:center; color:#888'>© {datetime.now().year} Mộc Phát Furniture Analytics System | Powered by Streamlit</div>", unsafe_allow_html=True)
