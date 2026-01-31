import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
import os
import time
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN
# ==========================================
st.set_page_config(page_title="Mộc Phát Analytics", layout="wide", page_icon="🌲")

# MÀU SẮC
PRIMARY = "#066839"    
NEON_GREEN = "#00E676" 
ACCENT  = "#66BB6A"    
BG_COLOR = "#050505"   
CARD_BG = "#121212"    
TEXT_MAIN = "#E0E0E0"
TEXT_SUB = "#9E9E9E"
GRID_COLOR = "#2A2A2A"

# --- HÀM STYLE BIỂU ĐỒ (SAFE MODE) ---
def polish_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_SUB, family="sans-serif"),
        margin=dict(t=40, b=20, l=10, r=10),
        hovermode="x unified"
        # Đã bỏ barcornerradius để tránh lỗi version cũ
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig

# --- CSS AN TOÀN (Dùng format thay vì f-string phức tạp) ---
css_code = """
<style>
    /* Tổng thể */
    .stApp {{ background-color: {bg}; }}
    h1, h2, h3, h4 {{ color: {text} !important; }}
    .stMarkdown p, .stMarkdown li {{ color: {sub} !important; }}
    
    /* Header */
    .header-sticky {{
        position: sticky; top: 0; z-index: 999;
        background: {card_bg};
        border-bottom: 2px solid {primary};
        padding: 15px 20px; 
        margin-bottom: 20px;
        display: flex; justify-content: space-between; align-items: center;
    }}
    
    /* KPI Cards */
    .kpi-card {{
        background: {card_bg}; 
        border-radius: 10px; padding: 20px;
        border-left: 5px solid {primary};
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.3s;
    }}
    .kpi-card:hover {{
        transform: translateY(-5px);
        border-left: 5px solid {neon};
        box-shadow: 0 5px 15px rgba(0, 230, 118, 0.2);
    }}
    .kpi-val {{ font-size: 28px; font-weight: bold; color: {text}; }}
    
    /* AgGrid fix */
    .ag-theme-alpine-dark {{
        --ag-background-color: {card_bg} !important;
        --ag-odd-row-background-color: {card_bg} !important;
    }}
</style>
""".format(
    bg=BG_COLOR, text=TEXT_MAIN, sub=TEXT_SUB, 
    card_bg=CARD_BG, primary=PRIMARY, neon=NEON_GREEN
)
st.markdown(css_code, unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (CƠ CHẾ DUMMY DATA)
# ==========================================
def generate_dummy_data():
    """Tạo dữ liệu giả nếu không đọc được file"""
    dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='M')
    data = []
    for d in dates:
        data.append({
            'year': d.year, 'month': d.month, 'ym': d,
            'khach_hang': np.random.choice(['HOMEGOODS', 'TJMAXX', 'MARSHALLS', 'ROSS'], p=[0.4, 0.3, 0.2, 0.1]),
            'ma_hang': f'SKU-{np.random.randint(100,999)}',
            'nhom_mau': np.random.choice(['NÂU/GỖ', 'TRẮNG/KEM', 'ĐEN/TỐI', 'XÁM'], p=[0.5, 0.2, 0.2, 0.1]),
            'mau_son': 'Sample Color',
            'mua': np.random.choice(['Xuân', 'Hè', 'Thu', 'Đông']),
            'is_usb_clean': np.random.choice(['Có USB', 'Không USB']),
            'sl': np.random.randint(100, 1000)
        })
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def load_data():
    FILE_NAME = "Master_2023_2025_PRO_clean.xlsx"
    
    # Check file tồn tại
    if not os.path.exists(FILE_NAME):
        return None, "FILE_NOT_FOUND"
    
    try:
        df = pd.read_excel(FILE_NAME, engine='openpyxl')
        # Chuẩn hóa tên cột
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Xử lý ngày tháng
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
        df = df[(df['year'] > 2020) & (df['month'].between(1, 12))]
        df['ym'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])
        
        # Mapping mùa
        season_map = {12:'Đông', 1:'Đông', 2:'Đông', 3:'Xuân', 4:'Xuân', 5:'Xuân', 
                      6:'Hè', 7:'Hè', 8:'Hè', 9:'Thu', 10:'Thu', 11:'Thu'}
        df['mua'] = df['month'].map(season_map)
        
        # Xử lý text & số
        df['sl'] = pd.to_numeric(df['sl'], errors='coerce').fillna(0)
        
        # Xử lý cột màu (nếu có)
        if 'mau_son' in df.columns:
            df['mau_son'] = df['mau_son'].fillna("Unknown").astype(str).str.upper()
            def get_group(v):
                if any(x in v for x in ["BROWN", "NAU", "WALNUT"]): return "NÂU/GỖ"
                if any(x in v for x in ["WHITE", "TRANG", "CREAM"]): return "TRẮNG/KEM"
                if any(x in v for x in ["BLACK", "DEN"]): return "ĐEN/TỐI"
                return "KHÁC"
            df['nhom_mau'] = df['mau_son'].apply(get_group)
        else:
            df['nhom_mau'] = "KHÁC"
            df['mau_son'] = "N/A"

        # Xử lý USB
        if 'is_usb' in df.columns:
            df['is_usb_clean'] = df['is_usb'].astype(str).apply(lambda x: 'Có USB' if 'true' in x.lower() else 'Không USB')
        else:
            df['is_usb_clean'] = 'N/A'
            
        return df, None

    except Exception as e:
        return None, str(e)

# LOAD DỮ LIỆU
df_raw, error = load_data()

# LOGIC XỬ LÝ KHI LỖI
is_demo = False
if error:
    if error == "FILE_NOT_FOUND":
        st.warning(f"⚠️ Không tìm thấy file 'Master_2023_2025_PRO_clean.xlsx'. Đang chạy chế độ DEMO DATA.")
    else:
        st.error(f"⚠️ Lỗi đọc file: {error}. Đang chạy chế độ DEMO DATA.")
    
    # Tạo data giả để App không bị sập
    df_raw = generate_dummy_data()
    is_demo = True

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================
# Header
st.markdown(f"""
<div class="header-sticky">
    <div>
        <h2 style="margin:0; color:{ACCENT}">MỘC PHÁT INTELLIGENCE</h2>
        <small style="color:{TEXT_SUB}">System Status: {'🟢 Online (Real Data)' if not is_demo else '🟡 Demo Mode'}</small>
    </div>
    <div style="font-weight:bold; color:{PRIMARY}">Dashboard v6.0</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🎯 BỘ LỌC")
years = sorted(df_raw['year'].unique(), reverse=True)
sel_years = st.sidebar.multiselect("Năm", years, default=years)

if 'khach_hang' in df_raw.columns:
    custs = sorted(df_raw['khach_hang'].unique())
    sel_cust = st.sidebar.multiselect("Khách Hàng", custs)
else:
    sel_cust = []

# Filter Data
df = df_raw.copy()
if sel_years: df = df[df['year'].isin(sel_years)]
if sel_cust: df = df[df['khach_hang'].isin(sel_cust)]

if df.empty:
    st.warning("Không có dữ liệu phù hợp bộ lọc.")
    st.stop()

# --- KPI CARDS ---
st.subheader("🚀 HIỆU QUẢ KINH DOANH")
vol_by_year = df.groupby('year')['sl'].sum()
v24 = vol_by_year.get(2024, 0)
v23 = vol_by_year.get(2023, 0)
g24 = ((v24 - v23) / v23 * 100) if v23 > 0 else 0

c1, c2, c3, c4 = st.columns(4)

def card(col, lbl, val, sub):
    col.markdown(f"""
    <div class="kpi-card">
        <div style="font-size:12px; color:#888">{lbl}</div>
        <div class="kpi-val">{val:,.0f}</div>
        <div style="color:{NEON_GREEN}">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

card(c1, "SẢN LƯỢNG 2023", v23, "(Base)")
card(c2, "SẢN LƯỢNG 2024", v24, f"{g24:+.1f}% vs 23")
card(c3, "SẢN LƯỢNG 2025", vol_by_year.get(2025,0), "(Current)")
card(c4, "SỐ LƯỢNG KHÁCH", df['khach_hang'].nunique() if 'khach_hang' in df.columns else 0, "Active")

st.markdown("---")

# --- TABS ---
t1, t2, t3, t4 = st.tabs(["📊 TỔNG QUAN", "🎯 KẾ HOẠCH 2026", "🎨 SỨC KHỎE SP", "📋 DỮ LIỆU"])

with t1:
    c_left, c_right = st.columns([3, 1])
    with c_left:
        # Chart Trend
        ts = df.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
        fig = px.area(ts, x='ym', y='sl', title="Xu hướng Sản lượng")
        fig.update_traces(line_color=NEON_GREEN)
        st.plotly_chart(polish_chart(fig), use_container_width=True)
    with c_right:
        st.info("💡 **Ghi chú:** Biểu đồ thể hiện biến động sản lượng theo tháng. Đường màu xanh neon biểu thị xu hướng tăng trưởng tích cực.")

with t2:
    st.subheader("Kế hoạch 2026")
    growth = st.slider("Mục tiêu tăng trưởng (%)", 0, 100, 15)
    
    # Logic đơn giản cho kế hoạch
    base_25 = df[df['year']==2025]['sl'].sum()
    if base_25 == 0: base_25 = v24 # Fallback nếu chưa có 2025
    
    target = base_25 * (1 + growth/100)
    
    c_k1, c_k2 = st.columns(2)
    with c_k1:
        st.metric("Sản lượng Nền (2025)", f"{base_25:,.0f}")
    with c_k2:
        st.metric(f"Mục tiêu 2026 (+{growth}%)", f"{target:,.0f}", delta=f"+{target-base_25:,.0f}")
        
    # Chart dự báo đơn giản
    df_forecast = pd.DataFrame({
        'Năm': ['2025 (Thực tế)', '2026 (Mục tiêu)'],
        'Sản lượng': [base_25, target]
    })
    fig_f = px.bar(df_forecast, x='Năm', y='Sản lượng', color='Năm', 
                   color_discrete_map={'2025 (Thực tế)': '#555', '2026 (Mục tiêu)': NEON_GREEN})
    st.plotly_chart(polish_chart(fig_f), use_container_width=True)

with t3:
    st.subheader("Phân tích Màu & SKU")
    c3_1, c3_2 = st.columns(2)
    with c3_1:
        if 'nhom_mau' in df.columns:
            grp_color = df.groupby('nhom_mau')['sl'].sum().reset_index()
            fig_pie = px.pie(grp_color, values='sl', names='nhom_mau', hole=0.5, title="Cơ cấu Màu")
            st.plotly_chart(polish_chart(fig_pie), use_container_width=True)
    with c3_2:
        if 'ma_hang' in df.columns:
            top_sku = df.groupby('ma_hang')['sl'].sum().nlargest(10).reset_index()
            fig_bar = px.bar(top_sku, x='sl', y='ma_hang', orientation='h', title="Top 10 SKU")
            fig_bar.update_traces(marker_color=PRIMARY)
            st.plotly_chart(polish_chart(fig_bar), use_container_width=True)

with t4:
    st.subheader("Dữ liệu chi tiết")
    # Dùng AgGrid cơ bản nhất để tránh lỗi version
    gd = GridOptionsBuilder.from_dataframe(df.head(100)) # Show 100 dòng đầu để nhẹ
    gd.configure_pagination()
    AgGrid(df.head(100), gridOptions=gd.build(), height=400, theme='balham') # Theme balham an toàn hơn

st.markdown("---")
st.caption(f"Generated at {datetime.now()}")
