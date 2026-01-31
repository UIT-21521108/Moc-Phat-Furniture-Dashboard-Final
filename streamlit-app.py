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
# 1. CẤU HÌNH GIAO DIỆN (GOLDEN GLASS & SOFT NEON)
# ==========================================
st.set_page_config(page_title="Mộc Phát Analytics Pro", layout="wide", page_icon="🌲")

# BẢNG MÀU
PRIMARY_NEON = "#00E676" 
GOLD_GRADIENT = "linear-gradient(135deg, #FFA726, #F57C00)"
BG_DARK = "#050505"    
TEXT_MAIN = "#FFFFFF"
TEXT_SUB = "#CFD8DC"
GRID_COLOR = "rgba(255, 255, 255, 0.05)"

# --- CSS VISUAL EFFECTS ---
css_style = """
<style>
    /* 1. NỀN DEEP NEBULA */
    .stApp {{
        background-color: {bg_dark};
        background-image: 
            radial-gradient(at 20% 20%, rgba(0, 230, 118, 0.05) 0px, transparent 50%),
            radial-gradient(at 80% 80%, rgba(255, 167, 38, 0.05) 0px, transparent 50%);
        background-attachment: fixed;
    }}

    /* 2. HEADER: SOFT NEON (GIẢM CHÓI) */
    .header-container {{
        text-align: center;
        padding: 30px 0;
        margin-bottom: 20px;
    }}
    .glow-logo {{
        filter: drop-shadow(0 0 15px {primary});
        margin-bottom: 10px;
    }}
    .neon-text {{
        font-family: 'Segoe UI', sans-serif;
        font-weight: 900;
        font-size: 40px;
        color: #fff;
        letter-spacing: 2px;
        /* Ánh sáng dịu hơn */
        text-shadow: 
            0 0 5px {primary},
            0 0 15px rgba(0, 230, 118, 0.4);
        text-transform: uppercase;
    }}

    /* 3. TIÊU ĐỀ PHỤ NỔI BẬT (ANTI-CHÌM) */
    h3 {{
        color: #FFFFFF !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.9); /* Bóng đổ cực sâu */
        border-left: 5px solid {primary};
        padding-left: 15px;
        margin-top: 30px !important;
    }}

    /* 4. GLASSMORPHISM CARDS & GRADIENT */
    .glass-card {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 22px;
        transition: all 0.3s ease;
    }}
    
    /* Card Kế hoạch màu Vàng Gradient */
    .gold-glass-card {{
        background: linear-gradient(135deg, rgba(255, 167, 38, 0.2), rgba(245, 124, 0, 0.1));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 167, 38, 0.3);
        border-radius: 20px;
        padding: 22px;
    }}

    .kpi-title {{ font-size: 12px; color: {text_sub}; text-transform: uppercase; letter-spacing: 1px; opacity: 0.8; }}
    .kpi-value {{ font-size: 32px; font-weight: 800; color: #fff; margin: 5px 0; }}

    /* 5. TABS */
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{ 
        background-color: rgba(255,255,255,0.03); 
        color: {text_sub}; 
        border-radius: 10px; 
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{ 
        border: 1px solid {primary};
        color: {primary};
        background: rgba(0, 230, 118, 0.1);
    }}
</style>
""".format(
    bg_dark=BG_DARK, primary=PRIMARY_NEON, 
    text_sub=TEXT_SUB
)
st.markdown(css_style, unsafe_allow_html=True)

# --- HÀM STYLE BIỂU ĐỒ ---
def polish_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E0E0E0", family="Segoe UI"),
        margin=dict(t=40, b=20, l=10, r=10),
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig

# ==========================================
# 2. LOAD DATA
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    FILE_NAME = "Master_2023_2025_PRO_clean.xlsx"
    if not os.path.exists(FILE_NAME): return None, f"⚠️ File {FILE_NAME} not found."
    try:
        df = pd.read_excel(FILE_NAME, engine='openpyxl')
        df.columns = [str(c).strip().lower() for c in df.columns]
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
        df['ym'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])
        df['sl'] = pd.to_numeric(df['sl'], errors='coerce').fillna(0)
        
        # Mùa vụ
        season_map = {12:'Đông', 1:'Đông', 2:'Đông', 3:'Xuân', 4:'Xuân', 5:'Xuân', 
                      6:'Hè', 7:'Hè', 8:'Hè', 9:'Thu', 10:'Thu', 11:'Thu'}
        df['mua'] = df['month'].map(season_map)
        
        # Nhóm màu
        def get_color_group(v):
            v = str(v).upper()
            if any(x in v for x in ["BROWN", "NAU", "WALNUT"]): return "NÂU/GỖ"
            if any(x in v for x in ["WHITE", "TRANG", "CREAM"]): return "TRẮNG/KEM"
            if any(x in v for x in ["BLACK", "DEN"]): return "ĐEN/TỐI"
            return "MÀU KHÁC"
        df['nhom_mau'] = df['mau_son'].apply(get_color_group) if 'mau_son' in df.columns else "MÀU KHÁC"
        
        return df, None
    except Exception as e: return None, str(e)

df_raw, error = load_data()
if error: st.error(error); st.stop()

# ==========================================
# 3. HEADER (SOFT NEON)
# ==========================================
def get_base64_logo(path):
    if os.path.exists(path):
        with open(path, "rb") as f: return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_base64_logo("mocphat_logo.png")
logo_img = f'<img src="data:image/png;base64,{logo_b64}" height="70" class="glow-logo">' if logo_b64 else '🌲'

st.markdown(f"""
<div class="header-container">
    {logo_img}
    <div class="neon-text">MỘC PHÁT INTELLIGENCE</div>
    <div class="sub-text">EXECUTIVE ANALYTICS SUITE</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🎯 BỘ LỌC")
years = sorted(df_raw['year'].unique(), reverse=True)
sel_years = st.sidebar.multiselect("Năm", years, default=years)
df = df_raw[df_raw['year'].isin(sel_years)] if sel_years else df_raw

# ==========================================
# 4. KPI CARDS (GLASS STYLE)
# ==========================================
st.subheader("🚀 Hiệu quả Kinh doanh")
v23 = df[df['year']==2023]['sl'].sum()
v24 = df[df['year']==2024]['sl'].sum()
v25 = df[df['year']==2025]['sl'].sum()
g24 = ((v24 - v23)/v23*100) if v23 > 0 else 0

c1, c2, c3, c4 = st.columns(4)

def kpi_box(col, lbl, val, sub_val, is_base=False):
    color = PRIMARY_NEON if sub_val >= 0 else "#FF5252"
    sub_text = f"▲ {sub_val:.1f}% vs 23" if not is_base else "Base Year"
    col.markdown(f"""
    <div class="glass-card">
        <div class="kpi-title">{lbl}</div>
        <div class="kpi-value">{val:,.0f}</div>
        <div style="color:{color if not is_base else '#aaa'}; font-size:14px; font-weight:700;">{sub_text}</div>
    </div>
    """, unsafe_allow_html=True)

kpi_box(c1, "SẢN LƯỢNG 2023", v23, 0, True)
kpi_box(c2, "SẢN LƯỢNG 2024", v24, g24)
kpi_box(c3, "SẢN LƯỢNG 2025", v25, 0, True)
kpi_box(c4, "ĐỐI TÁC ACTIVE", df['khach_hang'].nunique() if 'khach_hang' in df.columns else 0, 0, True)

st.markdown("---")

# ==========================================
# 5. TABS PHÂN TÍCH
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 TỔNG QUAN", "🎯 KẾ HOẠCH 2026", "🎨 SỨC KHỎE SP", "📋 DỮ LIỆU"])

with tab1:
    st.subheader("📈 Xu hướng & Phát hiện Bất thường")
    ts = df.groupby('ym')['sl'].sum().reset_index()
    fig = px.area(ts, x='ym', y='sl', title="Sản lượng theo tháng")
    fig.update_traces(line_color=PRIMARY_NEON, fillcolor="rgba(0, 230, 118, 0.1)")
    st.plotly_chart(polish_chart(fig), use_container_width=True)

with tab2:
    st.subheader("🎯 Lập Kế Hoạch 2026")
    col_in, col_res = st.columns([1, 2])
    with col_in:
        target = st.slider("Mục tiêu tăng trưởng (%)", 0, 100, 15)
    
    base_25 = df[df['year']==2025]['sl'].sum()
    target_val = base_25 * (1 + target/100)
    
    with col_res:
        # Khung màu Vàng Gradient như bản trước
        st.markdown(f"""
        <div class="gold-glass-card" style="display:flex; justify-content:space-around; align-items:center;">
            <div style="text-align:center;"><small>2025 BASE</small><br><b style="font-size:24px;">{base_25:,.0f}</b></div>
            <div style="font-size:30px; color:#FFA726;">➔</div>
            <div style="text-align:center;"><small>2026 TARGET</small><br><b style="font-size:24px; color:#00E676;">{target_val:,.0f}</b></div>
            <div style="text-align:center;"><small>GAP (+{target}%)</small><br><b style="font-size:24px; color:#FFA726;">+{target_val-base_25:,.0f}</b></div>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.subheader("🎨 Sunburst Chart (Màu chi tiết)")
    if 'nhom_mau' in df.columns:
        fig_sun = px.sunburst(df, path=['nhom_mau', 'mau_son'], values='sl')
        st.plotly_chart(polish_chart(fig_sun), use_container_width=True)

with tab4:
    st.subheader("Tra cứu dữ liệu")
    gb = GridOptionsBuilder.from_dataframe(df.head(100))
    gb.configure_pagination()
    grid_opt = gb.build()
    st.markdown('<div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:15px;">', unsafe_allow_html=True)
    AgGrid(df.head(100), gridOptions=grid_opt, theme='alpine-dark')
    st.markdown('</div>', unsafe_allow_html=True)

st.caption(f"© 2026 Mộc Phát Furniture | Golden Glass Edition | Updated: {datetime.now().strftime('%d/%m/%Y')}")
