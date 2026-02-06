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
# 1. CẤU HÌNH GIAO DIỆN (CYBER GLASS V2.0)
# ==========================================
st.set_page_config(page_title="Mộc Phát Strategic Hub", layout="wide", page_icon="🌲")

# BẢNG MÀU CYBERPUNK
NEON_GREEN = "#00E676"   # Growth, Success
NEON_BLUE  = "#2979FF"   # Info, Structure
NEON_PINK  = "#D500F9"   # Warning, Highlight
NEON_RED   = "#FF1744"   # Danger
BG_DARK    = "#050505"
TEXT_MAIN  = "#FFFFFF"
TEXT_SUB   = "#B0BEC5"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"

# --- CSS VISUAL EFFECTS (HUD STYLE) ---
# Lưu ý: Dùng {{ }} cho CSS trong f-string
st.markdown(f"""
<style>
    /* 1. NỀN TECH GRID */
    .stApp {{
        background-color: {BG_DARK};
        background-image: 
            linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 30px 30px;
    }}

    /* 2. HEADER GLASSMORPHISM */
    .header-sticky {{
        position: sticky; top: 0; z-index: 999;
        background: rgba(5, 5, 5, 0.85);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(0, 230, 118, 0.3);
        padding: 15px 20px; 
        margin-bottom: 25px;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}

    /* 3. KPI CARDS (HUD STYLE) */
    .kpi-card {{
        background: rgba(20, 20, 20, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 4px; /* Góc vuông hơn cho cảm giác tech */
        padding: 15px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }}
    .kpi-card::before {{
        content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%;
        background: {NEON_GREEN};
        box-shadow: 0 0 10px {NEON_GREEN};
    }}
    .kpi-card:hover {{
        border-color: {NEON_GREEN};
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0, 230, 118, 0.15);
    }}
    .kpi-lbl {{ font-size: 11px; color: {NEON_BLUE}; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; margin-bottom: 5px; }}
    .kpi-val {{ font-size: 28px; font-weight: 800; color: {TEXT_MAIN}; font-family: 'Consolas', monospace; }}
    .kpi-sub {{ font-size: 12px; margin-top: 5px; }}

    /* 4. TABS & AGGRID */
    .stTabs [data-baseweb="tab-list"] {{ gap: 5px; background: transparent; }}
    .stTabs [data-baseweb="tab"] {{ 
        background-color: rgba(255,255,255,0.05); 
        color: {TEXT_SUB}; 
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 4px;
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{ 
        border: 1px solid {NEON_GREEN};
        color: {NEON_GREEN};
        background: rgba(0, 230, 118, 0.1);
        box-shadow: 0 0 10px rgba(0, 230, 118, 0.2);
    }}
    
    /* 5. INSIGHT BOX */
    .insight-box {{
        border: 1px solid {NEON_BLUE};
        background: rgba(41, 121, 255, 0.05);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-family: 'Segoe UI', sans-serif;
    }}

    /* AGGRID THEME */
    .ag-theme-alpine-dark {{
        --ag-background-color: transparent !important;
        --ag-header-background-color: rgba(255,255,255,0.05) !important;
        --ag-odd-row-background-color: rgba(255,255,255,0.02) !important;
        --ag-foreground-color: {TEXT_SUB} !important;
        --ag-border-color: rgba(255,255,255,0.1) !important;
        font-family: 'Segoe UI', sans-serif !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- HÀM STYLE BIỂU ĐỒ (DARK MODE) ---
def polish_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_SUB, family="Segoe UI"),
        margin=dict(t=40, b=20, l=10, r=10),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig

# --- HÀM RENDER AGGRID ---
def render_dark_aggrid(dataframe, height=400):
    gb = GridOptionsBuilder.from_dataframe(dataframe)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('single', use_checkbox=True)
    gb.configure_default_column(resizable=True, filterable=True, sortable=True)
    # Format số
    for col in dataframe.select_dtypes(include=['number']).columns:
        gb.configure_column(col, type=["numericColumn", "numberColumnFilter"], precision=0)
    
    gridOptions = gb.build()
    st.markdown('<div class="kpi-card" style="padding:0; border:none;">', unsafe_allow_html=True)
    AgGrid(dataframe, gridOptions=gridOptions, height=height, theme='alpine-dark', enable_enterprise_modules=False)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (STRATEGIC LOGIC)
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    FILE_NAME = "Master_2023_2025_PRO_clean.xlsx"
    if not os.path.exists(FILE_NAME): return None, f"⚠️ Không tìm thấy file {FILE_NAME}"

    try:
        df = pd.read_excel(FILE_NAME, engine='openpyxl')
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Ép kiểu dữ liệu
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
        df = df[(df['year'] > 2020) & (df['month'].between(1, 12))]
        df['ym'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])
        df['sl'] = pd.to_numeric(df['sl'], errors='coerce').fillna(0)
        
        # Mùa vụ
        season_map = {12:'Đông', 1:'Đông', 2:'Đông', 3:'Xuân', 4:'Xuân', 5:'Xuân', 
                      6:'Hè', 7:'Hè', 8:'Hè', 9:'Thu', 10:'Thu', 11:'Thu'}
        df['mua'] = df['month'].map(season_map)
        
        # --- LOGIC 1: MẪU MỚI vs MẪU CŨ (Dựa trên lịch sử) ---
        first_seen = df.groupby('ma_hang')['year'].min().reset_index()
        first_seen.rename(columns={'year': 'first_year'}, inplace=True)
        df = df.merge(first_seen, on='ma_hang', how='left')
        df['loai_mau'] = np.where(df['year'] == df['first_year'], 'Mẫu Mới (New)', 'Mẫu Cũ (Repeat)')
        
        # --- LOGIC 2: PHÂN LOẠI SKU (Runner/Repeater/Stranger) ---
        sku_stats = df.groupby('ma_hang').agg(
            total_vol=('sl', 'sum'),
            months_active=('ym', 'nunique')
        ).reset_index()
        
        vol_80 = sku_stats['total_vol'].quantile(0.8) # Top 20% sản lượng
        
        def classify_sku(row):
            if row['total_vol'] >= vol_80 and row['months_active'] >= 6: return "RUNNER (Trụ cột)"
            if row['total_vol'] < vol_80 and row['months_active'] >= 6: return "REPEATER (Ổn định)"
            return "STRANGER (Thời vụ/Mới)"
            
        sku_stats['sku_class'] = sku_stats.apply(classify_sku, axis=1)
        df = df.merge(sku_stats[['ma_hang', 'sku_class']], on='ma_hang', how='left')

        # Xử lý màu sắc & USB
        def categorize_color(v):
            v = str(v).upper()
            if any(x in v for x in ["BROWN", "NAU", "WALNUT"]): return "NÂU/GỖ"
            if any(x in v for x in ["WHITE", "TRANG", "CREAM"]): return "TRẮNG/KEM"
            if any(x in v for x in ["BLACK", "DEN"]): return "ĐEN/TỐI"
            return "MÀU KHÁC"
        
        df['nhom_mau'] = df['mau_son'].apply(categorize_color) if 'mau_son' in df.columns else "MÀU KHÁC"
        df['is_usb'] = df['ma_hang'].astype(str).str.contains("USB", case=False) | df['mo_ta'].astype(str).str.contains("USB", case=False)

        return df, None
    except Exception as e:
        return None, str(e)

df_raw, error = load_data()
if error: st.error(error); st.stop()

# ==========================================
# 3. HEADER & FILTER
# ==========================================
def get_base64_logo(path):
    if os.path.exists(path):
        with open(path, "rb") as f: return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_base64_logo("mocphat_logo.png")
logo_img = f'<img src="data:image/png;base64,{logo_b64}" height="50">' if logo_b64 else "🌲"

st.markdown(f"""
<div class="header-sticky">
    <div style="display:flex; gap:15px; align-items:center;">
        {logo_img}
        <div>
            <div style="font-size:24px; font-weight:800; color:{NEON_GREEN}; letter-spacing:1px; text-shadow:0 0 10px {NEON_GREEN};">MỘC PHÁT ANALYTICS</div>
            <div style="font-size:11px; color:{TEXT_SUB}; letter-spacing:2px;">STRATEGIC V2.0 // CYBER GLASS</div>
        </div>
    </div>
    <div style="text-align:right;">
        <span style="color:{NEON_BLUE}; font-size:12px; font-family:'Consolas';">SYSTEM: ONLINE ●</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🧬 DATA FILTERS")
years = sorted(df_raw['year'].unique(), reverse=True)
sel_years = st.sidebar.multiselect("Năm Phân Tích", years, default=years[:1])
df = df_raw[df_raw['year'].isin(sel_years)] if sel_years else df_raw
if df.empty: st.warning("NO DATA AVAILABLE"); st.stop()

# ==========================================
# 4. KPI HUD
# ==========================================
st.markdown(f"<h3 style='border-left:4px solid {NEON_GREEN}; padding-left:10px; color:white;'>EXECUTIVE SUMMARY</h3>", unsafe_allow_html=True)

curr_year = df['year'].max()
vol_curr = df['sl'].sum()
try:
    vol_prev = df_raw[df_raw['year'] == curr_year - 1]['sl'].sum()
    growth = ((vol_curr - vol_prev) / vol_prev * 100) if vol_prev > 0 else 0
except: growth = 0

# Tỷ lệ mẫu mới
mix = df.groupby('loai_mau')['sl'].sum()
new_ratio = (mix.get('Mẫu Mới (New)', 0) / mix.sum() * 100) if mix.sum() > 0 else 0

c1, c2, c3, c4 = st.columns(4)

def kpi_hud(col, label, value, delta, suffix, color=NEON_GREEN):
    icon = "▲" if delta >= 0 else "▼"
    d_color = NEON_GREEN if delta >= 0 else NEON_RED
    col.markdown(f"""
    <div class="kpi-card" style="border-top: 1px solid {color}40;">
        <div class="kpi-lbl" style="color:{color}">{label}</div>
        <div class="kpi-val">{value}</div>
        <div class="kpi-sub" style="color:{d_color}">
            {icon} {abs(delta):.1f}% <span style="color:{TEXT_SUB}">{suffix}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

kpi_hud(c1, f"SẢN LƯỢNG {curr_year}", f"{vol_curr:,.0f}", growth, "vs Năm trước", NEON_GREEN)
kpi_hud(c2, "ACTIVE SKUS", f"{df['ma_hang'].nunique():,.0f}", 0, "Mã hàng", NEON_BLUE)
status_mix = NEON_GREEN if new_ratio <= 35 else NEON_PINK
kpi_hud(c3, "TỶ LỆ MẪU MỚI", f"{new_ratio:.1f}%", new_ratio-30, "Target < 30%", status_mix)
kpi_hud(c4, "SỐ LƯỢNG KHÁCH", f"{df['khach_hang'].nunique()}", 0, "Đối tác", NEON_BLUE)

st.markdown("---")

# ==========================================
# 5. STRATEGIC TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔭 1. CHIẾN LƯỢC & XU HƯỚNG", 
    "🧬 2. MA TRẬN SẢN PHẨM", 
    "🌍 3. KHÁCH HÀNG & THỊ TRƯỜNG", 
    "🧪 4. MÔ PHỎNG LOGISTICS"
])

# --- TAB 1: CHIẾN LƯỢC ---
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**📉 Xu hướng Sản lượng theo tháng**")
        trend = df.groupby('ym')['sl'].sum().reset_index()
        fig = px.area(trend, x='ym', y='sl', markers=True)
        fig.update_traces(line_color=NEON_GREEN, fillcolor="rgba(0, 230, 118, 0.1)")
        st.plotly_chart(polish_chart(fig), use_container_width=True)
    
    with col2:
        st.markdown("**🍩 Tỷ trọng Chiến lược (Mới vs Cũ)**")
        grp_mix = df.groupby('loai_mau')['sl'].sum().reset_index()
        fig_pie = px.pie(grp_mix, values='sl', names='loai_mau', hole=0.6,
                         color='loai_mau',
                         color_discrete_map={'Mẫu Cũ (Repeat)': NEON_GREEN, 'Mẫu Mới (New)': NEON_PINK})
        st.plotly_chart(polish_chart(fig_pie), use_container_width=True)
        
        st.markdown(f"""
        <div class="insight-box">
            <b style="color:{NEON_BLUE}">💡 Insight:</b> Tỷ lệ mẫu mới hiện tại là <b>{new_ratio:.1f}%</b>.
            {"✅ Nằm trong vùng an toàn (dưới 35%), giúp tối ưu hiệu suất." if new_ratio <= 35 else 
             "⚠️ Vượt ngưỡng an toàn! Cần đàm phán dời lịch mẫu mới để tránh nghẽn chuyền."}
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2: MA TRẬN SẢN PHẨM (SKU Matrix) ---
with tab2:
    st.markdown(f"""
    <div class="insight-box" style="border-left: 4px solid {NEON_PINK}">
        <b>🎯 Chiến lược Tồn kho:</b> Phân loại SKU dựa trên Sản lượng (Volume) và Độ ổn định (Frequency).
        <br>• <b>RUNNER (Xanh):</b> Sản xuất Stock (Make-to-Stock).
        <br>• <b>STRANGER (Xám):</b> Sản xuất theo đơn (Make-to-Order) để tránh tồn kho chết.
    </div>
    """, unsafe_allow_html=True)
    
    # Lấy data SKU Stats
    sku_matrix = df.groupby(['ma_hang', 'sku_class', 'nhom_mau']).agg(
        vol=('sl', 'sum'),
        freq=('ym', 'nunique')
    ).reset_index()

    c2_1, c2_2 = st.columns([3, 1])
    
    with c2_1:
        fig_scatter = px.scatter(sku_matrix, x='freq', y='vol', color='sku_class', size='vol',
                                 hover_name='ma_hang', log_y=True,
                                 color_discrete_map={
                                     "RUNNER (Trụ cột)": NEON_GREEN,
                                     "REPEATER (Ổn định)": NEON_BLUE,
                                     "STRANGER (Thời vụ/Mới)": "#757575"
                                 },
                                 title="Ma trận Sản phẩm: Volume vs Frequency")
        fig_scatter.add_vline(x=6, line_dash="dash", line_color="white", annotation_text="Ngưỡng ổn định (6 tháng)")
        st.plotly_chart(polish_chart(fig_scatter), use_container_width=True)
        
    with c2_2:
        st.markdown("**Phân bổ Số lượng SKU**")
        count_sku = sku_matrix['sku_class'].value_counts().reset_index()
        count_sku.columns = ['Phân loại', 'Số lượng']
        fig_bar = px.bar(count_sku, x='Phân loại', y='Số lượng', color='Phân loại',
                         color_discrete_map={
                             "RUNNER (Trụ cột)": NEON_GREEN,
                             "REPEATER (Ổn định)": NEON_BLUE,
                             "STRANGER (Thời vụ/Mới)": "#757575"
                         })
        st.plotly_chart(polish_chart(fig_bar), use_container_width=True)

# --- TAB 3: KHÁCH HÀNG ---
with tab3:
    col3_1, col3_2 = st.columns([2, 1])
    
    with col3_1:
        st.markdown("**🏆 Top Khách hàng (Pareto)**")
        cust_stats = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
        cust_stats['cum_perc'] = cust_stats['sl'].cumsum() / cust_stats['sl'].sum()
        
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(x=cust_stats.head(15)['khach_hang'], y=cust_stats.head(15)['sl'], 
                                    name='Sản lượng', marker_color=NEON_BLUE))
        fig_pareto.add_trace(go.Scatter(x=cust_stats.head(15)['khach_hang'], y=cust_stats.head(15)['cum_perc'], 
                                        name='% Tích lũy', yaxis='y2', line=dict(color=NEON_PINK, width=2)))
        
        fig_pareto.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 1.1], showgrid=False),
                                 template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(polish_chart(fig_pareto), use_container_width=True)
        
    with col3_2:
        st.markdown("**Danh sách Top Khách Hàng**")
        render_dark_aggrid(cust_stats.head(10), height=400)

# --- TAB 4: MÔ PHỎNG LOGISTICS (SIMULATION) ---
with tab4:
    st.subheader("🧪 Mô phỏng Kế hoạch Logistics 2026")
    
    base_data = df[df['year'] == curr_year]
    base_vol = base_data['sl'].sum()
    
    # Tính container (Giả sử 1 cont 40ft ~= 800 sản phẩm nội thất trung bình - Con số ước lượng)
    # Tốt nhất là lấy từ cột sl_container nếu có
    if 'sl_container' in df.columns:
        base_cont = base_data['sl_container'].sum()
        avg_items_per_cont = base_vol / base_cont if base_cont > 0 else 800
    else:
        avg_items_per_cont = 800 # Mặc định
        base_cont = base_vol / avg_items_per_cont

    c4_1, c4_2 = st.columns([1, 2])
    
    with c4_1:
        st.markdown("##### 🎛️ Tham số đầu vào")
        sim_growth = st.slider("Dự báo tăng trưởng (%)", -20, 50, 15)
        sim_usb_rate = st.slider("Tỷ lệ hàng có USB (%)", 0, 100, int(base_data['is_usb'].mean()*100))
        
        target_vol = base_vol * (1 + sim_growth/100)
        target_cont = target_vol / avg_items_per_cont
        target_usb = target_vol * (sim_usb_rate/100)
        
    with c4_2:
        st.markdown("##### 📊 Kết quả Dự báo")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='kpi-val' style='color:{NEON_GREEN}'>{target_vol:,.0f}</div>", unsafe_allow_html=True)
            st.markdown("Sản lượng Mục tiêu")
        with m2:
            st.markdown(f"<div class='kpi-val' style='color:{NEON_BLUE}'>{target_cont:,.1f}</div>", unsafe_allow_html=True)
            st.markdown("Số Container (Est)")
        with m3:
            st.markdown(f"<div class='kpi-val' style='color:{NEON_PINK}'>{target_usb:,.0f}</div>", unsafe_allow_html=True)
            st.markdown("Bộ phụ kiện USB")
            
        st.markdown(f"""
        <div style="margin-top:20px; padding:15px; border:1px dashed {TEXT_SUB}; border-radius:8px;">
            <b>📝 Kế hoạch hành động:</b><br>
            • Cần đặt chỗ (booking) trung bình <b>{target_cont/12:,.1f}</b> container/tháng.<br>
            • Đàm phán nhà cung cấp phụ kiện USB cho lô hàng <b>{target_usb:,.0f}</b> bộ.<br>
            • Nếu tăng trưởng > 20%, cần kích hoạt phương án thuê kho ngoài (3PL).
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"<div style='text-align:center; color:{TEXT_SUB}; font-size:12px;'>© 2026 MỘC PHÁT FURNITURE | SYSTEM STATUS: NORMAL</div>", unsafe_allow_html=True)
