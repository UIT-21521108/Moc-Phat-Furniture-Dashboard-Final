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
# 1. CẤU HÌNH GIAO DIỆN (DEEP AURORA GLASS)
# ==========================================
st.set_page_config(page_title="Mộc Phát Analytics Pro", layout="wide", page_icon="🌲")

# BẢNG MÀU
PRIMARY_NEON = "#00E676"  # Xanh Neon Chính
ACCENT_BLUE = "#2979FF"   # Ánh xanh dương phụ trợ
TEXT_MAIN = "#FFFFFF"
TEXT_SUB = "#E0E0E0"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"

# --- CSS VISUAL EFFECTS ---
# Sử dụng {{ }} để escape dấu ngoặc nhọn trong f-string
st.markdown(f"""
<style>
    /* 1. NỀN DEEP AURORA (ÁNH XANH HUYỀN ẢO) */
    .stApp {{
        background-color: #050505;
        /* Tạo 2 luồng ánh sáng xanh lục và xanh dương giao thoa trên nền tối */
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(0, 230, 118, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 85% 30%, rgba(41, 121, 255, 0.06) 0%, transparent 50%);
        background-attachment: fixed;
        background-size: cover;
    }}

    /* 2. HEADER VÔ HÌNH (NO FRAME) */
    .header-container {{
        text-align: center;
        padding: 40px 0 20px 0;
        margin-bottom: 30px;
        /* Không có viền, không có nền, chỉ có ánh sáng */
    }}
    
    .glow-logo {{
        filter: drop-shadow(0 0 20px {PRIMARY_NEON});
        transition: transform 0.5s ease;
    }}
    .glow-logo:hover {{ transform: scale(1.05); filter: drop-shadow(0 0 30px {PRIMARY_NEON}); }}

    .neon-title {{
        font-family: 'Segoe UI', sans-serif;
        font-weight: 900;
        font-size: 42px;
        color: #fff;
        margin-top: 15px;
        text-transform: uppercase;
        letter-spacing: 2px;
        /* Soft Glow: Bóng tỏa nhẹ, không gắt */
        text-shadow: 
            0 0 10px rgba(0, 230, 118, 0.6),
            0 0 30px rgba(0, 230, 118, 0.2);
    }}
    
    .sub-title {{
        font-size: 14px; color: {TEXT_SUB}; letter-spacing: 4px; font-weight: 300;
        text-shadow: 0 2px 4px rgba(0,0,0,0.8);
        opacity: 0.9;
    }}

    /* 3. SUBHEADERS (Tiêu đề phụ nổi bật) */
    h3 {{
        color: #fff !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        /* Bóng chữ tạo độ nổi 3D tách biệt khỏi nền */
        text-shadow: 0 4px 10px rgba(0,0,0,1); 
        border-left: 4px solid {PRIMARY_NEON};
        padding-left: 15px;
        margin-top: 35px !important;
        margin-bottom: 20px !important;
    }}

    /* 4. GLASSMORPHISM CARDS (KHUNG KÍNH MỜ) */
    .glass-card {{
        background: rgba(255, 255, 255, 0.03); /* Nền cực trong */
        backdrop-filter: blur(16px);           /* Làm mờ hậu cảnh mạnh */
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08); /* Viền kính mỏng */
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2); /* Bóng sâu */
        transition: all 0.3s ease;
    }}
    .glass-card:hover {{
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(0, 230, 118, 0.4);
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(0, 230, 118, 0.1);
    }}

    /* 5. GOLD CARD (Cho phần Kế hoạch - Giữ nguyên như bạn thích) */
    .gold-glass-card {{
        background: linear-gradient(135deg, rgba(255, 167, 38, 0.15), rgba(245, 124, 0, 0.05));
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 167, 38, 0.4);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 40px rgba(255, 167, 38, 0.1);
    }}

    /* Typography trong Card */
    .kpi-lbl {{ font-size: 13px; color: {TEXT_SUB}; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; opacity: 0.9; }}
    .kpi-val {{ font-size: 36px; font-weight: 800; color: #fff; margin: 8px 0; text-shadow: 0 2px 5px rgba(0,0,0,0.5); }}

    /* 6. TABS & AGGRID TRONG SUỐT */
    .stTabs [data-baseweb="tab-list"] {{ gap: 15px; }}
    .stTabs [data-baseweb="tab"] {{ 
        background-color: rgba(255,255,255,0.02); 
        backdrop-filter: blur(10px);
        color: {TEXT_SUB}; 
        border-radius: 12px; 
        border: 1px solid rgba(255,255,255,0.05);
        padding: 10px 25px;
    }}
    .stTabs [aria-selected="true"] {{ 
        border: 1px solid {PRIMARY_NEON};
        color: {PRIMARY_NEON};
        background: rgba(0, 230, 118, 0.1);
        font-weight: bold;
    }}

    /* Chart Container (Bọc biểu đồ vào kính) */
    .chart-box {{
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 15px;
    }}

    /* AgGrid Dark Transparent */
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

# --- HÀM STYLE BIỂU ĐỒ (TRANSPARENT) ---
def polish_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_SUB, family="Segoe UI"),
        margin=dict(t=40, b=20, l=10, r=10),
        hovermode="x unified"
    )
    # Lưới sáng hơn chút để nổi trên nền kính
    fig.update_xaxes(showgrid=False, linecolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig

# ==========================================
# 2. LOAD DATA
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    FILE_NAME = "Master_2023_2025_PRO_clean.xlsx"
    if not os.path.exists(FILE_NAME): return None, f"⚠️ Không tìm thấy file {FILE_NAME}"

    try:
        df = pd.read_excel(FILE_NAME, engine='openpyxl')
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
        df = df[(df['year'] > 2020) & (df['month'].between(1, 12))]
        df['ym'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])
        
        season_map = {12:'Đông', 1:'Đông', 2:'Đông', 3:'Xuân', 4:'Xuân', 5:'Xuân', 
                      6:'Hè', 7:'Hè', 8:'Hè', 9:'Thu', 10:'Thu', 11:'Thu'}
        df['mua'] = df['month'].map(season_map)
        
        cols_text = ['khach_hang', 'ma_hang', 'mau_son', 'khu_vuc', 'dim', 'mo_ta']
        for c in cols_text:
            if c not in df.columns: df[c] = "Unknown"
            else: df[c] = df[c].fillna("Unknown").astype(str).str.upper()
            
        df['sl'] = pd.to_numeric(df['sl'], errors='coerce').fillna(0)

        def categorize_detailed_color(v):
            v = v.strip()
            if any(x in v for x in ["BROWN", "COCOA", "BRONZE", "UMBER", "NAU", "WALNUT", "ESPRESSO"]): return "NÂU/GỖ"
            if any(x in v for x in ["WHITE", "CREAM", "IVORY", "TRANG", "OFF WHITE", "WASH"]): return "TRẮNG/KEM"
            if any(x in v for x in ["BLACK", "DEN", "CHARCOAL", "EBONY"]): return "ĐEN/TỐI"
            if any(x in v for x in ["GREY", "GRAY", "XAM", "SLATE"]): return "XÁM"
            if any(x in v for x in ["NATURAL", "OAK", "PINE", "HONEY", "TU NHIEN"]): return "TỰ NHIÊN"
            if any(x in v for x in ["BLUE", "NAVY"]): return "XANH DƯƠNG"
            if any(x in v for x in ["GREEN", "SAGE"]): return "XANH LÁ"
            return "MÀU KHÁC"
        
        df['nhom_mau'] = df['mau_son'].apply(categorize_detailed_color)
        df['is_usb_clean'] = df['is_usb'].astype(str).apply(lambda x: 'Có USB' if 'true' in x.lower() else 'Không USB') if 'is_usb' in df.columns else 'N/A'

        return df, None
    except Exception as e:
        return None, str(e)

df_raw, error = load_data()
if error: st.error(error); st.stop()

# ==========================================
# 3. HEADER (INVISIBLE & SOFT NEON)
# ==========================================
def get_base64_logo(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_base64_logo("mocphat_logo.png")
logo_img = f'<img src="data:image/png;base64,{logo_b64}" height="80" class="glow-logo">' if logo_b64 else '<span style="font-size:70px">🌲</span>'

# Header container: Không khung, chỉ có chữ nổi trên nền Aurora
st.markdown(f"""
<div class="header-container">
    {logo_img}
    <div class="neon-title">MỘC PHÁT INTELLIGENCE</div>
    <div class="sub-title">PREMIUM ANALYTICS SUITE</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🎯 BỘ LỌC")
years = sorted(df_raw['year'].unique(), reverse=True)
sel_years = st.sidebar.multiselect("Năm", years, default=years)
sel_cust = st.sidebar.multiselect("Khách Hàng", sorted(df_raw['khach_hang'].unique()))

df = df_raw.copy()
if sel_years: df = df[df['year'].isin(sel_years)]
if sel_cust: df = df[df['khach_hang'].isin(sel_cust)]
if df.empty: st.warning("Không có dữ liệu!"); st.stop()

# ==========================================
# 4. KPI CARDS (GLASS EFFECT)
# ==========================================
st.subheader("🚀 Hiệu quả Kinh doanh")
vol_by_year = df.groupby('year')['sl'].sum()
v24 = vol_by_year.get(2024, 0)
v23 = vol_by_year.get(2023, 0)
g24 = ((v24 - v23) / v23 * 100) if v23 > 0 else 0

c1, c2, c3, c4 = st.columns(4)

def kpi_card(col, lbl, val, sub_val, sub_lbl):
    val_str = f"{val:,.0f}" 
    color = PRIMARY_NEON if sub_val >= 0 else "#FF5252"
    icon = "▲" if sub_val >= 0 else "▼"
    
    # Sử dụng class "glass-card"
    col.markdown(f"""
    <div class="glass-card">
        <div class="kpi-lbl">{lbl}</div>
        <div class="kpi-val">{val_str}</div>
        <div style="font-size:14px; font-weight:700; margin-top:5px; color:{color}">
            {icon} {abs(sub_val):.1f}% <span style="color:{TEXT_SUB}; font-weight:normal; font-size:13px; opacity:0.8">{sub_lbl}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

kpi_card(c1, "SẢN LƯỢNG 2023", v23, 0, "Base Year")
kpi_card(c2, "SẢN LƯỢNG 2024", v24, g24, "vs 2023")
kpi_card(c3, "SẢN LƯỢNG 2025", vol_by_year.get(2025,0), 0, "Real-time")
kpi_card(c4, "ĐỐI TÁC KHÁCH HÀNG", df['khach_hang'].nunique(), 0, "Active")

st.markdown("---")

# ==========================================
# 5. TABS PHÂN TÍCH (GLASS CONTAINERS)
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 TỔNG QUAN", "🎯 KẾ HOẠCH 2026", "🎨 SỨC KHỎE SP", "🌡️ MÙA VỤ", "⚖️ KHÁCH HÀNG", "📋 DỮ LIỆU"
])

# Hàm render AgGrid trong khung kính
def render_glass_aggrid(dataframe, height=400):
    gb = GridOptionsBuilder.from_dataframe(dataframe)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('multiple', use_checkbox=True)
    gb.configure_default_column(resizable=True, filterable=True, sortable=True)
    for col in dataframe.select_dtypes(include=['number']).columns:
        gb.configure_column(col, type=["numericColumn", "numberColumnFilter"], precision=0)
    gridOptions = gb.build()
    
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    AgGrid(dataframe, gridOptions=gridOptions, height=height, theme='alpine-dark', enable_enterprise_modules=False)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 1: TỔNG QUAN ---
with tab1:
    c1_left, c1_right = st.columns([3, 1])
    with c1_left:
        st.subheader("📈 Xu hướng & Phát hiện Bất thường")
        ts_data = df.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['sl'], mode='lines+markers', name='Thực tế', 
                                 line=dict(color=PRIMARY_NEON, width=3, shape='spline'),
                                 fill='tozeroy', fillcolor='rgba(0, 230, 118, 0.15)')) 
        ts_data['ma3'] = ts_data['sl'].rolling(window=3).mean()
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['ma3'], mode='lines', name='TB 3 tháng', line=dict(color='#FFA726', dash='dot')))
        
        std = ts_data['sl'].rolling(window=3).std()
        upper = ts_data['ma3'] + (1.8 * std)
        anomalies = ts_data[ts_data['sl'] > upper]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies['ym'], y=anomalies['sl'], mode='markers', name='Đột biến', 
                                     marker=dict(color='#FF5252', size=14, symbol='star', line=dict(color='white', width=1))))
        
        # Bọc biểu đồ trong khung kính
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(polish_chart(fig), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c1_right:
        if not ts_data.empty:
            last_m = ts_data.iloc[-1]
            prev_m = ts_data.iloc[-2] if len(ts_data) > 1 else last_m
            mom = ((last_m['sl'] - prev_m['sl'])/prev_m['sl']*100) if prev_m['sl']>0 else 0
            sl_fmt = f"{last_m['sl']:,.0f}"
            
            # Insight box dùng glass-card
            st.subheader("💡 Phân tích nhanh")
            st.markdown(f"""
            <div class="glass-card" style="border-left: 3px solid {PRIMARY_NEON}">
                <div style="font-size:15px; color:#fff; line-height:1.8">
                • Tháng <b>{last_m['ym'].strftime('%m/%Y')}</b>: <br>
                  <span style="font-size:24px; font-weight:bold; color:#fff">{sl_fmt}</span> SP<br>
                • Biến động: <b style="color:{PRIMARY_NEON if mom>0 else '#FF5252'}">{mom:+.1f}%</b>.<br>
                • Phát hiện <b style="color:#FFA726">{len(anomalies)}</b> bất thường.
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 2: KẾ HOẠCH 2026 ---
with tab2:
    st.subheader("🎯 Lập Kế Hoạch 2026")
    col_input, col_info = st.columns([1, 2])
    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        growth_target = st.slider("Mục tiêu Tăng trưởng (%)", 0, 100, 15, 5)
        st.markdown('</div>', unsafe_allow_html=True)
        growth_factor = 1 + (growth_target / 100)
    
    base_2025 = df_raw[df_raw['year'] == 2025].copy()
    if not base_2025.empty:
        sl_2025_total = base_2025['sl'].sum()
        sl_2026_target = sl_2025_total * growth_factor
        
        v25 = f"{sl_2025_total:,.0f}"
        v26 = f"{sl_2026_target:,.0f}"
        v_gap = f"{sl_2026_target - sl_2025_total:,.0f}"
        
        with col_info:
            # SỬ DỤNG CLASS GOLD-GLASS-CARD (VÀNG GRADIENT)
            st.markdown(f"""
            <div class="gold-glass-card" style="display:flex; justify-content:space-around; align-items:center;">
                <div style="text-align:center"><div style="font-size:13px; color:#ddd; font-weight:600">2025 BASE</div><div style="font-size:28px; font-weight:bold; color:#fff">{v25}</div></div>
                <div style="font-size:24px; color:#FFF">➔</div>
                <div style="text-align:center"><div style="font-size:13px; color:#ddd; font-weight:600">2026 TARGET</div><div style="font-size:28px; font-weight:bold; color:#fff">{v26}</div></div>
                <div style="text-align:center"><div style="font-size:13px; color:#ddd; font-weight:600">GAP (+{growth_target}%)</div><div style="font-size:28px; font-weight:bold; color:#fff">+{v_gap}</div></div>
            </div>
            """, unsafe_allow_html=True)
            
        monthly_2025 = base_2025.groupby('month')['sl'].sum().reset_index()
        monthly_2026 = monthly_2025.copy()
        monthly_2026['sl'] = monthly_2026['sl'] * growth_factor
        monthly_2026['Type'] = 'Mục tiêu 2026'
        monthly_2025['Type'] = 'Thực tế 2025'
        combined_forecast = pd.concat([monthly_2025, monthly_2026])
        
        fig_forecast = px.line(combined_forecast, x='month', y='sl', color='Type', markers=True, 
                               color_discrete_map={'Thực tế 2025': '#757575', 'Mục tiêu 2026': PRIMARY_NEON})
        fig_forecast.update_traces(line=dict(width=3))
        
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(polish_chart(fig_forecast), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: SỨC KHỎE SP ---
with tab3:
    col_sun, col_bar = st.columns([2, 1])
    with col_sun:
        st.subheader("🎨 Phân tích Màu Sắc")
        color_data = df.groupby(['nhom_mau', 'mau_son'])['sl'].sum().reset_index()
        color_data = color_data[color_data['sl'] > color_data['sl'].sum() * 0.01] 
        fig_sun = px.sunburst(color_data, path=['nhom_mau', 'mau_son'], values='sl',
                              color_discrete_sequence=px.colors.qualitative.Set3)
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(polish_chart(fig_sun), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_bar:
        st.subheader("🏆 Top Màu Sơn")
        top_colors = df.groupby('mau_son')['sl'].sum().nlargest(10).sort_values(ascending=True).reset_index()
        fig_bar_col = px.bar(top_colors, x='sl', y='mau_son', orientation='h', text_auto='.2s', 
                             color='sl', color_continuous_scale='Greens')
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(polish_chart(fig_bar_col), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c_sku, c_usb = st.columns(2)
    with c_sku:
        st.subheader("Top 10 SKU")
        top_sku = df.groupby('ma_hang')['sl'].sum().nlargest(10).sort_values(ascending=True).reset_index()
        fig_sku = px.bar(top_sku, x='sl', y='ma_hang', orientation='h', text_auto='.2s', color_discrete_sequence=[PRIMARY_NEON])
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(polish_chart(fig_sku), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c_usb:
        st.subheader("Tỷ trọng USB")
        usb_trend = df.groupby(['year', 'is_usb_clean'])['sl'].sum().reset_index()
        fig_usb = px.bar(usb_trend, x='year', y='sl', color='is_usb_clean', barmode='group',
                         color_discrete_map={'Có USB': '#FFA726', 'Không USB': '#333'})
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(polish_chart(fig_usb), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 4: MÙA VỤ ---
with tab4:
    st.subheader("🌡️ Bản đồ nhiệt (Heatmap)")
    heat_data = df.groupby(['mua', 'nhom_mau'])['sl'].sum().reset_index()
    heat_data['share'] = heat_data['sl'] / heat_data.groupby('mua')['sl'].transform('sum')
    pivot = heat_data.pivot(index='mua', columns='nhom_mau', values='share').fillna(0).reindex(['Xuân', 'Hè', 'Thu', 'Đông'])
    fig_heat = px.imshow(pivot, text_auto='.0%', aspect="auto", color_continuous_scale='Greens', origin='upper')
    
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(polish_chart(fig_heat), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 5: KHÁCH HÀNG ---
with tab5:
    c5_1, c5_2 = st.columns([2, 1])
    with c5_1:
        st.subheader("Nguyên tắc Pareto (80/20)")
        pareto = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
        pareto['cum'] = pareto['sl'].cumsum()
        pareto['perc'] = pareto['cum'] / pareto['sl'].sum() * 100
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=pareto['khach_hang'], y=pareto['sl'], name='Sản lượng', marker_color=PRIMARY_NEON))
        fig_p.add_trace(go.Scatter(x=pareto['khach_hang'], y=pareto['perc'], name='% Tích lũy', yaxis='y2', line=dict(color='#FF5252', width=2)))
        fig_p.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110]), showlegend=False)
        
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(polish_chart(fig_p), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c5_2:
        st.subheader("Top Tăng Trưởng")
        curr_y, prev_y = df['year'].max(), df['year'].max()-1
        v_c = df[df['year']==curr_y].groupby('khach_hang')['sl'].sum()
        v_p = df[df['year']==prev_y].groupby('khach_hang')['sl'].sum()
        growth = ((v_c - v_p)/v_p*100).fillna(0).sort_values(ascending=False).reset_index()
        growth.columns = ['Khách Hàng', '% Tăng Trưởng']
        render_glass_aggrid(growth.head(10), height=400)

# --- TAB 6: DỮ LIỆU ---
with tab6:
    st.subheader("Tra cứu dữ liệu chi tiết")
    grid_df = df.groupby(['ma_hang', 'khach_hang', 'mau_son', 'nhom_mau', 'year']).agg(Tong_SL=('sl', 'sum')).reset_index().sort_values('Tong_SL', ascending=False)
    render_glass_aggrid(grid_df, height=600)

st.markdown("---")
st.caption(f"© 2026 Mộc Phát Furniture | Deep Aurora Glass Edition | Updated: {datetime.now().strftime('%d/%m/%Y')}")
