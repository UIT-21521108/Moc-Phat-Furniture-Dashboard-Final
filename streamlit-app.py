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
# 1. CẤU HÌNH GIAO DIỆN (NEON CENTERED EDITION)
# ==========================================
st.set_page_config(page_title="Mộc Phát Analytics Pro", layout="wide", page_icon="🌲")

# BẢNG MÀU
PRIMARY = "#00C853"    
NEON_GREEN = "#00E676" 
BG_DARK = "#050505"    
TEXT_MAIN = "#FFFFFF"  # Đổi sang trắng tinh cho nổi
TEXT_SUB = "#E0E0E0"   # Xám sáng hơn chút
GRID_COLOR = "rgba(255, 255, 255, 0.08)"

# --- CSS VISUAL EFFECTS (CLEAN & POP) ---
css_style = """
<style>
    /* 1. NỀN DEEP NEBULA */
    .stApp {{
        background-color: {bg_dark};
        background-image: 
            radial-gradient(at 50% 0%, rgba(0, 200, 83, 0.1) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(0, 230, 118, 0.05) 0px, transparent 50%);
        background-attachment: fixed;
    }}

    /* 2. HEADER CĂN GIỮA & LOGO NEON */
    .header-sticky {{
        position: sticky; top: 0; z-index: 999;
        background: rgba(10, 10, 10, 0.85);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(0, 230, 118, 0.3); /* Viền sáng hơn chút */
        padding: 15px 0; 
        margin-bottom: 30px;
        display: flex; 
        align-items: center; 
        justify-content: center; /* CĂN GIỮA */
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        position: relative;
    }}
    
    /* Hiệu ứng Logo Neon */
    .glow-logo {{
        filter: drop-shadow(0 0 8px {neon}); /* Phát sáng */
        transition: transform 0.3s;
    }}
    .glow-logo:hover {{
        transform: scale(1.1);
        filter: drop-shadow(0 0 15px {neon});
    }}

    /* Badge nằm tuyệt đối bên phải */
    .header-badge {{
        position: absolute;
        right: 25px;
        top: 50%;
        transform: translateY(-50%);
        border: 1px solid {primary};
        color: {neon};
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        background: rgba(0, 200, 83, 0.1);
    }}

    /* 3. TIÊU ĐỀ PHỤ (SUBHEADER) NỔI BẬT */
    /* Target vào h3 của Streamlit (st.subheader) */
    h3 {{
        color: {neon} !important;          /* Màu xanh Neon */
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(0, 230, 118, 0.3); /* Bóng sáng */
        border-left: 4px solid {primary};  /* Vạch kẻ bên trái */
        padding-left: 12px;
        margin-top: 20px !important;
        margin-bottom: 10px !important;
    }}
    
    /* Target vào h1, h2 */
    h1, h2 {{
        color: {text_main} !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }}

    /* 4. KPI CARDS (Depth & Glow) */
    .kpi-card {{
        background: rgba(30, 30, 30, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid {primary};
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-5px);
        border-left-color: {neon};
        box-shadow: 0 8px 30px rgba(0, 230, 118, 0.15);
        background: rgba(45, 45, 45, 0.8);
        border-top: 1px solid rgba(0, 230, 118, 0.3);
    }}
    .kpi-lbl {{ font-size: 13px; color: {text_sub}; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }}
    .kpi-val {{ font-size: 32px; font-weight: 800; color: {text_main}; letter-spacing: -1px; margin: 5px 0; }}

    /* 5. TABS & AGGRID */
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; background: transparent; }}
    .stTabs [data-baseweb="tab"] {{ 
        background-color: rgba(255,255,255,0.05); 
        color: {text_sub}; 
        border-radius: 8px; 
        border: 1px solid transparent;
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{ 
        border: 1px solid {neon};
        color: {neon};
        background: rgba(0, 230, 118, 0.08);
        font-weight: bold;
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.1);
    }}
    
    /* AgGrid Dark Theme Clean */
    .ag-theme-alpine-dark {{
        --ag-background-color: #0F0F0F !important;
        --ag-header-background-color: #181818 !important;
        --ag-odd-row-background-color: #0F0F0F !important;
        --ag-foreground-color: {text_sub} !important;
        --ag-border-color: #333 !important;
        font-family: 'Segoe UI', sans-serif !important;
    }}
</style>
""".format(
    bg_dark=BG_DARK, primary=PRIMARY, neon=NEON_GREEN, 
    text_main=TEXT_MAIN, text_sub=TEXT_SUB
)
st.markdown(css_style, unsafe_allow_html=True)

# --- HÀM STYLE BIỂU ĐỒ ---
def polish_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_SUB, family="Segoe UI"),
        margin=dict(t=40, b=20, l=10, r=10),
        hovermode="x unified",
    )
    # Lưới sáng hơn chút để dễ nhìn trên nền đen
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
# 3. HEADER (CENTERED & NEON LOGO)
# ==========================================
def get_base64_logo(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_base64_logo("mocphat_logo.png")
# Thêm class glow-logo cho ảnh
logo_img = f'<img src="data:image/png;base64,{logo_b64}" height="60" class="glow-logo">' if logo_b64 else '<span style="font-size:50px">🌲</span>'

st.markdown(f"""
<div class="header-sticky">
    <div style="text-align:center;">
        {logo_img}
        <div style="font-size:28px; font-weight:900; color:{NEON_GREEN}; letter-spacing:1px; margin-top:5px; text-shadow:0 0 10px rgba(0,230,118,0.5);">
            MỘC PHÁT INTELLIGENCE
        </div>
        <div style="font-size:14px; color:{TEXT_SUB}; letter-spacing:2px; font-weight:300;">
            EXECUTIVE DASHBOARD
        </div>
    </div>
    <div class="header-badge">
        MASTER 2023-2025
    </div>
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
# 4. KPI CARDS
# ==========================================
st.subheader("🚀 HIỆU QUẢ KINH DOANH")
vol_by_year = df.groupby('year')['sl'].sum()
v24 = vol_by_year.get(2024, 0)
v23 = vol_by_year.get(2023, 0)
g24 = ((v24 - v23) / v23 * 100) if v23 > 0 else 0

c1, c2, c3, c4 = st.columns(4)

def kpi_card(col, lbl, val, sub_val, sub_lbl):
    val_str = f"{val:,.0f}" 
    color = NEON_GREEN if sub_val >= 0 else "#FF5252"
    icon = "▲" if sub_val >= 0 else "▼"
    
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-lbl">{lbl}</div>
        <div class="kpi-val">{val_str}</div>
        <div style="font-size:14px; font-weight:700; margin-top:8px; color:{color}">
            {icon} {abs(sub_val):.1f}% <span style="color:{TEXT_SUB}; font-weight:normal; font-size:13px">{sub_lbl}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

kpi_card(c1, "SẢN LƯỢNG 2023", v23, 0, "(Base)")
kpi_card(c2, "SẢN LƯỢNG 2024", v24, g24, "vs 2023")
kpi_card(c3, "SẢN LƯỢNG 2025", vol_by_year.get(2025,0), 0, "(Real-time)")
kpi_card(c4, "ĐỐI TÁC KHÁCH HÀNG", df['khach_hang'].nunique(), 0, "Active")

st.markdown("---")

# ==========================================
# 5. TABS PHÂN TÍCH
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 TỔNG QUAN", "🎯 KẾ HOẠCH 2026", "🎨 SỨC KHỎE SP", "🌡️ MÙA VỤ", "⚖️ KHÁCH HÀNG", "📋 DỮ LIỆU"
])

def render_dark_aggrid(dataframe, height=400):
    gb = GridOptionsBuilder.from_dataframe(dataframe)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('multiple', use_checkbox=True)
    gb.configure_default_column(resizable=True, filterable=True, sortable=True)
    for col in dataframe.select_dtypes(include=['number']).columns:
        gb.configure_column(col, type=["numericColumn", "numberColumnFilter"], precision=0)
    gridOptions = gb.build()
    AgGrid(dataframe, gridOptions=gridOptions, height=height, theme='alpine-dark', enable_enterprise_modules=False)

# --- TAB 1: TỔNG QUAN ---
with tab1:
    c1_left, c1_right = st.columns([3, 1])
    with c1_left:
        st.subheader("📈 Xu hướng & Phát hiện Bất thường")
        ts_data = df.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
        
        fig = go.Figure()
        # Area chart clean
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['sl'], mode='lines+markers', name='Thực tế', 
                                 line=dict(color=NEON_GREEN, width=3, shape='spline'),
                                 fill='tozeroy', fillcolor='rgba(0, 230, 118, 0.1)')) 
        
        ts_data['ma3'] = ts_data['sl'].rolling(window=3).mean()
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['ma3'], mode='lines', name='TB 3 tháng', line=dict(color='#FFA726', dash='dot')))
        
        # Anomaly
        std = ts_data['sl'].rolling(window=3).std()
        upper = ts_data['ma3'] + (1.8 * std)
        anomalies = ts_data[ts_data['sl'] > upper]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies['ym'], y=anomalies['sl'], mode='markers', name='Đột biến', 
                                     marker=dict(color='#FF5252', size=14, symbol='star', line=dict(color='white', width=1))))
            
        st.plotly_chart(polish_chart(fig), use_container_width=True)

    with c1_right:
        if not ts_data.empty:
            last_m = ts_data.iloc[-1]
            prev_m = ts_data.iloc[-2] if len(ts_data) > 1 else last_m
            mom = ((last_m['sl'] - prev_m['sl'])/prev_m['sl']*100) if prev_m['sl']>0 else 0
            sl_fmt = f"{last_m['sl']:,.0f}"
            
            st.markdown(f"""
            <div style="background:rgba(20, 20, 20, 0.6); border-left:4px solid {NEON_GREEN}; padding:20px; border-radius:12px; border:1px solid rgba(255,255,255,0.08);">
                <div style="color:{NEON_GREEN}; font-weight:800; margin-bottom:12px; font-size:16px;">🤖 AI QUICK STATS</div>
                <div style="font-size:15px; color:{TEXT_MAIN}; line-height:1.8">
                • Tháng <b>{last_m['ym'].strftime('%m/%Y')}</b>: <br>
                  <span style="font-size:24px; font-weight:bold; color:{TEXT_MAIN}">{sl_fmt}</span> SP<br>
                • Biến động: <b style="color:{NEON_GREEN if mom>0 else '#FF5252'}">{mom:+.1f}%</b> so với tháng trước.<br>
                • Phát hiện <b style="color:#FFA726">{len(anomalies)}</b> điểm bất thường.
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 2: KẾ HOẠCH 2026 ---
with tab2:
    st.subheader("🎯 Lập Kế Hoạch 2026")
    col_input, col_info = st.columns([1, 2])
    with col_input:
        growth_target = st.slider("Mục tiêu Tăng trưởng (%)", 0, 100, 15, 5)
        growth_factor = 1 + (growth_target / 100)
    
    base_2025 = df_raw[df_raw['year'] == 2025].copy()
    if not base_2025.empty:
        sl_2025_total = base_2025['sl'].sum()
        sl_2026_target = sl_2025_total * growth_factor
        
        v25 = f"{sl_2025_total:,.0f}"
        v26 = f"{sl_2026_target:,.0f}"
        v_gap = f"{sl_2026_target - sl_2025_total:,.0f}"
        
        with col_info:
            st.markdown(f"""
            <div style="background:rgba(255,167,38,0.05); border:1px solid #FFA726; padding:20px; border-radius:12px; display:flex; justify-content:space-around; align-items:center;">
                <div style="text-align:center"><div style="font-size:13px; color:#aaa; font-weight:600">2025 BASE</div><div style="font-size:28px; font-weight:bold">{v25}</div></div>
                <div style="font-size:24px; color:#FFA726">➔</div>
                <div style="text-align:center"><div style="font-size:13px; color:#aaa; font-weight:600">2026 TARGET</div><div style="font-size:28px; font-weight:bold; color:{NEON_GREEN}">{v26}</div></div>
                <div style="text-align:center"><div style="font-size:13px; color:#aaa; font-weight:600">GAP (+{growth_target}%)</div><div style="font-size:28px; font-weight:bold; color:#FFA726">+{v_gap}</div></div>
            </div>
            """, unsafe_allow_html=True)
            
        monthly_2025 = base_2025.groupby('month')['sl'].sum().reset_index()
        monthly_2026 = monthly_2025.copy()
        monthly_2026['sl'] = monthly_2026['sl'] * growth_factor
        monthly_2026['Type'] = 'Mục tiêu 2026'
        monthly_2025['Type'] = 'Thực tế 2025'
        combined_forecast = pd.concat([monthly_2025, monthly_2026])
        
        fig_forecast = px.line(combined_forecast, x='month', y='sl', color='Type', markers=True, 
                               color_discrete_map={'Thực tế 2025': '#555', 'Mục tiêu 2026': NEON_GREEN})
        fig_forecast.update_traces(line=dict(width=3))
        st.plotly_chart(polish_chart(fig_forecast), use_container_width=True)

# --- TAB 3: SỨC KHỎE SP ---
with tab3:
    col_sun, col_bar = st.columns([2, 1])
    with col_sun:
        st.subheader("🎨 Phân tích Màu Sắc")
        color_data = df.groupby(['nhom_mau', 'mau_son'])['sl'].sum().reset_index()
        color_data = color_data[color_data['sl'] > color_data['sl'].sum() * 0.01] 
        fig_sun = px.sunburst(color_data, path=['nhom_mau', 'mau_son'], values='sl',
                              color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(polish_chart(fig_sun), use_container_width=True)
        
    with col_bar:
        st.subheader("🏆 Top Màu Sơn")
        top_colors = df.groupby('mau_son')['sl'].sum().nlargest(10).sort_values(ascending=True).reset_index()
        fig_bar_col = px.bar(top_colors, x='sl', y='mau_son', orientation='h', text_auto='.2s', 
                             color='sl', color_continuous_scale='Greens')
        st.plotly_chart(polish_chart(fig_bar_col), use_container_width=True)

    c_sku, c_usb = st.columns(2)
    with c_sku:
        st.subheader("Top 10 SKU")
        top_sku = df.groupby('ma_hang')['sl'].sum().nlargest(10).sort_values(ascending=True).reset_index()
        fig_sku = px.bar(top_sku, x='sl', y='ma_hang', orientation='h', text_auto='.2s', color_discrete_sequence=[PRIMARY])
        st.plotly_chart(polish_chart(fig_sku), use_container_width=True)
    with c_usb:
        st.subheader("Tỷ trọng USB")
        usb_trend = df.groupby(['year', 'is_usb_clean'])['sl'].sum().reset_index()
        fig_usb = px.bar(usb_trend, x='year', y='sl', color='is_usb_clean', barmode='group',
                         color_discrete_map={'Có USB': '#FFA726', 'Không USB': '#333'})
        st.plotly_chart(polish_chart(fig_usb), use_container_width=True)

# --- TAB 4: MÙA VỤ ---
with tab4:
    st.subheader("🌡️ Bản đồ nhiệt (Heatmap)")
    heat_data = df.groupby(['mua', 'nhom_mau'])['sl'].sum().reset_index()
    heat_data['share'] = heat_data['sl'] / heat_data.groupby('mua')['sl'].transform('sum')
    pivot = heat_data.pivot(index='mua', columns='nhom_mau', values='share').fillna(0).reindex(['Xuân', 'Hè', 'Thu', 'Đông'])
    fig_heat = px.imshow(pivot, text_auto='.0%', aspect="auto", color_continuous_scale='Greens', origin='upper')
    st.plotly_chart(polish_chart(fig_heat), use_container_width=True)

# --- TAB 5: KHÁCH HÀNG ---
with tab5:
    c5_1, c5_2 = st.columns([2, 1])
    with c5_1:
        st.subheader("Nguyên tắc Pareto (80/20)")
        pareto = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
        pareto['cum'] = pareto['sl'].cumsum()
        pareto['perc'] = pareto['cum'] / pareto['sl'].sum() * 100
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=pareto['khach_hang'], y=pareto['sl'], name='Sản lượng', marker_color=PRIMARY))
        fig_p.add_trace(go.Scatter(x=pareto['khach_hang'], y=pareto['perc'], name='% Tích lũy', yaxis='y2', line=dict(color='#FF5252', width=2)))
        fig_p.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110]), showlegend=False)
        st.plotly_chart(polish_chart(fig_p), use_container_width=True)
    with c5_2:
        st.subheader("Top Tăng Trưởng")
        curr_y, prev_y = df['year'].max(), df['year'].max()-1
        v_c = df[df['year']==curr_y].groupby('khach_hang')['sl'].sum()
        v_p = df[df['year']==prev_y].groupby('khach_hang')['sl'].sum()
        growth = ((v_c - v_p)/v_p*100).fillna(0).sort_values(ascending=False).reset_index()
        growth.columns = ['Khách Hàng', '% Tăng Trưởng']
        render_dark_aggrid(growth.head(10), height=400)

# --- TAB 6: DỮ LIỆU ---
with tab6:
    st.subheader("Tra cứu dữ liệu chi tiết")
    grid_df = df.groupby(['ma_hang', 'khach_hang', 'mau_son', 'nhom_mau', 'year']).agg(Tong_SL=('sl', 'sum')).reset_index().sort_values('Tong_SL', ascending=False)
    render_dark_aggrid(grid_df, height=600)

st.markdown("---")
st.caption(f"© 2026 Mộc Phát Furniture | Cyber Glass Edition | Updated: {datetime.now().strftime('%d/%m/%Y')}")
