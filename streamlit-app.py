import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
import os
from datetime import datetime

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN (FULL GLOW EDITION)
# ==========================================
st.set_page_config(page_title="Mộc Phát Analytics", layout="wide", page_icon="🌲")

# Bảng màu Mộc Phát Premium
PRIMARY = "#066839"     # Xanh Mộc Phát gốc
NEON_GREEN = "#00E676"  # Xanh Neon (Tăng trưởng/Glow)
NEON_RED = "#FF5252"    # Đỏ Neon (Sụt giảm)
ACCENT  = "#66BB6A"     # Xanh lá sáng
BG_COLOR = "#050505"    # Đen sâu
CARD_BG = "#121212"     # Nền card
TEXT_MAIN = "#E0E0E0"
TEXT_SUB = "#9E9E9E"
GRID_COLOR = "#2A2A2A"

def get_base64_logo(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

def fmt_num(n):
    return f"{n:,.0f}"

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
    fig.update_xaxes(showgrid=False, linecolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig

# --- CSS CAO CẤP (FULL GLOW EFFECT) ---
st.markdown(f"""
<style>
    /* 1. Nền & Chữ */
    .stApp {{ background-color: {BG_COLOR}; }}
    h1, h2, h3, h4 {{ color: {TEXT_MAIN} !important; font-family: 'Segoe UI', sans-serif; }}
    .stMarkdown p, .stMarkdown li {{ color: {TEXT_SUB} !important; }}
    
    /* 2. Header Sticky */
    .header-sticky {{
        position: sticky; top: 15px; z-index: 999;
        background: rgba(18, 18, 18, 0.95);
        backdrop-filter: blur(10px);
        border-bottom: 2px solid {PRIMARY};
        padding: 15px 25px;
        margin: -50px 0px 25px 0px;
        border-radius: 16px;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }}
    .app-title {{ font-size: 28px; font-weight: 800; color: {ACCENT}; margin: 5px 0 0 0; text-transform: uppercase; letter-spacing: 1.5px; }}

    /* =============================================
       3. HIỆU ỨNG GLOW CHO TẤT CẢ CÁC THẺ (CARD)
       ============================================= */
    
    /* A. KPI CARDS */
    .kpi-card {{
        background: {CARD_BG}; 
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 4px solid {PRIMARY};
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }}
    .kpi-card:hover {{ 
        transform: translateY(-5px); 
        border-left-color: {NEON_GREEN};
        border-top: 1px solid rgba(0, 230, 118, 0.5);
        border-right: 1px solid rgba(0, 230, 118, 0.5);
        border-bottom: 1px solid rgba(0, 230, 118, 0.5);
        box-shadow: 0 0 20px rgba(0, 230, 118, 0.4); /* Glow xanh */
    }}
    .kpi-lbl {{ font-size: 13px; color: {TEXT_SUB}; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 5px; }}
    .kpi-val {{ font-size: 28px; font-weight: 800; color: {TEXT_MAIN}; transition: color 0.3s; }}
    .kpi-card:hover .kpi-val {{ color: {NEON_GREEN}; }}

    /* B. TABS (NAVIGATION) */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 40px;
        background-color: {CARD_BG};
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 6px;
        color: {TEXT_SUB};
        padding: 0 20px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
    }}
    /* Hover Tab */
    .stTabs [data-baseweb="tab"]:hover {{
        border-color: {NEON_GREEN} !important;
        color: {NEON_GREEN} !important;
        background-color: rgba(0, 230, 118, 0.1) !important;
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.6);
        transform: translateY(-2px);
    }}
    /* Active Tab */
    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY} !important;
        color: #FFFFFF !important;
        border: 1px solid {NEON_GREEN} !important;
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.4);
    }}

    /* C. INSIGHT & FORECAST BOX */
    .insight-box, .forecast-box {{
        padding: 15px; border-radius: 10px; margin-bottom: 20px;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }}
    
    .insight-box {{ background: rgba(6, 104, 57, 0.15); border: 1px solid {PRIMARY}; }}
    .insight-box:hover {{
        box-shadow: 0 0 20px rgba(0, 230, 118, 0.3);
        border-color: {NEON_GREEN};
        transform: scale(1.01);
    }}

    .forecast-box {{ background: rgba(255, 167, 38, 0.08); border: 1px solid #FFA726; }}
    .forecast-box:hover {{
        box-shadow: 0 0 20px rgba(255, 167, 38, 0.3);
        transform: scale(1.01);
    }}

    /* 6. DataFrame */
    [data-testid="stDataFrame"] {{
        background-color: {CARD_BG};
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }}
</style>
""", unsafe_allow_html=True)

# Header
logo_b64 = get_base64_logo("mocphat_logo.png")
logo_img = f'<img src="data:image/png;base64,{logo_b64}" height="65">' if logo_b64 else "🌲"
st.markdown(f"""
<div class="header-sticky">
    <div style="text-align: center;">
        {logo_img}
        <div class="app-title">MỘC PHÁT FURNITURE</div>
        <div style="font-size:14px; color:{TEXT_SUB}; letter-spacing: 0.5px; margin-top: 5px;">Hệ thống Phân tích Sản xuất & Kinh doanh</div>
    </div>
</div>
""", unsafe_allow_html=True)

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
# 3. SIDEBAR
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
# 4. KPI CARDS
# ==========================================
st.subheader("🚀 HIỆU QUẢ KINH DOANH")

# Tính toán số liệu
vol_by_year = df.groupby('year')['sl'].sum()
v23 = vol_by_year.get(2023, 0)
v24 = vol_by_year.get(2024, 0)
v25 = vol_by_year.get(2025, 0)

# Tính tăng trưởng
g24 = ((v24 - v23) / v23 * 100) if v23 > 0 else 0
g25 = ((v25 - v24) / v24 * 100) if v24 > 0 else 0

c1, c2, c3, c4 = st.columns(4)

def kpi_card(col, lbl, val, growth_val, sub_lbl):
    # Logic màu sắc: Dương -> Xanh, Âm -> Đỏ
    color = NEON_GREEN if growth_val >= 0 else NEON_RED
    icon = "▲" if growth_val >= 0 else "▼"
    
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-lbl">{lbl}</div>
        <div class="kpi-val">{fmt_num(val)}</div>
        <div style="font-size:13px; font-weight:600; margin-top:5px; color:{color}">
            {icon} {abs(growth_val):.1f}% <span style="color:{TEXT_SUB}; font-weight:normal; font-size:12px">{sub_lbl}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

kpi_card(c1, "SẢN LƯỢNG 2023", v23, 0, "(Gốc)")
kpi_card(c2, "SẢN LƯỢNG 2024", v24, g24, "so với 2023")
kpi_card(c3, "SẢN LƯỢNG 2025", v25, g25, "so với 2024")
kpi_card(c4, "KHÁCH HÀNG", df['khach_hang'].nunique(), 0, "Đối tác hoạt động")

st.markdown("---")

# ==========================================
# 5. TABS PHÂN TÍCH
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 TỔNG QUAN", "📈 DỰ BÁO TĂNG TRƯỞNG", "📦 SẢN PHẨM", "🎨 MÀU SẮC", "⚖️ KHÁCH HÀNG", "📋 DỮ LIỆU"
])

def render_table(dataframe, height=400):
    column_config = {}
    for col in dataframe.select_dtypes(include=['number']).columns:
        column_config[col] = st.column_config.NumberColumn(format="%d")
    
    st.dataframe(
        dataframe,
        height=height,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )

# --- TAB 1: TỔNG QUAN ---
with tab1:
    c1_left, c1_right = st.columns([3, 1])
    with c1_left:
        st.subheader("📈 Xu hướng Sản lượng")
        ts_data = df.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['sl'], mode='lines+markers', name='Thực tế', 
                                 line=dict(color=NEON_GREEN, width=3, shape='spline'),
                                 fill='tozeroy', fillcolor='rgba(0, 230, 118, 0.1)')) 
        
        ts_data['ma3'] = ts_data['sl'].rolling(window=3).mean()
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['ma3'], mode='lines', name='TB 3 tháng', line=dict(color='#FFA726', dash='dot')))
        
        st.plotly_chart(polish_chart(fig), width="stretch")

    with c1_right:
        if not ts_data.empty:
            last_m = ts_data.iloc[-1]
            prev_m = ts_data.iloc[-2] if len(ts_data) > 1 else last_m
            mom = ((last_m['sl'] - prev_m['sl'])/prev_m['sl']*100) if prev_m['sl']>0 else 0
            
            st.markdown(f"""
            <div class="insight-box">
                <div style="font-weight:bold; color:{ACCENT}; margin-bottom:10px">🤖 Phân tích nhanh:</div>
                <div style="font-size:14px; color:{TEXT_MAIN}; line-height:1.6">
                • Tháng <b>{last_m['ym'].strftime('%m/%Y')}</b> đạt <b>{fmt_num(last_m['sl'])}</b> sản phẩm.<br>
                • Biến động tháng: <b style="color:{NEON_GREEN if mom>0 else NEON_RED}">{mom:+.1f}%</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 2: DỰ BÁO TĂNG TRƯỞNG ---
with tab2:
    st.subheader("🎯 Dự báo Tăng trưởng 2026")
    st.caption("Mô hình dự báo dựa trên xu hướng lịch sử và chỉ số mùa vụ.")
    
    # 1. Prepare Data
    hist_data = df_raw.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
    hist_data['idx'] = np.arange(len(hist_data))
    hist_data['month'] = hist_data['ym'].dt.month
    
    x_hist = hist_data['idx'].values
    y_hist = hist_data['sl'].values
    
    if len(x_hist) > 1:
        # Trend
        z = np.polyfit(x_hist, y_hist, 1)
        p = np.poly1d(z)
        hist_data['trend'] = p(x_hist)
        hist_data['season'] = hist_data['sl'] / hist_data['trend']
        season_indices = hist_data.groupby('month')['season'].mean().to_dict()
        
        # Forecast
        future_dates = pd.date_range(start='2026-01-01', periods=12, freq='MS')
        future_idx = np.arange(len(hist_data), len(hist_data) + 12)
        trend_2026 = p(future_idx)
        
        pred_2026 = []
        for i, m in enumerate(future_dates.month):
            pred_2026.append(trend_2026[i] * season_indices.get(m, 1.0))
        pred_2026 = np.array(pred_2026)
        
        # Controls
        col_in, col_out = st.columns([1, 2])
        with col_in:
            st.info("Kéo thanh trượt để đặt mục tiêu tăng trưởng (Stretch Goal).")
            adj = st.slider("Điều chỉnh mục tiêu (+/- %)", -20, 50, 5, 5)
            final_pred = pred_2026 * (1 + adj/100)
            final_total = final_pred.sum()
            
        with col_out:
            v2025 = df_raw[df_raw['year']==2025]['sl'].sum()
            growth_g = ((final_total - v2025)/v2025)*100
            
            st.markdown(f"""
            <div class="forecast-box">
                <h4 style="margin:0; color:{NEON_GREEN}">MỤC TIÊU 2026 (Đã điều chỉnh +{adj}%)</h4>
                <div style="display:flex; justify-content:space-between; margin-top:15px; text-align:center;">
                    <div><div style="font-size:12px; color:{TEXT_SUB}">Thực tế 2025</div><div style="font-size:22px; font-weight:bold">{fmt_num(v2025)}</div></div>
                    <div><div style="font-size:12px; color:{TEXT_SUB}">Mục tiêu 2026</div><div style="font-size:22px; font-weight:bold; color:{NEON_GREEN}">{fmt_num(final_total)}</div></div>
                    <div><div style="font-size:12px; color:{TEXT_SUB}">Tăng trưởng</div><div style="font-size:22px; font-weight:bold; color:#FFA726">+{growth_g:.1f}%</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Plot
        viz_df = pd.concat([
            hist_data[hist_data['ym'].dt.year==2025][['ym','sl']].assign(Type='Thực tế 2025'),
            pd.DataFrame({'ym': future_dates, 'sl': final_pred, 'Type': 'Mục tiêu 2026'})
        ])
        
        fig_fore = px.line(viz_df, x='ym', y='sl', color='Type', markers=True, 
                           color_discrete_map={'Thực tế 2025': '#757575', 'Mục tiêu 2026': NEON_GREEN})
        fig_fore.update_traces(line=dict(width=3))
        st.plotly_chart(polish_chart(fig_fore), width="stretch")

# --- TAB 3: SẢN PHẨM ---
with tab3:
    c3_1, c3_2 = st.columns([2, 1])
    with c3_1:
        st.subheader("Cơ cấu Sản phẩm")
        color_data = df.groupby(['nhom_mau', 'mau_son'])['sl'].sum().reset_index()
        color_data = color_data[color_data['sl'] > color_data['sl'].sum()*0.005]
        fig_sun = px.sunburst(color_data, path=['nhom_mau', 'mau_son'], values='sl', 
                              color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(polish_chart(fig_sun), width="stretch")
        
    with c3_2:
        st.subheader("Top Mã Hàng")
        top_sku = df.groupby('ma_hang')['sl'].sum().nlargest(10).sort_values(ascending=True).reset_index()
        fig_sku = px.bar(top_sku, x='sl', y='ma_hang', orientation='h', text_auto='.2s', color_discrete_sequence=[PRIMARY])
        st.plotly_chart(polish_chart(fig_sku), width="stretch")

# --- TAB 4: MÀU SẮC ---
with tab4:
    st.subheader("Bản đồ nhiệt Mùa vụ & Màu sắc")
    heat = df.groupby(['mua', 'nhom_mau'])['sl'].sum().reset_index()
    heat['share'] = heat['sl'] / heat.groupby('mua')['sl'].transform('sum')
    pivot = heat.pivot(index='mua', columns='nhom_mau', values='share').fillna(0).reindex(['Xuân', 'Hè', 'Thu', 'Đông'])
    fig_heat = px.imshow(pivot, text_auto='.0%', aspect="auto", color_continuous_scale='Greens', origin='upper')
    st.plotly_chart(polish_chart(fig_heat), width="stretch")

# --- TAB 5: KHÁCH HÀNG ---
with tab5:
    c5_1, c5_2 = st.columns([2, 1])
    with c5_1:
        st.subheader("Biểu đồ Pareto")
        pareto = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
        pareto['cum'] = pareto['sl'].cumsum() / pareto['sl'].sum() * 100
        
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=pareto.head(20)['khach_hang'], y=pareto.head(20)['sl'], name='Sản lượng', marker_color=PRIMARY))
        fig_p.add_trace(go.Scatter(x=pareto.head(20)['khach_hang'], y=pareto.head(20)['cum'], name='% Tích lũy', yaxis='y2', line=dict(color=NEON_RED)))
        fig_p.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110]), showlegend=False)
        st.plotly_chart(polish_chart(fig_p), width="stretch")
        
    with c5_2:
        st.subheader("Tăng trưởng Khách hàng")
        curr = df['year'].max()
        prev = curr - 1
        v_c = df[df['year']==curr].groupby('khach_hang')['sl'].sum()
        v_p = df[df['year']==prev].groupby('khach_hang')['sl'].sum()
        growth = ((v_c - v_p)/v_p*100).fillna(0).sort_values(ascending=False).reset_index()
        growth.columns = ['Khách Hàng', '% Tăng Trưởng']
        render_table(growth.head(15), height=400)

# --- TAB 6: DỮ LIỆU ---
with tab6:
    st.subheader("Dữ liệu chi tiết")
    grid_df = df.groupby(['ma_hang', 'khach_hang', 'mau_son', 'year']).agg(Tong_SL=('sl', 'sum')).reset_index().sort_values('Tong_SL', ascending=False)
    render_table(grid_df, height=600)

st.markdown("---")
st.caption(f"© 2026 Mộc Phát Furniture | Analytics Suite | Updated: {datetime.now().strftime('%d/%m/%Y')}")
