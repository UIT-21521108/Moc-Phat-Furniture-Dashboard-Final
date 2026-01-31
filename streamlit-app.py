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
# 1. CẤU HÌNH GIAO DIỆN (PREMIUM NEON DARK - STABLE)
# ==========================================
st.set_page_config(page_title="Mộc Phát Analytics", layout="wide", page_icon="🌲")

# BẢNG MÀU NEON DARK
PRIMARY = "#066839"    
NEON_GREEN = "#00E676" 
ACCENT  = "#66BB6A"    
BG_COLOR = "#050505"   
CARD_BG = "#121212"    
TEXT_MAIN = "#E0E0E0"
TEXT_SUB = "#9E9E9E"
GRID_COLOR = "#2A2A2A"

# --- HÀM STYLE BIỂU ĐỒ ---
def polish_chart(fig):
    """Làm đẹp biểu đồ: Xóa nền trắng, chỉnh màu chữ"""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_SUB, family="sans-serif"),
        margin=dict(t=40, b=20, l=10, r=10),
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig

# --- CSS CAO CẤP (Đã sửa lỗi cú pháp f-string) ---
st.markdown(f"""
<style>
    /* 1. Nền & Chữ */
    .stApp {{ background-color: {BG_COLOR}; }}
    h1, h2, h3, h4 {{ color: {TEXT_MAIN} !important; }}
    .stMarkdown p, .stMarkdown li {{ color: {TEXT_SUB} !important; }}
    
    /* 2. Header Sticky */
    .header-sticky {{
        position: sticky; top: 0; z-index: 999;
        background: rgba(18, 18, 18, 0.95);
        border-bottom: 2px solid {PRIMARY};
        padding: 15px 25px; 
        margin-bottom: 20px;
        border-radius: 0 0 15px 15px;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }}
    .header-sticky:hover {{
        border-bottom: 2px solid {NEON_GREEN};
        box-shadow: 0 0 25px rgba(0, 230, 118, 0.2);
    }}

    /* 3. KPI Cards - Hiệu ứng Glow khi di chuột */
    .kpi-card {{
        background: {CARD_BG}; 
        border-radius: 16px;
        padding: 20px;
        border-left: 5px solid {PRIMARY};
        box-shadow: 0 4px 10px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-5px);
        border-left: 5px solid {NEON_GREEN};
        box-shadow: 0 0 20px rgba(0, 230, 118, 0.2);
    }}
    .kpi-val {{ font-size: 28px; font-weight: bold; color: {TEXT_MAIN}; }}
    .kpi-card:hover .kpi-val {{ color: {NEON_GREEN}; }}

    /* 4. Insight Box */
    .insight-box {{
        background: linear-gradient(135deg, rgba(6, 104, 57, 0.2), rgba(0,0,0,0)); 
        border: 1px solid {PRIMARY};
        padding: 15px; border-radius: 12px; margin-bottom: 20px;
    }}
    
    /* 5. AgGrid Dark Fix */
    .ag-theme-alpine-dark {{
        --ag-background-color: {CARD_BG} !important;
        --ag-header-background-color: #1A1A1A !important;
        --ag-odd-row-background-color: {CARD_BG} !important;
        --ag-foreground-color: {TEXT_SUB} !important;
        --ag-border-color: #333 !important;
    }}
    
    /* 6. Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{ background-color: {CARD_BG}; border-radius: 5px; }}
    .stTabs [aria-selected="true"] {{ background-color: {PRIMARY}; color: white; }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    FILE_NAME = "Master_2023_2025_PRO_clean.xlsx"
    
    if not os.path.exists(FILE_NAME):
        return None, f"⚠️ Không tìm thấy file '{FILE_NAME}'"
    
    try:
        df = pd.read_excel(FILE_NAME, engine='openpyxl')
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
        
        # Xử lý cột màu
        if 'mau_son' in df.columns:
            df['mau_son'] = df['mau_son'].fillna("Unknown").astype(str).str.upper()
            def get_group(v):
                if any(x in v for x in ["BROWN", "NAU", "WALNUT", "COCOA"]): return "NÂU/GỖ"
                if any(x in v for x in ["WHITE", "TRANG", "CREAM", "IVORY"]): return "TRẮNG/KEM"
                if any(x in v for x in ["BLACK", "DEN", "CHARCOAL"]): return "ĐEN/TỐI"
                if any(x in v for x in ["GREY", "XAM"]): return "XÁM"
                if any(x in v for x in ["NATURAL", "TU NHIEN", "OAK"]): return "TỰ NHIÊN"
                return "MÀU KHÁC"
            df['nhom_mau'] = df['mau_son'].apply(get_group)
        else:
            df['nhom_mau'] = "N/A"

        # Xử lý USB
        if 'is_usb' in df.columns:
            df['is_usb_clean'] = df['is_usb'].astype(str).apply(lambda x: 'Có USB' if 'true' in x.lower() else 'Không USB')
        else:
            df['is_usb_clean'] = 'N/A'
            
        return df, None

    except Exception as e:
        return None, f"Lỗi đọc file: {str(e)}"

# LOAD DỮ LIỆU
df_raw, error = load_data()

# LOGIC XỬ LÝ KHI LỖI (Fallback Data)
if error:
    st.error(error)
    # Tạo data giả để App không bị trắng trơn
    st.warning("Đang hiển thị dữ liệu mẫu (Demo Mode) do không đọc được file gốc.")
    dates = pd.date_range('2023-01-01', '2025-12-31', freq='M')
    data = []
    for d in dates:
        data.append({
            'year': d.year, 'month': d.month, 'ym': d,
            'khach_hang': np.random.choice(['HOMEGOODS', 'TJMAXX', 'MARSHALLS'], p=[0.5, 0.3, 0.2]),
            'ma_hang': f'SKU-{np.random.randint(100,999)}',
            'nhom_mau': np.random.choice(['NÂU/GỖ', 'TRẮNG/KEM', 'ĐEN/TỐI'], p=[0.6, 0.2, 0.2]),
            'mau_son': 'Sample Color',
            'mua': np.random.choice(['Xuân', 'Hè']),
            'is_usb_clean': 'Không USB',
            'sl': np.random.randint(100, 1000)
        })
    df_raw = pd.DataFrame(data)

# ==========================================
# 3. HEADER & SIDEBAR
# ==========================================
def get_base64_logo(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_base64_logo("mocphat_logo.png")
logo_html = f'<img src="data:image/png;base64,{logo_b64}" height="50">' if logo_b64 else "🌲"

st.markdown(f"""
<div class="header-sticky">
    <div style="display:flex; gap:15px; align-items:center;">
        {logo_html}
        <div>
            <h3 style="margin:0; color:{ACCENT}">MỘC PHÁT INTELLIGENCE</h3>
            <small style="color:{TEXT_SUB}">Real-time Manufacturing Analytics</small>
        </div>
    </div>
    <div style="text-align:right;">
        <span style="font-weight:bold; color:{ACCENT}; font-size:14px;">Master 2023-2025</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Filter
st.sidebar.markdown("### 🎯 BỘ LỌC")
years = sorted(df_raw['year'].unique(), reverse=True)
sel_years = st.sidebar.multiselect("Năm", years, default=years)

if 'khach_hang' in df_raw.columns:
    custs = sorted(df_raw['khach_hang'].unique())
    sel_cust = st.sidebar.multiselect("Khách Hàng", custs)
else:
    sel_cust = []

# Filter Logic
df = df_raw.copy()
if sel_years: df = df[df['year'].isin(sel_years)]
if sel_cust: df = df[df['khach_hang'].isin(sel_cust)]

if df.empty:
    st.warning("Không có dữ liệu phù hợp bộ lọc.")
    st.stop()

# ==========================================
# 4. KPI CARDS
# ==========================================
st.subheader("🚀 HIỆU QUẢ KINH DOANH")
vol_by_year = df.groupby('year')['sl'].sum()
v24 = vol_by_year.get(2024, 0)
v23 = vol_by_year.get(2023, 0)
g24 = ((v24 - v23) / v23 * 100) if v23 > 0 else 0

c1, c2, c3, c4 = st.columns(4)

def card(col, lbl, val, sub):
    color_sub = NEON_GREEN if "pc" in str(sub) or "+" in str(sub) else "#EF5350"
    col.markdown(f"""
    <div class="kpi-card">
        <div style="font-size:12px; color:#888; text-transform:uppercase">{lbl}</div>
        <div class="kpi-val">{val:,.0f}</div>
        <div style="color:{color_sub}; font-size:13px; font-weight:bold">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

card(c1, "SẢN LƯỢNG 2023", v23, "(Base Year)")
card(c2, "SẢN LƯỢNG 2024", v24, f"{g24:+.1f}% vs 23")
card(c3, "SẢN LƯỢNG 2025", vol_by_year.get(2025,0), "(Current)")
card(c4, "SỐ LƯỢNG KHÁCH", df['khach_hang'].nunique() if 'khach_hang' in df.columns else 0, "Active Partners")

st.markdown("---")

# ==========================================
# 5. TABS PHÂN TÍCH
# ==========================================
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📊 TỔNG QUAN", "🎯 KẾ HOẠCH 2026", "🎨 SỨC KHỎE SP", 
    "🌡️ MÙA VỤ", "⚖️ KHÁCH HÀNG", "📋 DỮ LIỆU"
])

def render_aggrid(dataframe, height=400):
    gb = GridOptionsBuilder.from_dataframe(dataframe)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('multiple', use_checkbox=True)
    gb.configure_default_column(resizable=True, filterable=True, sortable=True)
    for col in dataframe.select_dtypes(include=['number']).columns:
        gb.configure_column(col, type=["numericColumn", "numberColumnFilter"], precision=0)
    gridOptions = gb.build()
    AgGrid(dataframe, gridOptions=gridOptions, height=height, theme='alpine-dark', enable_enterprise_modules=False)

# --- TAB 1: TỔNG QUAN ---
with t1:
    c1_left, c1_right = st.columns([3, 1])
    with c1_left:
        ts_data = df.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
        fig = go.Figure()
        # Area Chart Neon
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['sl'], mode='lines+markers', name='Thực tế', 
                                 line=dict(color=NEON_GREEN, width=3, shape='spline'),
                                 fill='tozeroy', fillcolor='rgba(0, 230, 118, 0.1)')) 
        # Moving Avg
        ts_data['ma3'] = ts_data['sl'].rolling(window=3).mean()
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['ma3'], mode='lines', name='TB 3 tháng', 
                                 line=dict(color='#FFA726', dash='dot')))
        st.plotly_chart(polish_chart(fig), use_container_width=True)

    with c1_right:
        if not ts_data.empty:
            last_m = ts_data.iloc[-1]
            prev_m = ts_data.iloc[-2] if len(ts_data) > 1 else last_m
            mom = ((last_m['sl'] - prev_m['sl'])/prev_m['sl']*100) if prev_m['sl']>0 else 0
            st.markdown(f"""
            <div class="insight-box">
                <div style="color:{NEON_GREEN}; font-weight:bold; margin-bottom:10px">🤖 AI Insights:</div>
                <ul style="margin:0; padding-left:20px; font-size:14px; color: {TEXT_MAIN}">
                    <li>Tháng <b>{last_m['ym'].strftime('%m/%Y')}</b>: <b>{fmt_num(last_m['sl'])}</b> SP.</li>
                    <li>Biến động: <b style="color:{NEON_GREEN if mom>0 else '#EF5350'}">{mom:+.1f}%</b> so với tháng trước.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 2: KẾ HOẠCH 2026 ---
with t2:
    st.subheader("🎯 Kịch bản 2026")
    col_input, col_view = st.columns([1, 2])
    with col_input:
        growth = st.slider("Mục tiêu tăng trưởng (%)", 0, 100, 15, 5)
    
    with col_view:
        base_25 = df[df['year']==2025]['sl'].sum()
        if base_25 == 0: base_25 = v24 # Fallback
        target = base_25 * (1 + growth/100)
        
        st.markdown(f"""
        <div style="background:{rgba(255, 167, 38, 0.1)}; border:1px solid #FFA726; padding:15px; border-radius:10px; display:flex; justify-content:space-between">
            <div><small>2025 Base</small><br><b>{fmt_num(base_25)}</b></div>
            <div><small style="color:{NEON_GREEN}">2026 Target</small><br><b style="color:{NEON_GREEN}; font-size:20px">{fmt_num(target)}</b></div>
            <div><small>Tăng thêm</small><br><b>+{fmt_num(target - base_25)}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Chart so sánh
        d_chart = pd.DataFrame({'Năm': ['2025', '2026'], 'SL': [base_25, target]})
        fig_bar = px.bar(d_chart, x='Năm', y='SL', color='Năm', 
                         color_discrete_map={'2025': '#555', '2026': NEON_GREEN})
        st.plotly_chart(polish_chart(fig_bar), use_container_width=True)

# --- TAB 3: SỨC KHỎE SP ---
with t3:
    col_sun, col_sku = st.columns(2)
    with col_sun:
        st.caption("Cơ cấu Màu sắc")
        if 'nhom_mau' in df.columns:
            # Sunburst
            color_data = df.groupby(['nhom_mau', 'mau_son'])['sl'].sum().reset_index()
            # Lọc màu nhỏ để đỡ rối
            total_sl = color_data['sl'].sum()
            color_data = color_data[color_data['sl'] > total_sl*0.01] 
            
            fig_sun = px.sunburst(color_data, path=['nhom_mau', 'mau_son'], values='sl', 
                                  color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(polish_chart(fig_sun), use_container_width=True)
            
    with col_sku:
        st.caption("Top 10 SKU")
        if 'ma_hang' in df.columns:
            top_sku = df.groupby('ma_hang')['sl'].sum().nlargest(10).reset_index().sort_values('sl')
            fig_sku = px.bar(top_sku, x='sl', y='ma_hang', orientation='h')
            fig_sku.update_traces(marker_color=PRIMARY)
            st.plotly_chart(polish_chart(fig_sku), use_container_width=True)

# --- TAB 4: MÙA VỤ ---
with t4:
    st.subheader("Bản đồ nhiệt Mùa vụ")
    if 'mua' in df.columns and 'nhom_mau' in df.columns:
        hm = df.groupby(['mua', 'nhom_mau'])['sl'].sum().reset_index()
        hm_pivot = hm.pivot(index='mua', columns='nhom_mau', values='sl').fillna(0)
        # Sắp xếp mùa
        hm_pivot = hm_pivot.reindex(['Xuân', 'Hè', 'Thu', 'Đông'])
        
        fig_hm = px.imshow(hm_pivot, aspect="auto", color_continuous_scale='Greens', origin='upper')
        st.plotly_chart(polish_chart(fig_hm), use_container_width=True)

# --- TAB 5: KHÁCH HÀNG ---
with t5:
    c5_1, c5_2 = st.columns([2, 1])
    with c5_1:
        st.caption("Pareto Khách Hàng")
        pareto = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
        fig_p = px.bar(pareto, x='khach_hang', y='sl')
        fig_p.update_traces(marker_color=PRIMARY)
        st.plotly_chart(polish_chart(fig_p), use_container_width=True)
    with c5_2:
        st.caption("Chi tiết tăng trưởng")
        # Đơn giản hóa bảng tăng trưởng
        curr = df['year'].max()
        prev = curr - 1
        d_curr = df[df['year']==curr].groupby('khach_hang')['sl'].sum()
        d_prev = df[df['year']==prev].groupby('khach_hang')['sl'].sum()
        growth_df = ((d_curr - d_prev)/d_prev*100).fillna(0).sort_values(ascending=False).reset_index()
        growth_df.columns = ['Khách Hàng', '% Tăng']
        render_aggrid(growth_df.head(10), height=400)

# --- TAB 6: DỮ LIỆU ---
with t6:
    st.subheader("Dữ liệu chi tiết")
    # Group lại cho gọn
    grid_df = df.groupby(['ma_hang', 'khach_hang', 'mau_son', 'nhom_mau', 'year']).agg(SL=('sl', 'sum')).reset_index().sort_values('SL', ascending=False)
    render_aggrid(grid_df, height=600)

st.markdown("---")
st.caption(f"© 2026 Mộc Phát Analytics | Last Update: {datetime.now().strftime('%d/%m/%Y')}")
