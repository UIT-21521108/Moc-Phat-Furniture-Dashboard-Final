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
# 1. CẤU HÌNH GIAO DIỆN (DEEP AURORA PRO)
# ==========================================
st.set_page_config(page_title="Mộc Phát Analytics Pro", layout="wide", page_icon="🌲")

# BẢNG MÀU NEON CHIẾN LƯỢC
NEON_GREEN = "#00E676"   # Tốt / Mẫu Cũ / Tăng trưởng
NEON_ORANGE = "#FFA726"  # Cảnh báo / Mẫu Mới
NEON_BLUE = "#2979FF"    # Thông tin / SKU
NEON_RED = "#FF5252"     # Nguy hiểm
BG_COLOR = "#050505"
TEXT_SUB = "#E0E0E0"
GRID_COLOR = "rgba(255, 255, 255, 0.1)"

# --- CSS VISUAL EFFECTS ---
st.markdown(f"""
<style>
    /* 1. NỀN DEEP AURORA */
    .stApp {{
        background-color: {BG_COLOR};
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(0, 230, 118, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(41, 121, 255, 0.05) 0%, transparent 40%);
        background-attachment: fixed;
    }}

    /* 2. HEADER */
    .header-container {{ text-align: center; padding: 30px 0; margin-bottom: 20px; }}
    .neon-title {{
        font-family: 'Segoe UI', sans-serif; font-weight: 900; font-size: 40px; color: #fff;
        text-transform: uppercase; letter-spacing: 2px;
        text-shadow: 0 0 15px rgba(0, 230, 118, 0.5);
    }}
    .sub-title {{ font-size: 14px; color: {TEXT_SUB}; letter-spacing: 3px; font-weight: 300; opacity: 0.8; }}

    /* 3. GLASS CARDS (KPI) */
    .glass-card {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }}
    .glass-card:hover {{ transform: translateY(-5px); border-color: {NEON_GREEN}; }}
    .kpi-lbl {{ font-size: 12px; color: {TEXT_SUB}; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }}
    .kpi-val {{ font-size: 32px; font-weight: 800; color: #fff; margin: 5px 0; text-shadow: 0 0 10px rgba(255,255,255,0.1); }}
    
    /* 4. STORYTELLING BOX */
    .story-box {{
        background: rgba(41, 121, 255, 0.05);
        border-left: 4px solid {NEON_BLUE};
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-style: italic;
        color: {TEXT_SUB};
    }}
    
    /* 5. TABS & AGGRID */
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{ background-color: rgba(255,255,255,0.02); border-radius: 8px; color: {TEXT_SUB}; border: 1px solid rgba(255,255,255,0.05); }}
    .stTabs [aria-selected="true"] {{ border: 1px solid {NEON_GREEN}; color: {NEON_GREEN}; background: rgba(0, 230, 118, 0.1); }}
    
    .chart-box {{
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 15px;
        height: 100%;
    }}

    /* AGGRID THEME OVERRIDE */
    .ag-theme-alpine-dark {
        --ag-background-color: transparent !important;
        --ag-header-background-color: rgba(255,255,255,0.05) !important;
        --ag-odd-row-background-color: rgba(255,255,255,0.02) !important;
        --ag-foreground-color: #E0E0E0 !important;
        --ag-border-color: rgba(255,255,255,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HÀM STYLE BIỂU ĐỒ ---
def polish_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_SUB, family="Segoe UI"),
        margin=dict(t=40, b=20, l=10, r=10),
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig

# --- HÀM RENDER AGGRID (ĐÃ BỔ SUNG LẠI) ---
def render_glass_aggrid(dataframe, height=400):
    gb = GridOptionsBuilder.from_dataframe(dataframe)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('multiple', use_checkbox=True)
    gb.configure_default_column(resizable=True, filterable=True, sortable=True)
    
    # Định dạng số
    for col in dataframe.select_dtypes(include=['number']).columns:
        gb.configure_column(col, type=["numericColumn", "numberColumnFilter"], precision=0)
        
    gridOptions = gb.build()
    
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    AgGrid(dataframe, gridOptions=gridOptions, height=height, theme='alpine-dark', enable_enterprise_modules=False)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (LOGIC THỰC TẾ 100%)
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    FILE_NAME = "Master_2023_2025_PRO_clean.xlsx"
    if not os.path.exists(FILE_NAME): return None, f"⚠️ Không tìm thấy file {FILE_NAME}"

    try:
        df = pd.read_excel(FILE_NAME, engine='openpyxl')
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Xử lý thời gian
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
        df = df[(df['year'] > 2020) & (df['month'].between(1, 12))]
        df['ym'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])
        df['sl'] = pd.to_numeric(df['sl'], errors='coerce').fillna(0)
        
        # Mùa vụ
        season_map = {12:'Đông', 1:'Đông', 2:'Đông', 3:'Xuân', 4:'Xuân', 5:'Xuân', 6:'Hè', 7:'Hè', 8:'Hè', 9:'Thu', 10:'Thu', 11:'Thu'}
        df['mua'] = df['month'].map(season_map)
        
        # --- LOGIC MẪU MỚI / CŨ ---
        first_seen = df.groupby('ma_hang')['year'].min().reset_index()
        first_seen.rename(columns={'year': 'first_year'}, inplace=True)
        df = df.merge(first_seen, on='ma_hang', how='left')
        df['loai_mau'] = np.where(df['year'] == df['first_year'], 'Mẫu Mới (New)', 'Mẫu Cũ (Repeat)')

        # Xử lý nhóm màu
        def categorize_color(v):
            v = str(v).upper()
            if any(x in v for x in ["BROWN", "NAU", "WALNUT", "COCOA"]): return "NÂU/GỖ"
            if any(x in v for x in ["WHITE", "TRANG", "CREAM", "IVORY"]): return "TRẮNG/KEM"
            if any(x in v for x in ["BLACK", "DEN", "CHARCOAL"]): return "ĐEN/TỐI"
            return "MÀU KHÁC"
        
        df['nhom_mau'] = df['mau_son'].apply(categorize_color) if 'mau_son' in df.columns else "MÀU KHÁC"
        df['is_usb_clean'] = df['is_usb'].astype(str).apply(lambda x: 'Có USB' if 'true' in x.lower() else 'Không USB') if 'is_usb' in df.columns else 'N/A'

        return df, None
    except Exception as e:
        return None, str(e)

df_raw, error = load_data()
if error: st.error(error); st.stop()

# ==========================================
# 3. HEADER
# ==========================================
logo_b64 = None
if os.path.exists("mocphat_logo.png"):
    with open("mocphat_logo.png", "rb") as f: logo_b64 = base64.b64encode(f.read()).decode()
logo_img = f'<img src="data:image/png;base64,{logo_b64}" height="80" class="glow-logo">' if logo_b64 else '<span style="font-size:70px">🌲</span>'

st.markdown(f"""
<div class="header-container">
    {logo_img}
    <div class="neon-title">MỘC PHÁT INTELLIGENCE</div>
    <div class="sub-title">STRATEGIC DATA STORYTELLING HUB</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🎯 BỘ LỌC")
years = sorted(df_raw['year'].unique(), reverse=True)
sel_years = st.sidebar.multiselect("Năm Phân Tích", years, default=years[:1]) 
df = df_raw[df_raw['year'].isin(sel_years)] if sel_years else df_raw
if df.empty: st.warning("Không có dữ liệu!"); st.stop()

# ==========================================
# 4. KPI CARDS (TỔNG QUAN)
# ==========================================
st.subheader("🚀 Chỉ Số Sức Khỏe Doanh Nghiệp")

curr_year = df['year'].max()
vol_curr = df['sl'].sum()
try:
    vol_prev = df_raw[df_raw['year'] == curr_year - 1]['sl'].sum()
    growth = ((vol_curr - vol_prev) / vol_prev * 100) if vol_prev > 0 else 0
except: growth = 0

mix_stats = df.groupby('loai_mau')['sl'].sum()
total_mix = mix_stats.sum()
new_ratio = (mix_stats.get('Mẫu Mới (New)', 0) / total_mix * 100) if total_mix > 0 else 0

c1, c2, c3, c4 = st.columns(4)

def kpi_card(col, lbl, val, sub_val, sub_lbl, type="neutral"):
    color = NEON_GREEN if type == "good" else (NEON_RED if type == "bad" else NEON_BLUE)
    icon = "▲" if sub_val >= 0 else "▼"
    col.markdown(f"""
    <div class="glass-card" style="border-top: 3px solid {color}">
        <div class="kpi-lbl">{lbl}</div>
        <div class="kpi-val">{val}</div>
        <div style="font-size:13px; color:{color}; font-weight:bold;">
            {icon} {abs(sub_val):.1f}% <span style="color:#aaa; font-weight:normal">{sub_lbl}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

kpi_card(c1, f"TỔNG SẢN LƯỢNG {curr_year}", f"{vol_curr:,.0f}", growth, "vs Năm trước", "good" if growth > 0 else "bad")
kpi_card(c2, "SỐ LƯỢNG MÃ HÀNG (SKU)", f"{df['ma_hang'].nunique():,.0f}", 0, "Đang hoạt động", "neutral")
status_mix = "good" if new_ratio <= 35 else "bad" 
kpi_card(c3, "TỶ LỆ MẪU MỚI (R&D)", f"{new_ratio:.1f}%", new_ratio - 30, "Mục tiêu < 30%", status_mix)
kpi_card(c4, "SỐ LƯỢNG KHÁCH HÀNG", f"{df['khach_hang'].nunique()}", 0, "Đối tác", "neutral")

st.markdown("---")

# ==========================================
# 5. CÂU CHUYỆN DỮ LIỆU (TABS)
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🛡️ 1. LA BÀN CHIẾN LƯỢC", 
    "🚧 2. SỨC KHỎE SẢN XUẤT", 
    "🌊 3. MÙA VỤ & MÀU SẮC", 
    "⚖️ 4. KHÁCH HÀNG & RỦI RO",
    "📋 5. DỮ LIỆU CHI TIẾT"
])

# --- TAB 1: CHIẾN LƯỢC 70/30 ---
with tab1:
    st.markdown(f"""
    <div class="story-box">
        <b>🔭 Câu chuyện Chiến lược:</b> Để đảm bảo mục tiêu tăng trưởng bền vững và tránh lặp lại khủng hoảng "vỡ trận" 2023, 
        chúng ta cần kiểm soát chặt chẽ tỷ lệ <b>Mẫu Mới (New SKU)</b>. 
        Dữ liệu dưới đây phân loại chính xác các mã hàng thành 'Mới' (lần đầu xuất hiện) và 'Cũ' (lặp lại) để giám sát tỷ lệ vàng 70/30.
    </div>
    """, unsafe_allow_html=True)

    c1_1, c1_2 = st.columns([2, 1])
    
    with c1_1:
        mix_trend = df.groupby(['month', 'loai_mau'])['sl'].sum().reset_index()
        fig_mix = px.bar(mix_trend, x='month', y='sl', color='loai_mau', 
                         title="Cơ cấu Sản xuất: Mẫu Mới vs Mẫu Cũ theo Tháng",
                         color_discrete_map={'Mẫu Cũ (Repeat)': NEON_GREEN, 'Mẫu Mới (New)': NEON_ORANGE},
                         barmode='stack')
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(polish_chart(fig_mix), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c1_2:
        grp_mix = df.groupby('loai_mau')['sl'].sum().reset_index()
        fig_donut = px.pie(grp_mix, values='sl', names='loai_mau', hole=0.6, 
                           title=f"Tỷ trọng Năm {curr_year}",
                           color='loai_mau',
                           color_discrete_map={'Mẫu Cũ (Repeat)': NEON_GREEN, 'Mẫu Mới (New)': NEON_ORANGE})
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(polish_chart(fig_donut), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: SỨC KHỎE SẢN XUẤT ---
with tab2:
    st.markdown(f"""
    <div class="story-box">
        <b>⚙️ Phân tích Điểm nghẽn:</b> Thay vì đoán mò, ta dùng chỉ số <b>Sản lượng trung bình trên mỗi SKU</b> để đo lường hiệu suất.
        Nếu số lượng SKU (cột xanh) tăng mà Sản lượng TB (đường vàng) giảm -> Dấu hiệu của sự "vụn vặt hóa", gây giảm năng suất Xưởng 1.
    </div>
    """, unsafe_allow_html=True)
    
    frag_data = df.groupby('month').agg(
        Total_Vol=('sl', 'sum'),
        SKU_Count=('ma_hang', 'nunique')
    ).reset_index()
    frag_data['Avg_Vol_SKU'] = frag_data['Total_Vol'] / frag_data['SKU_Count']
    
    c2_1, c2_2 = st.columns([2, 1])
    
    with c2_1:
        fig_eff = go.Figure()
        fig_eff.add_trace(go.Bar(x=frag_data['month'], y=frag_data['SKU_Count'], name='Số lượng SKU (Mã hàng)', 
                                 marker_color=NEON_BLUE, opacity=0.6))
        fig_eff.add_trace(go.Scatter(x=frag_data['month'], y=frag_data['Avg_Vol_SKU'], name='Sản lượng TB / SKU',
                                     mode='lines+markers', line=dict(color='#FFD740', width=3), yaxis='y2'))
        
        fig_eff.update_layout(
            title="Biểu đồ Hiệu suất Dòng hàng (Fragmentation Analysis)",
            yaxis=dict(title="Số lượng SKU", showgrid=False),
            yaxis2=dict(title="Sản lượng TB/SKU", overlaying='y', side='right', showgrid=True),
            legend=dict(x=0, y=1.1, orientation='h')
        )
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(polish_chart(fig_eff), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2_2:
        sku_stats = df.groupby('ma_hang')['sl'].sum().reset_index()
        low_limit = sku_stats['sl'].quantile(0.2)
        long_tail = sku_stats[sku_stats['sl'] <= low_limit]
        
        st.markdown(f"""
        <div class="chart-box" style="display:flex; flex-direction:column; justify-content:center; text-align:center">
            <h4 style="color:{NEON_ORANGE}">⚠️ Cảnh báo "Đuôi dài"</h4>
            <div style="font-size:40px; font-weight:bold; color:#fff">{len(long_tail)}</div>
            <div style="color:{TEXT_SUB}">Mã hàng kém hiệu quả<br>(Sản lượng < {low_limit:,.0f})</div>
            <hr style="border-color:rgba(255,255,255,0.1); width:50%">
            <p style="font-size:12px; color:#888; margin-top:10px">
                Nhóm này chiếm 20% danh mục nhưng đóng góp sản lượng không đáng kể. 
                Cần xem xét loại bỏ hoặc gộp đơn.
            </p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: MÙA VỤ & MÀU SẮC ---
with tab3:
    col_heat, col_trend = st.columns([1, 1])
    
    with col_heat:
        st.subheader("🌡️ Nhịp đập Mùa vụ (Heatmap)")
        heat_data = df.groupby(['month', 'year'])['sl'].sum().reset_index()
        heat_pivot = heat_data.pivot(index='month', columns='year', values='sl').fillna(0)
        fig_heat = px.imshow(heat_pivot, aspect="auto", color_continuous_scale='Greens', origin='upper')
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(polish_chart(fig_heat), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_trend:
        st.subheader("🎨 Xu hướng Màu sắc")
        color_trend = df.groupby(['month', 'nhom_mau'])['sl'].sum().reset_index()
        fig_color = px.area(color_trend, x='month', y='sl', color='nhom_mau', 
                            color_discrete_sequence=px.colors.qualitative.Pastel)
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(polish_chart(fig_color), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 4: KHÁCH HÀNG & RỦI RO ---
with tab4:
    st.markdown(f"""
    <div class="story-box">
        <b>⚖️ Quản trị Rủi ro:</b> Sử dụng nguyên tắc Pareto (80/20) để xác định mức độ phụ thuộc vào các khách hàng lớn (Key Accounts).
        Đồng thời theo dõi Top khách hàng tăng trưởng để tìm kiếm cơ hội mới.
    </div>
    """, unsafe_allow_html=True)
    
    c4_1, c4_2 = st.columns([2, 1])
    with c4_1:
        cust_stats = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
        cust_stats['cum_perc'] = cust_stats['sl'].cumsum() / cust_stats['sl'].sum() * 100
        
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(x=cust_stats.head(10)['khach_hang'], y=cust_stats.head(10)['sl'], name='Sản lượng', marker_color=NEON_GREEN))
        fig_pareto.add_trace(go.Scatter(x=cust_stats.head(10)['khach_hang'], y=cust_stats.head(10)['cum_perc'], name='% Tích lũy', yaxis='y2', line=dict(color=NEON_RED, width=2)))
        fig_pareto.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110], showgrid=False))
        
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(polish_chart(fig_pareto), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c4_2:
        st.subheader("Top Tăng Trưởng (YoY)")
        curr_y, prev_y = df['year'].max(), df['year'].max()-1
        v_c = df[df['year']==curr_y].groupby('khach_hang')['sl'].sum()
        v_p = df[df['year']==prev_y].groupby('khach_hang')['sl'].sum()
        growth_cust = ((v_c - v_p)/v_p*100).fillna(0).sort_values(ascending=False).reset_index()
        growth_cust.columns = ['Khách Hàng', '% Tăng']
        
        # SỬA LỖI: Gọi hàm render_glass_aggrid đã được định nghĩa
        render_glass_aggrid(growth_cust.head(10), height=400)

# --- TAB 5: DỮ LIỆU ---
with tab5:
    st.subheader("Tra cứu dữ liệu chi tiết")
    grid_df = df.groupby(['ma_hang', 'khach_hang', 'mau_son', 'nhom_mau', 'year', 'loai_mau']).agg(Tong_SL=('sl', 'sum')).reset_index().sort_values('Tong_SL', ascending=False)
    render_glass_aggrid(grid_df, height=600)

st.markdown("---")
st.caption(f"© 2026 Mộc Phát Furniture | Deep Aurora Storytelling Edition | Updated: {datetime.now().strftime('%d/%m/%Y')}")
