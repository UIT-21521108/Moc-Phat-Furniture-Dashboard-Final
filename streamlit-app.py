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
# 1. CẤU HÌNH & GIAO DIỆN
# ==========================================
st.set_page_config(page_title="Mộc Phát Analytics Pro", layout="wide", page_icon="🌲")

PRIMARY = "#066839"    # Xanh Mộc Phát
ACCENT  = "#1B7D4F"
BG_COLOR = "#F4F6F9"
WARNING = "#FF8C00"
DANGER = "#D32F2F"
SUCCESS = "#2E7D32"

# --- BẢNG MÀU CHI TIẾT (Logic tham khảo từ Copilot + Dữ liệu thực) ---
COLOR_MAP = {
    "NÂU/GỖ": ["#8B5A2B", "#A0522D", "#D2691E", "#CD853F"],
    "TRẮNG/KEM": ["#F5F5F5", "#F0E68C", "#FAEBD7", "#FFFFFF"],
    "ĐEN/TỐI": ["#000000", "#2F4F4F", "#1C1C1C"],
    "XÁM": ["#808080", "#A9A9A9", "#D3D3D3"],
    "TỰ NHIÊN": ["#DEB887", "#F4A460", "#DAA520"],
    "KHÁC": ["#B0C4DE", "#E6E6FA"]
}

def get_base64_logo(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

def fmt_num(n):
    return f"{n:,.0f}"

st.markdown(f"""
<style>
    .main {{ background-color: {BG_COLOR}; }}
    h1, h2, h3 {{ font-family: 'Segoe UI', sans-serif; color: #333; }}
    
    .header-sticky {{
        position: sticky; top: 0; z-index: 999;
        background: white; border-bottom: 3px solid {PRIMARY};
        padding: 12px 20px; margin: -60px -50px 20px -50px;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}
    .app-title {{ font-size: 24px; font-weight: 800; color: {PRIMARY}; margin: 0; }}
    
    .kpi-card {{
        background: white; border-radius: 12px; padding: 15px;
        border-left: 5px solid {PRIMARY};
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }}
    .kpi-card:hover {{ transform: translateY(-3px); }}
    .kpi-val {{ font-size: 26px; font-weight: 800; color: #2C3E50; }}
    .kpi-lbl {{ font-size: 13px; text-transform: uppercase; color: #7F8C8D; font-weight: 600; }}
    .kpi-sub {{ font-size: 13px; font-weight: 600; margin-top: 5px; }}
    .pos {{ color: {SUCCESS}; }} 
    .neg {{ color: {DANGER}; }}
    
    .insight-box {{
        background-color: #E8F5E9; border-left: 4px solid {PRIMARY};
        padding: 15px; border-radius: 5px; margin-bottom: 20px;
    }}
</style>
""", unsafe_allow_html=True)

logo_b64 = get_base64_logo("mocphat_logo.png")
logo_img = f'<img src="data:image/png;base64,{logo_b64}" height="45">' if logo_b64 else "🌲"
st.markdown(f"""
<div class="header-sticky">
    <div style="display:flex; gap:15px; align-items:center;">
        {logo_img}
        <div>
            <div class="app-title">MỘC PHÁT INTELLIGENCE</div>
            <div style="font-size:14px; color:#666;">Báo cáo Sản xuất & Kinh doanh (Detailed Analytics)</div>
        </div>
    </div>
    <div style="text-align:right;">
        <span style="font-weight:bold; color:{PRIMARY};">Dữ liệu Master 2023-2025</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (LOGIC CHI TIẾT MÀU)
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    FILE_NAME = "Master_2023_2025_PRO_clean.xlsx"
    if not os.path.exists(FILE_NAME): return None, f"⚠️ Không tìm thấy file {FILE_NAME}"

    try:
        df = pd.read_excel(FILE_NAME, engine='openpyxl')
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # 1. Thời gian
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
        df = df[(df['year'] > 2020) & (df['month'].between(1, 12))]
        df['ym'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])
        
        season_map = {12:'Đông', 1:'Đông', 2:'Đông', 3:'Xuân', 4:'Xuân', 5:'Xuân', 
                      6:'Hè', 7:'Hè', 8:'Hè', 9:'Thu', 10:'Thu', 11:'Thu'}
        df['mua'] = df['month'].map(season_map)
        
        # 2. Xử lý Text
        cols_text = ['khach_hang', 'ma_hang', 'mau_son', 'khu_vuc', 'dim', 'mo_ta']
        for c in cols_text:
            if c not in df.columns: df[c] = "Unknown"
            else: df[c] = df[c].fillna("Unknown").astype(str).str.upper() # Giữ nguyên tên màu gốc (mau_son) để chi tiết
            
        df['sl'] = pd.to_numeric(df['sl'], errors='coerce').fillna(0)

        # 3. LOGIC MÀU CHI TIẾT (Sửa lại logic gom nhóm để không bị mất chi tiết)
        def categorize_detailed_color(v):
            v = v.strip()
            # Trả về TUPLE (Nhóm Chính, Nhóm Chi tiết hơn)
            if any(x in v for x in ["BROWN", "COCOA", "BRONZE", "UMBER", "NAU", "WALNUT", "ESPRESSO"]): 
                return "NÂU/GỖ"
            if any(x in v for x in ["WHITE", "CREAM", "IVORY", "TRANG", "OFF WHITE", "WASH"]): 
                return "TRẮNG/KEM"
            if any(x in v for x in ["BLACK", "DEN", "CHARCOAL", "EBONY"]): 
                return "ĐEN/TỐI"
            if any(x in v for x in ["GREY", "GRAY", "XAM", "SLATE"]): 
                return "XÁM"
            if any(x in v for x in ["NATURAL", "OAK", "PINE", "HONEY", "TU NHIEN"]): 
                return "TỰ NHIÊN"
            if any(x in v for x in ["BLUE", "NAVY"]): return "XANH DƯƠNG"
            if any(x in v for x in ["GREEN", "SAGE"]): return "XANH LÁ"
            return "MÀU KHÁC"
        
        # Tạo cột Nhóm Màu Lớn nhưng vẫn giữ cột 'mau_son' gốc cho chart chi tiết
        df['nhom_mau'] = df['mau_son'].apply(categorize_detailed_color)

        # 4. USB Trend
        if 'is_usb' in df.columns:
            df['is_usb_clean'] = df['is_usb'].astype(str).apply(lambda x: 'Có USB' if 'true' in x.lower() else 'Không USB')
        else:
            df['is_usb_clean'] = 'N/A'

        return df, None
    except Exception as e:
        return None, str(e)

df_raw, error = load_data()
if error: st.error(error); st.stop()

# ==========================================
# 3. SIDEBAR BỘ LỌC
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
st.subheader("🚀 HIỆU QUẢ KINH DOANH (YoY)")
vol_by_year = df.groupby('year')['sl'].sum()
v24 = vol_by_year.get(2024, 0)
v23 = vol_by_year.get(2023, 0)
g24 = ((v24 - v23) / v23 * 100) if v23 > 0 else 0

c1, c2, c3, c4 = st.columns(4)

def kpi_card(col, year_label, val, growth_val, compare_label="so với năm trước"):
    color_class = "pos" if growth_val >= 0 else "neg"
    icon = "▲" if growth_val >= 0 else "▼"
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-lbl">{year_label}</div>
        <div class="kpi-val">{fmt_num(val)}</div>
        <div class="kpi-sub {color_class}">
            {icon} {abs(growth_val):.1f}% <span style="color:#888; font-weight:normal;">{compare_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

kpi_card(c1, "SẢN LƯỢNG 2023", v23, 0, "(Năm gốc)")
kpi_card(c2, "SẢN LƯỢNG 2024", v24, g24, "vs 2023")
kpi_card(c3, "SẢN LƯỢNG 2025", vol_by_year.get(2025,0), 0, "(Đang cập nhật)")
total_cust = df['khach_hang'].nunique()
kpi_card(c4, "KHÁCH HÀNG ACTIVE", total_cust, 0, "Đối tác")

st.markdown("---")

# ==========================================
# 5. TABS PHÂN TÍCH
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 TỔNG QUAN & DỰ BÁO", 
    "🎨 SỨC KHỎE SẢN PHẨM (CHI TIẾT)", 
    "🌡️ MÙA VỤ", 
    "⚖️ KHÁCH HÀNG",
    "📋 DỮ LIỆU"
])

# --- TAB 1: TỔNG QUAN & DỰ BÁO ---
with tab1:
    c1_left, c1_right = st.columns([3, 1])
    with c1_left:
        st.subheader("📈 Xu hướng & Phát hiện Bất thường")
        ts_data = df.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['sl'], mode='lines+markers', name='Thực tế', line=dict(color=PRIMARY, width=3)))
        
        # Moving Average
        ts_data['ma3'] = ts_data['sl'].rolling(window=3).mean()
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['ma3'], mode='lines', name='Trung bình 3 tháng', line=dict(color='orange', dash='dot')))
        
        # Anomaly Detection
        std = ts_data['sl'].rolling(window=3).std()
        upper = ts_data['ma3'] + (1.8 * std)
        anomalies = ts_data[ts_data['sl'] > upper]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies['ym'], y=anomalies['sl'], mode='markers', name='Đột biến', marker=dict(color=DANGER, size=12, symbol='star')))
            
        fig.update_layout(height=400, xaxis_title="Thời gian", yaxis_title="Sản lượng", template="plotly_white", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with c1_right:
        last_m = ts_data.iloc[-1]
        prev_m = ts_data.iloc[-2] if len(ts_data) > 1 else last_m
        mom = ((last_m['sl'] - prev_m['sl'])/prev_m['sl']*100) if prev_m['sl']>0 else 0
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">🤖 AI Phân tích nhanh:</div>
            <ul style="margin:0; padding-left:20px; font-size:14px;">
                <li>Tháng <b>{last_m['ym'].strftime('%m/%Y')}</b>: <b>{fmt_num(last_m['sl'])}</b> SP.</li>
                <li>Biến động tháng: <b style="color:{'green' if mom>0 else 'red'}">{mom:+.1f}%</b>.</li>
                <li>Phát hiện <b>{len(anomalies)}</b> điểm bất thường cần lưu ý về năng lực SX.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2: SỨC KHỎE SẢN PHẨM (NÂNG CẤP CHI TIẾT MÀU) ---
with tab2:
    st.subheader("🎨 Phân tích Màu sắc Đa tầng (Sunburst Chart)")
    st.caption("Biểu đồ này giúp bạn nhìn từ Nhóm màu lớn (Vòng trong) -> Chi tiết từng Màu sơn cụ thể (Vòng ngoài).")
    
    col_detail_1, col_detail_2 = st.columns([2, 1])
    
    with col_detail_1:
        # Chuẩn bị dữ liệu cho Sunburst: Nhóm Màu -> Màu Sơn (Gốc)
        # Lọc bỏ các màu quá nhỏ để biểu đồ đỡ rối
        color_data = df.groupby(['nhom_mau', 'mau_son'])['sl'].sum().reset_index()
        total_sl_color = color_data['sl'].sum()
        color_data = color_data[color_data['sl'] > (total_sl_color * 0.005)] # Chỉ hiện màu > 0.5% tỷ trọng
        
        fig_sun = px.sunburst(
            color_data, 
            path=['nhom_mau', 'mau_son'], 
            values='sl',
            color='nhom_mau',
            color_discrete_map={
                "NÂU/GỖ": "#8B4513", "TRẮNG/KEM": "#F5F5DC", 
                "ĐEN/TỐI": "#2F4F4F", "XÁM": "#778899", 
                "TỰ NHIÊN": "#DEB887", "MÀU KHÁC": "#B0C4DE"
            },
            title="Cấu trúc Màu sắc Chi tiết"
        )
        fig_sun.update_traces(textinfo="label+percent entry")
        fig_sun.update_layout(height=500, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_sun, use_container_width=True)
        
    with col_detail_2:
        st.subheader("Top 10 Màu Sơn Cụ thể")
        # Top 10 Specific Colors (Không phải nhóm)
        top_colors = df.groupby('mau_son')['sl'].sum().nlargest(10).sort_values(ascending=True).reset_index()
        fig_bar_col = px.bar(top_colors, x='sl', y='mau_son', orientation='h', 
                             text_auto='.2s', color='sl', color_continuous_scale='Greens')
        fig_bar_col.update_layout(height=500, xaxis_title="Sản lượng", yaxis_title="")
        st.plotly_chart(fig_bar_col, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 10 SKU & Xu hướng USB")
    c2_1, c2_2 = st.columns(2)
    with c2_1:
        top_sku = df.groupby('ma_hang')['sl'].sum().nlargest(10).sort_values(ascending=True).reset_index()
        fig_sku = px.bar(top_sku, x='sl', y='ma_hang', orientation='h', text_auto='.2s', color_discrete_sequence=[PRIMARY])
        st.plotly_chart(fig_sku, use_container_width=True)
    with c2_2:
        usb_trend = df.groupby(['year', 'is_usb_clean'])['sl'].sum().reset_index()
        fig_usb = px.bar(usb_trend, x='year', y='sl', color='is_usb_clean', barmode='group',
                         color_discrete_map={'Có USB': WARNING, 'Không USB': '#E0E0E0'})
        st.plotly_chart(fig_usb, use_container_width=True)

# --- TAB 3: MÙA VỤ ---
with tab3:
    st.subheader("🌡️ Heatmap: Màu sắc theo Mùa")
    heat_data = df.groupby(['mua', 'nhom_mau'])['sl'].sum().reset_index()
    heat_data['share'] = heat_data['sl'] / heat_data.groupby('mua')['sl'].transform('sum')
    pivot = heat_data.pivot(index='mua', columns='nhom_mau', values='share').fillna(0)
    pivot = pivot.reindex(['Xuân', 'Hè', 'Thu', 'Đông'])
    fig_heat = px.imshow(pivot, text_auto='.0%', aspect="auto", color_continuous_scale='Greens', origin='upper')
    st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 4: KHÁCH HÀNG ---
with tab4:
    c4_1, c4_2 = st.columns([2, 1])
    with c4_1:
        st.subheader("Pareto 80/20")
        pareto = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
        pareto['cum'] = pareto['sl'].cumsum()
        pareto['perc'] = pareto['cum'] / pareto['sl'].sum() * 100
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=pareto['khach_hang'], y=pareto['sl'], name='Sản lượng', marker_color=PRIMARY))
        fig_p.add_trace(go.Scatter(x=pareto['khach_hang'], y=pareto['perc'], name='% Tích lũy', yaxis='y2', line=dict(color=DANGER, width=2)))
        fig_p.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110]), showlegend=False, height=450)
        st.plotly_chart(fig_p, use_container_width=True)
    with c4_2:
        st.subheader("Tăng trưởng Khách hàng (%)")
        curr_y, prev_y = df['year'].max(), df['year'].max()-1
        v_c = df[df['year']==curr_y].groupby('khach_hang')['sl'].sum()
        v_p = df[df['year']==prev_y].groupby('khach_hang')['sl'].sum()
        growth = ((v_c - v_p)/v_p*100).fillna(0).sort_values(ascending=False)
        st.dataframe(growth.head(10).rename("% Growth"), height=400)

# --- TAB 5: DỮ LIỆU ---
with tab5:
    st.subheader("Tra cứu dữ liệu chi tiết")
    grid_df = df.groupby(['ma_hang', 'khach_hang', 'mau_son', 'nhom_mau', 'year']).agg(
        Tong_SL=('sl', 'sum'),
        So_Don=('ym', 'count')
    ).reset_index().sort_values('Tong_SL', ascending=False)
    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('multiple', use_checkbox=True)
    gb.configure_column("Tong_SL", type=["numericColumn", "numberColumnFilter"], precision=0)
    gb.configure_column("ma_hang", pinned=True)
    AgGrid(grid_df, gridOptions=gb.build(), height=600, fit_columns_on_grid_load=False)

st.markdown("---")
st.caption(f"© 2026 Mộc Phát Furniture | Ultimate Edition | Updated: {datetime.now().strftime('%d/%m/%Y')}")
