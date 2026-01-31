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
# 1. CẤU HÌNH GIAO DIỆN (DARK MODE PERFECT)
# ==========================================
st.set_page_config(page_title="Mộc Phát Analytics Pro", layout="wide", page_icon="🌲")

# Bảng màu Dark Mode Chuẩn
PRIMARY = "#066839"    # Xanh Mộc Phát
ACCENT  = "#66BB6A"    # Xanh lá sáng (Neon nhẹ)
BG_COLOR = "#0E1117"   # Nền đen sâu
CARD_BG = "#1E1E1E"    # Nền card xám đen
TEXT_MAIN = "#FAFAFA"  # Trắng
TEXT_SUB = "#B0BEC5"   # Xám xanh nhạt
GRID_COLOR = "#2A2A2A" # Màu đường kẻ lưới tối

def get_base64_logo(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

def fmt_num(n):
    return f"{n:,.0f}"

# --- HÀM LÀM ĐẸP BIỂU ĐỒ (QUAN TRỌNG) ---
def polish_chart(fig):
    """Hàm này xóa nền trắng, bo tròn viền và làm mềm biểu đồ"""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', # Nền trong suốt
        plot_bgcolor='rgba(0,0,0,0)',  # Nền vùng vẽ trong suốt
        font=dict(color=TEXT_SUB, family="Segoe UI"),
        margin=dict(t=40, b=20, l=20, r=20),
        hovermode="x unified",
        # Hiệu ứng bo tròn cột (Fake bằng marker line)
        barcornerradius=0 # Plotly chưa hỗ trợ border-radius trực tiếp cho Bar, ta xử lý bằng layout
    )
    # Tinh chỉnh trục
    fig.update_xaxes(showgrid=False, linecolor='#333')
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig

# --- CSS CAO CẤP ---
st.markdown(f"""
<style>
    /* Nền chính */
    .stApp {{ background-color: {BG_COLOR}; }}
    
    /* Chữ */
    h1, h2, h3, h4 {{ color: {TEXT_MAIN} !important; font-family: 'Segoe UI', sans-serif; }}
    .stMarkdown p {{ color: {TEXT_SUB} !important; }}
    
    /* Header Sticky Bo Tròn */
    .header-sticky {{
        position: sticky; top: 10px; z-index: 999;
        background: {CARD_BG}; 
        border-bottom: 2px solid {PRIMARY};
        padding: 15px 20px; 
        margin: -50px 0px 20px 0px;
        border-radius: 15px; /* Bo góc header */
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }}
    .app-title {{ font-size: 24px; font-weight: 800; color: {ACCENT}; margin: 0; }}
    
    /* KPI Cards Bo Tròn & Nổi */
    .kpi-card {{
        background: {CARD_BG}; 
        border-radius: 15px; /* Bo tròn card */
        padding: 20px;
        border-left: 5px solid {PRIMARY};
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }}
    .kpi-card:hover {{ transform: translateY(-5px); box-shadow: 0 6px 15px rgba(6, 104, 57, 0.3); }}
    .kpi-val {{ font-size: 28px; font-weight: 800; color: {TEXT_MAIN}; }}
    .kpi-lbl {{ font-size: 13px; text-transform: uppercase; color: {TEXT_SUB}; letter-spacing: 1px; }}
    
    /* Insight Box */
    .insight-box {{
        background-color: rgba(6, 104, 57, 0.15); 
        border: 1px solid {PRIMARY};
        padding: 15px; border-radius: 12px; margin-bottom: 20px;
    }}
    
    /* Forecast Box */
    .forecast-box {{
        background: linear-gradient(135deg, rgba(255, 167, 38, 0.1), rgba(0,0,0,0));
        border: 1px solid #FFA726;
        padding: 15px; border-radius: 12px; margin-bottom: 20px;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{ 
        background-color: {CARD_BG}; color: {TEXT_SUB}; border-radius: 8px; border: none;
    }}
    .stTabs [aria-selected="true"] {{ background-color: {PRIMARY}; color: white; font-weight: bold; }}
    
    /* Container cho biểu đồ để tạo hiệu ứng bo tròn nền */
    .chart-container {{
        background-color: {CARD_BG};
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }}
</style>
""", unsafe_allow_html=True)

# Header
logo_b64 = get_base64_logo("mocphat_logo.png")
logo_img = f'<img src="data:image/png;base64,{logo_b64}" height="50">' if logo_b64 else "🌲"
st.markdown(f"""
<div class="header-sticky">
    <div style="display:flex; gap:15px; align-items:center;">
        {logo_img}
        <div>
            <div class="app-title">MỘC PHÁT INTELLIGENCE</div>
            <div style="font-size:13px; color:{TEXT_SUB};">Dark Mode Analytics v5.0</div>
        </div>
    </div>
    <div style="text-align:right;">
        <span style="font-weight:bold; color:{ACCENT}; font-size:14px;">Master 2023-2025</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU
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
st.subheader("🚀 HIỆU QUẢ KINH DOANH (YoY)")
vol_by_year = df.groupby('year')['sl'].sum()
v24 = vol_by_year.get(2024, 0)
v23 = vol_by_year.get(2023, 0)
g24 = ((v24 - v23) / v23 * 100) if v23 > 0 else 0

c1, c2, c3, c4 = st.columns(4)

def kpi_card(col, year_label, val, growth_val, compare_label="so với năm trước"):
    color_class = "#66BB6A" if growth_val >= 0 else "#EF5350"
    icon = "▲" if growth_val >= 0 else "▼"
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-lbl">{year_label}</div>
        <div class="kpi-val">{fmt_num(val)}</div>
        <div class="kpi-sub" style="color: {color_class}">
            {icon} {abs(growth_val):.1f}% <span style="color:#A0A0A0; font-weight:normal;">{compare_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

kpi_card(c1, "SẢN LƯỢNG 2023", v23, 0, "(Năm gốc)")
kpi_card(c2, "SẢN LƯỢNG 2024", v24, g24, "vs 2023")
kpi_card(c3, "SẢN LƯỢNG 2025", vol_by_year.get(2025,0), 0, "(Hiện tại)")
kpi_card(c4, "KHÁCH HÀNG ACTIVE", df['khach_hang'].nunique(), 0, "Đối tác")

st.markdown("---")

# ==========================================
# 5. TABS PHÂN TÍCH
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 TỔNG QUAN", "🎯 KẾ HOẠCH 2026", "🎨 SỨC KHỎE SP", "🌡️ MÙA VỤ", "⚖️ KHÁCH HÀNG", "📋 DỮ LIỆU"
])

# --- TAB 1: TỔNG QUAN ---
with tab1:
    c1_left, c1_right = st.columns([3, 1])
    with c1_left:
        st.subheader("📈 Xu hướng & Phát hiện Bất thường")
        ts_data = df.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
        
        fig = go.Figure()
        # Area Chart với hiệu ứng gradient nhẹ (giả lập bằng fill)
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['sl'], mode='lines+markers', name='Thực tế', 
                                 line=dict(color=ACCENT, width=3, shape='spline'), # shape='spline' làm đường cong mềm hơn
                                 fill='tozeroy', fillcolor='rgba(102, 187, 106, 0.1)')) 
        
        ts_data['ma3'] = ts_data['sl'].rolling(window=3).mean()
        fig.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['ma3'], mode='lines', name='TB 3 tháng', line=dict(color='#FFA726', dash='dot')))
        
        # Anomaly
        std = ts_data['sl'].rolling(window=3).std()
        upper = ts_data['ma3'] + (1.8 * std)
        anomalies = ts_data[ts_data['sl'] > upper]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies['ym'], y=anomalies['sl'], mode='markers', name='Đột biến', marker=dict(color='#EF5350', size=12, symbol='star')))
            
        st.plotly_chart(polish_chart(fig), use_container_width=True)

    with c1_right:
        last_m = ts_data.iloc[-1]
        prev_m = ts_data.iloc[-2] if len(ts_data) > 1 else last_m
        mom = ((last_m['sl'] - prev_m['sl'])/prev_m['sl']*100) if prev_m['sl']>0 else 0
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">🤖 AI Phân tích nhanh:</div>
            <ul style="margin:0; padding-left:20px; font-size:14px; color: {TEXT_MAIN}">
                <li>Tháng <b>{last_m['ym'].strftime('%m/%Y')}</b>: <b>{fmt_num(last_m['sl'])}</b> SP.</li>
                <li>Biến động: <b style="color:{'#66BB6A' if mom>0 else '#EF5350'}">{mom:+.1f}%</b>.</li>
                <li>Phát hiện <b>{len(anomalies)}</b> điểm bất thường trong quá khứ.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2: KẾ HOẠCH 2026 ---
with tab2:
    st.subheader("🎯 Dự báo Kế hoạch 2026")
    col_input, col_info = st.columns([1, 2])
    with col_input:
        growth_target = st.slider("Mục tiêu Tăng trưởng (%)", 0, 100, 15, 5)
        growth_factor = 1 + (growth_target / 100)
    
    base_2025 = df_raw[df_raw['year'] == 2025].copy()
    if not base_2025.empty:
        sl_2025_total = base_2025['sl'].sum()
        sl_2026_target = sl_2025_total * growth_factor
        sl_increase = sl_2026_target - sl_2025_total
        
        with col_info:
            st.markdown(f"""
            <div class="forecast-box">
                <h4 style="margin:0; color:#FFA726">KỊCH BẢN +{growth_target}%</h4>
                <div style="display:flex; justify-content:space-between; margin-top:10px;">
                    <div><div style="font-size:12px; color:{TEXT_SUB}">2025 Base</div><div style="font-size:20px; font-weight:bold">{fmt_num(sl_2025_total)}</div></div>
                    <div><div style="font-size:12px; color:{TEXT_SUB}">2026 Target</div><div style="font-size:20px; font-weight:bold; color:{ACCENT}">{fmt_num(sl_2026_target)}</div></div>
                    <div><div style="font-size:12px; color:{TEXT_SUB}">Tăng thêm</div><div style="font-size:20px; font-weight:bold; color:#FFA726">+{fmt_num(sl_increase)}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        monthly_2025 = base_2025.groupby('month')['sl'].sum().reset_index()
        monthly_2026 = monthly_2025.copy()
        monthly_2026['sl'] = monthly_2026['sl'] * growth_factor
        monthly_2026['Type'] = 'Mục tiêu 2026'
        monthly_2025['Type'] = 'Thực tế 2025'
        combined_forecast = pd.concat([monthly_2025, monthly_2026])
        
        fig_forecast = px.line(combined_forecast, x='month', y='sl', color='Type', markers=True, 
                               color_discrete_map={'Thực tế 2025': '#757575', 'Mục tiêu 2026': ACCENT})
        fig_forecast.update_traces(line=dict(width=3))
        st.plotly_chart(polish_chart(fig_forecast), use_container_width=True)
        
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            color_2025 = base_2025.groupby('nhom_mau')['sl'].sum().reset_index()
            color_2025['sl_target'] = color_2025['sl'] * growth_factor
            fig_bar = px.bar(color_2025, x='sl_target', y='nhom_mau', orientation='h', text_auto='.2s', 
                             color_discrete_sequence=[ACCENT], title="Mục tiêu Màu Sắc")
            st.plotly_chart(polish_chart(fig_bar), use_container_width=True)
        with c_f2:
            cust_2025 = base_2025.groupby('khach_hang')['sl'].sum().nlargest(10).reset_index().sort_values('sl')
            cust_2025['sl_target'] = cust_2025['sl'] * growth_factor
            fig_bar2 = px.bar(cust_2025, x='sl_target', y='khach_hang', orientation='h', text_auto='.2s', 
                              color_discrete_sequence=['#FFA726'], title="Mục tiêu Khách Hàng")
            st.plotly_chart(polish_chart(fig_bar2), use_container_width=True)

# --- TAB 3: SỨC KHỎE SP ---
with tab3:
    st.subheader("🎨 Sunburst Chart")
    col_detail_1, col_detail_2 = st.columns([2, 1])
    
    with col_detail_1:
        color_data = df.groupby(['nhom_mau', 'mau_son'])['sl'].sum().reset_index()
        total_sl_color = color_data['sl'].sum()
        color_data = color_data[color_data['sl'] > (total_sl_color * 0.005)]
        
        fig_sun = px.sunburst(
            color_data, path=['nhom_mau', 'mau_son'], values='sl', color='nhom_mau',
            color_discrete_map={"NÂU/GỖ": "#8D6E63", "TRẮNG/KEM": "#FFF9C4", "ĐEN/TỐI": "#424242", "XÁM": "#90A4AE", "TỰ NHIÊN": "#FFCC80"}
        )
        st.plotly_chart(polish_chart(fig_sun), use_container_width=True)
        
    with col_detail_2:
        top_colors = df.groupby('mau_son')['sl'].sum().nlargest(10).sort_values(ascending=True).reset_index()
        fig_bar_col = px.bar(top_colors, x='sl', y='mau_son', orientation='h', text_auto='.2s', color='sl', color_continuous_scale='Greens')
        st.plotly_chart(polish_chart(fig_bar_col), use_container_width=True)

    c2_1, c2_2 = st.columns(2)
    with c2_1:
        top_sku = df.groupby('ma_hang')['sl'].sum().nlargest(10).sort_values(ascending=True).reset_index()
        fig_sku = px.bar(top_sku, x='sl', y='ma_hang', orientation='h', text_auto='.2s', color_discrete_sequence=[PRIMARY])
        st.plotly_chart(polish_chart(fig_sku), use_container_width=True)
    with c2_2:
        usb_trend = df.groupby(['year', 'is_usb_clean'])['sl'].sum().reset_index()
        fig_usb = px.bar(usb_trend, x='year', y='sl', color='is_usb_clean', barmode='group',
                         color_discrete_map={'Có USB': '#FFA726', 'Không USB': '#424242'})
        st.plotly_chart(polish_chart(fig_usb), use_container_width=True)

# --- TAB 4: MÙA VỤ ---
with tab4:
    st.subheader("🌡️ Heatmap Mùa vụ")
    heat_data = df.groupby(['mua', 'nhom_mau'])['sl'].sum().reset_index()
    heat_data['share'] = heat_data['sl'] / heat_data.groupby('mua')['sl'].transform('sum')
    pivot = heat_data.pivot(index='mua', columns='nhom_mau', values='share').fillna(0).reindex(['Xuân', 'Hè', 'Thu', 'Đông'])
    fig_heat = px.imshow(pivot, text_auto='.0%', aspect="auto", color_continuous_scale='Greens', origin='upper')
    st.plotly_chart(polish_chart(fig_heat), use_container_width=True)

# --- TAB 5: KHÁCH HÀNG ---
with tab5:
    c4_1, c4_2 = st.columns([2, 1])
    with c4_1:
        pareto = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
        pareto['cum'] = pareto['sl'].cumsum()
        pareto['perc'] = pareto['cum'] / pareto['sl'].sum() * 100
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=pareto['khach_hang'], y=pareto['sl'], name='Sản lượng', marker_color=PRIMARY))
        fig_p.add_trace(go.Scatter(x=pareto['khach_hang'], y=pareto['perc'], name='% Tích lũy', yaxis='y2', line=dict(color='#EF5350', width=2)))
        fig_p.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110]), showlegend=False)
        st.plotly_chart(polish_chart(fig_p), use_container_width=True)
    with c4_2:
        curr_y, prev_y = df['year'].max(), df['year'].max()-1
        v_c = df[df['year']==curr_y].groupby('khach_hang')['sl'].sum()
        v_p = df[df['year']==prev_y].groupby('khach_hang')['sl'].sum()
        growth = ((v_c - v_p)/v_p*100).fillna(0).sort_values(ascending=False)
        st.dataframe(growth.head(10).rename("% Growth"), height=400)

# --- TAB 6: DỮ LIỆU ---
with tab6:
    st.subheader("Tra cứu dữ liệu")
    grid_df = df.groupby(['ma_hang', 'khach_hang', 'mau_son', 'nhom_mau', 'year']).agg(Tong_SL=('sl', 'sum')).reset_index().sort_values('Tong_SL', ascending=False)
    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('multiple', use_checkbox=True)
    gb.configure_column("Tong_SL", type=["numericColumn", "numberColumnFilter"], precision=0)
    AgGrid(grid_df, gridOptions=gb.build(), height=600, theme='balham-dark')

st.markdown("---")
st.caption(f"© 2026 Mộc Phát Furniture | Dark Mode Perfect | Updated: {datetime.now().strftime('%d/%m/%Y')}")
