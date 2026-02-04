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
# 1. CẤU HÌNH & GIAO DIỆN (DEEP AURORA)
# ==========================================
st.set_page_config(page_title="Mộc Phát Strategy Hub", layout="wide", page_icon="🌲")

# BẢNG MÀU CHIẾN LƯỢC
PRIMARY = "#00E676"    # Tăng trưởng / Tốt
WARNING = "#FFA726"    # Cảnh báo / Trung bình
DANGER  = "#FF5252"    # Nguy hiểm / Giảm
INFO    = "#2979FF"    # Thông tin
BG_DARK = "#050505"
TEXT_MAIN = "#FFFFFF"
TEXT_SUB = "#B0BEC5"

# --- CSS CAO CẤP ---
st.markdown(f"""
<style>
    /* Nền Aurora */
    .stApp {{
        background-color: {BG_DARK};
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(0, 230, 118, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(41, 121, 255, 0.05) 0%, transparent 40%);
        background-attachment: fixed;
    }}

    /* Header */
    .header-container {{ text-align: center; padding: 40px 0 20px 0; }}
    .neon-title {{
        font-family: 'Segoe UI', sans-serif; font-weight: 900; font-size: 40px; color: #fff;
        text-transform: uppercase; letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(0, 230, 118, 0.4);
    }}
    .sub-title {{ font-size: 14px; color: {TEXT_SUB}; letter-spacing: 4px; font-weight: 300; margin-top:5px; }}

    /* Report Card (Khung chứa Insight - Quan trọng) */
    .report-card {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        border-left: 4px solid {INFO}; /* Mặc định là Info */
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }}
    .report-title {{ font-weight: 700; font-size: 16px; color: {TEXT_MAIN}; display: flex; align-items: center; gap: 10px; }}
    .report-content {{ font-size: 14px; color: {TEXT_SUB}; line-height: 1.6; margin-top: 10px; text-align: justify; }}
    .highlight {{ color: {PRIMARY}; font-weight: bold; }}
    .warn {{ color: {WARNING}; font-weight: bold; }}
    .danger {{ color: {DANGER}; font-weight: bold; }}

    /* Glass Card cho Chart */
    .glass-box {{
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 15px;
        height: 100%;
    }}

    /* Tùy chỉnh Tab */
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{ background: rgba(255,255,255,0.03); border-radius: 8px; color: {TEXT_SUB}; }}
    .stTabs [aria-selected="true"] {{ background: rgba(0, 230, 118, 0.1); border: 1px solid {PRIMARY}; color: {PRIMARY}; }}
</style>
""", unsafe_allow_html=True)

# --- HÀM STYLE CHART ---
def polish_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_SUB, family="Segoe UI"),
        margin=dict(t=40, b=20, l=10, r=10),
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    return fig

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & LOGIC INSIGHT (CORE)
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
        df['sl'] = pd.to_numeric(df['sl'], errors='coerce').fillna(0)
        
        # Mùa vụ
        season_map = {12:'Đông', 1:'Đông', 2:'Đông', 3:'Xuân', 4:'Xuân', 5:'Xuân', 6:'Hè', 7:'Hè', 8:'Hè', 9:'Thu', 10:'Thu', 11:'Thu'}
        df['mua'] = df['month'].map(season_map)
        
        # Xử lý màu & USB (Giữ nguyên logic cũ)
        def get_color_group(v):
            v = str(v).upper()
            if any(x in v for x in ["BROWN", "NAU", "WALNUT"]): return "NÂU/GỖ"
            if any(x in v for x in ["WHITE", "TRANG", "CREAM"]): return "TRẮNG/KEM"
            if any(x in v for x in ["BLACK", "DEN"]): return "ĐEN/TỐI"
            if any(x in v for x in ["GREY", "XAM"]): return "XÁM"
            if any(x in v for x in ["NATURAL", "TU NHIEN"]): return "TỰ NHIÊN"
            return "MÀU KHÁC"
        df['nhom_mau'] = df['mau_son'].apply(get_color_group) if 'mau_son' in df.columns else "MÀU KHÁC"
        df['is_usb_clean'] = df['is_usb'].astype(str).apply(lambda x: 'Có USB' if 'true' in x.lower() else 'Không USB') if 'is_usb' in df.columns else 'N/A'
        
        return df, None
    except Exception as e: return None, str(e)

df_raw, error = load_data()
if error: st.error(error); st.stop()

# --- HÀM TẠO TEXT REPORT (QUAN TRỌNG) ---
def generate_insight_box(title, content, type="info"):
    colors = {"success": PRIMARY, "warning": WARNING, "danger": DANGER, "info": INFO}
    icon = {"success": "🚀", "warning": "⚠️", "danger": "🔥", "info": "💡"}
    border_color = colors.get(type, INFO)
    
    st.markdown(f"""
    <div class="report-card" style="border-left: 4px solid {border_color};">
        <div class="report-title" style="color:{border_color}">{icon[type]} {title}</div>
        <div class="report-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 3. HEADER & SIDEBAR
# ==========================================
logo_b64 = None
if os.path.exists("mocphat_logo.png"):
    with open("mocphat_logo.png", "rb") as f: logo_b64 = base64.b64encode(f.read()).decode()
logo_img = f'<img src="data:image/png;base64,{logo_b64}" height="70">' if logo_b64 else '🌲'

st.markdown(f"""
<div class="header-container">
    {logo_img}
    <div class="neon-title">MỘC PHÁT INTELLIGENCE</div>
    <div class="sub-title">BÁO CÁO CHIẾN LƯỢC SẢN XUẤT</div>
</div>
""", unsafe_allow_html=True)

years = sorted(df_raw['year'].unique(), reverse=True)
sel_years = st.sidebar.multiselect("Năm", years, default=years)
df = df_raw[df_raw['year'].isin(sel_years)] if sel_years else df_raw
if df.empty: st.warning("Chưa có dữ liệu."); st.stop()

# ==========================================
# 4. KPI SUMMARY (HIỆU SUẤT)
# ==========================================
st.subheader("1. Tổng quan Hiệu suất")
v_total = df['sl'].sum()
avg_month = df.groupby('ym')['sl'].sum().mean()
curr_year = df['year'].max()
v_curr = df[df['year']==curr_year]['sl'].sum()
v_prev = df[df['year']==curr_year-1]['sl'].sum()
growth = ((v_curr - v_prev)/v_prev*100) if v_prev > 0 else 0

col_kpi, col_insight = st.columns([3, 1])

with col_kpi:
    c1, c2, c3 = st.columns(3)
    def card(c, lbl, v, s):
        c.markdown(f"""<div class="glass-box" style="text-align:center">
            <div style="font-size:12px; color:#aaa">{lbl}</div>
            <div style="font-size:28px; font-weight:bold; color:#fff">{v:,.0f}</div>
            <div style="font-size:14px; color:{PRIMARY}">{s}</div>
        </div>""", unsafe_allow_html=True)
    
    card(c1, f"TỔNG SẢN LƯỢNG {curr_year}", v_curr, f"{growth:+.1f}% so với năm trước")
    card(c2, "TRUNG BÌNH THÁNG", avg_month, "Sản phẩm/tháng")
    card(c3, "SỐ LƯỢNG SKU", df['ma_hang'].nunique(), "Mã hàng đang chạy")

with col_insight:
    msg = f"Năm <b>{curr_year}</b> đang ghi nhận mức tăng trưởng <span class='{'highlight' if growth>0 else 'danger'}'>{growth:+.1f}%</span>. "
    msg += "Điều này cho thấy nhu cầu thị trường đang phục hồi tốt." if growth > 0 else "Cần rà soát lại nguyên nhân sụt giảm đơn hàng."
    generate_insight_box("Đánh giá Tăng trưởng", msg, "success" if growth > 0 else "danger")

st.markdown("---")

# ==========================================
# 5. PHÂN TÍCH CHUYÊN SÂU
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 NHỊP ĐẬP MÙA VỤ", "🎨 CHIẾN LƯỢC SẢN PHẨM", "⚖️ QUẢN TRỊ RỦI RO"])

# --- TAB 1: MÙA VỤ ---
with tab1:
    c_chart, c_text = st.columns([2, 1])
    with c_chart:
        # Biểu đồ Heatmap
        heat = df.groupby(['month', 'year'])['sl'].sum().reset_index()
        heat_pivot = heat.pivot(index='month', columns='year', values='sl').fillna(0)
        fig_h = px.imshow(heat_pivot, aspect="auto", color_continuous_scale='Greens', title="Bản đồ nhiệt Sản lượng theo Tháng")
        st.plotly_chart(polish_chart(fig_h), use_container_width=True)
    
    with c_text:
        # Tự động tìm tháng cao điểm
        avg_monthly = df.groupby('month')['sl'].mean()
        peak_month = avg_monthly.idxmax()
        low_month = avg_monthly.idxmin()
        peak_val = avg_monthly.max()
        
        insight_season = f"""
        Theo dữ liệu lịch sử, <b>Tháng {peak_month}</b> luôn là tháng đạt đỉnh sản lượng trung bình ({peak_val:,.0f} SP).<br><br>
        Trong khi đó, <b>Tháng {low_month}</b> thường là vùng trũng. <br><br>
        👉 <b>Khuyến nghị:</b> <br>
        - Lên kế hoạch nhập nguyên vật liệu từ <b>Tháng {peak_month-2 if peak_month>2 else 12}</b> để đón đầu sóng.
        - Tận dụng tháng thấp điểm để bảo trì máy móc và đào tạo nhân sự.
        """
        generate_insight_box("Quy luật Mùa vụ", insight_season, "info")

# --- TAB 2: SẢN PHẨM ---
with tab2:
    c_prod_1, c_prod_2 = st.columns([1, 1])
    
    with c_prod_1:
        st.markdown("##### 🎨 Xu hướng Màu sắc")
        color_trend = df.groupby('nhom_mau')['sl'].sum().reset_index().sort_values('sl', ascending=False)
        fig_c = px.pie(color_trend, values='sl', names='nhom_mau', hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(polish_chart(fig_c), use_container_width=True)
        
        top_color = color_trend.iloc[0]
        insight_color = f"""
        Thị trường đang chuộng nhóm màu <b>{top_color['nhom_mau']}</b>, chiếm tỷ trọng chủ đạo.
        Các nhóm màu khác cần được xem xét lại nếu tồn kho cao.
        """
        generate_insight_box("Thị hiếu Thẩm mỹ", insight_color, "warning")

    with c_prod_2:
        st.markdown("##### 🔌 Phân tích Tính năng (USB)")
        usb_trend = df.groupby(['year', 'is_usb_clean'])['sl'].sum().reset_index()
        fig_u = px.bar(usb_trend, x='year', y='sl', color='is_usb_clean', barmode='group', 
                       color_discrete_map={'Có USB': WARNING, 'Không USB': '#424242'})
        st.plotly_chart(polish_chart(fig_u), use_container_width=True)
        
        # Logic USB Growth
        u_curr = usb_trend[(usb_trend['year']==curr_year) & (usb_trend['is_usb_clean']=='Có USB')]['sl'].sum()
        u_prev = usb_trend[(usb_trend['year']==curr_year-1) & (usb_trend['is_usb_clean']=='Có USB')]['sl'].sum()
        u_growth = ((u_curr - u_prev)/u_prev*100) if u_prev>0 else 0
        
        insight_usb = f"""
        Dòng sản phẩm tích hợp USB đang tăng trưởng <b>{u_growth:+.1f}%</b>. 
        Đây là tín hiệu cho thấy khách hàng ngày càng quan tâm đến tính năng công nghệ tiện ích.
        """
        generate_insight_box("Động lực Công nghệ", insight_usb, "success" if u_growth > 0 else "danger")

# --- TAB 3: QUẢN TRỊ RỦI RO (PARETO) ---
with tab3:
    col_pareto, col_risk = st.columns([2, 1])
    
    with col_pareto:
        cust_data = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
        cust_data['cum_perc'] = (cust_data['sl'].cumsum() / cust_data['sl'].sum()) * 100
        
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=cust_data['khach_hang'].head(10), y=cust_data['sl'].head(10), name='Sản lượng', marker_color=PRIMARY))
        fig_p.add_trace(go.Scatter(x=cust_data['khach_hang'].head(10), y=cust_data['cum_perc'].head(10), name='% Tích lũy', yaxis='y2', line=dict(color=DANGER, width=2)))
        fig_p.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110]), showlegend=False, title="Biểu đồ Pareto (Top 10 Khách hàng)")
        st.plotly_chart(polish_chart(fig_p), use_container_width=True)
        
    with col_risk:
        top_3_share = cust_data.head(3)['sl'].sum() / cust_data['sl'].sum() * 100
        top1_name = cust_data.iloc[0]['khach_hang']
        
        risk_level = "CAO" if top_3_share > 60 else "TRUNG BÌNH" if top_3_share > 40 else "THẤP"
        risk_color = "danger" if top_3_share > 60 else "warning" if top_3_share > 40 else "success"
        
        insight_risk = f"""
        Top 3 khách hàng lớn nhất đang nắm giữ <span class='highlight'>{top_3_share:.1f}%</span> tổng sản lượng.
        <br>Trong đó, <b>{top1_name}</b> là đối tác chi phối lớn nhất.<br><br>
        ⚠️ <b>Mức độ rủi ro phụ thuộc: {risk_level}</b>.<br>
        Cần mở rộng tệp khách hàng mới để giảm thiểu rủi ro nếu một trong các Key Account này cắt giảm đơn hàng.
        """
        generate_insight_box("Rủi ro Tập trung", insight_risk, risk_color)

st.markdown("---")
st.caption(f"© 2026 Mộc Phát Strategy Hub | Generated at: {datetime.now().strftime('%H:%M %d/%m/%Y')}")
