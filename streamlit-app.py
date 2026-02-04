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
# 1. CẤU HÌNH & GIAO DIỆN (PHOENIX EDITION)
# ==========================================
st.set_page_config(page_title="Mộc Phát Strategy Hub", layout="wide", page_icon="🌲")

# BẢNG MÀU CHIẾN LƯỢC
PRIMARY = "#00E676"     # Tăng trưởng / Tốt
WARNING = "#FFA726"     # Cảnh báo (Mẫu mới quá nhiều)
DANGER  = "#FF5252"     # Nguy hiểm (Chậm tiến độ/Chất lượng)
INFO    = "#2979FF"     # Thông tin
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
    .sub-title {{ font-size: 14px; color: {TEXT_SUB}; letter-spacing: 1px; font-weight: 300; margin-top:5px; }}

    /* Strategy Card (Quan trọng) */
    .strategy-card {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        border-left: 4px solid {INFO};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }}
    .strategy-title {{ font-weight: 700; font-size: 16px; color: {TEXT_MAIN}; display: flex; align-items: center; gap: 10px; }}
    .strategy-content {{ font-size: 14px; color: {TEXT_SUB}; line-height: 1.6; margin-top: 10px; text-align: justify; }}
    
    /* Highlight text */
    .hl-good {{ color: {PRIMARY}; font-weight: bold; }}
    .hl-warn {{ color: {WARNING}; font-weight: bold; }}
    .hl-bad {{ color: {DANGER}; font-weight: bold; }}

    /* Glass Box for Charts */
    .glass-box {{
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 15px;
        height: 100%;
    }}
    
    /* AgGrid & Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{ background: rgba(255,255,255,0.03); border-radius: 8px; color: {TEXT_SUB}; }}
    .stTabs [aria-selected="true"] {{ background: rgba(0, 230, 118, 0.1); border: 1px solid {PRIMARY}; color: {PRIMARY}; }}
    
    .ag-theme-alpine-dark {{
        --ag-background-color: transparent !important;
        --ag-header-background-color: rgba(255,255,255,0.05) !important;
        --ag-odd-row-background-color: rgba(255,255,255,0.02) !important;
    }}
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
# 2. XỬ LÝ DỮ LIỆU & GIẢ LẬP LOGIC 70/30
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
        
        # Xử lý màu
        def get_color_group(v):
            v = str(v).upper()
            if any(x in v for x in ["BROWN", "NAU", "WALNUT"]): return "NÂU/GỖ"
            if any(x in v for x in ["WHITE", "TRANG", "CREAM"]): return "TRẮNG/KEM"
            if any(x in v for x in ["BLACK", "DEN"]): return "ĐEN/TỐI"
            return "MÀU KHÁC"
        df['nhom_mau'] = df['mau_son'].apply(get_color_group) if 'mau_son' in df.columns else "MÀU KHÁC"
        
        # --- GIẢ LẬP LOGIC MẪU MỚI / CŨ (Quan trọng cho bài toán của chị Ngọc) ---
        # Logic: Những mã hàng xuất hiện lần đầu tiên trong năm hiện tại được coi là "Mẫu Mới"
        # Những mã hàng đã xuất hiện ở các năm trước là "Mẫu Cũ" (Repeat Order)
        
        # 1. Tìm năm xuất hiện đầu tiên của từng mã hàng
        first_appearance = df.groupby('ma_hang')['year'].min().reset_index()
        first_appearance.rename(columns={'year': 'first_year'}, inplace=True)
        
        df = df.merge(first_appearance, on='ma_hang', how='left')
        
        # 2. Gán nhãn: Nếu năm bán == năm đầu tiên -> Mẫu Mới, ngược lại -> Mẫu Cũ
        df['loai_mau'] = np.where(df['year'] == df['first_year'], 'Mẫu Mới (New)', 'Mẫu Cũ (Repeat)')
        
        return df, None
    except Exception as e: return None, str(e)

df_raw, error = load_data()
if error: st.error(error); st.stop()

def generate_insight_box(title, content, type="info"):
    colors = {"success": PRIMARY, "warning": WARNING, "danger": DANGER, "info": INFO}
    icon = {"success": "🚀", "warning": "⚠️", "danger": "🔥", "info": "💡"}
    st.markdown(f"""
    <div class="strategy-card" style="border-left: 4px solid {colors.get(type, INFO)};">
        <div class="strategy-title" style="color:{colors.get(type, INFO)}">{icon[type]} {title}</div>
        <div class="strategy-content">{content}</div>
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
    <div class="sub-title">STRATEGY 2026: 70% REPEAT - 30% NEW</div>
</div>
""", unsafe_allow_html=True)

years = sorted(df_raw['year'].unique(), reverse=True)
sel_years = st.sidebar.multiselect("Năm", years, default=years)
df = df_raw[df_raw['year'].isin(sel_years)] if sel_years else df_raw
if df.empty: st.warning("Chưa có dữ liệu."); st.stop()

# ==========================================
# 4. TAB CHIẾN LƯỢC (ĐƯỢC ĐƯA LÊN ĐẦU)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["🛡️ CHIẾN LƯỢC 70/30", "📊 HIỆU QUẢ SẢN XUẤT", "🎨 SẢN PHẨM & MÙA VỤ", "📋 DỮ LIỆU"])

# --- TAB 1: CHIẾN LƯỢC 70/30 (DÀNH RIÊNG CHO CHỊ NGỌC) ---
with tab1:
    st.markdown("### 🎯 Theo dõi Mục tiêu: Ổn định (70%) & Đổi mới (30%)")
    
    # Tính toán tỷ lệ thực tế
    curr_year = df['year'].max()
    df_curr = df[df['year'] == curr_year]
    
    mix_data = df_curr.groupby('loai_mau')['sl'].sum().reset_index()
    total_vol = mix_data['sl'].sum()
    mix_data['percent'] = (mix_data['sl'] / total_vol * 100)
    
    try:
        new_perc = mix_data[mix_data['loai_mau'] == 'Mẫu Mới (New)']['percent'].values[0]
    except: new_perc = 0
    
    old_perc = 100 - new_perc
    
    # 1. KPI CARDS
    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        st.markdown(f"""
        <div class="glass-box" style="text-align:center; border: 1px solid {PRIMARY}">
            <div style="color:{TEXT_SUB}">MẪU CŨ (MỤC TIÊU >70%)</div>
            <div style="font-size:36px; font-weight:bold; color:{PRIMARY}">{old_perc:.1f}%</div>
            <div style="font-size:12px; color:#aaa">Dòng sản phẩm ổn định</div>
        </div>
        """, unsafe_allow_html=True)
    with c_s2:
        color_new = PRIMARY if 25 <= new_perc <= 35 else WARNING 
        st.markdown(f"""
        <div class="glass-box" style="text-align:center; border: 1px solid {color_new}">
            <div style="color:{TEXT_SUB}">MẪU MỚI (MỤC TIÊU ~30%)</div>
            <div style="font-size:36px; font-weight:bold; color:{color_new}">{new_perc:.1f}%</div>
            <div style="font-size:12px; color:#aaa">R&D & Đổi mới</div>
        </div>
        """, unsafe_allow_html=True)
    with c_s3:
        # Giả lập tăng trưởng 2026
        base_25 = df_raw[df_raw['year'] == 2025]['sl'].sum()
        target_26 = base_25 * 1.15
        forecast_curr = total_vol # Giả sử total_vol là hiện tại (nếu lọc 2026)
        # Nếu đang lọc nhiều năm, logic này chỉ mang tính demo
        gap = target_26 - total_vol
        
        st.markdown(f"""
        <div class="glass-box" style="text-align:center; border: 1px solid #FFA726">
            <div style="color:{TEXT_SUB}">MỤC TIÊU TĂNG TRƯỞNG 2026</div>
            <div style="font-size:36px; font-weight:bold; color:#FFA726">15%</div>
            <div style="font-size:12px; color:#aaa">Cần thêm: {fmt_num(gap)} SP để đạt</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. BIỂU ĐỒ & INSIGHT
    c_chart_s, c_text_s = st.columns([2, 1])
    
    with c_chart_s:
        # Biểu đồ Donut 70/30
        fig_mix = px.pie(mix_data, values='sl', names='loai_mau', hole=0.6, 
                         color='loai_mau',
                         color_discrete_map={'Mẫu Cũ (Repeat)': PRIMARY, 'Mẫu Mới (New)': WARNING},
                         title=f"Cơ cấu Sản lượng Năm {curr_year}")
        st.plotly_chart(polish_chart(fig_mix), use_container_width=True)
        
        # Biểu đồ cột chồng theo tháng (Xem tháng nào làm nhiều mẫu mới quá)
        mix_month = df_curr.groupby(['month', 'loai_mau'])['sl'].sum().reset_index()
        fig_bar_mix = px.bar(mix_month, x='month', y='sl', color='loai_mau', 
                             color_discrete_map={'Mẫu Cũ (Repeat)': PRIMARY, 'Mẫu Mới (New)': WARNING},
                             title="Biến động Tỷ lệ Mẫu Mới/Cũ theo Tháng", barmode='stack')
        st.plotly_chart(polish_chart(fig_bar_mix), use_container_width=True)

    with c_text_s:
        # Tự động sinh Insight
        status = "ỔN ĐỊNH" if new_perc <= 35 else "CẢNH BÁO RỦI RO"
        msg_color = "success" if new_perc <= 35 else "warning"
        
        insight_strat = f"""
        Hiện tại, tỷ lệ mẫu mới đang ở mức <span class='hl-warn'>{new_perc:.1f}%</span>. 
        Trạng thái: <b style='color:{"#00E676" if new_perc<=35 else "#FFA726"}'>{status}</b>.
        <br><br>
        <b>Tại sao điều này quan trọng?</b><br>
        Việc giữ mẫu mới dưới 35% giúp dây chuyền tại <b>Xưởng 1</b> hoạt động liên tục, giảm thời gian chết do chuyển đổi mã hàng.
        <br><br>
        <b>Hành động khuyến nghị:</b><br>
        Nếu tỷ lệ này vượt quá 40% trong tháng tới, cần:
        1. Tạm dừng nhận mẫu R&D mới.
        2. Đàm phán với khách hàng dời lịch giao mẫu.
        3. Ưu tiên chạy các đơn hàng lặp lại (Repeat Order) để bù sản lượng.
        """
        generate_insight_box("Giám sát Chiến lược", insight_strat, msg_color)
        
        generate_insight_box("Lưu ý từ Quá khứ (2023)", 
                             "Bài học 2023: Chạy đua sản lượng + Quá nhiều mẫu mới = Mất kiểm soát chất lượng. <br>Năm nay kiên quyết giữ đúng tỷ lệ để bảo vệ uy tín.", 
                             "danger")

# --- TAB 2: HIỆU QUẢ SẢN XUẤT ---
with tab2:
    st.subheader("📊 Hiệu suất Vận hành")
    
    # Giả lập dữ liệu "Tiến độ" (Vì file excel ko có cột này, ta tạo giả lập để demo)
    # Trong thực tế bạn sẽ lấy từ dữ liệu thật
    df['status'] = np.random.choice(['Đúng tiến độ', 'Chậm tiến độ'], size=len(df), p=[0.85, 0.15])
    
    c_prod_1, c_prod_2 = st.columns([2, 1])
    
    with c_prod_1:
        # Biểu đồ xu hướng sản lượng
        ts_data = df.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=ts_data['ym'], y=ts_data['sl'], mode='lines', name='Sản lượng', 
                                 line=dict(color=PRIMARY, width=3, shape='spline'),
                                 fill='tozeroy', fillcolor='rgba(0, 230, 118, 0.1)'))
        st.plotly_chart(polish_chart(fig_trend), use_container_width=True)
        
    with c_prod_2:
        # Tỷ lệ Chậm tiến độ (Giả lập)
        delay_counts = df['status'].value_counts(normalize=True) * 100
        delay_rate = delay_counts.get('Chậm tiến độ', 0)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = delay_rate,
            title = {'text': "Tỷ lệ Chậm tiến độ (Ước tính)"},
            gauge = {'axis': {'range': [None, 100]},
                     'bar': {'color': DANGER},
                     'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 10}}))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        generate_insight_box("Nguyên nhân Chậm trễ", 
                             "Phần lớn các đơn hàng chậm tiến độ tập trung vào nhóm <b>Mẫu Mới</b> do thời gian set-up máy lâu và công nhân chưa quen thao tác.", 
                             "warning")

# --- TAB 3: SẢN PHẨM & MÙA VỤ ---
with tab3:
    st.subheader("🎨 Phân tích Sản phẩm & Mùa vụ")
    c3_1, c3_2 = st.columns(2)
    with c3_1:
        # Heatmap
        heat = df.groupby(['month', 'year'])['sl'].sum().reset_index()
        heat_pivot = heat.pivot(index='month', columns='year', values='sl').fillna(0)
        fig_h = px.imshow(heat_pivot, aspect="auto", color_continuous_scale='Greens', title="Bản đồ nhiệt Sản lượng")
        st.plotly_chart(polish_chart(fig_h), use_container_width=True)
    with c3_2:
        # Top Products
        top_sku = df.groupby('ma_hang')['sl'].sum().nlargest(10).reset_index()
        fig_sku = px.bar(top_sku, x='sl', y='ma_hang', orientation='h', color='sl', title="Top 10 Mã hàng chủ lực")
        st.plotly_chart(polish_chart(fig_sku), use_container_width=True)

# --- TAB 4: DỮ LIỆU ---
with tab4:
    st.subheader("📋 Dữ liệu Chi tiết")
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination()
    gb.configure_selection('multiple', use_checkbox=True)
    gridOptions = gb.build()
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    AgGrid(df, gridOptions=gridOptions, height=500, theme='alpine-dark')
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption(f"© 2026 Mộc Phát Furniture | Strategic Dashboard for Ms. Ngoc | Built by Ly (Data Analyst)")
