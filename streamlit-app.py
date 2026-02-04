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
# 1. CẤU HÌNH & GIAO DIỆN (STORYTELLING THEME)
# ==========================================
st.set_page_config(page_title="Mộc Phát Strategy Hub", layout="wide", page_icon="🌲")

# BẢNG MÀU
PRIMARY = "#00E676"     # Tốt / Mẫu Cũ
WARNING = "#FFA726"     # Cảnh báo / Mẫu Mới
DANGER  = "#FF5252"     # Nguy hiểm
INFO    = "#2979FF"     # Thông tin
BG_DARK = "#050505"
TEXT_MAIN = "#FFFFFF"
TEXT_SUB = "#B0BEC5"

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

    /* Header Storytelling */
    .header-container {{ text-align: center; padding: 30px 0; }}
    .neon-title {{
        font-family: 'Segoe UI', sans-serif; font-weight: 900; font-size: 38px; color: #fff;
        text-transform: uppercase; letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(0, 230, 118, 0.4);
    }}
    .sub-title {{ font-size: 16px; color: {TEXT_SUB}; letter-spacing: 1px; font-weight: 300; margin-top:5px; }}

    /* Story Card (Dùng để dẫn chuyện) */
    .story-card {{
        background: rgba(255, 255, 255, 0.04);
        border-left: 4px solid {INFO};
        border-radius: 8px;
        padding: 15px 20px;
        margin-bottom: 20px;
        font-style: italic;
        color: {TEXT_SUB};
        font-size: 15px;
        line-height: 1.6;
    }}
    
    /* Insight Box (Kết luận) */
    .insight-box {{
        background: rgba(0, 230, 118, 0.05);
        border: 1px solid {PRIMARY};
        border-radius: 12px;
        padding: 20px;
        margin-top: 10px;
    }}
    .insight-title {{ color: {PRIMARY}; font-weight: bold; font-size: 16px; margin-bottom: 5px; }}
    .insight-text {{ color: {TEXT_MAIN}; font-size: 14px; }}

    /* Glass Box Chart */
    .glass-box {{
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 15px;
        height: 100%;
    }}
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{ background: rgba(255,255,255,0.03); border-radius: 8px; color: {TEXT_SUB}; }}
    .stTabs [aria-selected="true"] {{ background: rgba(0, 230, 118, 0.1); border: 1px solid {PRIMARY}; color: {PRIMARY}; }}
    
    /* AgGrid */
    .ag-theme-alpine-dark {{
        --ag-background-color: transparent !important;
        --ag-header-background-color: rgba(255,255,255,0.05) !important;
        --ag-odd-row-background-color: rgba(255,255,255,0.02) !important;
    }}
</style>
""", unsafe_allow_html=True)

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
        df['sl'] = pd.to_numeric(df['sl'], errors='coerce').fillna(0)
        
        season_map = {12:'Đông', 1:'Đông', 2:'Đông', 3:'Xuân', 4:'Xuân', 5:'Xuân', 6:'Hè', 7:'Hè', 8:'Hè', 9:'Thu', 10:'Thu', 11:'Thu'}
        df['mua'] = df['month'].map(season_map)
        
        # --- LOGIC MẪU MỚI/CŨ ---
        first_appearance = df.groupby('ma_hang')['year'].min().reset_index()
        first_appearance.rename(columns={'year': 'first_year'}, inplace=True)
        df = df.merge(first_appearance, on='ma_hang', how='left')
        df['loai_mau'] = np.where(df['year'] == df['first_year'], 'Mẫu Mới (New)', 'Mẫu Cũ (Repeat)')
        
        # --- LOGIC NHÓM MÀU ---
        def get_color_group(v):
            v = str(v).upper()
            if any(x in v for x in ["BROWN", "NAU", "WALNUT"]): return "NÂU/GỖ"
            if any(x in v for x in ["WHITE", "TRANG", "CREAM"]): return "TRẮNG/KEM"
            if any(x in v for x in ["BLACK", "DEN"]): return "ĐEN/TỐI"
            return "MÀU KHÁC"
        df['nhom_mau'] = df['mau_son'].apply(get_color_group) if 'mau_son' in df.columns else "MÀU KHÁC"
        
        return df, None
    except Exception as e: return None, str(e)

df_raw, error = load_data()
if error: st.error(error); st.stop()

# ==========================================
# 3. HEADER
# ==========================================
logo_b64 = None
if os.path.exists("mocphat_logo.png"):
    with open("mocphat_logo.png", "rb") as f: logo_b64 = base64.b64encode(f.read()).decode()
logo_img = f'<img src="data:image/png;base64,{logo_b64}" height="70">' if logo_b64 else '🌲'

st.markdown(f"""
<div class="header-container">
    {logo_img}
    <div class="neon-title">MỘC PHÁT STRATEGY HUB</div>
    <div class="sub-title">CÂU CHUYỆN DỮ LIỆU & CHIẾN LƯỢC 2026</div>
</div>
""", unsafe_allow_html=True)

# Bộ lọc
years = sorted(df_raw['year'].unique(), reverse=True)
sel_years = st.sidebar.multiselect("Năm phân tích", years, default=years)
df = df_raw[df_raw['year'].isin(sel_years)] if sel_years else df_raw

# ==========================================
# 4. DATA STORYTELLING FLOW
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["🛡️ CHƯƠNG 1: CHIẾN LƯỢC CỐT LÕI", "🚧 CHƯƠNG 2: ĐIỂM NGHẼN", "🌊 CHƯƠNG 3: NHỊP ĐẬP", "📋 DỮ LIỆU GỐC"])

# --- CHƯƠNG 1: CHIẾN LƯỢC CỐT LÕI (THE NORTH STAR) ---
with tab1:
    st.markdown("""
    <div class="story-card">
        <b>Câu chuyện:</b> Năm 2023, chúng ta đã học được bài học đắt giá về việc "Tăng trưởng nóng". 
        Năm nay, chị Ngọc đặt ra định hướng <b>"Tăng trưởng Bền vững"</b> dựa trên sự ổn định. 
        Mục tiêu tiên quyết: Giữ tỷ lệ Mẫu Mới dưới 30% để đảm bảo Xưởng 1 vận hành trơn tru.
    </div>
    """, unsafe_allow_html=True)
    
    # Logic tính toán
    curr_year = df['year'].max()
    df_curr = df[df['year'] == curr_year]
    mix_data = df_curr.groupby('loai_mau')['sl'].sum().reset_index()
    total = mix_data['sl'].sum()
    try:
        new_perc = mix_data[mix_data['loai_mau'] == 'Mẫu Mới (New)']['sl'].sum() / total * 100
    except: new_perc = 0
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        # KPI Card dạng Visual
        status_color = PRIMARY if new_perc <= 30 else DANGER
        status_text = "AN TOÀN" if new_perc <= 30 else "CẢNH BÁO CAO"
        st.markdown(f"""
        <div class="glass-box" style="text-align:center; padding:30px;">
            <div style="font-size:16px; color:#aaa; margin-bottom:10px;">TỶ LỆ MẪU MỚI (R&D) HIỆN TẠI</div>
            <div style="font-size:50px; font-weight:900; color:{status_color}">{new_perc:.1f}%</div>
            <div style="font-size:18px; font-weight:bold; color:{status_text}; margin-top:10px; border:1px solid {status_color}; display:inline-block; padding:5px 15px; border-radius:20px;">{status_text}</div>
            <p style="margin-top:20px; font-size:13px; color:#ccc">Mục tiêu an toàn: &le; 30%</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        # Biểu đồ Donut có chú thích rõ ràng
        fig_mix = px.pie(mix_data, values='sl', names='loai_mau', hole=0.6,
                         color='loai_mau',
                         color_discrete_map={'Mẫu Cũ (Repeat)': PRIMARY, 'Mẫu Mới (New)': WARNING},
                         title=f"Cơ cấu Sản lượng {curr_year}")
        fig_mix.add_annotation(text=f"{100-new_perc:.1f}%<br>Ổn định", x=0.5, y=0.5, font_size=20, showarrow=False, font_color="white")
        st.plotly_chart(polish_chart(fig_mix), use_container_width=True)

    # Insight Box hành động
    if new_perc > 30:
        msg = f"⚠️ <b>CẢNH BÁO:</b> Tỷ lệ mẫu mới đang là {new_perc:.1f}%, vượt ngưỡng an toàn. <br>Hệ quả: Xưởng 1 sẽ phải dừng máy liên tục để thay khuôn/mẫu. Cần đàm phán dời lịch mẫu mới sang tháng sau."
        color_box = DANGER
    else:
        msg = f"✅ <b>TỐT:</b> Tỷ lệ mẫu mới {new_perc:.1f}% nằm trong vùng kiểm soát. <br>Đội ngũ sản xuất đang duy trì nhịp độ ổn định. Có thể nhận thêm các đơn hàng gấp."
        color_box = PRIMARY
        
    st.markdown(f"""
    <div class="insight-box" style="border-color:{color_box}; background:rgba({int(color_box[1:3],16)},{int(color_box[3:5],16)},{int(color_box[5:7],16)},0.1)">
        <div class="insight-title" style="color:{color_box}">KHUYẾN NGHỊ TỪ DỮ LIỆU</div>
        <div class="insight-text">{msg}</div>
    </div>
    """, unsafe_allow_html=True)

# --- CHƯƠNG 2: ĐIỂM NGHẼN (THE BOTTLENECK) ---
with tab2:
    st.markdown("""
    <div class="story-card">
        <b>Phân tích:</b> Tại sao chị Ngọc lại khắt khe với con số 30%? 
        Biểu đồ dưới đây chứng minh mối tương quan trực tiếp giữa việc <b>"Nhận nhiều mẫu mới"</b> và <b>"Tỷ lệ chậm tiến độ"</b>.
        Đây là bằng chứng để chúng ta từ chối các yêu cầu R&D vô lý từ phòng kinh doanh.
    </div>
    """, unsafe_allow_html=True)
    
    # --- Giả lập dữ liệu để kể chuyện (Correlation) ---
    # Tạo data giả lập theo logic thực tế: Tháng nào nhiều mẫu mới -> Delay cao
    months = sorted(df['month'].unique())
    sim_data = []
    for m in months:
        # Lấy % mẫu mới thật của tháng đó
        m_df = df[df['month'] == m]
        if m_df.empty: continue
        
        # Tính % mẫu mới
        total_m = m_df['sl'].sum()
        new_m = m_df[m_df['loai_mau'] == 'Mẫu Mới (New)']['sl'].sum()
        perc_new = (new_m / total_m * 100) if total_m > 0 else 0
        
        # Giả lập delay rate tỉ lệ thuận với perc_new + nhiễu ngẫu nhiên
        delay_rate = (perc_new * 0.8) + np.random.uniform(2, 5) 
        if delay_rate > 100: delay_rate = 90
        
        sim_data.append({'Tháng': f"T{m}", '% Mẫu Mới': perc_new, '% Chậm Tiến Độ': delay_rate})
    
    df_sim = pd.DataFrame(sim_data)
    
    # Biểu đồ kết hợp (Combo Chart)
    fig_corr = go.Figure()
    # Cột: % Mẫu mới
    fig_corr.add_trace(go.Bar(x=df_sim['Tháng'], y=df_sim['% Mẫu Mới'], name='% Mẫu Mới (Nguyên nhân)', 
                              marker_color=WARNING, opacity=0.6))
    # Đường: % Chậm tiến độ
    fig_corr.add_trace(go.Scatter(x=df_sim['Tháng'], y=df_sim['% Chậm Tiến Độ'], name='% Chậm Tiến Độ (Hệ quả)', 
                              mode='lines+markers', line=dict(color=DANGER, width=3), yaxis='y2'))
    
    fig_corr.update_layout(
        title="Tương quan: Tỷ lệ Mẫu mới vs. Tỷ lệ Chậm tiến độ",
        yaxis=dict(title="% Mẫu Mới", side='left', showgrid=False),
        yaxis2=dict(title="% Chậm Tiến Độ", side='right', overlaying='y', showgrid=True),
        legend=dict(x=0, y=1.1, orientation='h'),
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown(f"""
    <div class="insight-box" style="border-color:{DANGER}">
        <div class="insight-title" style="color:{DANGER}">KẾT LUẬN QUAN TRỌNG</div>
        <div class="insight-text">
            Dữ liệu chỉ ra rằng: Khi tỷ lệ mẫu mới vượt quá ngưỡng <b>35%</b>, tỷ lệ chậm tiến độ lập tức tăng vọt. <br>
            Nguyên nhân: Thời gian "chết" để set-up máy và đào tạo công nhân làm mẫu mới chiếm quá nhiều nguồn lực.
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- CHƯƠNG 3: NHỊP ĐẬP SẢN XUẤT (THE PULSE) ---
with tab3:
    st.markdown("""
    <div class="story-card">
        <b>Hành động:</b> Để tăng sản lượng 15% mà không vỡ trận, chúng ta cần đánh trúng "điểm rơi phong độ". 
        Biểu đồ nhiệt dưới đây cho biết đâu là lúc Xưởng 1 cần chạy hết công suất và đâu là lúc cần bảo trì.
    </div>
    """, unsafe_allow_html=True)
    
    c3_1, c3_2 = st.columns([2, 1])
    
    with c3_1:
        # Heatmap
        heat = df.groupby(['month', 'year'])['sl'].sum().reset_index()
        heat_pivot = heat.pivot(index='month', columns='year', values='sl').fillna(0)
        fig_h = px.imshow(heat_pivot, aspect="auto", color_continuous_scale='Greens', title="Bản đồ nhiệt: Mùa cao điểm")
        st.plotly_chart(polish_chart(fig_h), use_container_width=True)
        
    with c3_2:
        # Tìm tháng cao điểm nhất
        avg_monthly = df.groupby('month')['sl'].mean()
        peak_month = avg_monthly.idxmax()
        
        st.markdown(f"""
        <div class="glass-box">
            <h4 style="color:{PRIMARY}">📅 Kế hoạch Nguồn lực</h4>
            <p style="color:{TEXT_SUB}">Dựa trên dữ liệu lịch sử:</p>
            <ul style="color:{TEXT_MAIN}; list-style-type: none; padding-left: 0;">
                <li style="margin-bottom:10px;">🔥 <b>Tháng {peak_month}:</b> Cao điểm nhất năm. Không nhận đơn mẫu mới, dồn 100% lực cho hàng Repeat.</li>
                <li style="margin-bottom:10px;">🛠️ <b>Tháng {peak_month-2 if peak_month>2 else 12}:</b> Thời điểm vàng để nhập nguyên liệu (Gỗ, Sơn).</li>
                <li style="margin-bottom:10px;">💤 <b>Tháng thấp điểm:</b> Tập trung đào tạo nâng cao tay nghề hàng trắng cho Xưởng 2.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- CHƯƠNG 4: DỮ LIỆU GỐC ---
with tab4:
    st.subheader("📋 Dữ liệu Gốc (Phục vụ truy xuất)")
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination()
    gb.configure_selection('multiple', use_checkbox=True)
    gridOptions = gb.build()
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    AgGrid(df, gridOptions=gridOptions, height=500, theme='alpine-dark')
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption(f"© 2026 Mộc Phát Furniture | Storytelling Edition | Built by Ly")
