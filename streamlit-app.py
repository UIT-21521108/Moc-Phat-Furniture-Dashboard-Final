import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from datetime import datetime

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & BRANDING
# ==========================================
st.set_page_config(page_title="Moc Phat Dashboard Pro", layout="wide", page_icon="🌲")

# Màu thương hiệu
PRIMARY = "#066839"    # Xanh Mộc Phát
ACCENT  = "#1B7D4F"    # Xanh nhấn
BG_LIGHT = "#F0F2F6"

# CSS tùy chỉnh để giao diện chuyên nghiệp hơn
st.markdown(f"""
<style>
    /* Header Sticky */
    .header-sticky {{
        position: sticky; top: 0; z-index: 999; 
        background: white; border-bottom: 2px solid {PRIMARY};
        padding: 10px 0px; margin-bottom: 20px;
    }}
    .header-title {{ font-size: 28px; font-weight: 900; color: {PRIMARY}; margin: 0; }}
    .header-sub {{ font-size: 14px; color: #555; font-style: italic; }}
    
    /* KPI Cards */
    .kpi-box {{
        background: white; padding: 15px; border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); border-left: 5px solid {PRIMARY};
        text-align: center;
    }}
    .kpi-val {{ font-size: 24px; font-weight: bold; color: #333; }}
    .kpi-lbl {{ font-size: 14px; color: #666; text-transform: uppercase; }}
    .kpi-delta {{ font-size: 12px; font-weight: bold; }}
    .pos {{ color: green; }} .neg {{ color: red; }}
    
    /* AgGrid tweaking */
    .ag-theme-streamlit {{ --ag-header-background-color: {BG_LIGHT}; }}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"""
<div class="header-sticky">
    <div class="header-title">🌲 MỘC PHÁT FURNITURE DASHBOARD</div>
    <div class="header-sub">Hệ thống phân tích dữ liệu sản xuất & kinh doanh (Phiên bản Pro)</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (CACHING & PREP)
# ==========================================
@st.cache_data(ttl=3600)
def load_and_clean_data(uploaded_file):
    """Đọc file Excel/CSV và chuẩn hóa dữ liệu"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Chuẩn hóa tên cột (về chữ thường, bỏ khoảng trắng)
        df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
        
        # Các cột bắt buộc
        req_cols = ['sl', 'year', 'month']
        for c in req_cols:
            if c not in df.columns:
                return None, f"Thiếu cột bắt buộc: {c}"

        # Xử lý ngày tháng
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
        df = df[(df['year'] > 2000) & (df['month'] >= 1) & (df['month'] <= 12)]
        df['ym'] = pd.to_datetime(df.assign(day=1)[['year', 'month', 'day']])
        
        # Xử lý chuỗi
        str_cols = ['khach_hang', 'ma_hang', 'mau_son', 'xuong', 'khu_vuc']
        for c in str_cols:
            if c not in df.columns: df[c] = "Unknown"
            else: df[c] = df[c].fillna("Unknown").astype(str)

        # Logic phân loại Khu vực (Nếu chưa có)
        if df['khu_vuc'].iloc[0] == "Unknown":
            def get_region(cust):
                cust = cust.upper()
                if any(x in cust for x in ['TJX', 'MARSHALL', 'HOMEGOODS', 'WINNERS', 'MMX']): return "Bắc Mỹ"
                if any(x in cust for x in ['TK', 'TJX EUROPE']): return "Châu Âu"
                return "Khác"
            df['khu_vuc'] = df['khach_hang'].apply(get_region)

        # Logic nhóm màu
        def get_color_group(color):
            c = color.upper()
            if any(x in c for x in ['WHITE', 'CREAM', 'IVORY']): return 'WHITE'
            if any(x in c for x in ['BLACK', 'CHARCOAL']): return 'BLACK'
            if any(x in c for x in ['BROWN', 'WALNUT', 'ESPRESSO']): return 'BROWN'
            if any(x in c for x in ['GREY', 'GRAY', 'SLATE']): return 'GREY'
            if any(x in c for x in ['NATURAL', 'OAK', 'PINE']): return 'NATURAL'
            return 'OTHER'
        
        if 'nhom_mau' not in df.columns:
            df['nhom_mau'] = df['mau_son'].apply(get_color_group)

        return df, None
    except Exception as e:
        return None, str(e)

# ==========================================
# 3. SIDEBAR & BỘ LỌC
# ==========================================
st.sidebar.header("🛠️ BỘ LỌC DỮ LIỆU")

# Upload file
uploaded_file = st.sidebar.file_uploader("Tải file dữ liệu (Excel/CSV)", type=['xlsx', 'csv'])

if not uploaded_file:
    st.info("👋 Vui lòng tải file dữ liệu để bắt đầu.")
    st.stop()

df_raw, error = load_and_clean_data(uploaded_file)
if error:
    st.error(f"Lỗi đọc file: {error}")
    st.stop()

# Bộ lọc
with st.sidebar:
    st.subheader("Tiêu chí lọc")
    
    # 1. Năm
    all_years = sorted(df_raw['year'].unique())
    sel_years = st.multiselect("Năm", all_years, default=all_years)
    
    # 2. Xưởng (Quan trọng cho kịch bản tách xưởng)
    all_factories = sorted(df_raw['xuong'].unique())
    sel_factory = st.multiselect("Xưởng SX", all_factories, default=all_factories)
    
    # 3. Khách hàng
    all_customers = sorted(df_raw['khach_hang'].unique())
    sel_cust = st.multiselect("Khách hàng", all_customers, default=all_customers)
    
    # 4. SKU
    sku_search = st.text_input("Tìm Mã hàng (SKU)", placeholder="Nhập mã...")

# Áp dụng lọc
df = df_raw.copy()
if sel_years: df = df[df['year'].isin(sel_years)]
if sel_factory: df = df[df['xuong'].isin(sel_factory)]
if sel_cust: df = df[df['khach_hang'].isin(sel_cust)]
if sku_search: df = df[df['ma_hang'].str.contains(sku_search, case=False)]

if df.empty:
    st.warning("Không có dữ liệu phù hợp với bộ lọc.")
    st.stop()

# ==========================================
# 4. DASHBOARD CHÍNH
# ==========================================

# --- KPI SECTION ---
st.markdown("### 🚀 TỔNG QUAN HIỆU SUẤT")
k1, k2, k3, k4 = st.columns(4)

current_year = df['year'].max()
prev_year = current_year - 1

vol_curr = df[df['year'] == current_year]['sl'].sum()
vol_prev = df[df['year'] == prev_year]['sl'].sum()
growth_yoy = ((vol_curr - vol_prev) / vol_prev * 100) if vol_prev > 0 else 0

top_sku = df.groupby('ma_hang')['sl'].sum().idxmax()
top_cust = df.groupby('khach_hang')['sl'].sum().idxmax()
total_sku = df['ma_hang'].nunique()

def kpi_card(col, label, value, delta=None):
    delta_html = ""
    if delta is not None:
        color = "pos" if delta >= 0 else "neg"
        icon = "▲" if delta >= 0 else "▼"
        delta_html = f"<div class='kpi-delta {color}'>{icon} {abs(delta):.1f}% YoY</div>"
    
    col.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-val">{value}</div>
        <div class="kpi-lbl">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

kpi_card(k1, f"Tổng Sản Lượng {current_year}", f"{vol_curr:,.0f}", growth_yoy)
kpi_card(k2, "Mã Hàng (SKU) Active", f"{total_sku:,.0f}")
kpi_card(k3, "Top SKU Bán Chạy", top_sku)
kpi_card(k4, "Khách Hàng Lớn Nhất", top_cust)

st.markdown("---")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 BIỂU ĐỒ & XU HƯỚNG", "📋 QUẢN LÝ SKU (AG-GRID)", "🌍 THỊ TRƯỜNG & KHÁCH HÀNG"])

# === TAB 1: BIỂU ĐỒ ===
with tab1:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Diễn biến sản lượng theo tháng")
        trend_df = df.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
        fig_trend = px.area(trend_df, x='ym', y='sl', 
                            title="Xu hướng sản xuất",
                            labels={'ym': 'Thời gian', 'sl': 'Sản lượng'},
                            color_discrete_sequence=[PRIMARY])
        fig_trend.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with c2:
        st.subheader("Cơ cấu Nhóm Màu")
        color_df = df.groupby('nhom_mau')['sl'].sum().reset_index()
        fig_pie = px.pie(color_df, names='nhom_mau', values='sl', 
                         title="Tỷ trọng màu sắc",
                         color_discrete_sequence=px.colors.qualitative.Prism,
                         hole=0.4)
        fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Heatmap mùa vụ
    st.subheader("🔥 Heatmap: Mùa vụ & Mã hàng (Top 20)")
    top_20_sku = df.groupby('ma_hang')['sl'].sum().nlargest(20).index
    df_top = df[df['ma_hang'].isin(top_20_sku)].copy()
    df_top['month_str'] = df_top['month'].apply(lambda x: f"Tháng {x}")
    pivot = df_top.pivot_table(index='ma_hang', columns='month', values='sl', aggfunc='sum', fill_value=0)
    
    fig_heat = px.imshow(pivot, labels=dict(x="Tháng", y="SKU", color="Sản lượng"),
                         x=pivot.columns, y=pivot.index, aspect="auto", color_continuous_scale="Greens")
    st.plotly_chart(fig_heat, use_container_width=True)

# === TAB 2: AG-GRID & DEEP DIVE (FEATURE MỚI) ===
with tab2:
    st.markdown("### 🔍 Phân tích chi tiết Mã hàng (SKU Deep Dive)")
    st.caption("Chọn một hoặc nhiều dòng trong bảng bên dưới để xem biểu đồ chi tiết.")

    # Chuẩn bị dữ liệu cho Grid
    sku_stats = df.groupby(['ma_hang', 'khach_hang', 'nhom_mau']).agg(
        Tong_SL=('sl', 'sum'),
        Don_Hang_Cuoi=('ym', 'max')
    ).reset_index().sort_values('Tong_SL', ascending=False)
    
    sku_stats['Don_Hang_Cuoi'] = sku_stats['Don_Hang_Cuoi'].dt.strftime('%Y-%m')

    # Cấu hình AgGrid
    gb = GridOptionsBuilder.from_dataframe(sku_stats)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)
    gb.configure_column("Tong_SL", header_name="Tổng SL", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=0)
    gb.configure_column("ma_hang", header_name="Mã Hàng", pinned=True)
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()

    # Hiển thị Grid
    grid_response = AgGrid(
        sku_stats,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=True,
        theme='streamlit',
        height=400, 
        width='100%',
    )

    # Xử lý sự kiện chọn dòng
    selected = grid_response['selected_rows']
    
    # Do st_aggrid trả về list dict hoặc DataFrame tùy version, xử lý an toàn:
    if isinstance(selected, pd.DataFrame):
        selected_rows = selected.to_dict('records')
    else:
        selected_rows = selected

    if selected_rows:
        st.divider()
        st.subheader("📈 Phân tích SKU đang chọn")
        
        selected_skus = [row['ma_hang'] for row in selected_rows]
        df_sel = df[df['ma_hang'].isin(selected_skus)]
        
        # Vẽ biểu đồ so sánh
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Line chart so sánh các SKU theo thời gian
            df_line = df_sel.groupby(['ym', 'ma_hang'])['sl'].sum().reset_index()
            fig_sel_line = px.line(df_line, x='ym', y='sl', color='ma_hang', markers=True,
                                   title="So sánh biến động theo tháng")
            st.plotly_chart(fig_sel_line, use_container_width=True)
            
        with chart_col2:
            # Bar chart tổng quan
            fig_sel_bar = px.bar(df_sel.groupby('ma_hang')['sl'].sum().reset_index(), 
                                 x='ma_hang', y='sl', color='ma_hang',
                                 title="Tổng sản lượng so sánh")
            st.plotly_chart(fig_sel_bar, use_container_width=True)
            
        # Hiển thị chi tiết đơn hàng gần nhất
        st.write("Lịch sử đơn hàng chi tiết:")
        st.dataframe(df_sel[['ym', 'ma_hang', 'khach_hang', 'sl', 'xuong']].sort_values('ym', ascending=False).head(10), use_container_width=True)

# === TAB 3: KHÁCH HÀNG & PARETO ===
with tab3:
    c3, c4 = st.columns(2)
    
    with c3:
        st.subheader("Top 10 Khách hàng")
        top_cust_df = df.groupby('khach_hang')['sl'].sum().nlargest(10).reset_index().sort_values('sl', ascending=True)
        fig_cust = px.bar(top_cust_df, y='khach_hang', x='sl', orientation='h', text_auto='.2s',
                          color='sl', color_continuous_scale='Viridis')
        st.plotly_chart(fig_cust, use_container_width=True)
        
    with c4:
        st.subheader("Nguyên tắc Pareto (80/20)")
        pareto_df = df.groupby('khach_hang')['sl'].sum().sort_values(ascending=False).reset_index()
        pareto_df['cum_sl'] = pareto_df['sl'].cumsum()
        pareto_df['cum_perc'] = pareto_df['cum_sl'] / pareto_df['sl'].sum() * 100
        
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(x=pareto_df['khach_hang'], y=pareto_df['sl'], name='Sản lượng'))
        fig_pareto.add_trace(go.Scatter(x=pareto_df['khach_hang'], y=pareto_df['cum_perc'], name='Tích lũy %', yaxis='y2', mode='lines+markers'))
        
        fig_pareto.update_layout(
            yaxis=dict(title='Sản lượng'),
            yaxis2=dict(title='Tỷ lệ tích lũy (%)', overlaying='y', side='right', range=[0, 110]),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_pareto, use_container_width=True)
        
    st.info("💡 **Insight:** Tập trung chăm sóc nhóm khách hàng nằm bên trái đường cong Pareto (chiếm 80% sản lượng) để tối ưu hiệu quả kinh doanh.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(f"**Mộc Phát Furniture Data System** | Cập nhật lúc: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
