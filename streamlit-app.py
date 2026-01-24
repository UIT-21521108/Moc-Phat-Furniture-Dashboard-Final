
# app.py
# Báo cáo kinh doanh Mộc Phát Furniture (2023–2025)
# Tác giả: M365 Copilot cho Nguyễn Minh Lý

import os
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime

# Ag-Grid (bảng dữ liệu tương tác) - có thể không bắt buộc
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

# -------------------------
# THIẾT LẬP GIAO DIỆN
# -------------------------
st.set_page_config(page_title="Báo cáo kinh doanh Mộc Phát Furniture", layout="wide")

PRIMARY = "#00B8A9"
ACCENT  = "#F6416C"

# Bảng màu hiển thị giống màu thật cho các nhóm màu
COLOR_PALETTE = {
    "BROWN":   "#8B5A2B",  # nâu
    "WHITE":   "#F2F2F2",  # trắng (xám rất nhạt để thấy trên nền sáng/tối)
    "BLACK":   "#2E2E2E",  # đen
    "GREY":    "#9E9E9E",  # xám
    "GREEN":   "#2E7D32",  # xanh lá
    "BLUE":    "#1565C0",  # xanh dương
    "NATURAL": "#C4A484",  # gỗ tự nhiên/honey oak
    "PINK":    "#E57373",  # hồng
    "YELLOW":  "#FBC02D",  # vàng
    "RED":     "#D32F2F",  # đỏ
    "OTHER":   "#BDBDBD"   # khác
}

CUSTOM_CSS = """
<style>
h1, h2, h3, h4 { font-weight: 700 !important; }
.kpi-card {
  padding: 12px 16px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOT_TEMPLATE = 'plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white'

# -------------------------
# HÀM HỖ TRỢ
# -------------------------
@st.cache_data(show_spinner=False)
def load_data(file_or_buffer):
    """Đọc Excel/CSV, chuẩn hoá tên cột & kiểu dữ liệu."""
    if file_or_buffer is None:
        return None
    name = getattr(file_or_buffer, 'name', '') if file_or_buffer else ''
    if name.lower().endswith('.csv'):
        df = pd.read_csv(file_or_buffer)
    else:
        df = pd.read_excel(file_or_buffer, sheet_name=0, engine='openpyxl')
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ['sl','sl_container','month','year','w','d','h']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def bucket_color(v: str) -> str:
    """Gộp tên màu chi tiết vào nhóm màu chính để dễ xem."""
    v = str(v).upper()
    if ('BROWN' in v) or ('COCOA' in v) or ('BRONZE' in v) or ('UMBER' in v): return 'BROWN'
    if any(x in v for x in ['WHITE','OFF WHITE','WHT','IVORY','CREAM','GLOSS']): return 'WHITE'
    if 'BLACK' in v:  return 'BLACK'
    if ('GREY' in v) or ('GRAY' in v): return 'GREY'
    if any(x in v for x in ['GREEN','SAGE','KALE','OLIVE']): return 'GREEN'
    if ('NAVY' in v) or ('BLUE' in v): return 'BLUE'
    if any(x in v for x in ['NAT','OAK','WALNUT','HONEY','TEAK']): return 'NATURAL'
    if ('PINK' in v) or ('BLUSH' in v): return 'PINK'
    if ('YELL' in v) or ('MUSTARD' in v): return 'YELLOW'
    if 'RED' in v: return 'RED'
    return 'OTHER'

def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch, tạo các cờ và nhóm màu."""
    df = df.copy()
    for c in ['khach_hang','ma_hang','mo_ta','mau_son','sl','sl_container','month','year','is_usb']:
        if c not in df.columns: df[c] = np.nan
    df = df.dropna(subset=['year','month','sl'])
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['ym'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=1))

    text = (df['mo_ta'].fillna('') + ' ' + df['mau_son'].fillna('')).str.upper()
    df['usb_flag'] = df.get('is_usb', '').astype(str).str.contains('USB', case=False) | df['ma_hang'].fillna('').astype(str).str.contains('USB', case=False)
    df['ecom_flag'] = df['khach_hang'].fillna('').str.contains('ECOM', case=False)

    kh = df['khach_hang'].fillna('')
    conds = [
        kh.str.contains('TJX EUROPE|TK', case=False),
        kh.str.contains('TJMAXX|MARSHALL|HOMEGOODS|HOMESENSE|WINNERS|MMX|TJX UK|ECOM', case=False)
    ]
    df['region'] = np.select(conds, ['Europe','Bắc Mỹ'], default='Khác')

    df['hw_antique_brass']  = text.str.contains('ANTIQUE BRASS')
    df['hw_antique_bronze'] = text.str.contains('ANTIQUE BRONZE')
    df['hw_nickel']         = text.str.contains('NICKEL')
    df['hw_wood']           = text.str.contains('WOOD HARDWARE')

    df['color_bucket'] = df['mau_son'].fillna('').apply(bucket_color)
    return df

def apply_filters(base: pd.DataFrame) -> pd.DataFrame:
    """Bộ lọc bên trái."""
    with st.sidebar:
        st.header("Bộ lọc")
        years = sorted(base['year'].unique())
        year_sel = st.multiselect("Năm", options=years, default=years)
        cust_all = sorted(base['khach_hang'].dropna().unique().tolist())
        cust_sel = st.multiselect("Khách hàng", options=cust_all, default=cust_all[:10])
        reg_sel = st.multiselect("Khu vực", options=sorted(base['region'].unique()), default=list(base['region'].unique()))
        color_sel = st.multiselect("Nhóm màu", options=sorted(base['color_bucket'].unique()), default=list(base['color_bucket'].unique()))
        sku_query = st.text_input("Tìm theo mã sản phẩm (ví dụ: MP, MT001, BRN)")
        usb_only = st.checkbox("Chỉ sản phẩm có cổng sạc (USB)", value=False)
        ecom_only = st.checkbox("Chỉ đơn kênh Online", value=False)

    f = base[base['year'].isin(year_sel)]
    if cust_sel:  f = f[f['khach_hang'].isin(cust_sel)]
    if reg_sel:   f = f[f['region'].isin(reg_sel)]
    if color_sel: f = f[f['color_bucket'].isin(color_sel)]
    if sku_query:
        q = sku_query.strip().upper()
        f = f[f['ma_hang'].fillna('').str.upper().str.contains(q)]
    if usb_only:  f = f[f['usb_flag']]
    if ecom_only: f = f[f['ecom_flag']]
    return f

def excel_download(df: pd.DataFrame) -> bytes:
    """Xuất Excel dữ liệu đã lọc + vài bảng tóm tắt."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='DATA')
        df.groupby('year')['sl'].sum().reset_index().to_excel(writer, index=False, sheet_name='TONG_NAM')
        df.groupby(['year','color_bucket'])['sl'].sum().reset_index().to_excel(writer, index=False, sheet_name='MAU_NAM')
        df.groupby(['year','khach_hang'])['sl'].sum().reset_index().to_excel(writer, index=False, sheet_name='KHACH_NAM')
        df.groupby(['year','ma_hang'])['sl'].sum().reset_index().to_excel(writer, index=False, sheet_name='SKU_NAM')
    return output.getvalue()

def add_kpi_cards(df: pd.DataFrame):
    """Thẻ KPI ngắn gọn, dễ hiểu."""
    by_year = df.groupby('year')['sl'].sum().sort_index()
    t23, t24, t25 = [by_year.get(y, 0) for y in [2023, 2024, 2025]]
    yoy24 = (t24 - t23)/t23*100 if t23 else np.nan
    yoy25 = (t25 - t24)/t24*100 if t24 else np.nan

    last_ym = df['ym'].max() if not df.empty else None
    ytd, pytd, ytd_g = 0, 0, np.nan
    if last_ym is not None:
        y, m = last_ym.year, last_ym.month
        ytd  = df[(df['year']==y)   & (df['month']<=m)]['sl'].sum()
        pytd = df[(df['year']==y-1) & (df['month']<=m)]['sl'].sum()
        ytd_g = (ytd - pytd)/pytd*100 if pytd else np.nan

    usb_share = df.groupby('year')['usb_flag'].mean()
    ecom_share = df.groupby('year')['ecom_flag'].mean()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("Tổng sản lượng 2023", f"{int(t23):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("Tổng sản lượng 2024", f"{int(t24):,}", f"{yoy24:+.1f}% so với 2023" if not np.isnan(yoy24) else None)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("Tổng sản lượng 2025", f"{int(t25):,}", f"{yoy25:+.1f}% so với 2024" if not np.isnan(yoy25) else None)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("Lũy kế đến tháng gần nhất vs cùng kỳ", f"{int(ytd):,}", f"{ytd_g:+.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("Tỷ lệ sản phẩm có USB (2025)", f"{usb_share.get(2025,0)*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.metric("Tỷ lệ đơn kênh Online (2025)", f"{ecom_share.get(2025,0)*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

def anomaly_and_forecast(tr: pd.DataFrame, title_suffix: str=""):
    """Điểm bất thường (±2σ) + dự đoán 3 tháng (đơn giản)."""
    if tr.empty:
        return None, None
    s = tr.set_index('ym')['sl'].sort_index()
    roll = s.rolling(3, min_periods=2)
    mean = roll.mean(); std = roll.std().fillna(0)
    z = (s - mean)/std.replace(0, np.nan)
    anomalies = z.abs() > 2

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines+markers', name='Sản lượng'))
    fig1.add_trace(go.Scatter(x=s.index[anomalies], y=s[anomalies], mode='markers',
                              name='Bất thường', marker=dict(color=ACCENT, size=10)))
    fig1.update_layout(template=PLOT_TEMPLATE, title=f"Điểm bất thường (±2σ){' – ' + title_suffix if title_suffix else ''}",
                       xaxis_title="Thời gian (tháng)", yaxis_title="Sản lượng")

    # Dự đoán đơn giản: EWMA(span=3) + trung bình 3 tháng gần nhất
    span = 3
    ewma = s.ewm(span=span, adjust=False).mean()
    last3 = s.tail(3).mean() if len(s) >= 3 else s.mean()
    future_x = pd.date_range(s.index.max() + pd.offsets.MonthBegin(1), periods=3, freq='MS')
    f_ewma  = [ewma.iloc[-1]]*3
    f_naive = [last3]*3

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines', name='Lịch sử'))
    fig2.add_trace(go.Scatter(x=s.index, y=ewma.values, mode='lines', name=f"Đường mượt (EWMA {span})"))
    fig2.add_trace(go.Scatter(x=future_x, y=f_ewma, mode='lines+markers', name='Dự đoán (EWMA)', line=dict(dash='dash')))
    fig2.add_trace(go.Scatter(x=future_x, y=f_naive, mode='lines+markers', name='Dự đoán (TB 3 tháng)', line=dict(dash='dot')))
    fig2.update_layout(template=PLOT_TEMPLATE, title=f"Dự đoán 3 tháng{' – ' + title_suffix if title_suffix else ''}",
                       xaxis_title="Thời gian (tháng)", yaxis_title="Sản lượng")
    return fig1, fig2

def customer_segmentation(df: pd.DataFrame):
    """Phân nhóm KH đơn giản theo: gần đây - tần suất - tổng SL."""
    if df.empty:
        return pd.DataFrame()
    last_ym = df['ym'].max()
    grp = df.groupby('khach_hang').agg(
        units=('sl','sum'),
        months_active=('ym', lambda x: x.nunique()),
        last_purchase=('ym','max'),
    ).reset_index()
    grp['recency_m']   = ((last_ym - grp['last_purchase'])/np.timedelta64(1,'M')).round(1)  # số tháng từ lần mua gần nhất
    grp['frequency_m'] = grp['months_active']         # số tháng có đơn
    grp['monetary_u']  = grp['units']                 # tổng sản lượng
    r_cut = grp['recency_m'].median()
    f_cut = grp['frequency_m'].median()
    grp['segment'] = np.select([
        (grp['recency_m']<=r_cut) & (grp['frequency_m']>f_cut),
        (grp['recency_m']<=r_cut) & (grp['frequency_m']<=f_cut),
        (grp['recency_m']>r_cut)  & (grp['frequency_m']>f_cut),
    ], ['Nòng cốt','Ổn định','Có nguy cơ rời'], default='Ngủ đông')
    return grp

def pareto_share(df: pd.DataFrame, by_col: str='khach_hang'):
    """Bảng tích luỹ 80/20 theo khách hàng hoặc SKU."""
    if df.empty:
        return pd.DataFrame()
    s = df.groupby(by_col)['sl'].sum().sort_values(ascending=False).reset_index()
    s['cum_units'] = s['sl'].cumsum()
    total = s['sl'].sum()
    s['cum_share'] = s['cum_units']/total if total else 0
    return s

# -------------------------
# NGUỒN DỮ LIỆU (UPLOAD hoặc file mặc định)
# -------------------------
st.title("📊 Báo cáo kinh doanh Mộc Phát Furniture")
st.caption("Tải file Excel/CSV của bạn hoặc dùng sẵn tệp mặc định nếu có trong thư mục.")

# Giải thích nhanh (ngôn ngữ đời thường)
with st.expander("Giải thích nhanh các khái niệm (1 phút)"):
    st.markdown("""
- **Sản lượng**: số đơn vị giao hàng.  
- **Tỷ trọng**: phần trăm của từng nhóm trong tổng.  
- **Đơn kênh Online**: đơn qua kênh thương mại điện tử.  
- **Tay nắm/phụ kiện**: *đồng cổ = antique brass; niken = nickel; gỗ = wood hardware*.  
- **Giữ chân khách**: tỷ lệ khách quay lại theo từng tháng sau.
""")

with st.sidebar:
    st.header("Nguồn dữ liệu")
    up = st.file_uploader("Chọn tệp .xlsx / .csv", type=["xlsx","csv"])
    default_path = 'Master_2023_2025_PRO_clean.xlsx'
    if up is None and os.path.exists(default_path):
        st.info("Đang dùng tệp mặc định: Master_2023_2025_PRO_clean.xlsx")
        up = open(default_path, 'rb')
    if up is None:
        st.stop()

raw = load_data(up)
if raw is None or raw.empty:
    st.warning("Không đọc được dữ liệu. Vui lòng kiểm tra file.")
    st.stop()

base = prep_data(raw)
f = apply_filters(base)

# Thẻ KPI
add_kpi_cards(f)

# -------------------------
# CÁC TAB (dễ hiểu, từ cơ bản → nâng cao)
# -------------------------
T1, T2, T3, T4, T5, T6, T7, T8 = st.tabs([
    "Tổng quan", "Khách hàng", "Sản phẩm (SKU)", "Màu & Tay nắm",
    "Khu vực", "Container", "Biến động & Dự đoán", "Giữ chân & Dữ liệu"
])

# --- TAB 1: Tổng quan ---
with T1:
    st.subheader("Xu hướng theo tháng")
    tr = f.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
    if not tr.empty:
        fig = px.line(tr, x='ym', y='sl', template=PLOT_TEMPLATE)
        fig.update_traces(mode='lines+markers')
        fig.update_layout(xaxis_title="Thời gian (tháng)", yaxis_title="Sản lượng")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tỷ trọng màu theo năm (100%)**")
        color_tot = f.groupby(['year','color_bucket'])['sl'].sum().reset_index()
        if not color_tot.empty:
            pvt = color_tot.pivot(index='color_bucket', columns='year', values='sl').fillna(0)
            pvt = pvt.div(pvt.sum(axis=0), axis=1).reset_index().melt(id_vars='color_bucket', var_name='year', value_name='share')
            order = ["BROWN","WHITE","BLACK","GREY","NATURAL","GREEN","BLUE","PINK","YELLOW","RED","OTHER"]
            pvt['color_bucket'] = pd.Categorical(pvt['color_bucket'], categories=order, ordered=True)
            pvt = pvt.sort_values(['year','color_bucket'])
            fig = px.bar(
                pvt, x='year', y='share', color='color_bucket', barmode='stack',
                template=PLOT_TEMPLATE, color_discrete_map=COLOR_PALETTE
            )
            fig.update_yaxes(tickformat=',.0%', title="Tỷ trọng")
            fig.update_xaxes(title="Năm")
            fig.update_layout(legend_title_text="Màu")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Tỷ lệ USB & đơn kênh Online theo năm**")
        yuniq = sorted(f['year'].unique())
        shares = pd.DataFrame({
            'year': yuniq,
            'USB':  [f[f['year']==y]['usb_flag'].mean()  for y in yuniq],
            'Online': [f[f['year']==y]['ecom_flag'].mean() for y in yuniq]
        })
        m = shares.melt(id_vars='year', var_name='Chỉ tiêu', value_name='Tỷ lệ')
        fig = px.bar(m, x='year', y='Tỷ lệ', color='Chỉ tiêu', barmode='group', template=PLOT_TEMPLATE)
        fig.update_yaxes(tickformat=',.0%')
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: Khách hàng ---
with T2:
    st.subheader("Top khách hàng theo năm")
    cust_year = f.groupby(['year','khach_hang'])['sl'].sum().reset_index()
    cols = st.columns(2)
    for i, y in enumerate(sorted(cust_year['year'].unique())):
        t = cust_year[cust_year['year']==y].sort_values('sl', ascending=False).head(15)
        fig = px.bar(t, x='khach_hang', y='sl', title=f'Top 15 khách hàng {y}', template=PLOT_TEMPLATE)
        fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_title="Khách hàng", yaxis_title="Sản lượng")
        cols[i % 2].plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Quy tắc 80/20 theo khách hàng")
    pareto = pareto_share(f, 'khach_hang')
    if not pareto.empty:
        fig = px.line(pareto, x=pareto.index+1, y='cum_share', markers=True, title='Tích luỹ tỷ trọng (khách hàng)', template=PLOT_TEMPLATE)
        fig.add_hline(y=0.8, line_dash='dash', line_color=ACCENT)
        fig.update_yaxes(tickformat=',.0%')
        fig.update_xaxes(title="Số khách hàng theo thứ hạng", range=[1, max(5, len(pareto))])
        st.plotly_chart(fig, use_container_width=True)

    seg = customer_segmentation(f)
    if not seg.empty:
        c1, c2 = st.columns([2,1])
        with c1:
            fig = px.scatter(seg, x='recency_m', y='frequency_m', size='monetary_u', color='segment',
                             hover_data=['khach_hang','monetary_u'], template=PLOT_TEMPLATE)
            fig.update_layout(title='Phân nhóm khách hàng (gần đây – tần suất, kích thước = sản lượng)',
                              xaxis_title="Bao lâu rồi chưa mua (tháng)", yaxis_title="Số tháng có đơn")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(seg[['khach_hang','segment','recency_m','frequency_m','monetary_u']].sort_values('monetary_u', ascending=False).head(20))

# --- TAB 3: Sản phẩm (SKU) ---
with T3:
    st.subheader("Top SKU theo năm")
    sku_year = f.groupby(['year','ma_hang'])['sl'].sum().reset_index()
    cols = st.columns(2)
    for i, y in enumerate(sorted(sku_year['year'].unique())):
        s = sku_year[sku_year['year']==y].sort_values('sl', ascending=False).head(20)
        fig = px.bar(s, x='ma_hang', y='sl', title=f'Top 20 SKU {y}', template=PLOT_TEMPLATE)
        fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_title="SKU", yaxis_title="Sản lượng")
        cols[i % 2].plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Quy tắc 80/20 theo SKU")
    psku = pareto_share(f, 'ma_hang')
    if not psku.empty:
        fig = px.line(psku, x=psku.index+1, y='cum_share', markers=True, title='Tích luỹ tỷ trọng (SKU)', template=PLOT_TEMPLATE)
        fig.add_hline(y=0.8, line_dash='dash', line_color=ACCENT)
        fig.update_yaxes(tickformat=',.0%')
        fig.update_xaxes(title="Số SKU theo thứ hạng")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: Màu & Tay nắm ---
with T4:
    st.subheader("Tỷ trọng màu theo năm (100%)")
    color_tot = f.groupby(['year','color_bucket'])['sl'].sum().reset_index()
    if not color_tot.empty:
        pvt = color_tot.pivot(index='color_bucket', columns='year', values='sl').fillna(0)
        pvt = pvt.div(pvt.sum(axis=0), axis=1).reset_index().melt(id_vars='color_bucket', var_name='year', value_name='share')
        order = ["BROWN","WHITE","BLACK","GREY","NATURAL","GREEN","BLUE","PINK","YELLOW","RED","OTHER"]
        pvt['color_bucket'] = pd.Categorical(pvt['color_bucket'], categories=order, ordered=True)
        pvt = pvt.sort_values(['year','color_bucket'])
        fig = px.bar(
            pvt, x='year', y='share', color='color_bucket', barmode='stack',
            template=PLOT_TEMPLATE, color_discrete_map=COLOR_PALETTE
        )
        fig.update_yaxes(tickformat=',.0%', title="Tỷ trọng")
        fig.update_xaxes(title="Năm")
        fig.update_layout(legend_title_text="Màu")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Xu hướng sản lượng theo tháng – theo màu")
    trc = f.groupby(['ym','color_bucket'])['sl'].sum().reset_index()
    if not trc.empty:
        fig = px.line(trc, x='ym', y='sl', color='color_bucket', template=PLOT_TEMPLATE,
                      color_discrete_map=COLOR_PALETTE)
        fig.update_layout(legend_title_text="Màu", xaxis_title="Thời gian (tháng)", yaxis_title="Sản lượng")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Tay nắm/phụ kiện theo năm")
    hw_any = f[['hw_antique_brass','hw_antique_bronze','hw_nickel','hw_wood']].any(axis=1)
    hw = f[hw_any].groupby('year')[['hw_antique_brass','hw_antique_bronze','hw_nickel','hw_wood']].mean().reset_index()
    if not hw.empty:
        m = hw.melt(id_vars='year', var_name='Phụ kiện', value_name='Tỷ lệ')
        m['Phụ kiện'] = m['Phụ kiện'].map({
            'hw_antique_brass': 'Đồng cổ (antique brass)',
            'hw_antique_bronze': 'Đồng xước/bronze',
            'hw_nickel': 'Niken (nickel)',
            'hw_wood': 'Gỗ (wood hardware)'
        })
        fig = px.bar(m, x='year', y='Tỷ lệ', color='Phụ kiện', barmode='group', template=PLOT_TEMPLATE)
        fig.update_yaxes(tickformat=',.0%')
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 5: Khu vực ---
with T5:
    st.subheader("Tỷ trọng theo khu vực (năm)")
    reg = f.groupby(['year','region'])['sl'].sum().reset_index()
    if not reg.empty:
        pvt = reg.pivot(index='region', columns='year', values='sl').fillna(0)
        pvt = pvt.div(pvt.sum(axis=0), axis=1).reset_index().melt(id_vars='region', var_name='year', value_name='share')
        fig = px.bar(pvt, x='year', y='share', color='region', barmode='group', template=PLOT_TEMPLATE)
        fig.update_yaxes(tickformat=',.0%')
        fig.update_layout(legend_title_text="Khu vực", xaxis_title="Năm", yaxis_title="Tỷ trọng")
        st.plotly_chart(fig, use_container_width=True)

    tre = f.groupby(['ym','region'])['sl'].sum().reset_index()
    if not tre.empty:
        fig = px.area(tre, x='ym', y='sl', color='region', template=PLOT_TEMPLATE)
        fig.update_layout(legend_title_text="Khu vực", xaxis_title="Thời gian (tháng)", yaxis_title="Sản lượng")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 6: Container ---
with T6:
    st.subheader("Hiệu suất container & mô phỏng tiết kiệm")
    cont = f.dropna(subset=['sl_container']).copy()
    if not cont.empty:
        cont['units_per_container'] = cont['sl']/cont['sl_container']
        agg = cont.groupby('khach_hang')['units_per_container'].mean().sort_values(ascending=False).head(15).reset_index()
        fig = px.bar(agg, x='khach_hang', y='units_per_container', title='Top 15 khách hàng theo units/container', template=PLOT_TEMPLATE)
        fig.update_layout(xaxis_title="Khách hàng", yaxis_title="Đơn vị/container")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Mô phỏng:** đặt mục tiêu *đơn vị/container* để ước tính **số container tiết kiệm**")
        c1, c2 = st.columns(2)
        with c1:
            target = st.slider("Mục tiêu (đơn vị/container)", min_value=int(agg['units_per_container'].min()),
                               max_value=int(agg['units_per_container'].max())+200,
                               value=int(agg['units_per_container'].mean()))
        with c2:
            total_units = int(cont['sl'].sum())
            current_avg = cont['units_per_container'].mean()
            cur_conts = total_units/current_avg if current_avg else 0
            new_conts = total_units/target if target else 0
            saved = cur_conts - new_conts
            st.metric("Container ước tính tiết kiệm", f"{saved:.1f}")
    else:
        st.info("Không có dữ liệu container trong tập đang lọc.")

# --- TAB 7: Biến động & Dự đoán ---
with T7:
    st.subheader("Biến động & Dự đoán")
    tr_all = f.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
    fig_a, fig_f = anomaly_and_forecast(tr_all, 'Tổng sản lượng')
    if fig_a:
        st.plotly_chart(fig_a, use_container_width=True)
        st.plotly_chart(fig_f, use_container_width=True)

# --- TAB 8: Giữ chân & Dữ liệu ---
with T8:
    st.subheader("Giữ chân khách theo tháng đầu tiên mua")
    temp = f[['khach_hang','ym']].drop_duplicates()
    if not temp.empty:
        first = temp.groupby('khach_hang')['ym'].min().rename('cohort')
        temp = temp.join(first, on='khach_hang')
        temp['period'] = ((temp['ym'].dt.year - temp['cohort'].dt.year)*12 + (temp['ym'].dt.month - temp['cohort'].dt.month))
        coh = temp.groupby(['cohort','period'])['khach_hang'].nunique().reset_index(name='active_cust')
        base_count = coh[coh['period']==0][['cohort','active_cust']].rename(columns={'active_cust':'base'})
        coh = coh.merge(base_count, on='cohort')
        coh['retention'] = coh['active_cust']/coh['base']
        heat = coh.pivot(index='cohort', columns='period', values='retention').fillna(0)
        fig = px.imshow(heat, color_continuous_scale='Greens', aspect='auto', origin='lower', template=PLOT_TEMPLATE)
        fig.update_coloraxes(colorbar_title='Giữ chân')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Bảng dữ liệu (tương tác)")
    if AGGRID_AVAILABLE:
        gb = GridOptionsBuilder.from_dataframe(f)
        gb.configure_default_column(resizable=True, sortable=True, filter=True)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        grid_options = gb.build()
        AgGrid(f, gridOptions=grid_options,
               columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
               height=420)
    else:
        st.dataframe(f)

# Tải dữ liệu đã lọc
st.markdown("---")
colx, coly = st.columns([2,1])
with colx:
    st.write("**Tải dữ liệu đã lọc**")
    st.download_button("⬇️ CSV", data=f.to_csv(index=False).encode('utf-8-sig'),
                       file_name='filtered.csv', mime='text/csv')
    st.download_button("⬇️ Excel (DATA + tổng hợp)",
                       data=excel_download(f),
                       file_name='filtered.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
with coly:
    st.caption(f"Cập nhật: {datetime.now().strftime('%Y-%m-%d %H:%M')} • Giao diện: {PLOT_TEMPLATE}")
``
