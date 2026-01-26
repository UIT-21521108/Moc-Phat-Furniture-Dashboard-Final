
# app.py
# Báo cáo kinh doanh Mộc Phát Furniture – bản brand xanh, logo sticky, diễn giải sau biểu đồ
# Phân tích toàn bộ dữ liệu (kể cả ECOM), bộ lọc mượt, animation, hover chi tiết, Insight & Khu vực cải tiến

import os
import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime

# ========= THIẾT LẬP GIAO DIỆN / BRAND =========
st.set_page_config(page_title="Báo cáo kinh doanh Mộc Phát Furniture", layout="wide")

# Màu thương hiệu (Dartmouth green) + dải màu đồng hành
PRIMARY = "#066839"   # xanh thương hiệu
ACCENT  = "#1B7D4F"   # xanh đậm để nhấn
BRAND_COLORWAY = ["#066839", "#1B7D4F", "#0F5132", "#18392B", "#212224"]

# Bảng màu "đúng màu sơn" (không dùng brand palette)
COLOR_PALETTE = {
    "BROWN":   "#8B5A2B",
    "WHITE":   "#F2F2F2",
    "BLACK":   "#2E2E2E",
    "GREY":    "#9E9E9E",
    "GREEN":   "#2E7D32",
    "BLUE":    "#1565C0",
    "NATURAL": "#C4A484",
    "PINK":    "#E57373",
    "YELLOW":  "#FBC02D",
    "RED":     "#D32F2F",
    "OTHER":   "#BDBDBD"
}

# Plotly template + default màu brand
PLOT_TEMPLATE = 'plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white'
px.defaults.template = PLOT_TEMPLATE
px.defaults.color_discrete_sequence = BRAND_COLORWAY

# CSS hiện đại, nhấn brand + chữ nhỏ in đậm
CUSTOM_CSS = f"""
<style>
:root {{
  --brand: {PRIMARY};
  --brand-2: {ACCENT};
}}
html {{ scroll-behavior: smooth; }}

/* Chữ đậm, dễ đọc */
h1, h2, h3, h4 {{ font-weight: 800 !important; }}
section p, .stMarkdown p, .stCaption {{
  font-weight: 600 !important;
}}

/* KPI card */
.kpi-card {{
  padding: 14px 16px;
  border-radius: 12px;
  border: 1px solid rgba(0,0,0,0.06);
  background: linear-gradient(180deg, rgba(6,104,57,0.08), rgba(6,104,57,0.02));
}}

/* Sidebar tone nhẹ */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(6,104,57,0.08), rgba(6,104,57,0));
  border-right: 1px solid rgba(0,0,0,0.05);
}}

/* Nút/slider/radio theo brand */
.stButton>button, .stDownloadButton>button {{
  background: var(--brand) !important; border: none !important; color: #fff !important;
  font-weight: 700 !important; border-radius: 10px !important;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{ filter: brightness(0.95); }}
.stSlider [role='slider'] {{ border: 2px solid var(--brand) !important; }}
.stSlider .st-dq {{ background: var(--brand) !important; }}
.stRadio [role='radiogroup'] label span {{ font-weight: 700 !important; }}

/* Plotly container bo góc */
.js-plotly-plot {{ border-radius: 10px; }}

/* Bảng DataFrame tiêu đề đậm */
.stDataFrame thead tr th {{ font-weight: 800 !important; }}

/* Header sticky */
.header-sticky {{
  position: sticky; top: 0; z-index: 9999;
  backdrop-filter: blur(6px);
  background: {"#FFFFFF" if PLOT_TEMPLATE=="plotly_white" else "#0E1117"}E6;
  border-bottom: 1px solid rgba(0,0,0,0.08);
  padding: 8px 6px 6px 6px;
}}
.header-wrap {{ display:flex; align-items:center; gap:12px; }}
.header-logo {{ height: 40px; border-radius: 6px; }}
.header-title {{ font-size: 34px; font-weight: 900; line-height: 1.05; margin: 0; }}
.header-tagline {{ color: {PRIMARY}; font-weight: 800; margin-top: 2px; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ========= HÀM HỖ TRỢ =========
@st.cache_data(show_spinner=False)
def load_data(file_or_buffer):
    if file_or_buffer is None:
        return None
    name = getattr(file_or_buffer, 'name', '') if file_or_buffer else ''
    if name.lower().endswith('.csv'):
        df = pd.read_csv(file_or_buffer)
    else:
        df = pd.read_excel(file_or_buffer, sheet_name=0, engine='openpyxl')
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ['sl', 'sl_container', 'month', 'year', 'w', 'd', 'h']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def bucket_color(v: str) -> str:
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
    """Giữ toàn bộ dữ liệu (kể cả ECOM), tạo cờ/nhóm để phân tích."""
    df = df.copy()
    for c in ['khach_hang','ma_hang','mo_ta','mau_son','sl','sl_container','month','year','is_usb']:
        if c not in df.columns: df[c] = np.nan
    df = df.dropna(subset=['year','month','sl'])
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['ym'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=1))

    # Gộp mô tả + màu sơn để dò keyword (UPPER)
    text = (df['mo_ta'].fillna('') + ' ' + df['mau_son'].fillna('')).str.upper()

    # Cờ USB (đặc tính sản phẩm)
    df['usb_flag'] = df.get('is_usb', '').astype(str).str.contains('USB', case=False, na=False) | \
                     df['ma_hang'].fillna('').astype(str).str.contains('USB', case=False, na=False)

    # Khu vực: heuristic NA/EU/Other (giữ cả ECOM)
    kh = df['khach_hang'].fillna('')
    conds = [
        kh.str.contains('TJX EUROPE|TK', case=False, na=False),
        kh.str.contains('TJMAXX|MARSHALL|HOMEGOODS|HOMESENSE|WINNERS|MMX|TJX UK|ECOM', case=False, na=False)
    ]
    df['khu_vuc'] = np.select(conds, ['Châu Âu','Bắc Mỹ'], default='Khác')

    # Tay nắm/phụ kiện (đã sửa str.contains + na=False)
    df['pk_dong_co'] = text.str.contains('ANTIQUE BRASS', na=False, regex=False)
    df['pk_bronze']  = text.str.contains('ANTIQUE BRONZE', na=False, regex=False)
    df['pk_niken']   = text.str.contains('NICKEL', na=False, regex=False)
    df['pk_go']      = text.str.contains('WOOD HARDWARE', na=False, regex=False)

    # Nhóm màu
    df['nhom_mau'] = df['mau_son'].fillna('').apply(bucket_color)
    return df

def add_play_controls(fig, frame_ms=700, transition_ms=300):
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.15, x=1.0, xanchor="right",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": frame_ms, "redraw": True},
                                  "fromcurrent": True,
                                  "transition": {"duration": transition_ms, "easing":"linear"}}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode":"immediate"}]),
            ]
        )]
    )
    return fig

# ====== TIỆN ÍCH DIỄN GIẢI ======
def _fmt(n):
    try: return f"{int(n):,}"
    except: return f"{n:,}"

def explain_trend_monthly(df):
    if df.empty: return
    s = df.groupby('ym')['sl'].sum().sort_index()
    if s.empty: return
    peak_m, peak_v = s.idxmax(), s.max()
    low_m,  low_v  = s.idxmin(), s.min()
    last3 = s.tail(3).sum(); prev3 = s.tail(6).head(3).sum()
    delta3 = None if prev3==0 else (last3-prev3)/prev3*100
    st.markdown(
        f"**📌 Diễn giải:** Tháng cao nhất **{peak_m:%Y-%m}** đạt **{_fmt(peak_v)}**; "
        f"tháng thấp nhất **{low_m:%Y-%m}** **{_fmt(low_v)}**. "
        + (f"3 tháng gần nhất **{_fmt(last3)}** so với 3 tháng trước **{_fmt(prev3)}** "
           f"→ _{'tăng' if delta3 and delta3>0 else 'giảm' if delta3 and delta3<0 else 'ổn định'} "
           f"{'' if delta3 is None else f'{delta3:+.1f}%'}_.")
    )

def explain_color_100(col_df):
    if col_df.empty: return
    x = col_df.groupby(['year','nhom_mau'])['sl'].sum().reset_index()
    x['share'] = x['sl']/x.groupby('year')['sl'].transform('sum')
    txt = []
    for y, g in x.groupby('year'):
        g = g.sort_values('share', ascending=False).head(3)
        tri = ", ".join([f"{r['nhom_mau'].title()} {_fmt(round(r['share']*100,1))}%" for _,r in g.iterrows()])
        txt.append(f"**{y}:** Top màu {tri}.")
    st.markdown("**📌 Diễn giải:** " + " ".join(txt))

def explain_usb_share(df):
    y = sorted(df['year'].unique())
    if not y: return
    shares = {yy: df[df['year']==yy]['usb_flag'].mean() for yy in y}
    ytxt = " • ".join([f"**{yy}**: {shares[yy]*100:.1f}%" for yy in y])
    st.markdown(f"**📌 Diễn giải:** Tỷ lệ sản phẩm có cổng sạc (USB) — {ytxt}.")

def explain_top_list(df, by_col, title):
    if df.empty: return
    df = df.copy()
    df['year_total'] = df.groupby('year')['sl'].transform('sum')
    df['share'] = df['sl']/df['year_total']
    parts = []
    for y, g in df.groupby('year'):
        g = g.sort_values('sl', ascending=False).head(3)
        tri = "; ".join([f"{r[by_col]} ({_fmt(r['sl'])} – {r['share']*100:.1f}%)" for _,r in g.iterrows()])
        parts.append(f"**{y}:** {tri}")
    st.markdown(f"**📌 Diễn giải – {title}:** " + " • ".join(parts))

def explain_pareto(df, by_col):
    if df.empty: return
    s = df.groupby(by_col)['sl'].sum().sort_values(ascending=False)
    cum_share = (s.cumsum()/s.sum()).values
    n80 = int(np.searchsorted(cum_share, 0.8) + 1)
    st.markdown(f"**📌 Diễn giải:** Cần ~**{n80}** {('khách hàng' if by_col=='khach_hang' else 'SKU')} để đạt **80%** tổng sản lượng.")

def explain_region_100(reg_df):
    if reg_df.empty: return
    reg_df = reg_df.copy()
    reg_df['share'] = reg_df['sl']/reg_df.groupby('year')['sl'].transform('sum')
    lines = []
    for y, g in reg_df.groupby('year'):
        tri = "; ".join([f"{r['khu_vuc']} {r['share']*100:.1f}%" for _,r in g.sort_values('share', ascending=False).iterrows()])
        lines.append(f"**{y}:** {tri}")
    st.markdown("**📌 Diễn giải:** " + " • ".join(lines))

def explain_anomaly_forecast(tr_df):
    if tr_df.empty: return
    s = tr_df.set_index('ym')['sl'].sort_index()
    last_v = s.iloc[-1]
    last_3 = s.tail(3).mean()
    st.markdown(
        f"**📌 Diễn giải:** Tháng gần nhất **{_fmt(last_v)}**; "
        f"b/q 3 tháng gần nhất **{_fmt(int(last_3))}**. "
        "Dự đoán dùng EWMA (đường đứt) & trung bình 3 tháng (dấu chấm)."
    )

def apply_filters(base: pd.DataFrame):
    with st.sidebar:
        # Logo ở Sidebar
        try:
            st.image("mocphat_logo.png", use_column_width=True)
        except Exception:
            pass

        st.header("Bộ lọc")

        with st.expander("Thời gian & tuỳ chọn chung", expanded=True):
            years = sorted(base['year'].unique())
            year_sel = st.multiselect("Năm", options=years, default=years, key="flt_years")
            show_explain = st.toggle("🛈 Hiển thị giải thích trên biểu đồ", value=True)
            animate_on   = st.toggle("🎞️ Bật hiệu ứng động (animation)", value=True)

        with st.expander("Khách hàng", expanded=False):
            cust_all = sorted(base['khach_hang'].dropna().unique().tolist())
            default_cust = st.session_state.get("flt_cust_default", cust_all)
            cust_sel = st.multiselect("Chọn khách hàng", options=cust_all, default=default_cust, key="flt_customers")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Chọn tất cả KH"):
                    st.session_state["flt_customers"] = cust_all
                    st.session_state["flt_cust_default"] = cust_all
                    st.rerun()
            with c2:
                if st.button("Bỏ chọn KH"):
                    st.session_state["flt_customers"] = []
                    st.session_state["flt_cust_default"] = []
                    st.rerun()

        with st.expander("Khu vực & Nhóm màu", expanded=False):
            reg_sel   = st.multiselect("Khu vực", options=sorted(base['khu_vuc'].unique()),
                                       default=list(base['khu_vuc'].unique()), key="flt_regions")
            color_sel = st.multiselect("Nhóm màu", options=sorted(base['nhom_mau'].unique()),
                                       default=list(base['nhom_mau'].unique()), key="flt_colors")

        with st.expander("Tìm kiếm sản phẩm & tuỳ chọn khác", expanded=False):
            sku_query = st.text_input("Tìm theo mã sản phẩm (ví dụ: MP, MT001, BRN)", key="flt_sku")
            usb_only  = st.checkbox("Chỉ sản phẩm có cổng sạc (USB)", value=False, key="flt_usb")
            if st.button("🔄 Xoá toàn bộ lọc"):
                for k in ["flt_years","flt_customers","flt_regions","flt_colors","flt_sku","flt_usb","flt_cust_default"]:
                    if k in st.session_state: del st.session_state[k]
                st.rerun()

    f = base[base['year'].isin(year_sel)]
    if cust_sel:  f = f[f['khach_hang'].isin(cust_sel)]
    if reg_sel:   f = f[f['khu_vuc'].isin(reg_sel)]
    if color_sel: f = f[f['nhom_mau'].isin(color_sel)]
    if sku_query:
        q = sku_query.strip().upper()
        f = f[f['ma_hang'].fillna('').str.upper().str.contains(q, na=False)]
    if usb_only:  f = f[f['usb_flag']]
    return f, show_explain, animate_on

def excel_download(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='DATA')
        df.groupby('year')['sl'].sum().reset_index().to_excel(writer, index=False, sheet_name='TONG_NAM')
        df.groupby(['year','nhom_mau'])['sl'].sum().reset_index().to_excel(writer, index=False, sheet_name='MAU_NAM')
        df.groupby(['year','khach_hang'])['sl'].sum().reset_index().to_excel(writer, index=False, sheet_name='KHACH_NAM')
        df.groupby(['year','ma_hang'])['sl'].sum().reset_index().to_excel(writer, index=False, sheet_name='SKU_NAM')
    return output.getvalue()

def add_kpi_cards(df: pd.DataFrame):
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

    c1,c2,c3,c4 = st.columns(4)
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

def anomaly_and_forecast(tr: pd.DataFrame, title_suffix: str=""):
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

def pareto_share(df: pd.DataFrame, by_col: str='khach_hang'):
    if df.empty:
        return pd.DataFrame()
    s = df.groupby(by_col)['sl'].sum().sort_values(ascending=False).reset_index()
    s['cum_units'] = s['sl'].cumsum()
    total = s['sl'].sum()
    s['cum_share'] = s['cum_units']/total if total else 0
    return s

# ========= HEADER STICKY (logo + tiêu đề lớn + tagline mới) =========
def _logo_base64(path="mocphat_logo.png"):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

_logo64 = _logo_base64("mocphat_logo.png")
if _logo64:
    _img_tag = f"<img class='header-logo' src='data:image/png;base64,{_logo64}'/>"
else:
    _img_tag = f"<div class='header-logo' style='width:40px;height:40px;background:{PRIMARY};'></div>"

st.markdown(
    f"""
    <div class="header-sticky">
      <div class="header-wrap">
        {_img_tag}
        <div>
          <div class="header-title">Báo cáo kinh doanh Mộc Phát Furniture</div>
          <div class="header-tagline">Xuất khẩu đồ gỗ nội thất</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ========= NẠP DỮ LIỆU =========
st.caption("Tải file Excel/CSV của bạn hoặc dùng sẵn tệp mặc định nếu có trong thư mục.")

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
f, show_explain, animate_on = apply_filters(base)

# KPI
add_kpi_cards(f)

# ========= TABS =========
T1, T2, T3, T4, T5, T6, T7 = st.tabs([
    "Tổng quan", "Khách hàng", "Sản phẩm (SKU)", "Màu & Tay nắm",
    "Khu vực", "Biến động & Dự đoán", "Insight (Gợi ý vận hành)"
])

# --- TAB 1: Tổng quan ---
with T1:
    st.subheader("Xu hướng theo tháng")
    tr = f.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
    if not tr.empty:
        fig = px.line(tr, x='ym', y='sl')
        fig.update_traces(mode='lines+markers')
        fig.update_layout(xaxis_title="Thời gian (tháng)", yaxis_title="Sản lượng")
        if show_explain:
            fig.update_traces(hovertemplate="Tháng: %{x|%Y-%m}<br>Sản lượng: %{y:,}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, key="t1_trend")
        explain_trend_monthly(f)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tỷ trọng màu theo năm (100%)**")
        color_tot = f.groupby(['year','nhom_mau'])['sl'].sum().reset_index()
        if not color_tot.empty:
            color_tot['share'] = color_tot['sl']/color_tot.groupby('year')['sl'].transform('sum')
            pvt = (color_tot[['year','nhom_mau','share']]
                   .pivot(index='nhom_mau', columns='year', values='share').fillna(0)
                   .reset_index().melt(id_vars='nhom_mau', var_name='Năm', value_name='Tỷ trọng'))
            order = ["BROWN","WHITE","BLACK","GREY","NATURAL","GREEN","BLUE","PINK","YELLOW","RED","OTHER"]
            pvt['nhom_mau'] = pd.Categorical(pvt['nhom_mau'], categories=order, ordered=True)
            pvt = pvt.sort_values(['Năm','nhom_mau'])
            fig = px.bar(pvt, x='Năm', y='Tỷ trọng', color='nhom_mau',
                         barmode='stack', color_discrete_map=COLOR_PALETTE)
            fig.update_yaxes(tickformat=',.0%')
            if show_explain:
                fig.update_traces(hovertemplate="Năm: %{x}<br>Màu: %{legendgroup}<br>Tỷ trọng: %{y:.1%}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True, key="t1_colormix")
            explain_color_100(color_tot)

    with c2:
        st.markdown("**Tỷ lệ sản phẩm có cổng sạc (USB) theo năm**")
        yuniq = sorted(f['year'].unique())
        shares = pd.DataFrame({'Năm': yuniq,'Tỷ lệ USB': [f[f['year']==y]['usb_flag'].mean() for y in yuniq]})
        m_usb = shares.melt(id_vars='Năm', var_name='Chỉ tiêu', value_name='Tỷ lệ')
        fig = px.bar(m_usb, x='Năm', y='Tỷ lệ', color='Chỉ tiêu')
        fig.update_yaxes(tickformat=',.0%')
        if show_explain:
            fig.update_traces(hovertemplate="Năm: %{x}<br>% USB: %{y:.1%}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, key="t1_usbshare")
        explain_usb_share(f)

# --- TAB 2: Khách hàng ---
with T2:
    st.subheader("Khách hàng")
    # Bar‑race theo tháng (Top 10)
    if animate_on:
        topn = 10
        by_m = f.groupby(['ym','khach_hang'])['sl'].sum().reset_index()
        if not by_m.empty:
            by_m['ym_str'] = by_m['ym'].dt.strftime('%Y-%m')
            by_m['month_total'] = by_m.groupby('ym')['sl'].transform('sum')
            by_m['share'] = by_m['sl']/by_m['month_total']
            by_m = by_m.sort_values(['ym','sl'], ascending=[True, False])
            by_m['rank'] = by_m.groupby('ym')['sl'].rank(method='first', ascending=False).astype(int)
            by_m = by_m.groupby('ym').head(topn)
            by_m['label'] = np.where(by_m['rank']==1, " (khách hàng lớn nhất tháng)", "")
            fig = px.bar(by_m, x='sl', y='khach_hang', orientation='h',
                         animation_frame='ym_str', color='khach_hang',
                         title="Bar‑race: Top khách hàng theo từng tháng")
            fig.update_traces(
                hovertemplate="Tháng: %{animation_frame}<br>KH: %{y}%{customdata[2]}<br>Sản lượng: %{x:,}"
                              "<br>Tỷ trọng tháng: %{customdata[0]:.1%}<br>Thứ hạng: %{customdata[1]}<extra></extra>",
                customdata=np.stack([by_m['share'], by_m['rank'], by_m['label']], axis=-1)
            )
            fig.update_layout(xaxis_title="Sản lượng", yaxis_title="Khách hàng")
            fig = add_play_controls(fig, frame_ms=700, transition_ms=300)
            st.plotly_chart(fig, use_container_width=True, key="t2_bar_race_month")

    # Top KH theo năm
    cust_year = f.groupby(['year','khach_hang'])['sl'].sum().reset_index()
    if not cust_year.empty:
        cust_year['year_total'] = cust_year.groupby('year')['sl'].transform('sum')
        cust_year['share'] = cust_year['sl']/cust_year['year_total']
        cust_year = cust_year.sort_values(['year','sl'], ascending=[True, False])
        cust_year['rank'] = cust_year.groupby('year')['sl'].rank(method='first', ascending=False).astype(int)
        cols = st.columns(2)
        for i, y in enumerate(sorted(cust_year['year'].unique())):
            t = cust_year[cust_year['year']==y].head(15).copy()
            t['label'] = np.where(t['rank']==1, " (khách hàng lớn nhất năm)", "")
            fig = px.bar(t, x='khach_hang', y='sl', title=f'Top 15 khách hàng {y}')
            fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_title="Khách hàng", yaxis_title="Sản lượng")
            fig.update_traces(
                hovertemplate=("Năm: " + str(y) + "<br>KH: %{x}%{customdata[1]}"
                               "<br>Sản lượng: %{y:,}<br>Tỷ trọng năm: %{customdata[0]:.1%}"
                               "<br>Thứ hạng: %{customdata[2]}<extra></extra>"),
                customdata=np.stack([t['share'], t['label'], t['rank']], axis=-1)
            )
            cols[i % 2].plotly_chart(fig, use_container_width=True, key=f"t2_topcust_{y}")
        explain_top_list(cust_year, by_col='khach_hang', title='Khách hàng dẫn dắt')

    st.markdown("---")
    st.subheader("Quy tắc 80/20 theo khách hàng")
    pareto = pareto_share(f, 'khach_hang')
    if not pareto.empty:
        fig = px.line(pareto, x=pareto.index+1, y='cum_share', markers=True,
                      title='Tích luỹ tỷ trọng (khách hàng)')
        fig.add_hline(y=0.8, line_dash='dash', line_color=ACCENT)
        fig.update_yaxes(tickformat=',.0%'); fig.update_xaxes(title="Số khách hàng theo thứ hạng")
        fig.update_traces(hovertemplate="Xếp hạng KH: %{x}<br>Tích luỹ tỷ trọng: %{y:.1%}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, key="t2_pareto_cust")
        explain_pareto(f, by_col='khach_hang')

# --- TAB 3: Sản phẩm (SKU) ---
with T3:
    st.subheader("Sản phẩm (SKU)")
    # Bar‑race SKU theo tháng
    if animate_on:
        topn = 15
        by_m_sku = f.groupby(['ym','ma_hang'])['sl'].sum().reset_index()
        if not by_m_sku.empty:
            by_m_sku['ym_str'] = by_m_sku['ym'].dt.strftime('%Y-%m')
            by_m_sku['month_total'] = by_m_sku.groupby('ym')['sl'].transform('sum')
            by_m_sku['share'] = by_m_sku['sl']/by_m_sku['month_total']
            by_m_sku = by_m_sku.sort_values(['ym','sl'], ascending=[True, False])
            by_m_sku['rank'] = by_m_sku.groupby('ym')['sl'].rank(method='first', ascending=False).astype(int)
            by_m_sku = by_m_sku.groupby('ym').head(topn)
            by_m_sku['label'] = np.where(by_m_sku['rank']==1, " (SKU dẫn đầu tháng)", "")
            fig = px.bar(by_m_sku, x='sl', y='ma_hang', orientation='h',
                         animation_frame='ym_str', color='ma_hang',
                         title="Bar‑race: Top SKU theo từng tháng")
            fig.update_traces(
                hovertemplate="Tháng: %{animation_frame}<br>SKU: %{y}%{customdata[2]}<br>Sản lượng: %{x:,}"
                              "<br>Tỷ trọng tháng: %{customdata[0]:.1%}<br>Thứ hạng: %{customdata[1]}<extra></extra>",
                customdata=np.stack([by_m_sku['share'], by_m_sku['rank'], by_m_sku['label']], axis=-1)
            )
            fig.update_layout(xaxis_title="Sản lượng", yaxis_title="SKU")
            fig = add_play_controls(fig, frame_ms=700, transition_ms=300)
            st.plotly_chart(fig, use_container_width=True, key="t3_bar_race_month")

    # Top SKU theo năm
    sku_year = f.groupby(['year','ma_hang'])['sl'].sum().reset_index()
    if not sku_year.empty:
        sku_year['year_total'] = sku_year.groupby('year')['sl'].transform('sum')
        sku_year['share'] = sku_year['sl']/sku_year['year_total']
        sku_year = sku_year.sort_values(['year','sl'], ascending=[True, False])
        sku_year['rank'] = sku_year.groupby('year')['sl'].rank(method='first', ascending=False).astype(int)
        cols = st.columns(2)
        for i, y in enumerate(sorted(sku_year['year'].unique())):
            s = sku_year[sku_year['year']==y].head(20).copy()
            s['label'] = np.where(s['rank']==1, " (SKU dẫn đầu năm)", "")
            fig = px.bar(s, x='ma_hang', y='sl', title=f'Top 20 SKU {y}')
            fig.update_layout(xaxis={'categoryorder':'total descending'}, xaxis_title="SKU", yaxis_title="Sản lượng")
            fig.update_traces(
                hovertemplate=("Năm: " + str(y) + "<br>SKU: %{x}%{customdata[1]}"
                               "<br>Sản lượng: %{y:,}<br>Tỷ trọng năm: %{customdata[0]:.1%}"
                               "<br>Thứ hạng: %{customdata[2]}<extra></extra>"),
                customdata=np.stack([s['share'], s['label'], s['rank']], axis=-1)
            )
            cols[i % 2].plotly_chart(fig, use_container_width=True, key=f"t3_topsku_{y}")
        explain_top_list(sku_year, by_col='ma_hang', title='SKU dẫn dắt')

    st.markdown("---")
    st.subheader("Quy tắc 80/20 theo SKU")
    psku = pareto_share(f, 'ma_hang')
    if not psku.empty:
        fig = px.line(psku, x=psku.index+1, y='cum_share', markers=True,
                      title='Tích luỹ tỷ trọng (SKU)')
        fig.add_hline(y=0.8, line_dash='dash', line_color=ACCENT)
        fig.update_yaxes(tickformat=',.0%'); fig.update_xaxes(title="Số SKU theo thứ hạng")
        fig.update_traces(hovertemplate="Xếp hạng SKU: %{x}<br>Tích luỹ tỷ trọng: %{y:.1%}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, key="t3_pareto_sku")
        explain_pareto(f, by_col='ma_hang')

# --- TAB 4: Màu & Tay nắm ---
with T4:
    st.subheader("Tỷ trọng màu theo năm (100%)")
    col_tot = f.groupby(['year','nhom_mau'])['sl'].sum().reset_index()
    if not col_tot.empty:
        col_tot['share'] = col_tot['sl']/col_tot.groupby('year')['sl'].transform('sum')
        col_tot = col_tot.sort_values(['year','share'], ascending=[True, False])
        fig = px.bar(col_tot, x='year', y='share', color='nhom_mau',
                     barmode='stack', color_discrete_map=COLOR_PALETTE)
        fig.update_yaxes(tickformat=',.0%'); fig.update_layout(legend_title_text="Màu", xaxis_title="Năm", yaxis_title="Tỷ trọng")
        fig.update_traces(hovertemplate="Năm: %{x}<br>Màu: %{legendgroup}<br>Tỷ trọng: %{y:.1%}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, key="t4_colormix")
        explain_color_100(col_tot)

    st.markdown("---")
    st.subheader("Xu hướng sản lượng theo tháng – theo màu")
    trc = f.groupby(['ym','nhom_mau'])['sl'].sum().reset_index()
    if not trc.empty:
        fig = px.line(trc, x='ym', y='sl', color='nhom_mau', color_discrete_map=COLOR_PALETTE)
        fig.update_layout(legend_title_text="Màu", xaxis_title="Thời gian (tháng)", yaxis_title="Sản lượng")
        fig.update_traces(hovertemplate="Tháng: %{x|%Y-%m}<br>Màu: %{legendgroup}<br>Sản lượng: %{y:,}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, key="t4_color_trend")

    st.markdown("---")
    st.subheader("Tay nắm/phụ kiện theo năm")
    pk_any_cols = ['pk_dong_co','pk_bronze','pk_niken','pk_go']
    pk_any = f[pk_any_cols].any(axis=1)
    pk = f[pk_any].groupby('year')[pk_any_cols].mean().reset_index()
    if not pk.empty:
        m = pk.melt(id_vars='year', var_name='Phụ kiện', value_name='Tỷ lệ')
        m['Phụ kiện'] = m['Phụ kiện'].map({'pk_dong_co': 'Đồng cổ','pk_bronze': 'Bronze','pk_niken': 'Niken','pk_go': 'Gỗ'})
        fig = px.bar(m, x='year', y='Tỷ lệ', color='Phụ kiện', barmode='group')
        fig.update_yaxes(tickformat=',.0%'); fig.update_layout(xaxis_title="Năm")
        fig.update_traces(hovertemplate="Năm: %{x}<br>Phụ kiện: %{legendgroup}<br>Tỷ lệ xuất hiện: %{y:.1%}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, key="t4_hardware")

# --- TAB 5: Khu vực (mặc định Cột 100% theo năm) ---
with T5:
    st.subheader("Khu vực")
    view = st.radio("Chọn cách hiển thị",
                    ["Cột 100% theo năm", "Small multiples theo khu vực", "Cột theo quý", "Slope chart 2023→2025"],
                    index=0, horizontal=True, key="t5_view")

    reg = f.groupby(['year','khu_vuc'])['sl'].sum().reset_index()
    if reg.empty:
        st.info("Không có dữ liệu để hiển thị khu vực.")
    else:
        if view == "Cột 100% theo năm":
            reg['share'] = reg['sl']/reg.groupby('year')['sl'].transform('sum')
            fig = px.bar(reg, x='year', y='share', color='khu_vuc', barmode='stack')
            fig.update_yaxes(tickformat=',.0%', title="Tỷ trọng"); fig.update_xaxes(title="Năm")
            fig.update_layout(legend_title_text="Khu vực")
            fig.update_traces(hovertemplate="Năm: %{x}<br>Khu vực: %{legendgroup}<br>Tỷ trọng: %{y:.1%}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True, key="t5_bar_100")
            explain_region_100(reg)

        elif view == "Small multiples theo khu vực":
            m_reg = f.groupby(['ym','khu_vuc'])['sl'].sum().reset_index()
            if m_reg.empty:
                st.info("Không có dữ liệu theo tháng để hiển thị.")
            else:
                fig = px.line(m_reg, x='ym', y='sl', color='khu_vuc', facet_col='khu_vuc', facet_col_wrap=3)
                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                fig.update_xaxes(title="Thời gian (tháng)"); fig.update_yaxes(title="Sản lượng")
                fig.update_traces(mode='lines+markers',
                                  hovertemplate="Tháng: %{x|%Y-%m}<br>Khu vực: %{legendgroup}<br>Sản lượng: %{y:,}<extra></extra>")
                st.plotly_chart(fig, use_container_width=True, key="t5_small_multiples")

        elif view == "Cột theo quý":
            tmp = f.copy()
            tmp['quarter'] = pd.PeriodIndex(pd.to_datetime(dict(year=tmp['year'], month=tmp['month'], day=1)), freq='Q').astype(str)
            q = tmp.groupby(['year','quarter','khu_vuc'])['sl'].sum().reset_index()
            fig = px.bar(q, x='quarter', y='sl', color='khu_vuc', barmode='group')
            fig.update_layout(xaxis_title="Quý (năm-quý)", yaxis_title="Sản lượng", legend_title_text="Khu vực")
            fig.update_traces(hovertemplate="Quý: %{x}<br>Khu vực: %{legendgroup}<br>Sản lượng: %{y:,}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True, key="t5_quarter_grouped")

        elif view == "Slope chart 2023→2025":
            reg_2325 = reg[reg['year'].isin([2023, 2025])].copy()
            if reg_2325.empty or reg_2325['year'].nunique() < 2:
                st.info("Thiếu dữ liệu 2 mốc 2023 và 2025 để vẽ slope chart.")
            else:
                reg_2325['share'] = reg_2325['sl']/reg_2325.groupby('year')['sl'].transform('sum')
                wide = reg_2325.pivot(index='khu_vuc', columns='year', values='share').reset_index()
                long = wide.melt(id_vars='khu_vuc', var_name='Năm', value_name='Tỷ trọng')
                fig = px.line(long, x='Năm', y='Tỷ trọng', color='khu_vuc', markers=True)
                fig.update_yaxes(tickformat=',.0%'); fig.update_layout(legend_title_text="Khu vực")
                fig.update_traces(hovertemplate="Khu vực: %{legendgroup}<br>Năm: %{x}<br>Tỷ trọng: %{y:.1%}<extra></extra>")
                st.plotly_chart(fig, use_container_width=True, key="t5_slope_2325")

# --- TAB 6: Biến động & Dự đoán ---
with T6:
    st.subheader("Biến động & Dự đoán")
    tr_all = f.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
    fig_a, fig_f = anomaly_and_forecast(tr_all, 'Tổng sản lượng')
    if fig_a:
        if show_explain:
            fig_a.update_traces(hovertemplate="Tháng: %{x|%Y-%m}<br>Sản lượng: %{y:,}<extra></extra>")
            fig_f.update_traces(hovertemplate="Thời điểm: %{x|%Y-%m}<br>Giá trị: %{y:,}<extra></extra>")
        st.plotly_chart(fig_a, use_container_width=True, key="t6_anomaly")
        st.plotly_chart(fig_f, use_container_width=True, key="t6_forecast")
        explain_anomaly_forecast(tr_all)

# --- TAB 7: Insight (Gợi ý vận hành) ---
with T7:
    st.subheader("Insight (Gợi ý vận hành)")
    st.caption("Tổng hợp xu hướng theo **mùa–vùng** và **sức khỏe danh mục SKU**, kèm cảnh báo sớm.")

    # ==== A) Mùa–Màu–Vùng ====
    season_map = {12:'Đông',1:'Đông',2:'Đông', 3:'Xuân',4:'Xuân',5:'Xuân', 6:'Hè',7:'Hè',8:'Hè', 9:'Thu',10:'Thu',11:'Thu'}
    g = f.copy(); g['mua'] = g['month'].map(season_map)

    region_pick = st.selectbox("Chọn thị trường để xem heatmap màu theo mùa", ["Bắc Mỹ","Châu Âu"], index=0)
    g2 = g[g['khu_vuc'].isin([region_pick])].groupby(['mua','nhom_mau'])['sl'].sum().reset_index()
    if not g2.empty:
        g2['share'] = g2['sl']/g2.groupby('mua')['sl'].transform('sum')
        heat = g2.pivot(index='mua', columns='nhom_mau', values='share').fillna(0)
        heat = heat.reindex(index=['Xuân','Hè','Thu','Đông'])
        fig = px.imshow(heat, color_continuous_scale='YlGnBu', aspect='auto', origin='lower')
        fig.update_coloraxes(colorbar_title='Tỷ trọng')
        st.plotly_chart(fig, use_container_width=True, key="t7_season_color_heat")
        st.caption("Heatmap cho biết **màu nào trội theo mùa** ở thị trường đã chọn (dựa dữ liệu đang lọc).")

    st.markdown("---")

    # ==== B) SKU Health ====
    first_ym = f.groupby('ma_hang')['ym'].min().rename('first_ym')
    ff = f.join(first_ym, on='ma_hang')
    ff['is_new'] = ff['ym'] == ff['first_ym']

    m = ff.groupby('ym').agg(total_units=('sl','sum'), n_sku=('ma_hang','nunique')).reset_index()
    new_per_month = ff[ff['is_new']].groupby('ym')['ma_hang'].nunique().rename('new_sku_unique').reset_index()
    m = m.merge(new_per_month, on='ym', how='left').fillna({'new_sku_unique':0})
    m['new_sku_share'] = m['new_sku_unique']/m['n_sku']
    m['units_per_sku'] = m['total_units']/m['n_sku']
    hh = ff.groupby(['ym','ma_hang'])['sl'].sum().reset_index()
    hh['share'] = hh.groupby('ym')['sl'].transform(lambda s: s/s.sum())
    hhi = hh.groupby('ym')['share'].apply(lambda s: (s**2).sum()).rename('hhi').reset_index()
    m = m.merge(hhi, on='ym', how='left')

    c1, c2 = st.columns([2,1])
    with c1:
        threshold = st.slider("Ngưỡng cảnh báo tỷ lệ SKU mới", min_value=0, max_value=100, value=30, step=5,
                              help="Nếu tỷ lệ SKU mới trong tháng vượt ngưỡng này thì cảnh báo.", key="t7_thr")
        thr = threshold/100.0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m['ym'], y=m['new_sku_share'], mode='lines+markers', name='% SKU mới'))
        fig.add_hline(y=thr, line_dash='dash', line_color=ACCENT, annotation_text=f"Ngưỡng {threshold}%")
        fig.update_layout(title="% SKU mới theo tháng", xaxis_title="Tháng", yaxis_title="% SKU mới", template=PLOT_TEMPLATE)
        fig.update_yaxes(tickformat=',.0%')
        st.plotly_chart(fig, use_container_width=True, key="t7_newsku_share_line")
    with c2:
        y = ff.groupby('year').agg(total_units=('sl','sum'),
                                   n_sku=('ma_hang','nunique')).reset_index()
        new_per_year = ff[ff['is_new']].groupby('year')['ma_hang'].nunique().rename('new_sku_unique').reset_index()
        y = y.merge(new_per_year, on='year', how='left').fillna({'new_sku_unique':0})
        y['units_per_sku'] = y['total_units']/y['n_sku']
        yhh = ff.groupby(['year','ma_hang'])['sl'].sum().reset_index()
        yhh['share'] = yhh.groupby('year')['sl'].transform(lambda s: s/s.sum())
        hhi_y = yhh.groupby('year')['share'].apply(lambda s: (s**2).sum()).rename('hhi').reset_index()
        y = y.merge(hhi_y, on='year', how='left')
        st.write("**Tóm tắt theo năm**")
        for _, r in y.sort_values('year').iterrows():
            st.markdown(f"- **{int(r['year'])}** · SKU hoạt động: **{int(r['n_sku'])}**, SKU mới: **{int(r['new_sku_unique'])}**, "
                        f"Units/SKU: **{r['units_per_sku']:.0f}**, HHI: **{r['hhi']:.3f}**")

    risky = m[m['new_sku_share']>thr].copy()
    if not risky.empty:
        st.error("⚠️ Tháng có **tỷ lệ SKU mới** vượt ngưỡng:")
        for _, r in risky.sort_values('ym').iterrows():
            st.markdown(f"- {r['ym'].strftime('%Y-%m')}: %SKU mới **{r['new_sku_share']:.0%}**, "
                        f"SKU hoạt động **{int(r['n_sku'])}**, SKU mới **{int(r['new_sku_unique'])}**")

    st.markdown("---")
    st.write("**Tải dữ liệu đã lọc**")
    st.download_button("⬇️ CSV", data=f.to_csv(index=False).encode('utf-8-sig'),
                       file_name='filtered.csv', mime='text/csv', key="dl_csv_insight")
    st.download_button("⬇️ Excel (DATA + tổng hợp)",
                       data=excel_download(f),
                       file_name='filtered.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                       key="dl_xlsx_insight")
