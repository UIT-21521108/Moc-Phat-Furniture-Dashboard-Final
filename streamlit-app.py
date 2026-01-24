# app.py
# Streamlit Dashboard for Master_2023_2025_PRO_clean.xlsx
# Author: M365 Copilot for Nguyễn Minh Lý

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import os

st.set_page_config(page_title="M3Y Dashboard 2023–2025", layout="wide")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_data(file_or_buffer):
    """Load Excel/CSV and standardize columns."""
    if file_or_buffer is None:
        return None
    name = getattr(file_or_buffer, 'name', '') if file_or_buffer else ''
    if name.lower().endswith('.csv'):
        df = pd.read_csv(file_or_buffer)
    else:
        df = pd.read_excel(file_or_buffer, sheet_name=0, engine='openpyxl')
    df.columns = [str(c).strip().lower() for c in df.columns]
    # normalize numeric
    for c in ['sl','sl_container','month','year','w','d','h']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def bucket_color(v: str) -> str:
    v = str(v).upper()
    if ('BROWN' in v) or ('COCOA' in v) or ('BRONZE' in v) or ('UMBER' in v):
        return 'BROWN'
    if any(x in v for x in ['WHITE','OFF WHITE','WHT','IVORY','CREAM','GLOSS']):
        return 'WHITE'
    if 'BLACK' in v:
        return 'BLACK'
    if ('GREY' in v) or ('GRAY' in v):
        return 'GREY'
    if any(x in v for x in ['GREEN','SAGE','KALE','OLIVE']):
        return 'GREEN'
    if ('NAVY' in v) or ('BLUE' in v):
        return 'BLUE'
    if any(x in v for x in ['NAT','OAK','WALNUT','HONEY','TEAK']):
        return 'NATURAL'
    if ('PINK' in v) or ('BLUSH' in v):
        return 'PINK'
    if ('YELL' in v) or ('MUSTARD' in v):
        return 'YELLOW'
    if 'RED' in v:
        return 'RED'
    return 'OTHER'


def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Keep minimal useful columns
    for c in ['khach_hang','ma_hang','mo_ta','mau_son','sl','sl_container','month','year','is_usb']:
        if c not in df.columns:
            df[c] = np.nan
    df = df.dropna(subset=['year','month','sl'])
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['ym'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=1))

    text = (df['mo_ta'].fillna('') + ' ' + df['mau_son'].fillna('')).str.upper()
    df['usb_flag'] = df.get('is_usb', '').astype(str).str.contains('USB', case=False) | df['ma_hang'].fillna('').astype(str).str.contains('USB', case=False)
    df['ecom_flag'] = df['khach_hang'].fillna('').str.contains('ECOM', case=False)

    kh = df['khach_hang'].fillna('')
    conds = [kh.str.contains('TJX EUROPE|TK', case=False),
             kh.str.contains('TJMAXX|MARSHALL|HOMEGOODS|HOMESENSE|WINNERS|MMX|TJX UK|ECOM', case=False)]
    df['region'] = np.select(conds, ['Europe','North America'], default='Other')

    df['hw_antique_brass']  = text.str.contains('ANTIQUE BRASS')
    df['hw_antique_bronze'] = text.str.contains('ANTIQUE BRONZE')
    df['hw_nickel']         = text.str.contains('NICKEL')
    df['hw_wood']           = text.str.contains('WOOD HARDWARE')

    df['color_bucket'] = df['mau_son'].fillna('').apply(bucket_color)
    return df


def kpi_block(df: pd.DataFrame):
    by_year = df.groupby('year')['sl'].sum().sort_index()
    total_2023 = by_year.get(2023, 0)
    total_2024 = by_year.get(2024, 0)
    total_2025 = by_year.get(2025, 0)
    yoy_24 = (total_2024 - total_2023) / total_2023 * 100 if total_2023 else np.nan
    yoy_25 = (total_2025 - total_2024) / total_2024 * 100 if total_2024 else np.nan

    usb_share = df.groupby('year')['usb_flag'].mean()
    ecom_share = df.groupby('year')['ecom_flag'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tổng 2023", f"{int(total_2023):,}")
    c2.metric("Tổng 2024", f"{int(total_2024):,}", f"{yoy_24:+.1f}%" if not np.isnan(yoy_24) else None)
    c3.metric("Tổng 2025", f"{int(total_2025):,}", f"{yoy_25:+.1f}%" if not np.isnan(yoy_25) else None)
    c4.metric("USB share 2025", f"{usb_share.get(2025,0)*100:.1f}%")
    c5.metric("E‑com share 2025", f"{ecom_share.get(2025,0)*100:.1f}%")


def excel_download(df: pd.DataFrame) -> bytes:
    # export current filtered subset to Excel (DATA + 2 summary tabs)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='DATA')
        # quick summaries
        df.groupby('year')['sl'].sum().reset_index().to_excel(writer, index=False, sheet_name='KPI_YEAR')
        df.groupby(['year','color_bucket'])['sl'].sum().reset_index().to_excel(writer, index=False, sheet_name='COLOR_SUM')
    return output.getvalue()

# -------------------------
# UI
# -------------------------
st.title("📊 M3Y Analytics Web App (2023–2025)")
st.caption("Tải file Excel/CSV của bạn hoặc dùng sẵn Master_2023_2025_PRO_clean.xlsx trong thư mục (nếu có)")

# Sidebar: data source
with st.sidebar:
    st.header("Nguồn dữ liệu")
    up = st.file_uploader("Upload .xlsx / .csv", type=["xlsx","csv"])
    default_path = 'Master_2023_2025_PRO_clean.xlsx'
    if up is None and os.path.exists(default_path):
        st.info("Đang dùng file mặc định trong thư mục làm việc: Master_2023_2025_PRO_clean.xlsx")
        up = open(default_path, 'rb')

    if up is None:
        st.stop()

raw = load_data(up)
if raw is None or raw.empty:
    st.warning("Không đọc được dữ liệu. Vui lòng kiểm tra file.")
    st.stop()

base = prep_data(raw)

# Sidebar: filters
with st.sidebar:
    st.header("Bộ lọc")
    years = sorted(base['year'].unique())
    year_sel = st.multiselect("Năm", options=years, default=years)
    cust_all = sorted(base['khach_hang'].dropna().unique().tolist())
    cust_sel = st.multiselect("Khách hàng", options=cust_all, default=cust_all[:10])
    reg_sel = st.multiselect("Khu vực", options=sorted(base['region'].unique()), default=list(base['region'].unique()))
    color_sel = st.multiselect("Color bucket", options=sorted(base['color_bucket'].unique()), default=list(base['color_bucket'].unique()))
    usb_only = st.checkbox("Chỉ SKU có USB", value=False)
    ecom_only = st.checkbox("Chỉ kênh E‑commerce", value=False)

# Apply filters
f = base[base['year'].isin(year_sel)]
if cust_sel:
    f = f[f['khach_hang'].isin(cust_sel)]
if reg_sel:
    f = f[f['region'].isin(reg_sel)]
if color_sel:
    f = f[f['color_bucket'].isin(color_sel)]
if usb_only:
    f = f[f['usb_flag']]
if ecom_only:
    f = f[f['ecom_flag']]

# KPIs
kpi_block(f)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Tổng quan", "Khách hàng", "SKU", "Màu & Hardware", "Khu vực", "Container", "Dữ liệu"
])

with tab1:
    st.subheader("Xu hướng sản lượng theo tháng")
    tr = f.groupby('ym')['sl'].sum().reset_index().sort_values('ym')
    if not tr.empty:
        fig = px.line(tr, x='ym', y='sl', title=None)
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Color mix theo năm**")
        color_tot = f.groupby(['year','color_bucket'])['sl'].sum().reset_index()
        if not color_tot.empty:
            color_pvt = color_tot.pivot(index='color_bucket', columns='year', values='sl').fillna(0)
            color_pvt = color_pvt.div(color_pvt.sum(axis=0), axis=1).reset_index().melt(id_vars='color_bucket', var_name='year', value_name='share')
            fig = px.bar(color_pvt, x='year', y='share', color='color_bucket', barmode='stack')
            fig.update_yaxes(tickformat=',.0%')
            fig.update_layout(legend_title_text='Color bucket')
            st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.markdown("**USB & E‑commerce**")
        shares = pd.DataFrame({
            'year': sorted(f['year'].unique()),
            'USB_share': [f[f['year']==y]['usb_flag'].mean() for y in sorted(f['year'].unique())],
            'Ecom_share': [f[f['year']==y]['ecom_flag'].mean() for y in sorted(f['year'].unique())]
        })
        shares_m = shares.melt(id_vars='year', var_name='metric', value_name='share')
        fig = px.bar(shares_m, x='year', y='share', color='metric', barmode='group')
        fig.update_yaxes(tickformat=',.0%')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Top khách hàng (theo năm)")
    top_cust = f.groupby(['year','khach_hang'])['sl'].sum().reset_index()
    col1, col2 = st.columns(2)
    for idx, y in enumerate(sorted(top_cust['year'].unique())):
        t = top_cust[top_cust['year']==y].sort_values('sl', ascending=False).head(12)
        fig = px.bar(t, x='khach_hang', y='sl', title=f'Top 12 khách hàng {y}')
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        (col1 if idx%2==0 else col2).plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Top SKU theo năm")
    sku_year = f.groupby(['year','ma_hang'])['sl'].sum().reset_index()
    col1, col2 = st.columns(2)
    for idx, y in enumerate(sorted(sku_year['year'].unique())):
        s = sku_year[sku_year['year']==y].sort_values('sl', ascending=False).head(15)
        fig = px.bar(s, x='ma_hang', y='sl', title=f'Top 15 SKU {y}')
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        (col1 if idx%2==0 else col2).plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Hardware share theo năm")
    hw_any = f[['hw_antique_brass','hw_antique_bronze','hw_nickel','hw_wood']].any(axis=1)
    hw = f[hw_any].groupby('year')[['hw_antique_brass','hw_antique_bronze','hw_nickel','hw_wood']].mean().reset_index()
    if not hw.empty:
        m = hw.melt(id_vars='year', var_name='hardware', value_name='share')
        fig = px.bar(m, x='year', y='share', color='hardware', barmode='group')
        fig.update_yaxes(tickformat=',.0%')
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Region mix")
    reg = f.groupby(['year','region'])['sl'].sum().reset_index()
    if not reg.empty:
        pvt = reg.pivot(index='region', columns='year', values='sl').fillna(0)
        pvt = pvt.div(pvt.sum(axis=0), axis=1).reset_index().melt(id_vars='region', var_name='year', value_name='share')
        fig = px.bar(pvt, x='year', y='share', color='region', barmode='group')
        fig.update_yaxes(tickformat=',.0%')
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("Hiệu suất container (units/container)")
    cont = f.dropna(subset=['sl_container']).copy()
    if not cont.empty:
        cont['units_per_container'] = cont['sl']/cont['sl_container']
        agg = cont.groupby('khach_hang')['units_per_container'].mean().sort_values(ascending=False).head(15).reset_index()
        fig = px.bar(agg, x='khach_hang', y='units_per_container', title='Top 15 khách hàng theo units/container')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Bình quân (trong dataset đã lọc): {agg['units_per_container'].mean():.0f} units/container")
    else:
        st.info("Không có dữ liệu container trong tập đang lọc.")

with tab7:
    st.subheader("Dữ liệu đã lọc")
    st.dataframe(f.head(1000))
    b1, b2 = st.columns(2)
    with b1:
        st.download_button("⬇️ Tải CSV đang lọc", data=f.to_csv(index=False).encode('utf-8-sig'), file_name='filtered.csv', mime='text/csv')
    with b2:
        xbytes = excel_download(f)
        st.download_button("⬇️ Tải Excel (DATA + summary)", data=xbytes, file_name='filtered.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

st.success("Sẵn sàng! Dùng sidebar để lọc và khám phá insight theo thời gian thực.")
