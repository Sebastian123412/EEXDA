# pages/7_Residual_Load.py — Preturi + Residual Load pentru DE-LU si toti vecinii cu border.
# ENTSO-E: pret DA (A44) + forecast consum (A65) + vant/solar/hidro-ror (A69).
# RL = Consum − Vant − Solar − Hidro fir-de-apa (run-of-river = must-run, cost ~0).
# NU se scade hidro rezervor/pompaj (sunt dispecerizabile). Rezolutie ora/sfert.
# Dedesubt: media zilei (base) de pret si de RL. Aliniat CET. Token in Streamlit secrets ca ENTSOE_TOKEN.

import io, zipfile
import datetime as dt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Residual Load", page_icon="📉", layout="wide")
st.title("📉 Residual Load — DE-LU + vecini")

TZ = "Europe/Berlin"
ENTSOE = "https://web-api.tp.entsoe.eu/api"

# DE-LU si vecinii cu border (coduri EIC bidding zone)
ZONES = {
    "DE-LU": "10Y1001A1001A82H",
    "FR":    "10YFR-RTE------C",
    "NL":    "10YNL----------L",
    "BE":    "10YBE----------2",
    "AT":    "10YAT-APG------L",
    "PL":    "10YPL-AREA-----S",
    "CZ":    "10YCZ-CEPS-----N",
    "DK1":   "10YDK-1--------W",
    "DK2":   "10YDK-2--------M",
    "CH":    "10YCH-SWISSGRIDZ",
    "SE4":   "10Y1001A1001A47J",
    "NO2":   "10YNO-2--------T",
}

# ---------------- ENTSO-E: fetch + parse ----------------
def _ln(t): return t.split("}")[-1]

def _parse(content):
    docs = ([zipfile.ZipFile(io.BytesIO(content)).read(n)
             for n in zipfile.ZipFile(io.BytesIO(content)).namelist()]
            if content[:2] == b"PK" else [content])
    out = {}
    for doc in docs:
        try: root = ET.fromstring(doc)
        except ET.ParseError: continue
        for ts in [e for e in root.iter() if _ln(e.tag) == "TimeSeries"]:
            psr = "ALL"
            for c in ts.iter():
                if _ln(c.tag) == "psrType": psr = c.text
            for period in [e for e in ts.iter() if _ln(e.tag) == "Period"]:
                start = res = None
                for c in period:
                    if _ln(c.tag) == "timeInterval":
                        for cc in c:
                            if _ln(cc.tag) == "start": start = cc.text
                    elif _ln(c.tag) == "resolution": res = c.text
                if not start: continue
                step = pd.Timedelta(minutes=15 if res == "PT15M" else
                                    30 if res == "PT30M" else 60)
                t0 = pd.Timestamp(start); recs = {}
                for pt in period:
                    if _ln(pt.tag) != "Point": continue
                    pos = qty = None
                    for cc in pt:
                        if _ln(cc.tag) == "position": pos = int(cc.text)
                        elif _ln(cc.tag) in ("quantity", "price.amount"): qty = float(cc.text)
                    if pos is not None: recs[t0 + (pos - 1) * step] = qty
                if recs: out.setdefault(psr, []).append(pd.Series(recs).sort_index())
    return {k: pd.concat(v)[~pd.concat(v).index.duplicated()].sort_index() for k, v in out.items()}

def _token():
    tok = st.secrets.get("ENTSOE_TOKEN", "")
    if not tok: raise RuntimeError("Lipseste ENTSOE_TOKEN in Streamlit secrets.")
    return tok

@st.cache_data(ttl=1800, show_spinner=False)
def get_price(eic, s, e):
    p = {"documentType": "A44", "in_Domain": eic, "out_Domain": eic,
         "periodStart": s, "periodEnd": e, "securityToken": _token()}
    r = requests.get(ENTSOE, params=p, timeout=60)
    if r.status_code != 200: return pd.Series(dtype=float)
    d = _parse(r.content)
    return d.get("ALL", pd.Series(dtype=float)).tz_convert(TZ)

@st.cache_data(ttl=1800, show_spinner=False)
def get_load(eic, s, e):
    p = {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": eic,
         "periodStart": s, "periodEnd": e, "securityToken": _token()}
    r = requests.get(ENTSOE, params=p, timeout=60)
    if r.status_code != 200: return pd.Series(dtype=float)
    d = _parse(r.content)
    return d.get("ALL", pd.Series(dtype=float)).tz_convert(TZ)

@st.cache_data(ttl=1800, show_spinner=False)
def get_renewables(eic, s, e):
    """Forecast day-ahead: solar (B16), vant offshore (B18), vant onshore (B19),
    hidro fir-de-apa (B11). B11 poate lipsi pe A69 pt unele zone → tratat ca 0."""
    p = {"documentType": "A69", "processType": "A01", "in_Domain": eic,
         "periodStart": s, "periodEnd": e, "securityToken": _token()}
    r = requests.get(ENTSOE, params=p, timeout=60)
    if r.status_code != 200: return pd.DataFrame()
    d = _parse(r.content)
    cols = {}
    for k, name in [("B16", "solar"), ("B18", "woff"), ("B19", "won"), ("B11", "hydro_ror")]:
        if k in d: cols[name] = d[k].tz_convert(TZ)
    return pd.DataFrame(cols)

def to_grid(series, res):
    if series is None or len(series) == 0: return pd.Series(dtype=float)
    if res == "1h":
        return series.resample("1h").mean()
    return series.resample("15min").mean().interpolate(limit_direction="both")

# ---------------- UI ----------------
c1, c2, c3 = st.columns([3, 1, 1])
neighbors = c1.multiselect("Zone (DE-LU e mereu inclusa)",
                           [z for z in ZONES if z != "DE-LU"],
                           default=[z for z in ZONES if z != "DE-LU"])
res_label = c2.radio("Rezolutie", ["Oră", "Sfert"], horizontal=True)
res = "1h" if res_label == "Oră" else "15min"
days = c3.selectbox("Zile (pana mâine)", [3, 7, 14], index=1)

sub_hydro = st.checkbox("Scade hidro fir-de-apa (B11) din residual load", value=True,
                        help="Run-of-river = must-run, cost ~0. NU se scade rezervor/pompaj.")

sel = ["DE-LU"] + neighbors
start = dt.date.today() - dt.timedelta(days=days)
end = dt.date.today() + dt.timedelta(days=2)  # include mâine
s_utc = pd.Timestamp(start, tz=TZ).tz_convert("UTC").strftime("%Y%m%d%H%M")
e_utc = pd.Timestamp(end, tz=TZ).tz_convert("UTC").strftime("%Y%m%d%H%M")

st.caption(f"Interval: {start} → {end} | aliniat CET | prima incarcare mai lenta, apoi cache.")

# ---------------- Fetch ----------------
COMPONENTS = ["solar", "woff", "won"] + (["hydro_ror"] if sub_hydro else [])
price_cols, rl_cols, hydro_seen = {}, {}, []
progress = st.progress(0.0, text="Preiau date ENTSO-E...")
for i, z in enumerate(sel):
    eic = ZONES[z]
    try:
        pr = get_price(eic, s_utc, e_utc)
        if len(pr): price_cols[z] = to_grid(pr, res)
    except Exception as ex:
        st.warning(f"Preț {z}: {ex}")
    try:
        load = get_load(eic, s_utc, e_utc)
        ws = get_renewables(eic, s_utc, e_utc)
        if len(load):
            ren = pd.Series(0.0, index=load.index)
            for col in COMPONENTS:
                if col in ws.columns:
                    ren = ren.add(ws[col].reindex(load.index).interpolate(), fill_value=0)
                    if col == "hydro_ror": hydro_seen.append(z)
            rl = load - ren
            rl_cols[z] = to_grid(rl, res)
    except Exception as ex:
        st.warning(f"RL {z}: {ex}")
    progress.progress((i + 1) / len(sel), text=f"Preiau date ENTSO-E... {z}")
progress.empty()

price_df = pd.DataFrame(price_cols).dropna(how="all")
rl_df = pd.DataFrame(rl_cols).dropna(how="all")

if price_df.empty and rl_df.empty:
    st.error("Nu am primit date. Verifica ENTSOE_TOKEN in secrets."); st.stop()

if sub_hydro:
    if hydro_seen:
        st.success(f"Hidro fir-de-apa (B11) scazut pentru: {', '.join(hydro_seen)}.")
    else:
        st.info("Hidro fir-de-apa (B11) nu a venit pe forecast A69 pt niciuna din zone → "
                "RL neschimbat. (Pt hidro garantat ar trebui generarea realizata A75, dar aia e actual, nu forecast.)")

# ---------------- 1) PRETURI ----------------
st.subheader(f"① Preț DA ({res_label})")
if not price_df.empty:
    fig = go.Figure()
    for z in price_df.columns:
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df[z], mode="lines", name=z,
                                 line=dict(width=3 if z == "DE-LU" else 1)))
    fig.update_layout(yaxis_title="EUR/MWh", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Tabel preturi"):
        st.dataframe(price_df.round(1), use_container_width=True)

# ---------------- 2) RESIDUAL LOAD ----------------
comp_txt = "Consum − Vânt − Solar" + (" − Hidro fir-de-apă" if sub_hydro else "")
st.subheader(f"② Residual Load = {comp_txt} ({res_label})")
if not rl_df.empty:
    fig = go.Figure()
    for z in rl_df.columns:
        fig.add_trace(go.Scatter(x=rl_df.index, y=rl_df[z], mode="lines", name=z,
                                 line=dict(width=3 if z == "DE-LU" else 1)))
    fig.update_layout(yaxis_title="MW", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Tabel residual load (MW)"):
        st.dataframe(rl_df.round(0), use_container_width=True)
    st.caption("NO2/SE4/CH au vant/solar mic in ENTSO-E → RL ≈ consum. Normal.")

# ---------------- 3) MEDIA ZILEI (BASE) ----------------
st.subheader("③ Media zilei (base)")
cc1, cc2 = st.columns(2)
if not price_df.empty:
    dpr = price_df.groupby(price_df.index.date).mean().round(1)
    dpr.index.name = "Zi"
    cc1.markdown("**Preț mediu zilnic (EUR/MWh)**")
    cc1.dataframe(dpr, use_container_width=True)
if not rl_df.empty:
    drl = rl_df.groupby(rl_df.index.date).mean().round(0)
    drl.index.name = "Zi"
    cc2.markdown("**Residual load mediu zilnic (MW)**")
    cc2.dataframe(drl, use_container_width=True)

st.caption("Sursa: ENTSO-E Transparency. Consum=A65, Vant/Solar/Hidro-ror=A69 (day-ahead), Preț=A44.")
