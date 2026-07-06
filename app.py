# pages/1_Maine.py — Forecast preț DA pentru MAINE + preț realizat AZI
# Date: ENTSO-E (prognoze consum/vant/solar mâine) + Energy-Charts (actuals + preț azi) + JAO (context border).
# Token ENTSO-E: pus in Streamlit secrets ca ENTSOE_TOKEN (NU in cod).
#
# LOGICA: RL_forecast(mâine) = Consum_fc − Solar_fc − Vant_on_fc − Vant_off_fc.
# Curba preț(RL) o invatam empiric din ultimele 30 zile (Energy-Charts), apoi o aplicam pe RL de mâine.
# JAO intra ca ajustare de cuplaj (manuala acum) — full flow-based = pas ulterior.

import io, zipfile
import datetime as dt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Mâine — DE-LU", page_icon="🔮", layout="wide")
st.title("🔮 Forecast MAINE + preț realizat AZI (DE-LU)")

TZ = "Europe/Berlin"
EC = "https://api.energy-charts.info"
ENTSOE = "https://web-api.tp.entsoe.eu/api"
DE_LU = "10Y1001A1001A82H"        # EIC bidding zone DE-LU
PSR = {"B16": "Solar", "B18": "Vant offshore", "B19": "Vant onshore"}

# ---------------------------------------------------------------------------
# Energy-Charts (actuals + pret) — fara token
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def ec_public_power(start, end):
    r = requests.get(f"{EC}/public_power", params={"country": "de", "start": start, "end": end}, timeout=45)
    r.raise_for_status(); j = r.json()
    idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
    df = pd.DataFrame(index=idx)
    for pt in j.get("production_types", []):
        df[pt["name"]] = pt["data"]
    return df

@st.cache_data(ttl=900, show_spinner=False)
def ec_price(start, end):
    r = requests.get(f"{EC}/price", params={"bzn": "DE-LU", "start": start, "end": end}, timeout=45)
    r.raise_for_status(); j = r.json()
    idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
    return pd.Series(j["price"], index=idx, name="DA").astype(float)

def find_col(df, *kw):
    for c in df.columns:
        if all(k in c.lower() for k in kw):
            return c
    return None

def hourly(s):
    return s.resample("1h").mean()

# ---------------------------------------------------------------------------
# ENTSO-E — prognoze mâine (token din secrets)
# ---------------------------------------------------------------------------
def _lname(t):
    return t.split("}")[-1]

def _parse_entsoe(content: bytes) -> dict:
    """Returneaza {psrType|'ALL': Series(UTC)}. Suporta si raspuns ZIP."""
    docs = []
    if content[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(content))
        docs = [z.read(n) for n in z.namelist()]
    else:
        docs = [content]
    out = {}
    for doc in docs:
        try:
            root = ET.fromstring(doc)
        except ET.ParseError:
            continue
        for ts in [e for e in root.iter() if _lname(e.tag) == "TimeSeries"]:
            psr = "ALL"
            for c in ts.iter():
                if _lname(c.tag) == "psrType":
                    psr = c.text
            for period in [e for e in ts.iter() if _lname(e.tag) == "Period"]:
                start, res = None, None
                for c in period:
                    ln = _lname(c.tag)
                    if ln == "timeInterval":
                        for cc in c:
                            if _lname(cc.tag) == "start":
                                start = cc.text
                    elif ln == "resolution":
                        res = c.text
                if not start:
                    continue
                step = pd.Timedelta(minutes=15 if res == "PT15M" else 60)
                t0 = pd.Timestamp(start)  # tz-aware UTC (…Z)
                recs = {}
                for pt in period:
                    if _lname(pt.tag) != "Point":
                        continue
                    pos = qty = None
                    for cc in pt:
                        ln = _lname(cc.tag)
                        if ln == "position": pos = int(cc.text)
                        elif ln == "quantity": qty = float(cc.text)
                    if pos is not None:
                        recs[t0 + (pos - 1) * step] = qty
                if recs:
                    s = pd.Series(recs).sort_index()
                    out.setdefault(psr, []).append(s)
    return {k: pd.concat(v).sort_index() for k, v in out.items()}

@st.cache_data(ttl=1800, show_spinner=False)
def entsoe_fetch(params: dict) -> dict:
    token = st.secrets.get("ENTSOE_TOKEN", "")
    if not token:
        raise RuntimeError("Lipseste ENTSOE_TOKEN in Streamlit secrets.")
    p = dict(params); p["securityToken"] = token
    r = requests.get(ENTSOE, params=p, timeout=60)
    r.raise_for_status()
    return _parse_entsoe(r.content)

def _bounds(day: dt.date):
    start = pd.Timestamp(day, tz=TZ)
    end = start + pd.Timedelta(days=1)
    fmt = lambda t: t.tz_convert("UTC").strftime("%Y%m%d%H%M")
    return fmt(start), fmt(end)

def load_forecast(day):
    s, e = _bounds(day)
    d = entsoe_fetch({"documentType": "A65", "processType": "A01",
                      "outBiddingZone_Domain": DE_LU, "periodStart": s, "periodEnd": e})
    if "ALL" not in d:
        return pd.Series(dtype=float)
    return hourly(d["ALL"].tz_convert(TZ)).rename("Consum_fc")

def wind_solar_forecast(day):
    s, e = _bounds(day)
    d = entsoe_fetch({"documentType": "A69", "processType": "A01",
                      "in_Domain": DE_LU, "periodStart": s, "periodEnd": e})
    out = {}
    for psr, name in PSR.items():
        if psr in d:
            out[name] = hourly(d[psr].tz_convert(TZ))
    return pd.DataFrame(out)

# ---------------------------------------------------------------------------
# Curba empirica preț(RL) — invatata din ultimele 30 zile
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def learn_curve(days=30, n_bins=25):
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=days)
    pp = ec_public_power(start.isoformat(), end.isoformat())
    da = ec_price(start.isoformat(), end.isoformat())
    c_res = find_col(pp, "residual")
    if not c_res:
        return None
    d = pd.concat([hourly(da).rename("da"), hourly(pp)[c_res].rename("rl")], axis=1).dropna()
    if len(d) < 50:
        return None
    d = d.sort_values("rl")
    try:
        d["bin"] = pd.qcut(d["rl"], q=min(n_bins, d["rl"].nunique()), duplicates="drop")
    except Exception:
        d["bin"] = pd.cut(d["rl"], bins=10)
    g = d.groupby("bin", observed=True).agg(rl=("rl", "median"), da=("da", "median")).dropna()
    xs = g["rl"].values
    ys = np.maximum.accumulate(g["da"].values)  # monoton crescator
    return xs, ys

def predict_curve(rl, curve):
    xs, ys = curve
    return np.interp(np.asarray(rl, dtype=float), xs, ys)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
today = dt.date.today()
tomorrow = today + dt.timedelta(days=1)

st.sidebar.header("Ajustare cuplaj (JAO)")
st.sidebar.caption("Provizoriu manual. + trage prețul in sus (DE blocat pe export), "
                   "− in jos (import ieftin disponibil). Full flow-based = pas ulterior.")
coupling_adj = st.sidebar.slider("Ajustare (EUR/MWh)", -30.0, 30.0, 0.0, 1.0)

# --- 1) PRET REALIZAT AZI ---
st.subheader(f"⚡ Preț DA realizat — AZI ({today})")
try:
    px = ec_price((today - dt.timedelta(days=1)).isoformat(), (today + dt.timedelta(days=2)).isoformat())
    px_h = hourly(px)
    today_px = px_h[px_h.index.date == today]
    tom_px = px_h[px_h.index.date == tomorrow]  # exista doar dupa ~13:00 D-1
    if len(today_px):
        c1, c2, c3 = st.columns(3)
        c1.metric("Media azi", f"{today_px.mean():.1f}")
        c2.metric("Min / Max", f"{today_px.min():.0f} / {today_px.max():.0f}")
        c3.metric("Ore negative", f"{int((today_px < 0).sum())}")
        st.line_chart(today_px.rename("DA azi"))
    else:
        st.info("Prețul de azi nu e inca disponibil.")
except Exception as ex:
    st.error(f"Eroare preț azi: {ex}")
    tom_px = pd.Series(dtype=float)

st.divider()

# --- 2) FORECAST MAINE ---
st.subheader(f"🔮 Forecast preț DA — MAINE ({tomorrow})")
curve = learn_curve()
if curve is None:
    st.warning("Nu am putut invata curba (date insuficiente).")
    st.stop()

try:
    load_fc = load_forecast(tomorrow)
    ws_fc = wind_solar_forecast(tomorrow)
except Exception as ex:
    st.error(f"ENTSO-E: {ex}")
    st.info("Adauga ENTSOE_TOKEN in Streamlit secrets ca sa vezi forecast-ul de mâine.")
    st.stop()

if load_fc.empty or ws_fc.empty:
    st.warning("Prognozele ENTSO-E pentru mâine nu sunt inca publicate (revino dupa dimineata D-1).")
    st.stop()

df = pd.concat([load_fc, ws_fc], axis=1).dropna(how="all")
ren_cols = [c for c in ws_fc.columns]
df["Regenerabil_fc"] = df[ren_cols].sum(axis=1)
df["RL_fc"] = df["Consum_fc"] - df["Regenerabil_fc"]
df = df.dropna(subset=["RL_fc"])

df["Pret_fc"] = predict_curve(df["RL_fc"], curve) + coupling_adj

c1, c2, c3, c4 = st.columns(4)
c1.metric("Preț mediu mâine", f"{df['Pret_fc'].mean():.1f}")
c2.metric("Min / Max", f"{df['Pret_fc'].min():.0f} / {df['Pret_fc'].max():.0f}")
c3.metric("Ore negative prezise", f"{int((df['Pret_fc'] < 0).sum())}")
c4.metric("RL mediu mâine (MW)", f"{df['RL_fc'].mean():,.0f}")

st.markdown("**Profil preț prezis (24h)**")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Pret_fc"], mode="lines+markers", name="Forecast"))
if len(tom_px):
    fig.add_trace(go.Scatter(x=tom_px.index, y=tom_px.values, mode="lines+markers",
                             name="DA publicat", line=dict(dash="dash")))
fig.update_layout(yaxis_title="EUR/MWh", height=420)
st.plotly_chart(fig, use_container_width=True)
if len(tom_px):
    j = pd.concat([df["Pret_fc"], tom_px.rename("DA")], axis=1).dropna()
    if len(j):
        mae = (j["Pret_fc"] - j["DA"]).abs().mean()
        st.success(f"DA de mâine deja publicat → MAE forecast vs realizat = {mae:.1f} EUR/MWh "
                   f"(bias {j['Pret_fc'].mean()-j['DA'].mean():+.1f}).")

st.markdown("**Descompunere: consum vs regenerabile (input-urile forecast-ului)**")
st.line_chart(df[["Consum_fc", "Regenerabil_fc", "RL_fc"]])

with st.expander("📋 Tabel orar"):
    st.dataframe(df.round(1), use_container_width=True)

st.divider()
st.subheader("🔗 JAO — border (context)")
st.info("Aici afisam headroom-ul de net position al DE pentru mâine (publicat de JAO la 11:00 D-1). "
        "Momentan folosim maneta manuala din sidebar. Urmatorul pas: cablam fetch-ul la endpoint-ul "
        "tau JAO existent (din jao.html) ca sa umplem automat min/max net pos si maxBex, "
        "iar ajustarea de cuplaj sa devina calculata, nu manuala.")
st.caption("Date: ENTSO-E (prognoze) + Energy-Charts / Fraunhofer ISE, CC BY 4.0 (actuals & preț).")
