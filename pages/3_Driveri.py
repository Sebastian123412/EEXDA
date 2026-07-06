# pages/3_Driveri.py — Tablou de driveri de preț cu presiune ↑/↓ pentru MAINE
# Sageata = directia presiunii pe pret (rosu=bullish/scump, verde=bearish/ieftin).
# Master = residual load prognozat vs normal. Restul senzorilor explica de ce.
# Surse: ENTSO-E (prognoze) + Open-Meteo (meteo, fara cheie) + Energy-Charts (actuals).

import io, zipfile
import datetime as dt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Driveri preț", page_icon="🎛️", layout="wide")
st.title("🎛️ Driveri de preț — presiune pentru MAINE")

TZ = "Europe/Berlin"
EC = "https://api.energy-charts.info"
ENTSOE = "https://web-api.tp.entsoe.eu/api"
OM = "https://api.open-meteo.com/v1/forecast"
DE_LU = "10Y1001A1001A82H"
# Puncte reprezentative DE: Hamburg (nord/vant), Frankfurt (centru), Munchen (sud)
POINTS = [(53.55, 9.99), (50.11, 8.68), (48.14, 11.58)]

today = dt.date.today()
tomorrow = today + dt.timedelta(days=1)
week_ago = today - dt.timedelta(days=7)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hourly(x): return x.resample("1h").mean()

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
                step = pd.Timedelta(minutes=15 if res == "PT15M" else 60)
                t0 = pd.Timestamp(start); recs = {}
                for pt in period:
                    if _ln(pt.tag) != "Point": continue
                    pos = qty = None
                    for cc in pt:
                        if _ln(cc.tag) == "position": pos = int(cc.text)
                        elif _ln(cc.tag) == "quantity": qty = float(cc.text)
                    if pos is not None: recs[t0 + (pos - 1) * step] = qty
                if recs: out.setdefault(psr, []).append(pd.Series(recs).sort_index())
    return {k: pd.concat(v).sort_index() for k, v in out.items()}

@st.cache_data(ttl=1800, show_spinner=False)
def entsoe(params):
    token = st.secrets.get("ENTSOE_TOKEN", "")
    if not token: raise RuntimeError("Lipseste ENTSOE_TOKEN in secrets.")
    p = dict(params); p["securityToken"] = token
    r = requests.get(ENTSOE, params=p, timeout=60); r.raise_for_status()
    return _parse(r.content)

def _win():
    s = pd.Timestamp(week_ago, tz=TZ).tz_convert("UTC").strftime("%Y%m%d%H%M")
    e = pd.Timestamp(tomorrow + dt.timedelta(days=1), tz=TZ).tz_convert("UTC").strftime("%Y%m%d%H%M")
    return s, e

@st.cache_data(ttl=1800, show_spinner=False)
def get_forecasts():
    s, e = _win()
    load = entsoe({"documentType": "A65", "processType": "A01",
                   "outBiddingZone_Domain": DE_LU, "periodStart": s, "periodEnd": e})
    ws = entsoe({"documentType": "A69", "processType": "A01",
                 "in_Domain": DE_LU, "periodStart": s, "periodEnd": e})
    L = hourly(load["ALL"].tz_convert(TZ)) if "ALL" in load else pd.Series(dtype=float)
    sol = hourly(ws["B16"].tz_convert(TZ)) if "B16" in ws else pd.Series(0, index=L.index)
    won = hourly(ws["B19"].tz_convert(TZ)) if "B19" in ws else pd.Series(0, index=L.index)
    woff = hourly(ws["B18"].tz_convert(TZ)) if "B18" in ws else pd.Series(0, index=L.index)
    df = pd.concat([L.rename("load"), sol.rename("solar"),
                    won.rename("won"), woff.rename("woff")], axis=1)
    df["ren"] = df[["solar", "won", "woff"]].sum(axis=1)
    df["rl"] = df["load"] - df["ren"]
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def get_meteo():
    frames = []
    for lat, lon in POINTS:
        r = requests.get(OM, params={
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,wind_speed_100m,shortwave_radiation",
            "past_days": 7, "forecast_days": 2, "timezone": "Europe/Berlin"}, timeout=30)
        r.raise_for_status(); h = r.json()["hourly"]
        idx = pd.to_datetime(h["time"])
        frames.append(pd.DataFrame({"temp": h["temperature_2m"],
                                    "wind100": h["wind_speed_100m"],
                                    "rad": h["shortwave_radiation"]}, index=idx))
    avg = sum(frames) / len(frames)
    avg.index = avg.index.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
    return avg

@st.cache_data(ttl=900, show_spinner=False)
def get_energy_charts_now():
    s = week_ago.isoformat(); e = (tomorrow + dt.timedelta(days=1)).isoformat()
    pp = requests.get(f"{EC}/public_power", params={"country": "de", "start": s, "end": e}, timeout=45).json()
    px = requests.get(f"{EC}/price", params={"bzn": "DE-LU", "start": s, "end": e}, timeout=45).json()
    idx = pd.to_datetime(pp["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
    ren = tot = None
    for p in pp.get("production_types", []):
        nm = p["name"].lower()
        ser = pd.Series(p["data"], index=idx)
        if any(k in nm for k in ["solar", "wind", "hydro", "bio", "geo", "renew", "waste"]) and "residual" not in nm:
            ren = ser if ren is None else ren.add(ser, fill_value=0)
        if nm.strip() == "load":
            load_now = ser
    pidx = pd.to_datetime(px["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
    price = pd.Series(px["price"], index=pidx)
    return ren, price

def split(series):
    """current = media pe mâine, normal = media pe ultimele 7 zile (excl. azi)."""
    if series is None or series.empty: return np.nan, np.nan
    cur = series[series.index.date == tomorrow].mean()
    normal = series[(series.index.date >= week_ago) & (series.index.date < today)].mean()
    return cur, normal

def card(col, label, cur, normal, unit, bullish_when_high, fmt="{:.0f}", deadband=3.0):
    if pd.isna(cur) or pd.isna(normal) or normal == 0:
        col.metric(label, "—"); col.caption("date indisponibile"); return None
    dev = cur - normal
    pct = dev / abs(normal) * 100
    score = (1 if bullish_when_high else -1) * pct
    if score > deadband:   press, color = "↑ Bullish", "red"
    elif score < -deadband: press, color = "↓ Bearish", "green"
    else:                   press, color = "→ Neutru", "gray"
    col.metric(label, fmt.format(cur) + f" {unit}", f"{dev:+.0f} vs normal")
    col.markdown(f"Presiune: :{color}[{press}]")
    return score

# ---------------------------------------------------------------------------
# SIDEBAR — combustibili (manual, cu referinta ta)
# ---------------------------------------------------------------------------
st.sidebar.header("Combustibili & CO2 (manual)")
ttf   = st.sidebar.number_input("TTF azi (EUR/MWh)", 5.0, 200.0, 32.0, 0.5)
ttf_r = st.sidebar.number_input("TTF referinta", 5.0, 200.0, 32.0, 0.5)
coal   = st.sidebar.number_input("API2 azi ($/t)", 40.0, 400.0, 110.0, 1.0)
coal_r = st.sidebar.number_input("API2 referinta", 40.0, 400.0, 110.0, 1.0)
co2   = st.sidebar.number_input("EUA azi (EUR/t)", 20.0, 200.0, 75.0, 1.0)
co2_r = st.sidebar.number_input("EUA referinta", 20.0, 200.0, 75.0, 1.0)

# ---------------------------------------------------------------------------
# DATE
# ---------------------------------------------------------------------------
try:
    fc = get_forecasts()
except Exception as ex:
    st.error(f"ENTSO-E: {ex}. Adauga ENTSOE_TOKEN in secrets."); st.stop()

if fc["rl"].dropna().empty or fc[fc.index.date == tomorrow].empty:
    st.warning("Prognozele ENTSO-E pentru mâine nu sunt inca publicate. Revino dupa dimineata D-1.")
    st.stop()

# ---------------------------------------------------------------------------
# MASTER GAUGE — residual load
# ---------------------------------------------------------------------------
rl_cur, rl_normal = split(fc["rl"])
rl_dev = rl_cur - rl_normal
rl_pct = rl_dev / abs(rl_normal) * 100
if rl_pct > 3:    head, hc = "BULLISH", "red"
elif rl_pct < -3: head, hc = "BEARISH", "green"
else:             head, hc = "NEUTRU", "gray"

st.subheader(f"Bias residual load — MAINE ({tomorrow})")
c = st.columns([1, 2])
c[0].metric("RL mediu mâine (MW)", f"{rl_cur:,.0f}", f"{rl_dev:+,.0f} vs normal 7z")
c[1].markdown(f"### Presiune de baza: :{hc}[{head}]")
c[1].caption(f"RL cu {rl_pct:+.1f}% fata de media ultimelor 7 zile. "
             "RL sus = mai putin regenerabil / mai mult consum → termic marginal → preț sus.")
st.divider()

# ---------------------------------------------------------------------------
# CERERE & REGENERABILE (ENTSO-E)
# ---------------------------------------------------------------------------
st.markdown("### ⚡ Cerere & regenerabile (ENTSO-E, mâine vs normal 7z)")
cols = st.columns(4)
lc, ln = split(fc["load"]); card(cols[0], "Consum", lc, ln, "MW", True)
sc, sn = split(fc["solar"]); card(cols[1], "Solar", sc, sn, "MW", False)
oc, on = split(fc["won"]); card(cols[2], "Vant onshore", oc, on, "MW", False)
fc_, fn = split(fc["woff"]); card(cols[3], "Vant offshore", fc_, fn, "MW", False)

# ---------------------------------------------------------------------------
# METEO (Open-Meteo)
# ---------------------------------------------------------------------------
st.markdown("### 🌤️ Meteo (Open-Meteo, medie 3 puncte DE)")
try:
    m = get_meteo()
    cols = st.columns(3)
    tc, tn = split(m["temp"])
    # temperatura: presiune sezoniera (frig iarna / cald vara = bullish prin consum)
    dev_t = tc - tn
    if tn < 15:   bull_t = dev_t < 0   # mai frig ca normal → consum sus
    elif tn > 20: bull_t = dev_t > 0   # mai cald ca normal → AC sus
    else:         bull_t = None
    cols[0].metric("Temperatura", f"{tc:.1f} °C", f"{dev_t:+.1f} vs normal")
    if bull_t is None: cols[0].markdown("Presiune: :gray[→ Neutru]")
    else: cols[0].markdown("Presiune: " + (":red[↑ Bullish]" if bull_t else ":green[↓ Bearish]"))
    wc, wn = split(m["wind100"]); card(cols[1], "Vant 100m", wc, wn, "km/h", False, "{:.0f}")
    rc, rn = split(m["rad"]); card(cols[2], "Radiatie solara", rc, rn, "W/m²", False, "{:.0f}")
except Exception as ex:
    st.info(f"Meteo indisponibil: {ex}")

# ---------------------------------------------------------------------------
# COMBUSTIBILI (manual)
# ---------------------------------------------------------------------------
st.markdown("### 🔥 Combustibili & CO2 (nivel preț, vs referinta ta)")
cols = st.columns(3)
card(cols[0], "TTF gaz", ttf, ttf_r, "EUR/MWh", True, "{:.1f}", 2)
card(cols[1], "API2 carbune", coal, coal_r, "$/t", True, "{:.0f}", 2)
card(cols[2], "EUA CO2", co2, co2_r, "EUR/t", True, "{:.0f}", 2)

# ---------------------------------------------------------------------------
# SISTEM ACUM (Energy-Charts)
# ---------------------------------------------------------------------------
st.markdown("### 📡 Sistem acum (Energy-Charts)")
try:
    ren_now, price_now = get_energy_charts_now()
    ph = hourly(price_now)
    today_px = ph[ph.index.date == today]
    cols = st.columns(2)
    if len(today_px):
        cols[0].metric("Preț DA azi (mediu)", f"{today_px.mean():.1f} EUR/MWh")
    if ren_now is not None and not ren_now.empty:
        cols[1].metric("Regenerabil acum (MW)", f"{ren_now.dropna().iloc[-1]:,.0f}")
except Exception as ex:
    st.info(f"Energy-Charts indisponibil: {ex}")

st.divider()
st.caption("Outage-uri termice (UMM/REMIT) = urmatorul senzor de cablat — parsarea ENTSO-E de "
           "indisponibilitati e separata. JAO border headroom vine cand cablam fetch-ul tau existent.")
st.caption("Surse: ENTSO-E + Open-Meteo + Energy-Charts / Fraunhofer ISE (CC BY 4.0).")
