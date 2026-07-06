# pages/2_Mix_energetic.py — Mix energetic DE din DOUA surse (ENTSO-E + Energy-Charts)
# Cross-check intre surse: nu se potrivesc perfect (mapari/timing/revizuiri diferite).
# Token ENTSO-E: in Streamlit secrets ca ENTSOE_TOKEN.

import io, zipfile
import datetime as dt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Mix energetic DE", page_icon="🔋", layout="wide")
st.title("🔋 Mix energetic DE — ENTSO-E vs Energy-Charts")

TZ = "Europe/Berlin"
EC = "https://api.energy-charts.info"
ENTSOE = "https://web-api.tp.entsoe.eu/api"
DE_LU = "10Y1001A1001A82H"

# Coduri PsrType ENTSO-E (A75 = generare realizata pe tip)
PSR = {
    "B01": "Biomasa", "B02": "Lignit", "B04": "Gaz", "B05": "Carbune",
    "B06": "Pacura", "B09": "Geotermal", "B10": "Hidro pompaj",
    "B11": "Hidro fir apa", "B12": "Hidro rezervor", "B14": "Nuclear",
    "B15": "Alt regenerabil", "B16": "Solar", "B17": "Deseuri",
    "B18": "Vant offshore", "B19": "Vant onshore", "B20": "Altele",
}
RENEW = {"Biomasa", "Solar", "Vant offshore", "Vant onshore", "Hidro fir apa",
         "Hidro rezervor", "Geotermal", "Alt regenerabil", "Deseuri"}

# --- Energy-Charts (fara token) ---
@st.cache_data(ttl=1800, show_spinner=False)
def ec_public_power(start, end):
    r = requests.get(f"{EC}/public_power", params={"country": "de", "start": start, "end": end}, timeout=45)
    r.raise_for_status(); j = r.json()
    idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
    df = pd.DataFrame(index=idx)
    for pt in j.get("production_types", []):
        df[pt["name"]] = pt["data"]
    return df

def hourly(x):
    return x.resample("1h").mean()

# --- ENTSO-E ---
def _ln(t): return t.split("}")[-1]

def _parse(content):
    docs = [z for z in (zipfile.ZipFile(io.BytesIO(content)).read(n)
            for n in zipfile.ZipFile(io.BytesIO(content)).namelist())] if content[:2] == b"PK" else [content]
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
def entsoe_generation(start_utc, end_utc):
    token = st.secrets.get("ENTSOE_TOKEN", "")
    if not token: raise RuntimeError("Lipseste ENTSOE_TOKEN in secrets.")
    p = {"documentType": "A75", "processType": "A16", "in_Domain": DE_LU,
         "periodStart": start_utc, "periodEnd": end_utc, "securityToken": token}
    r = requests.get(ENTSOE, params=p, timeout=60); r.raise_for_status()
    d = _parse(r.content)
    cols = {}
    for psr, name in PSR.items():
        if psr in d:
            cols[name] = hourly(d[psr].tz_convert(TZ))
    return pd.DataFrame(cols)

# --- UI ---
days = st.selectbox("Interval (zile)", [1, 3, 7, 14], index=1)
end = dt.date.today() + dt.timedelta(days=1)
start = end - dt.timedelta(days=days)
s_iso, e_iso = start.isoformat(), end.isoformat()
s_utc = pd.Timestamp(start, tz=TZ).tz_convert("UTC").strftime("%Y%m%d%H%M")
e_utc = pd.Timestamp(end, tz=TZ).tz_convert("UTC").strftime("%Y%m%d%H%M")

# Energy-Charts
try:
    pp = ec_public_power(s_iso, e_iso)
except Exception as ex:
    st.error(f"Energy-Charts: {ex}"); st.stop()

gen_cols = [c for c in pp.columns
            if not any(k in c.lower() for k in ["load", "residual", "share", "cross", "import", "export"])]
gen = hourly(pp[gen_cols]).clip(lower=0)

st.subheader("📊 Mix generare (Energy-Charts) — stivuit")
st.area_chart(gen)

# Snapshot ultima ora + share regenerabil
latest = gen.iloc[-1].sort_values(ascending=False)
latest = latest[latest > 0]
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(f"**Snapshot ultima ora ({gen.index[-1]:%d.%m %H:%M})**")
    fig = go.Figure(go.Pie(labels=latest.index, values=latest.values, hole=0.5))
    fig.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    total = latest.sum()
    ren = sum(v for k, v in latest.items() if any(r.lower() in k.lower()
              for r in ["solar", "wind", "vant", "hydro", "hidro", "bio", "geo", "renew", "regenerabil", "waste", "deseu"]))
    st.metric("Generare totala (MW)", f"{total:,.0f}")
    st.metric("Cota regenerabil (%)", f"{100*ren/total:.0f}" if total else "—")

# Cross-check ENTSO-E vs Energy-Charts
st.divider()
st.subheader("🔍 Cross-check ENTSO-E vs Energy-Charts (medie pe interval, MW)")
try:
    ent = entsoe_generation(s_utc, e_utc)
    # mapare denumiri EC -> ENTSO-E
    pairs = {"Solar": "Solar", "Vant onshore": ("wind", "onshore"), "Vant offshore": ("wind", "offshore"),
             "Gaz": ("gas",), "Lignit": ("brown",), "Carbune": ("hard",),
             "Nuclear": ("nuclear",), "Biomasa": ("bio",)}
    rows = []
    for name, eckey in pairs.items():
        ec_col = None
        for c in gen.columns:
            keys = (eckey,) if isinstance(eckey, str) else eckey
            if all(k.lower() in c.lower() for k in keys):
                ec_col = c; break
        ec_val = gen[ec_col].mean() if ec_col else np.nan
        ent_val = ent[name].mean() if name in ent.columns else np.nan
        diff = ec_val - ent_val if (pd.notna(ec_val) and pd.notna(ent_val)) else np.nan
        rows.append({"Tehnologie": name, "Energy-Charts": round(ec_val, 0) if pd.notna(ec_val) else "—",
                     "ENTSO-E": round(ent_val, 0) if pd.notna(ent_val) else "—",
                     "Δ (EC−ENTSO)": round(diff, 0) if pd.notna(diff) else "—"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Diferentele sunt normale (mapari, timing, revizuiri). Le folosesti ca sanity-check, "
               "nu ca eroare. Daca una diverge MULT constant → verifica maparea sau ce sursa e mai proaspata.")
except Exception as ex:
    st.info(f"ENTSO-E indisponibil ({ex}). Afisez doar Energy-Charts. Adauga ENTSOE_TOKEN in secrets pentru cross-check.")

st.caption("Surse: ENTSO-E Transparency + Energy-Charts / Fraunhofer ISE (CC BY 4.0).")
