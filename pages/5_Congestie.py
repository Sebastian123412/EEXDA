# pages/5_Congestie.py — CAND se umple o interconexiune? Probabilitate de binding per ora mâine.
# Logica: border-ul se umple cand piata vrea sa treaca mai mult decat capacitatea. Cererea de transport
# ~ spread-ul fundamental necuplat, condus de divergenta de residual load intre zone.
# Invatam din istoric: P(binding) vs divergenta RL. Scoram mâine cu forecast DE (ENTSO-E) + climatologie vecin.

import io, zipfile
import datetime as dt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Congestie", page_icon="🚦", layout="wide")
st.title("🚦 Congestie — când se umple o interconexiune?")

TZ = "Europe/Berlin"
EC = "https://api.energy-charts.info"
ENTSOE = "https://web-api.tp.entsoe.eu/api"
DE_LU = "10Y1001A1001A82H"
BZN = {"Franta (FR)": "fr", "Olanda (NL)": "nl", "Belgia (BE)": "be", "Austria (AT)": "at",
       "Polonia (PL)": "pl", "Cehia (CZ)": "cz", "Danemarca V (DK1)": "dk1",
       "Elvetia (CH)": "ch", "Suedia (SE4)": "se4", "Norvegia (NO2)": "no2"}
PRICE_BZN = {"fr": "FR", "nl": "NL", "be": "BE", "at": "AT", "pl": "PL", "cz": "CZ",
             "dk1": "DK1", "ch": "CH", "se4": "SE4", "no2": "NO2"}

today = dt.date.today()
tomorrow = today + dt.timedelta(days=1)

# ---------------- Energy-Charts ----------------
@st.cache_data(ttl=1800, show_spinner=False)
def ec_price(bzn, s, e):
    r = requests.get(f"{EC}/price", params={"bzn": bzn, "start": s, "end": e}, timeout=45)
    r.raise_for_status(); j = r.json()
    idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
    return pd.Series(j["price"], index=idx).astype(float).resample("1h").mean()

@st.cache_data(ttl=1800, show_spinner=False)
def ec_residual(country, s, e):
    """Residual load (MW). Daca lipseste coloana, il calculez din Load - vant - solar."""
    r = requests.get(f"{EC}/public_power", params={"country": country, "start": s, "end": e}, timeout=45)
    r.raise_for_status(); j = r.json()
    idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
    cols = {}
    for p in j.get("production_types", []):
        cols[p["name"]] = pd.Series(p["data"], index=idx)
    df = pd.DataFrame(cols)
    res = next((c for c in df.columns if "residual" in c.lower()), None)
    if res:
        return df[res].resample("1h").mean()
    load = next((c for c in df.columns if c.lower().strip() == "load"), None)
    ren = [c for c in df.columns if any(k in c.lower() for k in ["solar", "wind"]) and "residual" not in c.lower()]
    if load and ren:
        return (df[load] - df[ren].sum(axis=1)).resample("1h").mean()
    return pd.Series(dtype=float)

# ---------------- ENTSO-E (forecast DE mâine) ----------------
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
def de_rl_forecast_tomorrow():
    token = st.secrets.get("ENTSOE_TOKEN", "")
    if not token: raise RuntimeError("Lipseste ENTSOE_TOKEN in secrets.")
    s = pd.Timestamp(tomorrow, tz=TZ).tz_convert("UTC").strftime("%Y%m%d%H%M")
    e = pd.Timestamp(tomorrow + dt.timedelta(days=1), tz=TZ).tz_convert("UTC").strftime("%Y%m%d%H%M")
    def call(params):
        p = dict(params); p["securityToken"] = token
        r = requests.get(ENTSOE, params=p, timeout=60); r.raise_for_status()
        return _parse(r.content)
    load = call({"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": DE_LU,
                 "periodStart": s, "periodEnd": e})
    ws = call({"documentType": "A69", "processType": "A01", "in_Domain": DE_LU,
               "periodStart": s, "periodEnd": e})
    L = load["ALL"].tz_convert(TZ).resample("1h").mean() if "ALL" in load else pd.Series(dtype=float)
    ren = None
    for k in ["B16", "B18", "B19"]:
        if k in ws:
            s2 = ws[k].tz_convert(TZ).resample("1h").mean()
            ren = s2 if ren is None else ren.add(s2, fill_value=0)
    if L.empty or ren is None: return pd.Series(dtype=float)
    return (L - ren.reindex(L.index).fillna(0)).rename("rl")

# ---------------- UI ----------------
c1, c2, c3 = st.columns([2, 1, 1])
label = c1.selectbox("Border", list(BZN))
days = c2.selectbox("Istoric (zile)", [30, 60, 90], index=1)
band = c3.number_input("Prag binding |spread| (EUR/MWh)", 0.5, 20.0, 3.0, 0.5)
country = BZN[label]

s_hist = (today - dt.timedelta(days=days)).isoformat()
e_hist = today.isoformat()

# ---------------- Istoric: eticheta binding + feature divergenta ----------------
try:
    de_p = ec_price("DE-LU", s_hist, e_hist)
    nb_p = ec_price(PRICE_BZN[country], s_hist, e_hist)
    de_rl = ec_residual("de", s_hist, e_hist)
    nb_rl = ec_residual(country, s_hist, e_hist)
except Exception as ex:
    st.error(f"Eroare date istorice: {ex}"); st.stop()

hist = pd.concat([de_p.rename("de_p"), nb_p.rename("nb_p"),
                  de_rl.rename("de_rl"), nb_rl.rename("nb_rl")], axis=1).dropna()
if len(hist) < 100:
    st.warning("Date insuficiente pentru acest border."); st.stop()

hist["binding"] = (hist["de_p"] - hist["nb_p"]).abs() >= band
# standardizare residual load fiecare zona
de_mu, de_sd = hist["de_rl"].mean(), hist["de_rl"].std()
nb_mu, nb_sd = hist["nb_rl"].mean(), hist["nb_rl"].std()
hist["z_de"] = (hist["de_rl"] - de_mu) / de_sd
hist["z_nb"] = (hist["nb_rl"] - nb_mu) / nb_sd
hist["div"] = hist["z_de"] - hist["z_nb"]   # + => DE relativ mai strans => import => border se umple

base_rate = hist["binding"].mean() * 100
st.metric(f"Rata de binding istorica — DE ↔ {label}", f"{base_rate:.0f}% din ore",
          help=f"Ore cu |spread| >= {band} EUR/MWh in ultimele {days} zile.")

# ---------------- Curba elasticitate: P(binding) vs divergenta ----------------
st.subheader("① Elasticitatea: cât divergență de residual load saturează border-ul")
bins = np.quantile(hist["div"], np.linspace(0, 1, 13))
bins = np.unique(bins)
hist["b"] = pd.cut(hist["div"], bins=bins, include_lowest=True)
curve = hist.groupby("b", observed=True).agg(div=("div", "mean"), p=("binding", "mean")).dropna()
fig = go.Figure(go.Scatter(x=curve["div"], y=curve["p"] * 100, mode="lines+markers"))
fig.update_layout(xaxis_title="Divergenta RL (z_DE − z_vecin)  →  DE relativ mai strans",
                  yaxis_title="P(binding) %", height=360)
st.plotly_chart(fig, use_container_width=True)
st.caption("Panta abrupta = border sensibil (se umple usor la divergenta mica). "
           "Aripa dreapta sus = DE strans vs vecin larg → import saturat. "
           "Aripa stanga sus = DE larg vs vecin strans → export saturat.")

# ---------------- Base rate pe ora ----------------
st.subheader("② Ce ore congestionează cronic")
by_hour = hist.groupby(hist.index.hour)["binding"].mean() * 100
fig2 = go.Figure(go.Bar(x=by_hour.index, y=by_hour.values))
fig2.update_layout(xaxis_title="Ora", yaxis_title="P(binding) %", height=300)
st.plotly_chart(fig2, use_container_width=True)

# ---------------- Forecast mâine ----------------
st.subheader(f"③ Probabilitate de binding — MAINE ({tomorrow})")
try:
    de_fc = de_rl_forecast_tomorrow()
except Exception as ex:
    st.info(f"Forecast DE indisponibil ({ex}). Adauga ENTSOE_TOKEN pentru scorarea de mâine.")
    st.stop()
if de_fc.empty:
    st.warning("Prognoza DE pentru mâine nu e inca publicata."); st.stop()

# climatologie vecin: RL mediu pe (zi saptamana, ora)
clim = nb_rl.groupby([nb_rl.index.dayofweek, nb_rl.index.hour]).mean()
z_de_fc = (de_fc - de_mu) / de_sd
z_nb_fc = pd.Series(
    [(clim.get((ts.dayofweek, ts.hour), nb_mu) - nb_mu) / nb_sd for ts in de_fc.index],
    index=de_fc.index)
div_fc = z_de_fc - z_nb_fc
# mapare pe curba invatata
p_fc = np.interp(div_fc.values, curve["div"].values, curve["p"].values,
                 left=curve["p"].values[0], right=curve["p"].values[-1]) * 100

out = pd.DataFrame({"RL_DE_fc": de_fc.values.round(0),
                    "Divergenta": div_fc.values.round(2),
                    "P(binding) %": p_fc.round(0)}, index=de_fc.index)
c = st.columns(3)
c[0].metric("Ore probabile binding (>50%)", f"{int((p_fc > 50).sum())}")
c[1].metric("P(binding) medie mâine", f"{p_fc.mean():.0f}%")
peak = out["P(binding) %"].idxmax()
c[2].metric("Ora cea mai riscanta", f"{peak:%H:%M} ({out.loc[peak,'P(binding) %']:.0f}%)")

fig3 = go.Figure(go.Bar(x=out.index, y=out["P(binding) %"],
                        marker=dict(color=out["P(binding) %"], colorscale="Reds")))
fig3.update_layout(yaxis_title="P(binding) %", height=340)
st.plotly_chart(fig3, use_container_width=True)
with st.expander("📋 Tabel orar mâine"):
    st.dataframe(out, use_container_width=True)

st.caption("Semnul divergentei = directia: + DE importa (border de import se umple), − DE exporta. "
           "Vecinul e pus la climatologie (medie pe zi-ora) — se ascute cand cablam forecast-ul lui "
           "ENTSO-E si capacitatea reala JAO de la 11:00.")
st.divider()
st.info("**Ce lipseste ca sa fie complet:** (1) capacitatea JAO la 11:00 ca ceiling — acum modelul "
        "invata din binding-ul realizat, care deja reflecta capacitatea istorica; cand o cablam, "
        "capacitatea intra ca feature si prinzi zilele cand JAO taie neasteptat maxBex. "
        "(2) shadow price pe CNEC = confirmarea cauzala directa a border-ului activ.")
st.caption("Surse: ENTSO-E (forecast) + Energy-Charts / Fraunhofer ISE (CC BY 4.0).")
