# pages/4_Interconexiuni.py — DOVADA ca interconexiunile misca pretul DE-LU
# Nivel 1: convergenta de pret (amprenta cuplarii). Nivel 2: spread vs flux. Nivel 3: event study.
# Surse: Energy-Charts /price (sigur) + /cbet (best-effort, parsare defensiva).

import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Interconexiuni", page_icon="🔗", layout="wide")
st.title("🔗 Interconexiuni — dovada ca misca prețul DE-LU")

TZ = "Europe/Berlin"
EC = "https://api.energy-charts.info"

# Vecinii DE-LU cu codurile de zona Energy-Charts
BZN = {"Franta (FR)": "FR", "Olanda (NL)": "NL", "Belgia (BE)": "BE",
       "Austria (AT)": "AT", "Polonia (PL)": "PL", "Cehia (CZ)": "CZ",
       "Danemarca V (DK1)": "DK1", "Elvetia (CH)": "CH",
       "Suedia (SE4)": "SE4", "Norvegia (NO2)": "NO2"}

@st.cache_data(ttl=1800, show_spinner=False)
def ec_price(bzn, start, end):
    r = requests.get(f"{EC}/price", params={"bzn": bzn, "start": start, "end": end}, timeout=45)
    r.raise_for_status(); j = r.json()
    idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
    return pd.Series(j["price"], index=idx).astype(float).resample("1h").mean()

@st.cache_data(ttl=1800, show_spinner=False)
def ec_cbet(start, end):
    """Flux comercial DE <-> vecini. Parsare defensiva (schema poate varia)."""
    try:
        j = requests.get(f"{EC}/cbet", params={"country": "de", "start": start, "end": end}, timeout=45).json()
    except Exception:
        return pd.DataFrame()
    if "unix_seconds" not in j:
        return pd.DataFrame()
    idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
    df = pd.DataFrame(index=idx)
    for key, val in j.items():
        if key == "unix_seconds":
            continue
        if isinstance(val, list) and val and isinstance(val[0], dict):
            for item in val:
                name = item.get("name") or item.get("country") or item.get("id")
                data = item.get("data")
                if name and isinstance(data, list) and len(data) == len(idx):
                    df[str(name)] = data
    return df.resample("1h").mean()

def match_flow(cbet_df, code, label):
    """Gaseste coloana de flux pt vecin dupa cod/nume."""
    if cbet_df.empty:
        return None
    country = label.split("(")[0].strip().lower()
    for c in cbet_df.columns:
        cl = c.lower()
        if code.lower() in cl or country[:4] in cl:
            return cbet_df[c]
    return None

# --- UI ---
c1, c2, c3 = st.columns([2, 1, 1])
sel = c1.multiselect("Vecini", list(BZN), default=["Franta (FR)", "Olanda (NL)", "Polonia (PL)"])
days = c2.selectbox("Interval (zile)", [14, 30, 60, 90], index=1)
band = c3.number_input("Banda convergenta (EUR/MWh)", 0.01, 5.0, 0.5, 0.25)

end = dt.date.today()
start = end - dt.timedelta(days=days)
s_iso, e_iso = start.isoformat(), end.isoformat()

try:
    de = ec_price("DE-LU", s_iso, e_iso)
except Exception as ex:
    st.error(f"Nu pot lua prețul DE-LU: {ex}"); st.stop()

if not sel:
    st.info("Alege cel putin un vecin."); st.stop()

cbet = ec_cbet(s_iso, e_iso)

# ---------------------------------------------------------------------------
# NIVEL 1 — convergenta de pret
# ---------------------------------------------------------------------------
st.subheader("① Convergența de preț — amprenta cuplării")
rows = []
spreads = {}
for label in sel:
    try:
        nb = ec_price(BZN[label], s_iso, e_iso)
    except Exception:
        continue
    j = pd.concat([de.rename("DE"), nb.rename("NB")], axis=1).dropna()
    if j.empty:
        continue
    sp = j["DE"] - j["NB"]
    spreads[label] = sp
    conv = (sp.abs() < band).mean() * 100
    rows.append({"Border": f"DE ↔ {label}",
                 "Convergenta %": round(conv, 1),
                 "|Spread| mediu": round(sp.abs().mean(), 1),
                 "Spread max": round(sp.abs().max(), 0),
                 "Corelatie": round(j["DE"].corr(j["NB"]), 3),
                 "Ore": len(j)})
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Convergenta % mare = border des cuplat (surplus curge liber, preturi egale). "
               "Restul orelor = congestie. Corelatia mare + convergenta mare = cuplare puternica. "
               "Corelatie mare DAR convergenta mica = zone care se misca la fel dar raman despartite de o limita.")

# ---------------------------------------------------------------------------
# NIVEL 1b — histograma spread (bimodalitatea = dovada)
# ---------------------------------------------------------------------------
focus = st.selectbox("Analiza detaliata pe border:", sel)
if focus in spreads:
    sp = spreads[focus]
    st.subheader(f"① Histograma spread DE − {focus}")
    fig = go.Figure(go.Histogram(x=sp.values, nbinsx=80))
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(xaxis_title="Spread (EUR/MWh)", yaxis_title="Ore", height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Vârful ascutit la zero = orele cuplate (pistolul fumegand). "
               "Masa departe de zero = congestie. Distributie bimodala = dovada cuplarii cu decuplari.")

    # ---------------------------------------------------------------------------
    # NIVEL 2 — spread vs flux (crosa de hochei)
    # ---------------------------------------------------------------------------
    flow = match_flow(cbet, BZN[focus], focus)
    st.subheader(f"② Spread vs flux comercial — DE ↔ {focus}")
    if flow is not None:
        jf = pd.concat([sp.rename("spread"), flow.rename("flux")], axis=1).dropna()
        if len(jf) > 10:
            fig = go.Figure(go.Scatter(x=jf["flux"], y=jf["spread"], mode="markers",
                                       marker=dict(size=5, opacity=0.4)))
            fig.update_layout(xaxis_title=f"Flux comercial DE↔{focus} (MW)",
                              yaxis_title="Spread DE − vecin (EUR/MWh)", height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Cauta 'crosa de hochei': fluxul creste cu spread-ul pana se satureaza, "
                       "apoi spread-ul urca la flux ~constant. Cotul = capacitatea limita "
                       "dezvaluita de date. Platoul vertical la marginea de flux = border binding.")
        else:
            st.info("Prea putine puncte flux/spread aliniate.")
    else:
        st.info("Fluxul comercial (cbet) pt acest vecin nu a fost recunoscut in raspuns. "
                "Il cablam cand confirmam schema — dovada Nivel 1 nu depinde de flux.")

    # ---------------------------------------------------------------------------
    # NIVEL 3 — event study (binding vs cuplat)
    # ---------------------------------------------------------------------------
    st.subheader(f"③ Event study — ore binding vs cuplate (DE ↔ {focus})")
    binding = sp[sp.abs() >= band]
    coupled = sp[sp.abs() < band]
    de_al = de.reindex(sp.index)
    c = st.columns(3)
    c[0].metric("Ore binding", f"{len(binding)} ({len(binding)/len(sp)*100:.0f}%)")
    c[1].metric("Preț DE mediu — binding", f"{de_al[sp.abs()>=band].mean():.1f}")
    c[2].metric("Preț DE mediu — cuplat", f"{de_al[sp.abs()<band].mean():.1f}")
    try:
        from scipy import stats
        de_b = de_al[sp.abs() >= band].dropna()
        de_c = de_al[sp.abs() < band].dropna()
        if len(de_b) > 5 and len(de_c) > 5:
            u, p = stats.mannwhitneyu(de_b, de_c, alternative="two-sided")
            st.write(f"Mann-Whitney: prețul DE in orele binding vs cuplate diferă "
                     f"{'semnificativ' if p < 0.05 else 'nesemnificativ'} (p = {p:.4f}).")
    except Exception:
        pass
    st.caption("Daca prețul DE in orele binding difera semnificativ de cel din orele cuplate, "
               "ai dovada ca decuplarea (congestia border-ului) chiar muta prețul DE, nu e coincidenta.")

# ---------------------------------------------------------------------------
# Serie temporala
# ---------------------------------------------------------------------------
st.subheader("Serie: DE-LU vs vecini")
plot = pd.concat([de.rename("DE-LU")] +
                 [ec_price(BZN[l], s_iso, e_iso).rename(l) for l in sel], axis=1).dropna()
st.line_chart(plot)

st.divider()
st.info("**Nivel 4 (dovada directa):** shadow prices JAO pe CNEC — un shadow price > 0 inseamna "
        "ca acea linie a re-modelat clearing-ul. Il cablam din integrarea ta JAO.\n\n"
        "**Nivel 5 (contrafactual):** pret 'island' din motorul tau merit-order vs DA realizat — "
        "diferenta explicata de border-ele binding = contributia cuplarii. Se leaga de pagina Motor.")
st.caption("Surse: Energy-Charts / Fraunhofer ISE (CC BY 4.0).")
