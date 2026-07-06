# app.py — DE-LU Day-Ahead Fundamental Forecaster (v1)
# Motor merit-order pentru piata germana. Sursa de date: Energy-Charts (Fraunhofer ISE),
# fara token necesar, licenta CC BY 4.0. Preturile combustibililor sunt input manual.
#
# NOTA DE INVATARE (citeste asta prima data):
# Pretul DA in Germania ~ costul marginal al ultimei unitati dispecerizate ca sa acopere
# "residual load"-ul (consum minus regenerabile). Dupa iesirea nuclearului (2023), NIVELUL
# pretului in orele termice e dat de care combustibil e la margine: lignit -> carbune -> gaz.
# Cine castiga depinde de clean dark spread (carbune+CO2) vs clean spark spread (gaz+CO2).
# Forma zilei (valea de la pranz vara, spike-urile de Dunkelflaute iarna) e data de regenerabile.

import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="DE-LU DA Forecaster", page_icon="⚡", layout="wide")

BASE = "https://api.energy-charts.info"
TZ = "Europe/Berlin"

# ----------------------------------------------------------------------------------
# 1) STRAT DE DATE — Energy-Charts (fara cheie)
# ----------------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def ec_public_power(country: str, start: str, end: str) -> pd.DataFrame:
    """Mix de productie + Load + Residual load, in MW. Rezolutie 15 min pentru DE."""
    r = requests.get(f"{BASE}/public_power",
                     params={"country": country, "start": start, "end": end}, timeout=45)
    r.raise_for_status()
    j = r.json()
    idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
    df = pd.DataFrame(index=idx)
    for pt in j.get("production_types", []):
        df[pt["name"]] = pt["data"]
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def ec_price(bzn: str, start: str, end: str) -> pd.Series:
    """Pret DA (EUR/MWh) pentru zona de licitatie."""
    r = requests.get(f"{BASE}/price",
                     params={"bzn": bzn, "start": start, "end": end}, timeout=45)
    r.raise_for_status()
    j = r.json()
    idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
    return pd.Series(j["price"], index=idx, name="DA").astype(float)

@st.cache_data(ttl=1800, show_spinner=False)
def ec_forecast(country: str, production_type: str) -> pd.Series:
    """Forecast Energy-Charts pentru solar / wind_onshore / wind_offshore / load."""
    try:
        r = requests.get(f"{BASE}/public_power_forecast",
                         params={"country": country, "production_type": production_type},
                         timeout=45)
        r.raise_for_status()
        j = r.json()
        idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
        # cheia variaza; luam primul array numeric care nu e timestampul
        for k, v in j.items():
            if k not in ("unix_seconds", "deprecated") and isinstance(v, list) and len(v) == len(idx):
                return pd.Series(v, index=idx, name=production_type).astype(float)
    except Exception:
        pass
    return pd.Series(dtype=float, name=production_type)

def find_col(df: pd.DataFrame, *keywords) -> str | None:
    """Gaseste o coloana care contine toate cuvintele cheie (case-insensitive)."""
    for c in df.columns:
        low = c.lower()
        if all(k in low for k in keywords):
            return c
    return None

def hourly(s: pd.Series | pd.DataFrame):
    """Aduce totul pe rezolutie orara (medie), ca sa aliniem MW cu preturile."""
    return s.resample("1h").mean()

# ----------------------------------------------------------------------------------
# 2) MOTOR MERIT-ORDER
# ----------------------------------------------------------------------------------
# SRMC (Short Run Marginal Cost, EUR/MWh_el) =
#   pret_combustibil_termic/randament + pret_CO2 * factor_emisie/randament + VOM
# factor_emisie: tCO2 / MWh_termic. randament: fractie (electric/termic).

def srmc(fuel_th: float, co2: float, ef: float, eff: float, vom: float) -> float:
    return fuel_th / eff + co2 * ef / eff + vom

def build_stack(p: dict) -> list[dict]:
    """Construieste stiva de unitati dispecerizabile, sortata dupa cost marginal."""
    coal_eur_per_t = p["coal_usd_t"] / p["eur_usd"]          # $/t -> EUR/t
    coal_th = coal_eur_per_t / p["coal_mwh_t"]               # EUR/t -> EUR/MWh_th

    layers = [
        {"tech": "Must-run (biomasa+hidro RoR)", "srmc": p["mustrun_cost"], "cap": p["cap_mustrun"]},
        {"tech": "Lignit",  "srmc": srmc(p["lignite_th"], p["co2"], p["ef_lignite"], p["eff_lignite"], p["vom_lignite"]), "cap": p["cap_lignite"]},
        {"tech": "Carbune", "srmc": srmc(coal_th,        p["co2"], p["ef_coal"],    p["eff_coal"],    p["vom_coal"]),    "cap": p["cap_coal"]},
        {"tech": "Gaz CCGT","srmc": srmc(p["ttf"],       p["co2"], p["ef_gas"],     p["eff_ccgt"],    p["vom_gas"]),     "cap": p["cap_ccgt"]},
        {"tech": "Gaz OCGT","srmc": srmc(p["ttf"],       p["co2"], p["ef_gas"],     p["eff_ocgt"],    p["vom_gas"]),     "cap": p["cap_ocgt"]},
        {"tech": "Pacura",  "srmc": srmc(p["oil_th"],    p["co2"], p["ef_oil"],     p["eff_oil"],     p["vom_oil"]),     "cap": p["cap_oil"]},
    ]
    layers.sort(key=lambda x: x["srmc"])
    cum = 0.0
    for L in layers:
        cum += L["cap"]
        L["cum"] = cum
    return layers

def marginal_price(rl_mw: float, stack: list[dict], floor: float, scarcity: float) -> float:
    """Pretul = SRMC-ul tehnologiei a carei capacitate cumulata acopera residual load-ul."""
    if rl_mw <= 0:
        return floor
    for L in stack:
        if rl_mw <= L["cum"]:
            return L["srmc"]
    return scarcity  # RL depaseste toata capacitatea dispecerizabila -> scarcity

def clean_spreads(power: float, p: dict):
    """Clean spark (gaz) si clean dark (carbune) — arata ce tehnologie e la margine."""
    coal_th = (p["coal_usd_t"] / p["eur_usd"]) / p["coal_mwh_t"]
    spark = power - srmc(p["ttf"],  p["co2"], p["ef_gas"],  p["eff_ccgt"], 0)
    dark  = power - srmc(coal_th,   p["co2"], p["ef_coal"], p["eff_coal"], 0)
    return spark, dark

# ----------------------------------------------------------------------------------
# 3) SIDEBAR — input-uri combustibili si parc
# ----------------------------------------------------------------------------------

st.sidebar.title("⚡ DE-LU DA Forecaster")
page = st.sidebar.radio("Sectiune", ["📊 Piata", "⚙️ Motor merit-order", "🎯 Forecast vs DA", "📚 Teorie"])

st.sidebar.header("Combustibili & CO2")
ttf        = st.sidebar.number_input("Gaz TTF (EUR/MWh_th)", 5.0, 200.0, 32.0, 0.5)
coal_usd_t = st.sidebar.number_input("Carbune API2 ($/t)", 40.0, 400.0, 110.0, 1.0)
eur_usd    = st.sidebar.number_input("EUR/USD", 0.90, 1.30, 1.08, 0.01)
coal_mwh_t = st.sidebar.number_input("Continut energetic carbune (MWh_th/t)", 5.0, 9.0, 6.978, 0.05,
                                     help="API2 6000 kcal/kg NAR ≈ 6.978 MWh/t. Ajusteaza dupa specificatie.")
co2        = st.sidebar.number_input("CO2 EUA (EUR/t)", 20.0, 200.0, 75.0, 1.0)
lignite_th = st.sidebar.number_input("Cost lignit (EUR/MWh_th)", 2.0, 20.0, 6.0, 0.5,
                                     help="Lignitul e local, cost ~ minerit; nu are piata lichida.")
oil_th     = st.sidebar.number_input("Cost pacura (EUR/MWh_th)", 20.0, 120.0, 55.0, 1.0)

with st.sidebar.expander("Randamente (η) & VOM"):
    eff_lignite = st.slider("η Lignit", 0.30, 0.46, 0.38, 0.01)
    eff_coal    = st.slider("η Carbune", 0.35, 0.50, 0.44, 0.01)
    eff_ccgt    = st.slider("η Gaz CCGT", 0.45, 0.62, 0.55, 0.01)
    eff_ocgt    = st.slider("η Gaz OCGT", 0.30, 0.42, 0.38, 0.01)
    eff_oil     = st.slider("η Pacura", 0.30, 0.42, 0.35, 0.01)
    vom_lignite = st.number_input("VOM lignit", 0.0, 10.0, 3.0, 0.5)
    vom_coal    = st.number_input("VOM carbune", 0.0, 10.0, 3.0, 0.5)
    vom_gas     = st.number_input("VOM gaz", 0.0, 10.0, 2.0, 0.5)
    vom_oil     = st.number_input("VOM pacura", 0.0, 10.0, 3.0, 0.5)

with st.sidebar.expander("Capacitati disponibile (MW)"):
    cap_mustrun = st.number_input("Must-run", 0, 20000, 8000, 500)
    cap_lignite = st.number_input("Lignit", 0, 30000, 14000, 500)
    cap_coal    = st.number_input("Carbune", 0, 30000, 13000, 500)
    cap_ccgt    = st.number_input("Gaz CCGT", 0, 40000, 20000, 500)
    cap_ocgt    = st.number_input("Gaz OCGT", 0, 30000, 12000, 500)
    cap_oil     = st.number_input("Pacura", 0, 10000, 4000, 500)

with st.sidebar.expander("Preturi de plafon"):
    floor    = st.number_input("Plafon jos / regenerabil (EUR/MWh)", -500.0, 20.0, -5.0, 5.0)
    scarcity = st.number_input("Pret scarcity (EUR/MWh)", 200.0, 4000.0, 500.0, 50.0)

# Factori de emisie ficsi (tCO2/MWh_th) — valori standard
EF = {"lignite": 0.36, "coal": 0.34, "gas": 0.20, "oil": 0.28}

P = dict(ttf=ttf, coal_usd_t=coal_usd_t, eur_usd=eur_usd, coal_mwh_t=coal_mwh_t, co2=co2,
         lignite_th=lignite_th, oil_th=oil_th,
         eff_lignite=eff_lignite, eff_coal=eff_coal, eff_ccgt=eff_ccgt, eff_ocgt=eff_ocgt, eff_oil=eff_oil,
         vom_lignite=vom_lignite, vom_coal=vom_coal, vom_gas=vom_gas, vom_oil=vom_oil,
         ef_lignite=EF["lignite"], ef_coal=EF["coal"], ef_gas=EF["gas"], ef_oil=EF["oil"],
         cap_mustrun=cap_mustrun, cap_lignite=cap_lignite, cap_coal=cap_coal,
         cap_ccgt=cap_ccgt, cap_ocgt=cap_ocgt, cap_oil=cap_oil, mustrun_cost=1.0)

stack = build_stack(P)

# ----------------------------------------------------------------------------------
# 4) PAGINI
# ----------------------------------------------------------------------------------

def load_market(days_back: int):
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=days_back)
    s, e = start.isoformat(), end.isoformat()
    pp = ec_public_power("de", s, e)
    da = ec_price("DE-LU", s, e)
    return pp, da

# ===== PAGINA: PIATA =====
if page == "📊 Piata":
    st.title("📊 Piata DE-LU — date reale")
    days = st.selectbox("Interval (zile in urma)", [3, 7, 14, 30], index=1)
    try:
        pp, da = load_market(days)
    except Exception as ex:
        st.error(f"Eroare la incarcarea datelor: {ex}")
        st.stop()

    c_load  = find_col(pp, "load") if not find_col(pp, "residual") else find_col(pp, "load")
    c_load  = find_col(pp, "load")
    c_res   = find_col(pp, "residual")
    c_won   = find_col(pp, "wind", "onshore")
    c_woff  = find_col(pp, "wind", "offshore")
    c_sol   = find_col(pp, "solar")

    pph = hourly(pp)
    dah = hourly(da)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("DA mediu (EUR/MWh)", f"{dah.mean():.1f}")
    k1.metric("DA min / max", f"{dah.min():.0f} / {dah.max():.0f}")
    if c_res:  k2.metric("Residual load mediu (MW)", f"{pph[c_res].mean():,.0f}")
    neg = (dah < 0).sum()
    k3.metric("Ore cu pret negativ", f"{int(neg)}")
    if c_sol:  k4.metric("Solar peak (MW)", f"{pph[c_sol].max():,.0f}")

    st.subheader("Pret DA")
    st.line_chart(dah)

    st.subheader("Residual load vs regenerabile")
    cols = [c for c in [c_res, c_won, c_woff, c_sol] if c]
    if cols:
        st.line_chart(pph[cols])

    st.subheader("Pret DA vs Residual load (scatter)")
    if c_res:
        j = pd.concat([dah.rename("DA"), pph[c_res].rename("RL")], axis=1).dropna()
        fig = go.Figure(go.Scatter(x=j["RL"], y=j["DA"], mode="markers",
                                   marker=dict(size=5, opacity=0.5)))
        fig.update_layout(xaxis_title="Residual load (MW)", yaxis_title="DA (EUR/MWh)", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Norul asta ESTE merit order-ul empiric. Panta = cat de scump devine sistemul "
                   "cand creste RL. Imprastierea verticala = variatia combustibililor/outage-urilor.")

# ===== PAGINA: MOTOR MERIT-ORDER =====
elif page == "⚙️ Motor merit-order":
    st.title("⚙️ Motor merit-order")
    st.markdown("Stiva de unitati sortata dupa cost marginal. Rasuceste butoanele din sidebar "
                "si uita-te cum se reordoneaza si cum se misca curba pret(RL).")

    dfstack = pd.DataFrame(stack)
    dfstack = dfstack.rename(columns={"tech": "Tehnologie", "srmc": "SRMC (EUR/MWh)",
                                      "cap": "Capacitate (MW)", "cum": "Cumulat (MW)"})
    dfstack["SRMC (EUR/MWh)"] = dfstack["SRMC (EUR/MWh)"].round(1)
    st.dataframe(dfstack, use_container_width=True, hide_index=True)

    # Fuel switching: la ce pret de energie gazul intra sub carbune?
    coal_th = (P["coal_usd_t"] / P["eur_usd"]) / P["coal_mwh_t"]
    srmc_ccgt = srmc(P["ttf"], P["co2"], EF["gas"], P["eff_ccgt"], P["vom_gas"])
    srmc_coal = srmc(coal_th, P["co2"], EF["coal"], P["eff_coal"], P["vom_coal"])
    marginal_tech = "GAZ (CCGT)" if srmc_ccgt < srmc_coal else "CARBUNE"

    c1, c2, c3 = st.columns(3)
    c1.metric("SRMC Gaz CCGT", f"{srmc_ccgt:.1f}")
    c2.metric("SRMC Carbune", f"{srmc_coal:.1f}")
    c3.metric("La margine (termic)", marginal_tech)

    st.subheader("Curba de oferta preț(residual load)")
    rl_grid = np.arange(0, sum(L["cap"] for L in stack) + 5000, 250)
    prices = [marginal_price(rl, stack, floor, scarcity) for rl in rl_grid]
    fig = go.Figure(go.Scatter(x=rl_grid, y=prices, mode="lines", line_shape="hv"))
    fig.update_layout(xaxis_title="Residual load (MW)", yaxis_title="Pret marginal (EUR/MWh)", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Aceasta e curba pe care o compari cu norul empiric din pagina Piata. "
               "Daca norul real sta mai sus/jos decat curba ta, ori combustibilii, ori "
               "capacitatile, ori randamentele tale sunt gresite — asta e feedback-ul de calibrare.")

# ===== PAGINA: FORECAST VS DA =====
elif page == "🎯 Forecast vs DA":
    st.title("🎯 Preț fundamental vs DA realizat")
    days = st.selectbox("Interval de backtest (zile)", [7, 14, 30], index=1)
    try:
        pp, da = load_market(days)
    except Exception as ex:
        st.error(f"Eroare: {ex}")
        st.stop()

    c_res = find_col(pp, "residual")
    if not c_res:
        st.warning("Nu am gasit coloana 'Residual load' in raspuns.")
        st.stop()

    pph = hourly(pp)
    dah = hourly(da)
    rl = pph[c_res]
    fund = rl.apply(lambda x: marginal_price(x, stack, floor, scarcity)).rename("Fundamental")

    j = pd.concat([dah.rename("DA"), fund], axis=1).dropna()
    err = j["Fundamental"] - j["DA"]
    mae = err.abs().mean()
    bias = err.mean()
    corr = j["DA"].corr(j["Fundamental"])

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (EUR/MWh)", f"{mae:.1f}")
    c2.metric("Bias (fund - DA)", f"{bias:+.1f}")
    c3.metric("Corelatie", f"{corr:.2f}")

    st.subheader("Serie: DA realizat vs Fundamental")
    st.line_chart(j)

    st.subheader("Scatter fundamental vs DA")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=j["DA"], y=j["Fundamental"], mode="markers", marker=dict(size=5, opacity=0.5)))
    lim = [min(j.min().min(), 0), j.max().max()]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash"), name="perfect"))
    fig.update_layout(xaxis_title="DA realizat", yaxis_title="Fundamental", height=420, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.info("**Interpretare:** un bias pozitiv constant = modelul tau supraestimeaza "
            "(probabil combustibili/CO2 prea scumpi sau capacitati prea mici). "
            "MAE mare doar in orele de varf = problema de coada (scarcity/peakere). "
            "MAE mare la pranz = problema pe zona negativa/solar. Aici incepe calibrarea.")

# ===== PAGINA: TEORIE =====
else:
    st.title("📚 Teorie — cum se formeaza prețul DA in DE-LU")
    st.markdown(r"""
### 1. Ecuatia de baza
**Residual Load (RL) = Consum − Eolian − Solar − must-run.**
Pretul DA ≈ costul marginal al ultimei unitati care acopera RL, ajustat de cuplajul cross-border (flow-based Core).

### 2. Cine e la margine (post-nuclear)
Merit order: **regenerabile (~0) → lignit → carbune → gaz CCGT → gaz OCGT → pacura/scarcity.**
- **Clean dark spread** = Pret − (carbune/η + CO2·EF/η) → profitul carbunelui.
- **Clean spark spread** = Pret − (gaz/η + CO2·EF/η) → profitul gazului.
- Cand gazul e ieftin fata de carbune+CO2, CCGT intra sub carbune si "taie" nivelul.

### 3. Doua regimuri
- **Regenerabil abundent** → RL mic → pret se prabuseste, adesea negativ (regula EEG suspenda subventia in orele negative → schimba biddarea).
- **Dunkelflaute** (vant slab + soare putin, iarna) → RL mare → peakere → spike-uri.

### 4. Unde e alpha
Banii NU vin din a avea dreptate in absolut, ci din a avea **mai multa** dreptate decat consensul.
Backtesteaza mereu *forecast-ul tau vs. curba/settlement de la momentul deciziei*, nu vs. realizat.
Edge-ul e cel mai ascutit pe scadentele front (Day/Weekend/Week), unde vizibilitatea meteo e buna.

### 5. Descompunerea erorii (cel mai profitabil exercitiu la inceput)
Cat din eroarea ta DA vine din: eroare de consum? de vant? de solar? de cross-border? de combustibil?
Iti spune unde sa investesti efort. Fara asta, calibrezi orbeste.
""")
    st.caption("Sursa date live: Energy-Charts / Fraunhofer ISE (CC BY 4.0).")

st.sidebar.divider()
st.sidebar.caption("Date: Energy-Charts (Fraunhofer ISE), CC BY 4.0. Preturi combustibili: input manual.")
