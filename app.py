# app.py — DE-LU Day-Ahead Fundamental Forecaster (v2, cu auto-calibrare)
# Sursa date: Energy-Charts (Fraunhofer ISE), fara token, CC BY 4.0.
# NOU: (1) curba empirica data-driven (fara parametri), (2) auto-calibrare structurala
# cu optimizer, (3) split train/test pentru control de supra-fitting.
#
# INVATARE: pretul DA ~ costul marginal al ultimei unitati care acopera residual load-ul
# (consum - regenerabile). Nivelul in orele termice = comutarea lignit->carbune->gaz,
# decisa de clean dark spread (carbune+CO2) vs clean spark spread (gaz+CO2).

import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="DE-LU DA Forecaster", page_icon="⚡", layout="wide")

BASE = "https://api.energy-charts.info"
TZ = "Europe/Berlin"
EF = {"lignite": 0.36, "coal": 0.34, "gas": 0.20, "oil": 0.28}  # tCO2/MWh_th

# ---------------------------------------------------------------------------
# 1) DATE — Energy-Charts (fara cheie)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def ec_public_power(country, start, end):
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
def ec_price(bzn, start, end):
    r = requests.get(f"{BASE}/price", params={"bzn": bzn, "start": start, "end": end}, timeout=45)
    r.raise_for_status()
    j = r.json()
    idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
    return pd.Series(j["price"], index=idx, name="DA").astype(float)

def find_col(df, *keywords):
    for c in df.columns:
        if all(k in c.lower() for k in keywords):
            return c
    return None

def hourly(s):
    return s.resample("1h").mean()

@st.cache_data(ttl=1800, show_spinner=False)
def load_market(days_back):
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=days_back)
    pp = ec_public_power("de", start.isoformat(), end.isoformat())
    da = ec_price("DE-LU", start.isoformat(), end.isoformat())
    return pp, da

# ---------------------------------------------------------------------------
# 2) MODEL STRUCTURAL (merit order)
# ---------------------------------------------------------------------------
def srmc(fuel_th, co2, ef, eff, vom):
    return fuel_th / eff + co2 * ef / eff + vom

def build_stack(p):
    coal_th = (p["coal_usd_t"] / p["eur_usd"]) / p["coal_mwh_t"]
    layers = [
        {"tech": "Must-run", "srmc": p["mustrun_cost"], "cap": p["cap_mustrun"]},
        {"tech": "Lignit",  "srmc": srmc(p["lignite_th"], p["co2"], EF["lignite"], p["eff_lignite"], p["vom_lignite"]), "cap": p["cap_lignite"]},
        {"tech": "Carbune", "srmc": srmc(coal_th,        p["co2"], EF["coal"],    p["eff_coal"],    p["vom_coal"]),    "cap": p["cap_coal"]},
        {"tech": "Gaz CCGT","srmc": srmc(p["ttf"],       p["co2"], EF["gas"],     p["eff_ccgt"],    p["vom_gas"]),     "cap": p["cap_ccgt"]},
        {"tech": "Gaz OCGT","srmc": srmc(p["ttf"],       p["co2"], EF["gas"],     p["eff_ocgt"],    p["vom_gas"]),     "cap": p["cap_ocgt"]},
        {"tech": "Pacura",  "srmc": srmc(p["oil_th"],    p["co2"], EF["oil"],     p["eff_oil"],     p["vom_oil"]),     "cap": p["cap_oil"]},
    ]
    layers.sort(key=lambda x: x["srmc"])
    cum = 0.0
    for L in layers:
        cum += L["cap"]; L["cum"] = cum
    return layers

def marginal_price_vec(rl_arr, stack, floor, scarcity):
    """Vectorizat: pentru fiecare RL, SRMC-ul primei tehnologii cu cumulat >= RL."""
    rl_arr = np.asarray(rl_arr, dtype=float)
    cums = np.array([L["cum"] for L in stack])
    srmcs = np.array([L["srmc"] for L in stack])
    idx = np.searchsorted(cums, rl_arr, side="left")
    over = idx >= len(srmcs)
    idx_c = np.clip(idx, 0, len(srmcs) - 1)
    out = np.where(over, scarcity, srmcs[idx_c])
    out = np.where(rl_arr <= 0, floor, out)
    return out

def clean_spreads(p):
    coal_th = (p["coal_usd_t"] / p["eur_usd"]) / p["coal_mwh_t"]
    srmc_ccgt = srmc(p["ttf"],  p["co2"], EF["gas"],  p["eff_ccgt"], p["vom_gas"])
    srmc_coal = srmc(coal_th,   p["co2"], EF["coal"], p["eff_coal"], p["vom_coal"])
    return srmc_ccgt, srmc_coal

# ---------------------------------------------------------------------------
# 3) MODEL EMPIRIC (data-driven, fara parametri)
# ---------------------------------------------------------------------------
def fit_empirical(rl, da, n_bins=25):
    d = pd.DataFrame({"rl": np.asarray(rl), "da": np.asarray(da)}).dropna().sort_values("rl")
    if len(d) < 10:
        return None
    try:
        d["bin"] = pd.qcut(d["rl"], q=min(n_bins, d["rl"].nunique()), duplicates="drop")
    except Exception:
        d["bin"] = pd.cut(d["rl"], bins=min(n_bins, 10))
    g = d.groupby("bin", observed=True).agg(rl=("rl", "median"), da=("da", "median")).dropna()
    if len(g) < 2:
        return None
    xs = g["rl"].values
    ys = np.maximum.accumulate(g["da"].values)  # monotonic crescator (merit order)
    return xs, ys

def predict_empirical(rl_new, model):
    xs, ys = model
    return np.interp(np.asarray(rl_new, dtype=float), xs, ys)

# ---------------------------------------------------------------------------
# 4) AUTO-CALIBRARE STRUCTURALA
# ---------------------------------------------------------------------------
def autocalibrate(rl_train, da_train, baseP):
    from scipy.optimize import differential_evolution
    rl = np.asarray(rl_train, dtype=float); da = np.asarray(da_train, dtype=float)
    # parametri liberi: eff_lignite, eff_coal, eff_ccgt, cap_mustrun, scarcity, floor
    bounds = [(0.30, 0.46), (0.35, 0.50), (0.45, 0.62), (0.0, 20000.0), (150.0, 1500.0), (-100.0, 10.0)]
    def obj(x):
        p = dict(baseP)
        p["eff_lignite"], p["eff_coal"], p["eff_ccgt"], p["cap_mustrun"] = x[0], x[1], x[2], x[3]
        stk = build_stack(p)
        pred = marginal_price_vec(rl, stk, x[5], x[4])
        return float(np.mean(np.abs(pred - da)))
    res = differential_evolution(obj, bounds, seed=42, maxiter=40, popsize=12,
                                 tol=1e-3, polish=True)
    return res.x, res.fun

def metrics(pred, actual):
    pred = np.asarray(pred, dtype=float); actual = np.asarray(actual, dtype=float)
    m = ~(np.isnan(pred) | np.isnan(actual))
    pred, actual = pred[m], actual[m]
    if len(pred) == 0:
        return dict(mae=np.nan, bias=np.nan, corr=np.nan)
    return dict(mae=np.mean(np.abs(pred - actual)),
                bias=np.mean(pred - actual),
                corr=np.corrcoef(pred, actual)[0, 1] if len(pred) > 1 else np.nan)

# ---------------------------------------------------------------------------
# 5) SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("⚡ DE-LU DA Forecaster")
page = st.sidebar.radio("Sectiune", ["📊 Piata", "⚙️ Motor merit-order", "🎯 Forecast vs DA", "📚 Teorie"])

st.sidebar.header("Combustibili & CO2")
ttf        = st.sidebar.number_input("Gaz TTF (EUR/MWh_th)", 5.0, 200.0, 32.0, 0.5)
coal_usd_t = st.sidebar.number_input("Carbune API2 ($/t)", 40.0, 400.0, 110.0, 1.0)
eur_usd    = st.sidebar.number_input("EUR/USD", 0.90, 1.30, 1.08, 0.01)
coal_mwh_t = st.sidebar.number_input("Continut energetic carbune (MWh_th/t)", 5.0, 9.0, 6.978, 0.05)
co2        = st.sidebar.number_input("CO2 EUA (EUR/t)", 20.0, 200.0, 75.0, 1.0)
lignite_th = st.sidebar.number_input("Cost lignit (EUR/MWh_th)", 2.0, 20.0, 6.0, 0.5)
oil_th     = st.sidebar.number_input("Cost pacura (EUR/MWh_th)", 20.0, 120.0, 55.0, 1.0)

with st.sidebar.expander("Randamente & VOM (start manual)"):
    eff_lignite = st.slider("η Lignit", 0.30, 0.46, 0.38, 0.01)
    eff_coal    = st.slider("η Carbune", 0.35, 0.50, 0.44, 0.01)
    eff_ccgt    = st.slider("η Gaz CCGT", 0.45, 0.62, 0.55, 0.01)
    eff_ocgt    = st.slider("η Gaz OCGT", 0.30, 0.42, 0.38, 0.01)
    eff_oil     = st.slider("η Pacura", 0.30, 0.42, 0.35, 0.01)
    vom_lignite = st.number_input("VOM lignit", 0.0, 10.0, 3.0, 0.5)
    vom_coal    = st.number_input("VOM carbune", 0.0, 10.0, 3.0, 0.5)
    vom_gas     = st.number_input("VOM gaz", 0.0, 10.0, 2.0, 0.5)
    vom_oil     = st.number_input("VOM pacura", 0.0, 10.0, 3.0, 0.5)

with st.sidebar.expander("Capacitati (MW)"):
    cap_mustrun = st.number_input("Must-run", 0, 20000, 8000, 500)
    cap_lignite = st.number_input("Lignit", 0, 30000, 14000, 500)
    cap_coal    = st.number_input("Carbune", 0, 30000, 13000, 500)
    cap_ccgt    = st.number_input("Gaz CCGT", 0, 40000, 20000, 500)
    cap_ocgt    = st.number_input("Gaz OCGT", 0, 30000, 12000, 500)
    cap_oil     = st.number_input("Pacura", 0, 10000, 4000, 500)

floor    = st.sidebar.number_input("Plafon jos (EUR/MWh)", -500.0, 20.0, -5.0, 5.0)
scarcity = st.sidebar.number_input("Pret scarcity (EUR/MWh)", 200.0, 4000.0, 500.0, 50.0)

P = dict(ttf=ttf, coal_usd_t=coal_usd_t, eur_usd=eur_usd, coal_mwh_t=coal_mwh_t, co2=co2,
         lignite_th=lignite_th, oil_th=oil_th, mustrun_cost=1.0,
         eff_lignite=eff_lignite, eff_coal=eff_coal, eff_ccgt=eff_ccgt, eff_ocgt=eff_ocgt, eff_oil=eff_oil,
         vom_lignite=vom_lignite, vom_coal=vom_coal, vom_gas=vom_gas, vom_oil=vom_oil,
         cap_mustrun=cap_mustrun, cap_lignite=cap_lignite, cap_coal=cap_coal,
         cap_ccgt=cap_ccgt, cap_ocgt=cap_ocgt, cap_oil=cap_oil)

stack_manual = build_stack(P)

# ---------------------------------------------------------------------------
# 6) PAGINI
# ---------------------------------------------------------------------------
if page == "📊 Piata":
    st.title("📊 Piata DE-LU — date reale")
    days = st.selectbox("Interval (zile)", [3, 7, 14, 30], index=1)
    try:
        pp, da = load_market(days)
    except Exception as ex:
        st.error(f"Eroare: {ex}"); st.stop()
    c_res = find_col(pp, "residual"); c_sol = find_col(pp, "solar")
    c_won = find_col(pp, "wind", "onshore"); c_woff = find_col(pp, "wind", "offshore")
    pph, dah = hourly(pp), hourly(da)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("DA mediu", f"{dah.mean():.1f}")
    if c_res: k2.metric("Residual load mediu (MW)", f"{pph[c_res].mean():,.0f}")
    k3.metric("Ore pret negativ", f"{int((dah < 0).sum())}")
    if c_sol: k4.metric("Solar peak (MW)", f"{pph[c_sol].max():,.0f}")
    st.subheader("Pret DA"); st.line_chart(dah)
    cols = [c for c in [c_res, c_won, c_woff, c_sol] if c]
    if cols:
        st.subheader("Residual load vs regenerabile"); st.line_chart(pph[cols])
    if c_res:
        st.subheader("DA vs Residual load (norul = merit order empiric)")
        j = pd.concat([dah.rename("DA"), pph[c_res].rename("RL")], axis=1).dropna()
        fig = go.Figure(go.Scatter(x=j["RL"], y=j["DA"], mode="markers", marker=dict(size=5, opacity=0.5)))
        fig.update_layout(xaxis_title="Residual load (MW)", yaxis_title="DA (EUR/MWh)", height=420)
        st.plotly_chart(fig, use_container_width=True)

elif page == "⚙️ Motor merit-order":
    st.title("⚙️ Motor merit-order")
    df = pd.DataFrame(stack_manual).rename(columns={"tech": "Tehnologie", "srmc": "SRMC",
                                                    "cap": "Capacitate MW", "cum": "Cumulat MW"})
    df["SRMC"] = df["SRMC"].round(1)
    st.dataframe(df, use_container_width=True, hide_index=True)
    srmc_ccgt, srmc_coal = clean_spreads(P)
    c1, c2, c3 = st.columns(3)
    c1.metric("SRMC Gaz CCGT", f"{srmc_ccgt:.1f}")
    c2.metric("SRMC Carbune", f"{srmc_coal:.1f}")
    c3.metric("La margine", "GAZ" if srmc_ccgt < srmc_coal else "CARBUNE")
    if "fit" in st.session_state:
        f = st.session_state["fit"]
        st.success(f"Parametri auto-calibrati: η_lignit={f[0]:.3f}, η_carbune={f[1]:.3f}, "
                   f"η_CCGT={f[2]:.3f}, must-run={f[3]:,.0f} MW, scarcity={f[4]:.0f}, floor={f[5]:.0f}")
    st.subheader("Curba de oferta preț(RL)")
    rl_grid = np.arange(0, sum(L["cap"] for L in stack_manual) + 5000, 250)
    prices = marginal_price_vec(rl_grid, stack_manual, floor, scarcity)
    fig = go.Figure(go.Scatter(x=rl_grid, y=prices, mode="lines", line_shape="hv"))
    fig.update_layout(xaxis_title="Residual load (MW)", yaxis_title="Pret marginal", height=420)
    st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Forecast vs DA":
    st.title("🎯 Forecast vs DA — 3 modele comparate")
    days = st.selectbox("Interval (zile)", [14, 30, 60], index=1)
    try:
        pp, da = load_market(days)
    except Exception as ex:
        st.error(f"Eroare: {ex}"); st.stop()
    c_res = find_col(pp, "residual")
    if not c_res:
        st.warning("Lipseste 'Residual load'."); st.stop()

    d = pd.concat([hourly(da).rename("DA"), hourly(pp)[c_res].rename("RL")], axis=1).dropna()
    n = len(d); split = int(n * 0.7)
    train, test = d.iloc[:split], d.iloc[split:]
    st.caption(f"{n} ore | train: primele {split} | test: ultimele {n - split} "
               f"(split temporal — testul e 'viitorul' pe care modelul nu l-a vazut)")

    # --- Model 1: structural manual ---
    pred_manual = marginal_price_vec(test["RL"], stack_manual, floor, scarcity)

    # --- Model 2: structural auto-calibrat ---
    st.subheader("Auto-calibrare structurala")
    if st.button("🔧 Calibreaza automat pe train"):
        with st.spinner("Optimizer ruleaza..."):
            x, fun = autocalibrate(train["RL"], train["DA"], P)
            st.session_state["fit"] = x
            st.success(f"Gata. MAE train = {fun:.1f} EUR/MWh")
    pred_auto = None
    if "fit" in st.session_state:
        f = st.session_state["fit"]
        pauto = dict(P); pauto["eff_lignite"], pauto["eff_coal"], pauto["eff_ccgt"], pauto["cap_mustrun"] = f[0], f[1], f[2], f[3]
        stk_auto = build_stack(pauto)
        pred_auto = marginal_price_vec(test["RL"], stk_auto, f[5], f[4])

    # --- Model 3: empiric ---
    emp = fit_empirical(train["RL"], train["DA"])
    pred_emp = predict_empirical(test["RL"], emp) if emp else None

    # --- Tabel de scoruri (pe test) ---
    rows = [("Structural manual", metrics(pred_manual, test["DA"]))]
    if pred_auto is not None: rows.append(("Structural auto", metrics(pred_auto, test["DA"])))
    if pred_emp is not None:  rows.append(("Empiric", metrics(pred_emp, test["DA"])))
    tbl = pd.DataFrame([{"Model": nm, "MAE test": f"{m['mae']:.1f}",
                         "Bias test": f"{m['bias']:+.1f}", "Corr test": f"{m['corr']:.2f}"}
                        for nm, m in rows])
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # --- Scatter + curbe ---
    st.subheader("Norul DA vs RL + curbele modelelor")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["RL"], y=d["DA"], mode="markers",
                             marker=dict(size=4, opacity=0.35), name="DA real"))
    rl_grid = np.linspace(d["RL"].min(), d["RL"].max(), 200)
    fig.add_trace(go.Scatter(x=rl_grid, y=marginal_price_vec(rl_grid, stack_manual, floor, scarcity),
                             mode="lines", name="Structural manual", line_shape="hv"))
    if pred_auto is not None:
        fig.add_trace(go.Scatter(x=rl_grid, y=marginal_price_vec(rl_grid, stk_auto, f[5], f[4]),
                                 mode="lines", name="Structural auto", line_shape="hv"))
    if emp:
        fig.add_trace(go.Scatter(x=rl_grid, y=predict_empirical(rl_grid, emp),
                                 mode="lines", name="Empiric", line=dict(width=3)))
    fig.update_layout(xaxis_title="Residual load (MW)", yaxis_title="EUR/MWh", height=460)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Compara MAE test intre modele. Empiricul e de obicei greu de batut fara efort. "
            "Daca structural-auto se apropie de empiric, ai un model interpretabil (iti da "
            "tehnologia marginala si clean spreads) la fel de bun ca cel data-driven — "
            "asta e combinatia castigatoare.")

else:
    st.title("📚 Teorie")
    st.markdown(r"""
### De ce NU calibrezi manual
Modelul structural are prea multi parametri corelati (randamente × capacitati × combustibili).
Ochiul nu poate optimiza 13 variabile simultan. Doua solutii corecte:
- **Empiric**: lasi datele sa deseneze curba preț(RL) — median per bin, monoton crescator. Fara parametri.
- **Auto-calibrare**: un optimizer (least-squares / differential evolution) gaseste parametrii care minimizeaza eroarea pe train.

### Train / test (anti supra-fitting)
Antrenezi pe primele 70% din ore, masori pe ultimele 30% pe care modelul NU le-a vazut.
Daca MAE-train e mic dar MAE-test e mare → ai memorat zgomot, nu ai invatat structura.

### Ecuatia
RL = Consum − Eolian − Solar − must-run. Pret DA ≈ SRMC-ul ultimei unitati care acopera RL.
Nivelul termic = comutarea lignit → carbune → gaz, decisa de clean dark vs clean spark spread.

### Unde e alpha
Nu conteaza sa ai dreptate in absolut, ci **mai mult** decat consensul. Backtesteaza forecast-ul
tau vs. curba/settlement de la momentul deciziei. Edge maxim pe front (Day/Week).
""")

st.sidebar.divider()
st.sidebar.caption("Date: Energy-Charts (Fraunhofer ISE), CC BY 4.0.")
