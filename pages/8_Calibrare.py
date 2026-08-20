# pages/8_Calibrare.py — Calibrare RL -> pret DA pentru DE-LU, pe istoric lung.
#
# CE FACE: trage RL prognozat (A65+A69, processType A01 = vintage day-ahead, lead fix D-1)
# si pretul DA realizat (A44) pe luni intregi, apoi fiteaza pret ~ bucket(RL) + efect_ora.
# Validare out-of-sample pe ultimele N zile. La final: fair value pentru maine.
#
# DE CE BUCKET + ORA: pe 192 ore de august, liniarul dadea r=0.89 dar reziduurile aveau
# structura clara pe ora (-25 EUR la 04:00, +33 EUR la 19:00, la ACELASI RL). Panta e ~2
# EUR/GW la mijloc si ~8 EUR/GW peste 40 GW. Ambele efecte sunt mari si sistematice.
#
# ATENTIE: RL aici NU scade biomasa (~4.5 GW, absenta din A69). E un offset constant,
# absorbit de intercept. Nu compara nivelul cu seria `rdl` de la Volue.

import io
import zipfile
import datetime as dt
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Calibrare RL→Preț", page_icon="📐", layout="wide")
st.title("📐 Calibrare RL → Preț — DE-LU")

TZ = "Europe/Berlin"
ENTSOE = "https://web-api.tp.entsoe.eu/api"
DE_LU = "10Y1001A1001A82H"

# ---------------------------------------------------------------------------
# ENTSO-E: parser (acelasi pattern ca pe celelalte pagini)
# ---------------------------------------------------------------------------
def _ln(t):
    return t.split("}")[-1]


def _parse(content):
    docs = ([zipfile.ZipFile(io.BytesIO(content)).read(n)
             for n in zipfile.ZipFile(io.BytesIO(content)).namelist()]
            if content[:2] == b"PK" else [content])
    out = {}
    for doc in docs:
        try:
            root = ET.fromstring(doc)
        except ET.ParseError:
            continue
        for ts in [e for e in root.iter() if _ln(e.tag) == "TimeSeries"]:
            psr = "ALL"
            for c in ts.iter():
                if _ln(c.tag) == "psrType":
                    psr = c.text
            for period in [e for e in ts.iter() if _ln(e.tag) == "Period"]:
                start = res = None
                for c in period:
                    if _ln(c.tag) == "timeInterval":
                        for cc in c:
                            if _ln(cc.tag) == "start":
                                start = cc.text
                    elif _ln(c.tag) == "resolution":
                        res = c.text
                if not start:
                    continue
                step = pd.Timedelta(minutes=15 if res == "PT15M" else
                                    30 if res == "PT30M" else 60)
                t0 = pd.Timestamp(start)
                recs = {}
                for pt in period:
                    if _ln(pt.tag) != "Point":
                        continue
                    pos = qty = None
                    for cc in pt:
                        if _ln(cc.tag) == "position":
                            pos = int(cc.text)
                        elif _ln(cc.tag) in ("quantity", "price.amount"):
                            qty = float(cc.text)
                    if pos is not None:
                        recs[t0 + (pos - 1) * step] = qty
                if recs:
                    out.setdefault(psr, []).append(pd.Series(recs).sort_index())
    res_out = {}
    for k, v in out.items():
        s = pd.concat(v)
        res_out[k] = s[~s.index.duplicated(keep="last")].sort_index()
    return res_out


def _token():
    tok = st.secrets.get("ENTSOE_TOKEN", "")
    if not tok:
        raise RuntimeError("Lipseste ENTSOE_TOKEN in Streamlit secrets.")
    return tok


def _call(params):
    params = dict(params, securityToken=_token())
    r = requests.get(ENTSOE, params=params, timeout=90)
    if r.status_code != 200:
        return {}
    return _parse(r.content)


def _months(a, b):
    """[a, b) impartit pe luni. ENTSO-E respinge intervale lungi pe unele documente."""
    cur = a
    while cur < b:
        nxt = min((cur + pd.DateOffset(months=1)).normalize(), b)
        yield cur, nxt
        cur = nxt


# ---------------------------------------------------------------------------
# Fetch pe istoric lung
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_history(start_date, end_date):
    """
    Returneaza DataFrame orar cu: price, load_fc, wind_fc, pv_fc, ror_fc, rl_fc.

    Toate componentele de RL sunt processType A01 = prognoza day-ahead, publicata
    inainte de licitatie. Deci pretul din acelasi rand NU e cunoscut cand a fost
    facuta prognoza. Asta face setul valid pentru backtest.
    """
    a = pd.Timestamp(start_date, tz=TZ)
    b = pd.Timestamp(end_date, tz=TZ)
    chunks = list(_months(a, b))

    price_parts, load_parts, ren_parts = [], [], []
    bar = st.progress(0.0, text="Preiau istoric ENTSO-E...")

    for i, (c0, c1) in enumerate(chunks):
        s = c0.tz_convert("UTC").strftime("%Y%m%d%H%M")
        e = c1.tz_convert("UTC").strftime("%Y%m%d%H%M")

        try:
            d = _call({"documentType": "A44", "in_Domain": DE_LU,
                       "out_Domain": DE_LU, "periodStart": s, "periodEnd": e})
            if "ALL" in d:
                price_parts.append(d["ALL"])
        except Exception:
            pass

        try:
            d = _call({"documentType": "A65", "processType": "A01",
                       "outBiddingZone_Domain": DE_LU,
                       "periodStart": s, "periodEnd": e})
            if "ALL" in d:
                load_parts.append(d["ALL"])
        except Exception:
            pass

        try:
            d = _call({"documentType": "A69", "processType": "A01",
                       "in_Domain": DE_LU, "periodStart": s, "periodEnd": e})
            if d:
                cols = {}
                for k, name in [("B16", "pv_fc"), ("B18", "woff"),
                                ("B19", "won"), ("B11", "ror_fc")]:
                    if k in d:
                        cols[name] = d[k]
                if cols:
                    ren_parts.append(pd.DataFrame(cols))
        except Exception:
            pass

        bar.progress((i + 1) / len(chunks),
                     text=f"Preiau istoric ENTSO-E... {c0:%Y-%m}")
    bar.empty()

    def _cat_series(parts, name):
        if not parts:
            return pd.Series(dtype=float, name=name)
        s = pd.concat(parts)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s.tz_convert(TZ).resample("1h").mean().rename(name)

    price = _cat_series(price_parts, "price")
    load = _cat_series(load_parts, "load_fc")

    if ren_parts:
        ren = pd.concat(ren_parts)
        ren = ren[~ren.index.duplicated(keep="last")].sort_index()
        ren = ren.tz_convert(TZ).resample("1h").mean()
    else:
        ren = pd.DataFrame()

    df = pd.concat([price, load], axis=1)
    for c in ["pv_fc", "woff", "won", "ror_fc"]:
        df[c] = ren[c] if c in ren.columns else 0.0
    df["wind_fc"] = df["woff"] + df["won"]

    # RL fara biomasa (A69 nu o publica). Offset constant, absorbit de intercept.
    df["rl_fc"] = df["load_fc"] - df["wind_fc"] - df["pv_fc"] - df["ror_fc"]
    df["hour"] = df.index.hour
    df["day"] = df.index.normalize()
    return df.dropna(subset=["rl_fc"])


# ---------------------------------------------------------------------------
# Model: bucket(RL) + efect fix pe ora, fitat impreuna prin lstsq
# ---------------------------------------------------------------------------
def build_design(rl, hour, edges):
    """Dummies pentru bucket RL si pentru ora. Prima categorie e omisa (baza)."""
    bk = np.clip(np.digitize(rl, edges) - 1, 0, len(edges) - 2)
    n = len(rl)
    nb, nh = len(edges) - 1, 24
    X = np.zeros((n, 1 + (nb - 1) + (nh - 1)))
    X[:, 0] = 1.0
    for j in range(1, nb):
        X[:, j] = (bk == j)
    for j in range(1, nh):
        X[:, nb - 1 + j] = (hour == j)
    return X, bk


def fit_model(df, edges, target="price"):
    X, _ = build_design(df["rl_fc"].values, df["hour"].values, edges)
    y = df[target].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def predict(df, edges, beta):
    X, _ = build_design(df["rl_fc"].values, df["hour"].values, edges)
    return X @ beta


# ---------------------------------------------------------------------------
# UI — parametri
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns([2, 1, 1])
default_start = dt.date.today() - dt.timedelta(days=540)
start_date = c1.date_input("Start istoric", value=default_start,
                           min_value=dt.date(2022, 1, 1))
test_days = c2.number_input("Zile de test (out-of-sample)", 14, 180, 60, step=7)
bucket_w = c3.selectbox("Latime bucket RL (GW)", [2.5, 5.0, 10.0], index=1)

st.caption(
    "Prognozele A65/A69 sunt processType A01 — publicate inainte de licitatie, "
    "deci pretul din acelasi rand nu era cunoscut. Prima incarcare dureaza "
    "(o cerere pe luna x 3 documente), apoi ramane in cache 24h."
)

if not st.button("Trage datele si fiteaza", type="primary"):
    st.stop()

end_date = dt.date.today() + dt.timedelta(days=2)
df = fetch_history(start_date, end_date)

if df.empty or df["price"].notna().sum() < 500:
    st.error("Prea putine date. Verifica ENTSOE_TOKEN si intervalul.")
    st.stop()

hist = df.dropna(subset=["price"]).copy()
st.success(
    f"{len(hist):,} ore cu pret, {hist.index.min():%Y-%m-%d} → {hist.index.max():%Y-%m-%d} "
    f"({hist['day'].nunique()} zile)"
)

lo = np.floor(hist["rl_fc"].quantile(0.001) / 1000 / bucket_w) * bucket_w
hi = np.ceil(hist["rl_fc"].quantile(0.999) / 1000 / bucket_w) * bucket_w
edges = np.arange(lo, hi + bucket_w, bucket_w) * 1000

# ---------------------------------------------------------------------------
# Split temporal — ultimele N zile sunt test, restul train
# ---------------------------------------------------------------------------
cut = hist["day"].max() - pd.Timedelta(days=int(test_days))
train = hist[hist["day"] <= cut]
test = hist[hist["day"] > cut]

if len(train) < 2000 or len(test) < 100:
    st.error("Split invalid — mareste istoricul sau micsoreaza zilele de test.")
    st.stop()

beta = fit_model(train, edges)
test = test.assign(pred=predict(test, edges, beta))
test = test.assign(res=test["price"] - test["pred"])

# Baseline onest: pretul aceleiasi ore, cu 7 zile inainte (persistenta saptamanala).
naive = hist["price"].shift(24 * 7).reindex(test.index)
naive_err = (test["price"] - naive).dropna()

st.subheader("① Validare out-of-sample")
m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE model (EUR/MWh)", f"{test['res'].abs().mean():.1f}")
m2.metric("MAE naiv (t−7 zile)", f"{naive_err.abs().mean():.1f}")
r2 = 1 - (test["res"] ** 2).sum() / ((test["price"] - test["price"].mean()) ** 2).sum()
m3.metric("R² out-of-sample", f"{r2:.3f}")
m4.metric("Bias (EUR/MWh)", f"{test['res'].mean():+.1f}")

if test["res"].abs().mean() >= naive_err.abs().mean():
    st.error(
        "Modelul NU bate baseline-ul naiv. Nu-l folosi pentru decizii. "
        "Cauza probabila: fereastra de train acopera regimuri diferite de combustibil "
        "(nivelul pretului vine din TTF/EUA, nu din RL). Incearca un istoric mai scurt, "
        "sau modeleaza pret − SRMC in loc de pret."
    )
else:
    st.success("Modelul bate baseline-ul naiv pe perioada de test.")

# Eroare agregata pe base zilnic — asta conteaza pentru contractul Day Base.
db = test.groupby("day").agg(price=("price", "mean"), pred=("pred", "mean"))
db["err"] = db["pred"] - db["price"]
st.caption(
    f"Pe **base zilnic** (unitatea contractului DB): MAE {db['err'].abs().mean():.2f} "
    f"EUR/MWh, bias {db['err'].mean():+.2f}, pe {len(db)} zile."
)

# ---------------------------------------------------------------------------
# Curba si efectul de ora
# ---------------------------------------------------------------------------
st.subheader("② Curba merit order empirica")
nb = len(edges) - 1
mids = (edges[:-1] + edges[1:]) / 2 / 1000
curve = np.concatenate([[0.0], beta[1:nb]]) + beta[0]
counts = np.histogram(train["rl_fc"], bins=edges)[0]

fig = go.Figure()
fig.add_trace(go.Scatter(x=mids[counts > 20], y=curve[counts > 20],
                         mode="lines+markers", name="Pret la ora de referinta"))
fig.update_layout(xaxis_title="Residual load (GW)", yaxis_title="EUR/MWh", height=420)
st.plotly_chart(fig, use_container_width=True)

slopes = np.diff(curve) / np.diff(mids)
ok = (counts[:-1] > 20) & (counts[1:] > 20)
if ok.any():
    st.caption(
        f"Panta: {np.nanmin(slopes[ok]):.1f} pana la **{np.nanmax(slopes[ok]):.1f} "
        f"EUR/MWh pe GW**. Daca raportul e mare, un model liniar te va costa exact "
        "in coada de sus — acolo unde se face si se pierde banul."
    )

st.subheader("③ Efect fix pe ora")
hr = np.concatenate([[0.0], beta[nb:nb + 23]])
fig2 = go.Figure(go.Bar(x=list(range(24)), y=hr))
fig2.update_layout(xaxis_title="Ora (CET)", yaxis_title="EUR/MWh fata de ora 0",
                   height=320)
st.plotly_chart(fig2, use_container_width=True)
st.caption(
    "La ACELASI residual load, pretul difera pe ora: costuri de pornire noaptea, "
    "prima de rampa seara. Daca amplitudinea depaseste ~20 EUR, un model doar-pe-RL "
    "lasa bani pe masa."
)

# ---------------------------------------------------------------------------
# Fair value pentru maine
# ---------------------------------------------------------------------------
st.subheader("④ Fair value — MAINE")
tomorrow = pd.Timestamp(dt.date.today() + dt.timedelta(days=1), tz=TZ)
fwd = df[(df.index >= tomorrow) & (df.index < tomorrow + pd.Timedelta(days=1))]

if fwd.empty or fwd["rl_fc"].isna().all():
    st.info("Prognoza pentru maine nu e inca publicata pe ENTSO-E.")
else:
    beta_full = fit_model(hist, edges)
    fwd = fwd.assign(fv=predict(fwd, edges, beta_full))
    pk = fwd.index.hour.isin(range(8, 20))

    k1, k2, k3 = st.columns(3)
    k1.metric("RL mediu (GW)", f"{fwd['rl_fc'].mean() / 1000:.1f}")
    k2.metric("FV Base (EUR/MWh)", f"{fwd['fv'].mean():.1f}")
    k3.metric("FV Peak (EUR/MWh)", f"{fwd.loc[pk, 'fv'].mean():.1f}")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=fwd.index, y=fwd["fv"], name="Fair value", mode="lines"))
    fig3.add_trace(go.Scatter(x=fwd.index, y=fwd["rl_fc"] / 1000, name="RL (GW)",
                              mode="lines", yaxis="y2", line=dict(dash="dot")))
    fig3.update_layout(height=380, yaxis_title="EUR/MWh",
                       yaxis2=dict(title="GW", overlaying="y", side="right"),
                       legend=dict(orientation="h"))
    st.plotly_chart(fig3, use_container_width=True)

    st.warning(
        f"Bias out-of-sample masurat: {db['err'].mean():+.2f} EUR/MWh pe base. "
        "Scade-l din FV inainte sa-l compari cu pretul EEX — si tine minte ca "
        "e estimat pe o singura fereastra, deci instabil."
    )

# ---------------------------------------------------------------------------
# Export — Streamlit Cloud pierde fisierele la redeploy
# ---------------------------------------------------------------------------
st.subheader("⑤ Export")
st.download_button(
    "Descarca datele (CSV)",
    hist.reset_index().to_csv(index=False).encode(),
    file_name=f"de_lu_rl_price_{hist.index.min():%Y%m%d}_{hist.index.max():%Y%m%d}.csv",
    mime="text/csv",
)
st.caption(
    "Streamlit Cloud sterge fisierele la redeploy, deci nu are rost sa scriu parquet "
    "pe disc. Descarca CSV-ul si tine-l undeva stabil — cand IT-ul face baza de date, "
    "asta devine sursa de adevar pentru backtest."
)
