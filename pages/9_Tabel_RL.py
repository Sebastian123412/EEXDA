# pages/9_Tabel_RL.py — Tabel de date: residual load DE-LU, 2 saptamani in spate.
# Componente desfacute (consum, vant, solar, hidro-ror) + RL, orar. Media zilei dedesubt.
# Toate seriile sunt processType A01 = prognoza day-ahead (vintage la lead fix D-1).
# Baza pentru coloane viitoare: pret DA, SRMC, deviatii.

import io
import zipfile
import datetime as dt
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Tabel RL", page_icon="🧾", layout="wide")
st.title("🧾 Tabel residual load — DE-LU")

TZ = "Europe/Berlin"
ENTSOE = "https://web-api.tp.entsoe.eu/api"
DE_LU = "10Y1001A1001A82H"

COMPONENTS = ["Vant onshore", "Vant offshore", "Solar", "Hidro ROR"]
COLS = ["Consum", "Vant onshore", "Vant offshore", "Vant total",
        "Solar", "Hidro ROR", "RL"]


# ---------------------------------------------------------------------------
# ENTSO-E
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
    return {k: pd.concat(v)[~pd.concat(v).index.duplicated(keep="last")].sort_index()
            for k, v in out.items()}


def _call(params):
    tok = st.secrets.get("ENTSOE_TOKEN", "")
    if not tok:
        raise RuntimeError("Lipseste ENTSOE_TOKEN in Streamlit secrets.")
    r = requests.get(ENTSOE, params=dict(params, securityToken=tok), timeout=90)
    return _parse(r.content) if r.status_code == 200 else {}


@st.cache_data(ttl=1800, show_spinner="Preiau date ENTSO-E...")
def get_data(s_utc, e_utc):
    """
    Consum (A65) + vant/solar/hidro-ror (A69), ambele processType A01.

    Returneaza si `missing`: numarul de ore in care o componenta lipsea din A69
    si a fost completata cu 0. Fara asta nu poti distinge "n-a produs" de
    "nu s-a publicat" — iar diferenta se vede direct in RL.
    """
    load = _call({"documentType": "A65", "processType": "A01",
                  "outBiddingZone_Domain": DE_LU,
                  "periodStart": s_utc, "periodEnd": e_utc}).get("ALL")

    ren = _call({"documentType": "A69", "processType": "A01", "in_Domain": DE_LU,
                 "periodStart": s_utc, "periodEnd": e_utc})

    if load is None or len(load) == 0:
        return pd.DataFrame(), pd.Series(dtype=int)

    df = pd.DataFrame({"Consum": load.tz_convert(TZ)}).resample("1h").mean()

    for psr, name in [("B19", "Vant onshore"), ("B18", "Vant offshore"),
                      ("B16", "Solar"), ("B11", "Hidro ROR")]:
        if psr in ren:
            df[name] = ren[psr].tz_convert(TZ).resample("1h").mean()
        else:
            df[name] = float("nan")

    # Cate ore au fost completate artificial, inainte de fillna.
    missing = df[COMPONENTS].isna().sum()

    # Orele lipsa din A69 (gauri de publicare, reindexare pe grila de consum)
    # ar propaga NaN in RL. Le tratam ca productie 0 — vezi avertismentul de mai jos.
    df["_gaps"] = df[COMPONENTS].isna().any(axis=1).astype(int)
    df[COMPONENTS] = df[COMPONENTS].fillna(0.0)

    df["Vant total"] = df["Vant onshore"] + df["Vant offshore"]
    df["RL"] = df["Consum"] - df["Vant total"] - df["Solar"] - df["Hidro ROR"]

    return df.dropna(subset=["Consum"]), missing


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
days = st.selectbox("Zile in spate", [7, 14, 21, 30], index=1)
unit = st.radio("Unitate", ["MW", "GW"], horizontal=True, index=1)

start = dt.date.today() - dt.timedelta(days=days)
end = dt.date.today() + dt.timedelta(days=2)          # include si maine
s_utc = pd.Timestamp(start, tz=TZ).tz_convert("UTC").strftime("%Y%m%d%H%M")
e_utc = pd.Timestamp(end, tz=TZ).tz_convert("UTC").strftime("%Y%m%d%H%M")

df, missing = get_data(s_utc, e_utc)
if df.empty:
    st.error("Nu am primit date. Verifica ENTSOE_TOKEN in secrets.")
    st.stop()

k = 1000.0 if unit == "GW" else 1.0
nd = 2 if unit == "GW" else 0

st.caption(
    f"{len(df)} ore, {df.index.min():%Y-%m-%d %H:%M} → {df.index.max():%Y-%m-%d %H:%M} (CET). "
    "RL = Consum − Vant − Solar − Hidro ROR. Biomasa (~4.5 GW) NU e scazuta — "
    "A69 nu o publica. E un offset constant, nu compara nivelul cu seria `rdl` de la Volue."
)

if missing.sum() > 0:
    st.warning(
        "Ore completate cu 0 pentru ca A69 nu le-a publicat: "
        + ", ".join(f"{n} ({int(v)}h)" for n, v in missing.items() if v > 0)
        + ". Atentie: 0 inseamna aici *nu s-a publicat*, nu *n-a produs*. "
        "O ora fara vant iti umfla RL-ul cu pana la 10 GW si arata perfect normal in tabel. "
        "Pentru vizualizare e acceptabil; inainte sa fitezi ceva pe datele astea, exclude-le."
    )
else:
    st.success("Toate orele au date complete pe componente — nicio valoare completata artificial.")

# ---------------------------------------------------------------------------
# 1) Tabel orar
# ---------------------------------------------------------------------------
st.subheader(f"① Orar ({unit})")
hourly = (df[COLS] / k).round(nd).fillna(0)
hourly.insert(0, "Gap", df["_gaps"].map({0: "", 1: "⚠"}))
hourly.index = hourly.index.strftime("%Y-%m-%d %H:%M")
hourly.index.name = "Ora (CET)"
st.dataframe(hourly, use_container_width=True, height=420)

# ---------------------------------------------------------------------------
# 2) Media zilei
# ---------------------------------------------------------------------------
st.subheader(f"② Media zilei ({unit})")
g = df.groupby(df.index.date)
daily = g[COLS].mean() / k
daily["RL min"] = g["RL"].min() / k
daily["RL max"] = g["RL"].max() / k
daily["Ore RL<0"] = g["RL"].apply(lambda s: int((s < 0).sum()))

# Peak = 08-20 CET, definitia contractului EEX Peak.
pk = df[df.index.hour.isin(range(8, 20))]
daily["RL peak"] = pk.groupby(pk.index.date)["RL"].mean() / k

daily["Ore lipsa"] = g["_gaps"].sum().astype(int)

daily = daily.round(nd).fillna(0)
daily.index.name = "Zi"
st.dataframe(daily, use_container_width=True)

st.caption(
    "`Ore RL<0` si `RL min` conteaza mai mult decat media: pe august, mutarea de la "
    "3-5 ore ieftine la 0 a urcat media zilnica de pret cu ~40 EUR/MWh. Media singura "
    "ascunde exact pragul asta. `Ore lipsa` = ore completate cu 0, nu masurate."
)

st.download_button(
    "Descarca CSV (orar)",
    df[COLS + ["_gaps"]].reset_index().to_csv(index=False).encode(),
    file_name=f"de_lu_rl_{start:%Y%m%d}_{end:%Y%m%d}.csv",
    mime="text/csv",
)
