# pages/10_Convergenta.py — Preturi DA (DE-LU + vecini) + convergenta pe sfert de ora.
#
# ① Tabel de preturi, sfert cu sfert, cu sferturile cuplate marcate.
# ② Matrice: pentru fiecare slot din zi (00:00 ... 23:45) si fiecare vecin,
#    in cate % din zile a avut acelasi pret ca DE-LU.
# ③ Detaliu pe fiecare tara in parte.
#
# Convergenta pe un sfert = |pret_vecin - pret_DE| <= prag.

import io
import zipfile
import datetime as dt
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Convergenta preturi", page_icon="🔗", layout="wide")
st.title("🔗 Preturi DA + convergenta cu DE-LU")

TZ = "Europe/Berlin"
ENTSOE = "https://web-api.tp.entsoe.eu/api"

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


@st.cache_data(ttl=1800, show_spinner=False)
def get_price(eic, s_utc, e_utc):
    tok = st.secrets.get("ENTSOE_TOKEN", "")
    if not tok:
        raise RuntimeError("Lipseste ENTSOE_TOKEN in Streamlit secrets.")
    p = {"documentType": "A44", "in_Domain": eic, "out_Domain": eic,
         "periodStart": s_utc, "periodEnd": e_utc, "securityToken": tok}
    r = requests.get(ENTSOE, params=p, timeout=90)
    if r.status_code != 200:
        return pd.Series(dtype=float)
    d = _parse(r.content)
    s = d.get("ALL", pd.Series(dtype=float))
    return s.tz_convert(TZ) if len(s) else s


def heat(v, lo=0.0, hi=100.0):
    """Verde proportional cu valoarea. Scris de mana ca sa nu ceara matplotlib."""
    if pd.isna(v):
        return "color:#bbb"
    t = min(max((v - lo) / (hi - lo), 0.0), 1.0)
    r = int(255 - 75 * t)
    g = int(255 - 15 * t)
    b = int(255 - 80 * t)
    return f"background-color: rgb({r},{g},{b})"


# ---------------------------------------------------------------------------
# Selector perioada
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns([2, 1, 1])

PERIODS = {
    "Maine": (0, 2),
    "Azi": (0, 1),
    "Ultima zi (ieri)": (1, 0),
    "3 zile": (3, 1),
    "7 zile": (7, 1),
    "14 zile": (14, 1),
    "30 zile": (30, 1),
    "90 zile": (90, 1),
}
period = c1.selectbox("Perioada", list(PERIODS.keys()), index=4)
res_label = c2.radio("Rezolutie", ["Sfert", "Oră"], horizontal=True)
res = "15min" if res_label == "Sfert" else "1h"
tol = c3.number_input("Prag convergenta (EUR/MWh)", 0.0, 20.0, 0.01, step=0.25,
                      help="0.01 = pret identic la cent. 1-2 = 'practic cuplat'.")

back, fwd = PERIODS[period]
start = dt.date.today() - dt.timedelta(days=back)
end = dt.date.today() + dt.timedelta(days=fwd)
s_utc = pd.Timestamp(start, tz=TZ).tz_convert("UTC").strftime("%Y%m%d%H%M")
e_utc = pd.Timestamp(end, tz=TZ).tz_convert("UTC").strftime("%Y%m%d%H%M")

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
data = {}
bar = st.progress(0.0, text="Preiau preturi...")
for i, (z, eic) in enumerate(ZONES.items()):
    try:
        s = get_price(eic, s_utc, e_utc)
        if len(s):
            # Zonele care publica orar sunt propagate pe cele 4 sferturi:
            # pretul orar chiar se aplica fiecarui sfert din ora.
            data[z] = (s.resample(res).mean().ffill() if res == "15min"
                       else s.resample(res).mean())
    except Exception as ex:
        st.warning(f"{z}: {ex}")
    bar.progress((i + 1) / len(ZONES), text=f"Preiau preturi... {z}")
bar.empty()

px = pd.DataFrame(data).dropna(how="all")
if "DE-LU" not in px.columns or px["DE-LU"].notna().sum() < 4:
    st.error("Lipsesc preturile DE-LU. Verifica ENTSOE_TOKEN si perioada.")
    st.stop()

cols = ["DE-LU"] + sorted([c for c in px.columns if c != "DE-LU"])
px = px[cols]
de = px["DE-LU"]
neigh = [z for z in px.columns if z != "DE-LU"]

# Slot in zi: "00:00", "00:15", ... "23:45"
px_slot = px.index.strftime("%H:%M")
n_days = px.index.normalize().nunique()
unit = "sferturi" if res == "15min" else "ore"

st.caption(
    f"{len(px)} {unit} pe {n_days} zile, "
    f"{px.index.min():%Y-%m-%d %H:%M} → {px.index.max():%Y-%m-%d %H:%M} (CET). "
    "Convergenta = |pret vecin − pret DE-LU| ≤ prag."
)

# ---------------------------------------------------------------------------
# ① Tabel de preturi
# ---------------------------------------------------------------------------
st.subheader(f"① Preturi DA — {res_label.lower()} cu {res_label.lower()} (EUR/MWh)")

show = px.round(2).copy()
show.index = show.index.strftime("%Y-%m-%d %H:%M")
show.index.name = "CET"


def _mark(col):
    if col.name == "DE-LU":
        return ["background-color: #e8eef7; font-weight: 600"] * len(col)
    d = (px[col.name] - de).abs()
    return ["background-color: #d8f0d8" if (pd.notna(v) and v <= tol) else ""
            for v in d]


st.dataframe(show.style.apply(_mark, axis=0).format("{:.2f}", na_rep="—"),
             use_container_width=True, height=440)

# ---------------------------------------------------------------------------
# ② Matrice: % convergenta pe fiecare slot din zi x fiecare vecin
# ---------------------------------------------------------------------------
st.subheader(f"② % convergenta pe {'sfert' if res == '15min' else 'ora'} × zona")

conv = pd.DataFrame(
    {z: ((px[z] - de).abs() <= tol) & px[z].notna() & de.notna() for z in neigh},
    index=px.index,
)
valid = pd.DataFrame({z: px[z].notna() & de.notna() for z in neigh}, index=px.index)

num = conv.groupby(px_slot).sum()
den = valid.groupby(px_slot).sum()
matrix = (100 * num / den.replace(0, np.nan)).round(1)
matrix.index.name = "Slot CET"

st.dataframe(
    matrix.style.map(heat).format("{:.0f}%", na_rep="—"),
    use_container_width=True,
    height=min(700, 40 + 26 * len(matrix)),
)

if n_days == 1:
    st.info(
        "O singura zi in perioada → fiecare slot are un singur punct, deci "
        "valorile sunt doar 0% sau 100%. Alege 7+ zile ca procentele sa insemne ceva."
    )

st.caption(
    "Coloanele reci pe randurile de dimineata (05:00–08:00) si de seara (18:00–21:00) "
    "sunt tipice: cuplarea se rupe la rampe, adica exact in orele care decid media zilei. "
    "O convergenta globala de 70% poate insemna 95% noaptea si 30% la ora 19."
)

# ---------------------------------------------------------------------------
# ③ Detaliu pe fiecare tara
# ---------------------------------------------------------------------------
st.subheader("③ Detaliu pe zona")

summary = []
for z in neigh:
    both = px[[z]].join(de.rename("DE")).dropna()
    if both.empty:
        continue
    d = both[z] - both["DE"]
    c = d.abs() <= tol
    summary.append({
        "Zona": z,
        f"n {unit}": len(both),
        "Cuplate": int(c.sum()),
        "% conv": 100 * c.mean(),
        "Spread mediu": d.mean(),
        "Spread |mediu|": d.abs().mean(),
        "% vecin mai scump": 100 * (d > tol).mean(),
        "Spread min": d.min(),
        "Spread max": d.max(),
    })

summ = pd.DataFrame(summary).sort_values("% conv", ascending=False).set_index("Zona")
st.dataframe(
    summ.round(2), use_container_width=True,
    column_config={"% conv": st.column_config.ProgressColumn(
        "% conv", min_value=0, max_value=100, format="%.1f%%")},
)

st.caption(
    "`Spread mediu` are semn: pozitiv = vecinul e in medie mai scump ca DE-LU. "
    "Cuplarea e asimetrica — o zona poate fi cuplata 70% din timp si totusi sistematic "
    "mai scumpa in rest. Procentul singur ascunde asta, semnul nu."
)

for z in summ.index:
    both = px[[z]].join(de.rename("DE")).dropna()
    d = both[z] - both["DE"]
    with st.expander(f"{z} — {summ.loc[z, '% conv']:.1f}% cuplat, "
                     f"spread mediu {summ.loc[z, 'Spread mediu']:+.2f} EUR/MWh"):
        slot = both.index.strftime("%H:%M")
        det = pd.DataFrame({
            f"n {unit}": d.groupby(slot).size(),
            "Cuplate": (d.abs() <= tol).groupby(slot).sum().astype(int),
            "% conv": 100 * (d.abs() <= tol).groupby(slot).mean(),
            "Spread mediu": d.groupby(slot).mean(),
            "Spread min": d.groupby(slot).min(),
            "Spread max": d.groupby(slot).max(),
        })
        det.index.name = "Slot CET"
        st.dataframe(
            det.round(2), use_container_width=True,
            height=min(600, 40 + 26 * len(det)),
            column_config={"% conv": st.column_config.ProgressColumn(
                "% conv", min_value=0, max_value=100, format="%.0f%%")},
        )

st.download_button(
    "Descarca preturile (CSV)",
    px.reset_index().to_csv(index=False).encode(),
    file_name=f"da_prices_{start:%Y%m%d}_{end:%Y%m%d}.csv",
    mime="text/csv",
)
