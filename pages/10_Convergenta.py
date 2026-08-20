# pages/10_Convergenta.py — Cat de des are DE-LU acelasi pret DA ca fiecare vecin.
# Sfert cu sfert (sau ora, daca alegi). Plus: cand NU converg, cine e mai scump si cu cat.
#
# NOTA: media aritmetica intre pretul DE si al unui vecin nu e calculata intentionat.
# Pe orele cuplate media = pretul DE (identice). Pe cele decuplate da un numar care nu
# se tranzactioneaza si nu deconteaza nimic. Ce are continut e SPREAD-ul si semnul lui.

import io
import zipfile
import datetime as dt
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Convergenta preturi", page_icon="🔗", layout="wide")
st.title("🔗 Convergență preț DA — DE-LU vs vecini")

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


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns([2, 1, 1])
days = c1.selectbox("Zile in spate", [7, 14, 30, 60, 90], index=2)
res_label = c2.radio("Rezolutie", ["Sfert", "Oră"], horizontal=True)
res = "15min" if res_label == "Sfert" else "1h"
tol = c3.number_input("Prag convergenta (EUR/MWh)", 0.0, 10.0, 0.01, step=0.5,
                      help="0.01 = pret identic la cent. 1-2 = 'practic cuplat'.")

start = dt.date.today() - dt.timedelta(days=days)
end = dt.date.today() + dt.timedelta(days=1)
s_utc = pd.Timestamp(start, tz=TZ).tz_convert("UTC").strftime("%Y%m%d%H%M")
e_utc = pd.Timestamp(end, tz=TZ).tz_convert("UTC").strftime("%Y%m%d%H%M")

st.caption(
    "Pretul DA german e pe sferturi de ora din oct. 2025; unele zone publica inca orar. "
    "Pe 'Sfert', seriile orare sunt replicate pe cele 4 sferturi (forward-fill), ceea ce "
    "e corect: pretul orar SE aplica tuturor sferturilor din ora."
)

data = {}
bar = st.progress(0.0, text="Preiau preturi...")
for i, (z, eic) in enumerate(ZONES.items()):
    try:
        s = get_price(eic, s_utc, e_utc)
        if len(s):
            data[z] = (s.resample(res).mean().ffill() if res == "15min"
                       else s.resample(res).mean())
    except Exception as ex:
        st.warning(f"{z}: {ex}")
    bar.progress((i + 1) / len(ZONES), text=f"Preiau preturi... {z}")
bar.empty()

px = pd.DataFrame(data).dropna(how="all")
if "DE-LU" not in px.columns or px["DE-LU"].notna().sum() < 24:
    st.error("Lipsesc preturile DE-LU. Verifica tokenul si intervalul.")
    st.stop()

de = px["DE-LU"]

# ---------------------------------------------------------------------------
# 1) Clasament convergenta
# ---------------------------------------------------------------------------
st.subheader(f"① Cât de des are DE-LU același preț ({res_label.lower()} cu {res_label.lower()})")

rows = []
for z in px.columns:
    if z == "DE-LU":
        continue
    both = px[[z]].join(de.rename("DE")).dropna()
    if len(both) < 24:
        continue
    d = both[z] - both["DE"]
    conv = (d.abs() <= tol)
    rows.append({
        "Zona": z,
        "n": len(both),
        "Convergenta %": 100 * conv.mean(),
        "Spread mediu": d.mean(),
        "Spread |mediu|": d.abs().mean(),
        "% vecin mai scump": 100 * (d > tol).mean(),
        "Spread max": d.max(),
        "Spread min": d.min(),
        "Corelatie": both[z].corr(both["DE"]),
    })

tab = pd.DataFrame(rows).sort_values("Convergenta %", ascending=False)
st.dataframe(
    tab.set_index("Zona").round(2),
    use_container_width=True,
    column_config={"Convergenta %": st.column_config.ProgressColumn(
        "Convergenta %", min_value=0, max_value=100, format="%.1f%%")},
)

st.caption(
    "`Spread mediu` cu semn: pozitiv = vecinul e in medie mai scump ca DE. "
    "Cuplarea e adesea asimetrica — o zona poate fi cuplata 70% din timp si totusi "
    "sistematic mai scumpa in restul de 30%. Media ascunde asta, semnul nu."
)

# ---------------------------------------------------------------------------
# 2) Convergenta pe ora din zi
# ---------------------------------------------------------------------------
st.subheader("② Convergență pe ora din zi")
top = tab.head(6)["Zona"].tolist()
fig = go.Figure()
for z in top:
    both = px[[z]].join(de.rename("DE")).dropna()
    conv = (both[z] - both["DE"]).abs() <= tol
    byh = conv.groupby(both.index.hour).mean() * 100
    fig.add_trace(go.Scatter(x=byh.index, y=byh.values, mode="lines+markers", name=z))
fig.update_layout(xaxis_title="Ora (CET)", yaxis_title="% sferturi cuplate",
                  height=400, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Cuplarea se rupe aproape mereu la rampa de dimineata si la varful de seara — "
    "exact orele care decid media zilei. O convergenta anuala de 70% poate insemna "
    "95% noaptea si 30% la ora 19."
)

# ---------------------------------------------------------------------------
# 3) Vecinul ca predictor, nu ca medie
# ---------------------------------------------------------------------------
st.subheader("③ Vecinul ca predictor pentru DE")
st.caption(
    "Cat de bine aproximeaza pretul unui vecin pretul DE, raportat la a folosi "
    "media DE din perioada. Skill > 0 = vecinul aduce informatie. Asta e alternativa "
    "utila la a face media celor doua preturi — media nu se deconteaza nicaieri."
)

base_mae = (de - de.mean()).abs().mean()
pred = []
for z in px.columns:
    if z == "DE-LU":
        continue
    both = px[[z]].join(de.rename("DE")).dropna()
    if len(both) < 24:
        continue
    mae = (both[z] - both["DE"]).abs().mean()
    pred.append({"Zona": z, "MAE ca proxy": mae,
                 "Skill vs medie DE": 1 - mae / base_mae})
pr = pd.DataFrame(pred).sort_values("Skill vs medie DE", ascending=False)
st.dataframe(pr.set_index("Zona").round(3), use_container_width=True)

best = pr.iloc[0]
st.info(
    f"Cel mai bun proxy pe perioada: **{best['Zona']}**, MAE {best['MAE ca proxy']:.2f} "
    f"EUR/MWh. Un ansamblu (media mai multor vecini) bate de obicei oricare singur — "
    "dar doar daca vecinii se decupleaza *independent*. Daca se decupleaza toti in "
    "aceleasi ore de varf, media nu ajuta deloc."
)

st.download_button(
    "Descarca preturile (CSV)",
    px.reset_index().to_csv(index=False).encode(),
    file_name=f"da_prices_{start:%Y%m%d}_{end:%Y%m%d}.csv",
    mime="text/csv",
)
