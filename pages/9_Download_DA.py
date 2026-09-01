# pages/9_Download_DA.py — Descarcator de preturi day-ahead ENTSO-E, multi-zona,
# multi-an. Atat. Nu salveaza nimic nicaieri: alegi zone si ani, apesi, primesti
# un CSV. Analiza se face in alta parte.
#
# CE FACE PARSERUL, si de ce nu poti sari peste el:
#   * curveType A03 — pozitiile vin rare si pretul tine pana la urmatoarea
#     pozitie. Fara reindexare + ffill pierzi ore intregi, in tacere.
#   * mai multe TimeSeries pe acelasi interval (Sequence 1 / Sequence 2 in
#     exportul din GUI). Se pastreaza prima — aia e seria SDAC cuplata.
#   * rezolutii mixte: pana in 2025 e PT60M, dupa e PT15M. Iesirea orara
#     mediaza sferturile, ceea ce e exact corect: fiecare sfert are aceeasi
#     durata, deci media orara e chiar pretul baseload al orei.
#
# LIMITA ENTSO-E: 60 de cereri in 60 de secunde, altfel IP banat pana la 10 min.
# Un an per cerere, deci 6 zone x 6 ani = 36 de cereri. MIN_GAP tine marja.

from __future__ import annotations

import time
from datetime import date
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Download DA", layout="wide")

API = "https://web-api.tp.entsoe.eu/api"
TZ = "Europe/Berlin"          # CET/CEST — corect pentru DE, FR, AT, IT, ES, HU
MIN_GAP = 2.0
FREQ = {"PT15M": "15min", "PT30M": "30min", "PT60M": "60min", "PT1H": "60min"}

EIC = {
    "DE-LU": "10Y1001A1001A82H", "FR": "10YFR-RTE------C",
    "AT": "10YAT-APG------L",    "ES": "10YES-REE------0",
    "HU": "10YHU-MAVIR----U",    "IT-NORD": "10Y1001A1001A73I",
    "IT-CNOR": "10Y1001A1001A70O", "IT-CSUD": "10Y1001A1001A71M",
    "IT-SUD": "10Y1001A1001A788", "IT-SICI": "10Y1001A1001A75E",
    "IT-SARD": "10Y1001A1001A74G", "CZ": "10YCZ-CEPS-----N",
    "PL": "10YPL-AREA-----S",    "SK": "10YSK-SEPS-----K",
    "CH": "10YCH-SWISSGRIDZ",    "NL": "10YNL----------L",
    "BE": "10YBE----------2",    "SI": "10YSI-ELES-----O",
    "RO": "10YRO-TEL------P",    "RS": "10YCS-SERBIATSOV",
    "BG": "10YCA-BULGARIA-R",    "HR": "10YHR-HEP------M",
    "GR": "10YGR-HTSO-----Y",    "PT": "10YPT-REN------W",
}
DEFAULT = ["DE-LU", "FR", "AT", "ES", "HU", "IT-NORD"]


def parse_a44(xml_text: str) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    ns = {"n": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    f = lambda el, t: el.find(f"n:{t}", ns) if ns else el.find(t)
    fa = lambda el, t: el.findall(f"n:{t}", ns) if ns else el.findall(t)

    reason = f(root, "Reason")
    if reason is not None and not fa(root, "TimeSeries"):
        t = f(reason, "text")
        raise RuntimeError(f"ENTSO-E fara date: {t.text if t is not None else '?'}")

    frames, seq = [], 0
    for ts in fa(root, "TimeSeries"):
        seq += 1
        for per in fa(ts, "Period"):
            res = f(per, "resolution").text
            freq = FREQ.get(res)
            if freq is None:
                continue
            ti = f(per, "timeInterval")
            idx = pd.date_range(pd.Timestamp(f(ti, "start").text).tz_convert("UTC"),
                                pd.Timestamp(f(ti, "end").text).tz_convert("UTC"),
                                freq=freq, inclusive="left")
            if len(idx) == 0:
                continue
            pts = [(int(f(p, "position").text), float(f(p, "price.amount").text))
                   for p in fa(per, "Point")]
            pts = [(p, v) for p, v in pts if 1 <= p <= len(idx)]
            if not pts:
                continue
            s = pd.Series(np.nan, index=range(1, len(idx) + 1), dtype=float)
            s.loc[[p for p, _ in pts]] = [v for _, v in pts]
            frames.append(pd.DataFrame({"ts_utc": idx, "price": s.ffill().to_numpy(),
                                        "resolution": res, "seq": seq}))
    if not frames:
        return pd.DataFrame(columns=["ts_utc", "price", "resolution", "seq"])
    return pd.concat(frames, ignore_index=True).dropna(subset=["price"])


_last = [0.0]


def fetch_year(zone: str, year: int, token: str) -> pd.DataFrame:
    params = {"securityToken": token, "documentType": "A44", "processType": "A01",
              "in_Domain": EIC[zone], "out_Domain": EIC[zone],
              "periodStart": f"{year}01010000", "periodEnd": f"{year+1}01010000"}
    err = None
    for attempt in range(4):
        gap = time.monotonic() - _last[0]
        if gap < MIN_GAP:
            time.sleep(MIN_GAP - gap)
        _last[0] = time.monotonic()
        try:
            r = requests.get(API, params=params, timeout=180)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return parse_a44(r.text)
        except Exception as exc:                      # noqa: BLE001
            err = exc
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(str(err))


def to_series(df: pd.DataFrame, out_res: str) -> pd.Series:
    """Un chunk brut -> o serie. Pastreaza seq-ul cel mai mic, prefera PT15M
    unde exista, apoi reesantioneaza la rezolutia ceruta."""
    d = (df.sort_values(["ts_utc", "resolution", "seq"])
           .drop_duplicates(["ts_utc", "resolution"], keep="first"))
    fine = d[d["resolution"] == "PT15M"]
    hour = d[d["resolution"].isin(["PT60M", "PT1H"])]
    parts = []
    if not fine.empty:
        parts.append(fine.set_index("ts_utc")["price"])
    if not hour.empty:
        h = hour.set_index("ts_utc")["price"]
        if not fine.empty:
            h = h[~h.index.floor("h").isin(fine.index.floor("h").unique())]
        parts.append(h)
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s.resample(out_res).mean()


# --------------------------------------------------------------------------- #
st.title("Descarca preturi day-ahead ENTSO-E")

token = st.secrets.get("ENTSOE_TOKEN", "")
if not token:
    st.error("Lipseste ENTSOE_TOKEN din secrets.")
    st.stop()

a, b, c, d = st.columns([2.4, 1, 1, 1.1])
zones = a.multiselect("Zone", list(EIC), default=DEFAULT)
y0 = b.number_input("Din", 2015, date.today().year, 2021, step=1)
y1 = c.number_input("Pana in", 2015, date.today().year, date.today().year, step=1)
out_res = d.selectbox("Rezolutie iesire", ["60min", "15min", "D"], index=0)

years = list(range(int(y0), int(y1) + 1))
n = len(zones) * len(years)
st.caption(f"{len(zones)} zone x {len(years)} ani = {n} cereri, "
           f"~{n*MIN_GAP/60:.0f}-{n*1.0:.0f} minute. Nu se salveaza nimic: "
           f"cand se termina, apesi download.")

if st.button("Descarca", type="primary", disabled=not zones):
    bar, box, notes = st.progress(0.0), st.empty(), []
    cols, i = {}, 0
    for z in zones:
        chunks = []
        for y in years:
            i += 1
            bar.progress(i / n, text=f"{z} {y}  ({i}/{n})")
            try:
                df = fetch_year(z, y, token)
                if not df.empty:
                    chunks.append(df)
                    if df["seq"].max() > 1:
                        notes.append(f"{z} {y}: {int(df['seq'].max())} secvente, "
                                     "am pastrat prima")
            except Exception as exc:                  # noqa: BLE001
                notes.append(f"{z} {y}: ESUAT — {str(exc)[:120]}")
            box.info(" · ".join(notes[-4:]) if notes else "")
        if chunks:
            cols[z] = to_series(pd.concat(chunks, ignore_index=True), out_res)

    if not cols:
        st.error("Nu am obtinut nimic.")
    else:
        px = pd.DataFrame(cols).sort_index()
        px.index = px.index.tz_convert(TZ)
        px.index.name = "timestamp_local"
        st.session_state["dl"] = px
        st.session_state["dl_notes"] = notes

px = st.session_state.get("dl")
if px is not None:
    st.success(f"{len(px):,} randuri x {len(px.columns)} zone  |  "
               f"{px.index.min():%Y-%m-%d} - {px.index.max():%Y-%m-%d}")
    for nt in st.session_state.get("dl_notes", []):
        st.caption(nt)
    st.download_button("CSV", px.round(2).to_csv().encode(),
                       f"da_{'-'.join(px.columns)}_{px.index.min():%Y%m%d}_"
                       f"{px.index.max():%Y%m%d}_{out_res}.csv", "text/csv",
                       type="primary")
    st.caption("Format lat: o coloana per zona, ora locala CET/CEST cu DST tratat. "
               "Orar pe 6 ani = ~52.600 randuri, sub limita Excel.")
