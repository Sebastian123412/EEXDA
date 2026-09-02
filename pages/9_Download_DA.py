# pages/9_Download_ENTSOE.py — Descarcator generic ENTSO-E, multi-zona, multi-an.
#
# CE FACE: alegi setul de date, zonele si anii, apesi, primesti un CSV lat cu o
# coloana per serie. Nu salveaza nimic. Analiza se face in alta parte.
#
# DE CE NU EXPORTUL DIN GUI: la load si generare GUI-ul sparge un an in fisiere
# lunare. Sase tari x sase ani = 432 de fisiere. API-ul da un an per cerere.
#
# LIMITA: 60 de cereri in 60 de secunde, altfel IP banat pana la 10 minute.
# MIN_GAP=2s tine marja. Un job de 6 zone x 6 ani = 36 de cereri.
#
# CAPCANE TRATATE IN PARSER:
#   * curveType A03 — pozitiile vin rare si valoarea tine pana la urmatoarea.
#     Fara reindexare + ffill pierzi ore intregi, in tacere.
#   * mai multe TimeSeries pe acelasi interval (Sequence 1/2 din GUI). Se
#     pastreaza prima; la preturi e seria SDAC cuplata.
#   * rezolutii mixte: orar pana in 2025, sferturi dupa. Iesirea orara mediaza.
#   * A75: seriile cu outBiddingZone_Domain sunt CONSUM (pompaj), nu producție.
#     Amestecate, subestimeaza flota de pompaj si umfla generarea.
#
# ATENTIE LA AGREGARE, MAI TARZIU: la load si generare unitatea e MW. Suma
# sferturilor NU e energie — trebuie MW x 0,25 h. O insumare naiva pe serii
# mixte da un salt de 4x la trecerea la 15 minute, care arata ca o creștere
# reala de consum. De asta iesirea implicita e orara.

from __future__ import annotations

import time
from datetime import date
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Download ENTSO-E", layout="wide")

API = "https://web-api.tp.entsoe.eu/api"
TZ = "Europe/Berlin"
MIN_GAP = 2.0
MAX_RETRIES = 4
FREQ = {"PT15M": "15min", "PT30M": "30min", "PT60M": "60min",
        "PT1H": "60min", "P1D": "1D"}

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

# Production types. Names kept short because they become column suffixes.
PSR = {
    "B01": "biomass",   "B02": "lignite",    "B03": "coalgas",  "B04": "gas",
    "B05": "hardcoal",  "B06": "oil",        "B07": "oilshale", "B08": "peat",
    "B09": "geo",       "B10": "hydro_pump", "B11": "hydro_ror","B12": "hydro_res",
    "B13": "marine",    "B14": "nuclear",    "B15": "other_res","B16": "solar",
    "B17": "waste",     "B18": "wind_off",   "B19": "wind_on",  "B20": "other",
    "B25": "energy_storage",
}

# dataset key: (label, documentType, processType, domain style, value tag,
#               split by psrType, psr filter or None)
DATASETS = {
    "prices":     ("Day-ahead prices (A44)",            "A44", "A01", "in_out",  "price.amount", False, None),
    "load_act":   ("Actual total load (A65/A16)",        "A65", "A16", "out_bz",  "quantity",     False, None),
    "load_fc":    ("Day-ahead load forecast (A65/A01)",  "A65", "A01", "out_bz",  "quantity",     False, None),
    "ws_fc":      ("Wind & solar forecast (A69/A01)",    "A69", "A01", "in_dom",  "quantity",     True,  ["B16","B18","B19"]),
    "gen_act":    ("Actual generation per type (A75/A16)","A75", "A16", "in_dom", "quantity",     True,  None),
    "flows":      ("Physical cross-border flows (A11)",  "A11", None,  "pair",    "quantity",     False, None),
}

# Interconnections that actually exist, so the flow job does not fire 30 requests
# for borders like DE-ES that have no interconnector.
BORDERS = {
    ("DE-LU","FR"), ("DE-LU","AT"), ("DE-LU","CZ"), ("DE-LU","PL"), ("DE-LU","NL"),
    ("DE-LU","BE"), ("DE-LU","CH"), ("FR","ES"), ("FR","IT-NORD"), ("FR","CH"),
    ("FR","BE"), ("AT","IT-NORD"), ("AT","HU"), ("AT","CH"), ("AT","CZ"),
    ("AT","SI"), ("HU","SK"), ("HU","RO"), ("HU","HR"), ("HU","RS"), ("HU","SI"),
    ("IT-NORD","CH"), ("IT-NORD","SI"), ("ES","PT"), ("CZ","PL"), ("CZ","SK"),
    ("SK","PL"), ("SI","HR"), ("RO","BG"), ("RO","RS"), ("BG","GR"), ("BG","RS"),
}


# --------------------------------------------------------------------------- #
def parse_doc(xml_text: str, value_tag: str) -> pd.DataFrame:
    """MarketDocument -> [ts_utc, value, resolution, seq, psr, flow]."""
    root = ET.fromstring(xml_text)
    ns = {"n": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    f  = lambda el, t: el.find(f"n:{t}", ns) if ns else el.find(t)
    fa = lambda el, t: el.findall(f"n:{t}", ns) if ns else el.findall(t)

    reason = f(root, "Reason")
    if reason is not None and not fa(root, "TimeSeries"):
        t = f(reason, "text")
        raise RuntimeError(f"ENTSO-E: {t.text if t is not None else 'no data'}")

    out, seq = [], 0
    for ts in fa(root, "TimeSeries"):
        seq += 1
        psr = None
        mk = f(ts, "MktPSRType")
        if mk is not None:
            p = f(mk, "psrType")
            if p is not None:
                psr = p.text
        flow = ('gen'  if f(ts, "inBiddingZone_Domain.mRID")  is not None else
                'cons' if f(ts, "outBiddingZone_Domain.mRID") is not None else None)
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
            pts = []
            for pt in fa(per, "Point"):
                v = f(pt, value_tag)
                if v is not None:
                    pts.append((int(f(pt, "position").text), float(v.text)))
            pts = [(p, v) for p, v in pts if 1 <= p <= len(idx)]
            if not pts:
                continue
            s = pd.Series(np.nan, index=range(1, len(idx) + 1), dtype=float)
            s.loc[[p for p, _ in pts]] = [v for _, v in pts]
            out.append(pd.DataFrame({"ts_utc": idx, "value": s.ffill().to_numpy(),
                                     "resolution": res, "seq": seq,
                                     "psr": psr, "flow": flow}))
    if not out:
        return pd.DataFrame(columns=["ts_utc","value","resolution","seq","psr","flow"])
    return pd.concat(out, ignore_index=True).dropna(subset=["value"])


_last = [0.0]


def fetch(dsk: str, target, year: int, token: str) -> pd.DataFrame:
    """One calendar year for one zone (or one ordered zone pair for flows)."""
    _, doc, proc, style, vtag, _, _ = DATASETS[dsk]
    p = {"securityToken": token, "documentType": doc,
         "periodStart": f"{year}01010000", "periodEnd": f"{year+1}01010000"}
    if proc:
        p["processType"] = proc
    if style == "in_out":
        p["in_Domain"] = p["out_Domain"] = EIC[target]
    elif style == "out_bz":
        p["outBiddingZone_Domain"] = EIC[target]
    elif style == "in_dom":
        p["in_Domain"] = EIC[target]
    elif style == "pair":
        a, b = target
        p["out_Domain"], p["in_Domain"] = EIC[a], EIC[b]   # flow from a to b
    err = None
    for attempt in range(MAX_RETRIES):
        gap = time.monotonic() - _last[0]
        if gap < MIN_GAP:
            time.sleep(MIN_GAP - gap)
        _last[0] = time.monotonic()
        try:
            r = requests.get(API, params=p, timeout=180)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            if r.status_code == 400:
                raise RuntimeError("HTTP 400 — no data or bad parameters")
            r.raise_for_status()
            return parse_doc(r.text, vtag)
        except Exception as exc:                       # noqa: BLE001
            err = exc
            if isinstance(exc, RuntimeError):
                raise
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(str(err))


def to_series(df: pd.DataFrame, out_res: str) -> pd.Series:
    """One (target, psr, flow) slice -> a single resampled series."""
    d = (df.sort_values(["ts_utc", "resolution", "seq"])
           .drop_duplicates(["ts_utc", "resolution"], keep="first"))
    fine = d[d["resolution"] == "PT15M"]
    hour = d[d["resolution"].isin(["PT60M", "PT1H"])]
    parts = []
    if not fine.empty:
        parts.append(fine.set_index("ts_utc")["value"])
    if not hour.empty:
        h = hour.set_index("ts_utc")["value"]
        if not fine.empty:
            h = h[~h.index.floor("h").isin(fine.index.floor("h").unique())]
        parts.append(h)
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s.resample(out_res).mean()


# --------------------------------------------------------------------------- #
st.title("Descarca date ENTSO-E")

token = st.secrets.get("ENTSOE_TOKEN", "")
if not token:
    st.error("Lipseste ENTSOE_TOKEN din secrets.")
    st.stop()

dsk = st.selectbox("Set de date", list(DATASETS),
                   format_func=lambda k: DATASETS[k][0])
label, doc, proc, style, vtag, split, psrfilt = DATASETS[dsk]

a, b, c, d_ = st.columns([2.4, 1, 1, 1.1])
if style == "pair":
    zones = a.multiselect("Zone", list(EIC), default=DEFAULT, key="zp")
    only_borders = a.checkbox("Doar granite care exista fizic", True)
    targets = [(x, y) for x in zones for y in zones if x != y
               and (not only_borders or (x, y) in BORDERS or (y, x) in BORDERS)]
    a.caption(f"{len(targets)} direcții. ENTSO-E publica fiecare sens separat.")
else:
    zones = a.multiselect("Zone", list(EIC), default=DEFAULT)
    targets = zones
y0 = b.number_input("Din", 2015, date.today().year, 2021, step=1)
y1 = c.number_input("Pana in", 2015, date.today().year, date.today().year, step=1)
out_res = d_.selectbox("Rezolutie iesire", ["60min", "15min", "D"], index=0)

if split:
    keep = st.multiselect("Tipuri de producție",
                          psrfilt or list(PSR), default=psrfilt or ["B16","B18","B19"],
                          format_func=lambda p: f"{p} {PSR.get(p, p)}")
else:
    keep = None

years = list(range(int(y0), int(y1) + 1))
n = len(targets) * len(years)
st.caption(f"{len(targets)} ținte x {len(years)} ani = **{n} cereri**, "
           f"~{n*MIN_GAP/60:.0f}-{n*1.0:.0f} minute. Limita ENTSO-E e 60 cereri / 60s. "
           "Nu se salveaza nimic — la final apesi download.")

if st.button("Descarca", type="primary", disabled=not targets):
    bar, box, notes, cols = st.progress(0.0), st.empty(), [], {}
    i = 0
    for tgt in targets:
        name = tgt if isinstance(tgt, str) else f"{tgt[0]}>{tgt[1]}"
        chunks = []
        for y in years:
            i += 1
            bar.progress(i / n, text=f"{name} {y}  ({i}/{n})")
            try:
                df = fetch(dsk, tgt, y, token)
                if not df.empty:
                    chunks.append(df)
                    if df["seq"].max() > 1 and not split:
                        notes.append(f"{name} {y}: {int(df['seq'].max())} secvente, prima pastrata")
            except Exception as exc:                   # noqa: BLE001
                notes.append(f"{name} {y}: {str(exc)[:90]}")
            box.info(" · ".join(notes[-3:]) if notes else "")
        if not chunks:
            continue
        allc = pd.concat(chunks, ignore_index=True)
        if split:
            for (psr, fl), g in allc.groupby(["psr", "flow"], dropna=False):
                if keep and psr not in keep:
                    continue
                suffix = PSR.get(psr, psr or "na")
                if fl == "cons":
                    suffix += "_cons"          # pumped-storage consumption, not generation
                cols[f"{name}_{suffix}"] = to_series(g, out_res)
        else:
            cols[name] = to_series(allc, out_res)

    if not cols:
        st.error("Nu am obtinut nimic. Verifica zonele si anii — nu toate seriile "
                 "exista pentru toate zonele si toti anii.")
    else:
        wide = pd.DataFrame(cols).sort_index()
        wide.index = wide.index.tz_convert(TZ)
        wide.index.name = "timestamp_local"
        st.session_state["dl"] = wide
        st.session_state["dl_notes"] = notes
        st.session_state["dl_label"] = dsk

wide = st.session_state.get("dl")
if wide is not None:
    st.success(f"{len(wide):,} randuri x {len(wide.columns)} serii  |  "
               f"{wide.index.min():%Y-%m-%d} - {wide.index.max():%Y-%m-%d}")
    comp = (100 * wide.notna().mean()).round(1).sort_values()
    thin = comp[comp < 95]
    if len(thin):
        st.warning("Serii sub 95% completitudine: " +
                   ", ".join(f"{k} {v:.0f}%" for k, v in thin.items()) +
                   ". O medie lunara calculata pe date incomplete arata normala si e "
                   "parțiala — verifica inainte sa agreghezi.")
    for nt in st.session_state.get("dl_notes", []):
        st.caption(nt)
    st.download_button(
        "CSV", wide.round(2).to_csv().encode(),
        f"{st.session_state.get('dl_label','entsoe')}_{wide.index.min():%Y%m%d}_"
        f"{wide.index.max():%Y%m%d}_{out_res}.csv", "text/csv", type="primary")
    st.caption("Format lat, ora locala CET/CEST cu DST tratat. Coloanele de load si "
               "generare sunt in MW — la agregare, energia e MW mediu x orele perioadei, "
               "nu suma valorilor.")
