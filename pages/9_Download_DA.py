# pages/9_Download_ENTSOE.py — Descarcator generic ENTSO-E, multi-zona, multi-an.
#
# CE FACE: alegi setul de date, zonele si anii, apesi, primesti un CSV lat cu o
# coloana per serie. Nu salveaza nimic. Analiza se face in alta parte.
#
# DE CE NU EXPORTUL DIN GUI: la load si generare GUI-ul sparge un an in fisiere
# lunare. API-ul da un an per cerere.
#
# LIMITA: 60 de cereri in 60 de secunde, altfel IP banat pana la 10 minute.
#
# ERORI: ENTSO-E raspunde 400 atat pentru parametri greșiți cat si pentru lipsa
# de date, si pune motivul REAL in corpul raspunsului. Codul il extrage si il
# afiseaza. Fara asta nu poti distinge o limita de interval de un entitlement
# lipsa, si sunt remedii complet diferite.
#
# INTERVALE: periodEnd se opreste la 12-31 23:00, nu la 01-01 al anului urmator.
# A65 are restrictie de un an per interval si granita exacta o declanseaza. Iar
# pentru anul curent, capatul e ziua de azi — altfel ceri date din viitor.
#
# CAPCANE IN PARSER:
#   * curveType A03 — pozitiile vin rare si valoarea tine pana la urmatoarea.
#     Fara reindexare + ffill pierzi ore intregi, in tacere.
#   * mai multe TimeSeries pe acelasi interval (Sequence 1/2 din GUI). Se
#     pastreaza prima; la preturi e seria SDAC cuplata.
#   * rezolutii mixte: orar pana in 2025, sferturi dupa. Iesirea orara mediaza.
#   * A75: seriile cu outBiddingZone_Domain sunt CONSUM (pompaj), nu producție.
#     Amestecate, subestimeaza flota de pompaj si umfla generarea.
#
# UNITATI: preturile sunt EUR/MWh. Load si generare sunt MW. Suma sferturilor NU
# e energie — trebuie MW x 0,25 h. De asta iesirea implicita e orara.

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
MIN_GAP = 1.2
MAX_RETRIES = 4
STEP_NS = {"PT15M": 900_000_000_000, "PT30M": 1_800_000_000_000,
           "PT60M": 3_600_000_000_000, "PT1H": 3_600_000_000_000,
           "P1D": 86_400_000_000_000}

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
DEFAULT = ["DE-LU"]

PSR = {
    "B01": "biomass",   "B02": "lignite",    "B03": "coalgas",  "B04": "gas",
    "B05": "hardcoal",  "B06": "oil",        "B07": "oilshale", "B08": "peat",
    "B09": "geo",        "B10": "hydro_pump", "B11": "hydro_ror", "B12": "hydro_res",
    "B13": "marine",    "B14": "nuclear",    "B15": "other_res", "B16": "solar",
    "B17": "waste",     "B18": "wind_off",   "B19": "wind_on",  "B20": "other",
    "B25": "storage",
}

# key: (label, documentType, processType, domain style, value tag, split, psr filter)
DATASETS = {
    "prices":    ("Day-ahead prices (A44)",             "A44", "A01", "in_out", "price.amount", False, None),
    "load_act":  ("Actual total load (A65/A16)",         "A65", "A16", "out_bz", "quantity",     False, None),
    "load_fc":   ("Day-ahead load forecast (A65/A01)",   "A65", "A01", "out_bz", "quantity",     False, None),
    "ws_fc":     ("Wind & solar forecast (A69/A01)",     "A69", "A01", "in_dom", "quantity",     True,  ["B16", "B18", "B19"]),
    "gen_act":   ("Actual generation per type (A75/A16)", "A75", "A16", "in_dom", "quantity",    True,  None),
    "flows":     ("Physical cross-border flows (A11)",   "A11", None,  "pair",   "quantity",     False, None),
}

BORDERS = {
    ("DE-LU", "FR"), ("DE-LU", "AT"), ("DE-LU", "CZ"), ("DE-LU", "PL"), ("DE-LU", "NL"),
    ("DE-LU", "BE"), ("DE-LU", "CH"), ("FR", "ES"), ("FR", "IT-NORD"), ("FR", "CH"),
    ("FR", "BE"), ("AT", "IT-NORD"), ("AT", "HU"), ("AT", "CH"), ("AT", "CZ"),
    ("AT", "SI"), ("HU", "SK"), ("HU", "RO"), ("HU", "HR"), ("HU", "RS"), ("HU", "SI"),
    ("IT-NORD", "CH"), ("IT-NORD", "SI"), ("ES", "PT"), ("CZ", "PL"), ("CZ", "SK"),
    ("SK", "PL"), ("SI", "HR"), ("RO", "BG"), ("RO", "RS"), ("BG", "GR"), ("BG", "RS"),
}


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def reason_text(xml_text: str) -> str:
    """ENTSO-E puts the real explanation in a Reason/text node. Pull it out."""
    try:
        root = ET.fromstring(xml_text)
        ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
        code_t = f"{{{ns}}}code" if ns else "code"
        text_t = f"{{{ns}}}text" if ns else "text"
        code = root.find(f".//{code_t}")
        txt = root.find(f".//{text_t}")
        bits = [b.text.strip() for b in (code, txt) if b is not None and b.text]
        if bits:
            return " · ".join(bits)
    except Exception:                                  # noqa: BLE001
        pass
    return xml_text.strip()[:300] or "răspuns gol"


def parse_doc(xml_text: str, value_tag: str) -> pd.DataFrame:
    """MarketDocument -> [ts_utc, value, resolution, seq, psr, flow].

    numpy timestamp arithmetic and one concat at the end. ENTSO-E returns one
    TimeSeries per day, so a year of generation across 18 production types is
    ~6,500 TimeSeries in a single document — a DataFrame per Period is ~10x
    slower, measured.
    """
    root = ET.fromstring(xml_text)
    ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    q = (lambda t: f"{{{ns}}}{t}") if ns else (lambda t: t)
    TS, PER, PT = q("TimeSeries"), q("Period"), q("Point")
    RES, TI, ST, EN = q("resolution"), q("timeInterval"), q("start"), q("end")
    POS, VAL = q("position"), q(value_tag)
    MK, PSRT = q("MktPSRType"), q("psrType")
    INBZ, OUTBZ = q("inBiddingZone_Domain.mRID"), q("outBiddingZone_Domain.mRID")

    series = root.findall(f".//{TS}")
    if not series:
        raise RuntimeError(reason_text(xml_text))

    ts_parts, v_parts, meta = [], [], []
    for seq, ts in enumerate(series, start=1):
        mk = ts.find(MK)
        psr = None
        if mk is not None:
            p = mk.find(PSRT)
            if p is not None:
                psr = p.text
        flow = ('gen' if ts.find(INBZ) is not None
                else 'cons' if ts.find(OUTBZ) is not None else None)
        for per in ts.findall(PER):
            rn = per.find(RES)
            if rn is None:
                continue
            step = STEP_NS.get(rn.text)
            if step is None:
                continue
            ti = per.find(TI)
            s_ns = pd.Timestamp(ti.find(ST).text).tz_convert("UTC").value
            e_ns = pd.Timestamp(ti.find(EN).text).tz_convert("UTC").value
            n = int((e_ns - s_ns) // step)
            if n <= 0:
                continue
            pos, val = [], []
            for pnt in per.findall(PT):
                v = pnt.find(VAL)
                if v is not None:
                    pos.append(int(pnt.find(POS).text))
                    val.append(float(v.text))
            if not pos:
                continue
            p = np.asarray(pos, dtype=np.int64) - 1
            keep = (p >= 0) & (p < n)
            if not keep.any():
                continue
            arr = np.full(n, np.nan)
            arr[p[keep]] = np.asarray(val, dtype=float)[keep]
            mask = ~np.isnan(arr)
            if not mask.all():                         # curveType A03 forward fill
                ix = np.where(mask, np.arange(n), 0)
                np.maximum.accumulate(ix, out=ix)
                arr = arr[ix]
                arr[:int(np.argmax(mask))] = np.nan
            ts_parts.append(s_ns + np.arange(n, dtype=np.int64) * step)
            v_parts.append(arr)
            meta.append((rn.text, seq, psr, flow, n))

    if not ts_parts:
        return pd.DataFrame(columns=["ts_utc", "value", "resolution", "seq", "psr", "flow"])
    reps = np.array([m[4] for m in meta])
    return pd.DataFrame({
        "ts_utc": pd.to_datetime(np.concatenate(ts_parts), utc=True),
        "value": np.concatenate(v_parts),
        "resolution": np.repeat([m[0] for m in meta], reps),
        "seq": np.repeat([m[1] for m in meta], reps),
        "psr": np.repeat([m[2] for m in meta], reps),
        "flow": np.repeat([m[3] for m in meta], reps),
    }).dropna(subset=["value"])


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
_last = [0.0]


def period_bounds(year: int) -> tuple[str, str]:
    """Start of year to 12-31 23:00, or to today for the current year.

    Two bugs live here if you get it wrong. Ending at 01-01 of the next year
    hits the one-year interval limit exactly, and for the current year it asks
    for data that does not exist yet.
    """
    today = date.today()
    start = f"{year}01010000"
    if year > today.year:
        raise RuntimeError(f"{year} e in viitor")
    if year == today.year:
        return start, today.strftime("%Y%m%d") + "0000"
    return start, f"{year}12312300"


def fetch(dsk: str, target, year: int, token: str) -> pd.DataFrame:
    _, doc, proc, style, vtag, _, _ = DATASETS[dsk]
    ps, pe = period_bounds(year)
    p = {"securityToken": token, "documentType": doc,
         "periodStart": ps, "periodEnd": pe}
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
        p["out_Domain"], p["in_Domain"] = EIC[a], EIC[b]

    err = None
    for attempt in range(MAX_RETRIES):
        gap = time.monotonic() - _last[0]
        if gap < MIN_GAP:
            time.sleep(MIN_GAP - gap)
        _last[0] = time.monotonic()
        try:
            r = requests.get(API, params=p, timeout=240)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            if r.status_code in (400, 401, 403):
                raise RuntimeError(f"HTTP {r.status_code} · {reason_text(r.text)}")
            r.raise_for_status()
            return parse_doc(r.text, vtag)
        except RuntimeError:
            raise
        except Exception as exc:                       # noqa: BLE001
            err = exc
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(str(err))


def to_series(df: pd.DataFrame, out_res: str) -> pd.Series:
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
# UI
# --------------------------------------------------------------------------- #
st.title("Descarca date ENTSO-E")

token = st.secrets.get("ENTSOE_TOKEN", "")
if not token:
    st.error("Lipseste ENTSOE_TOKEN din secrets.")
    st.stop()

dsk = st.selectbox("Set de date", list(DATASETS), format_func=lambda k: DATASETS[k][0])
label, doc, proc, style, vtag, split, psrfilt = DATASETS[dsk]

a, b, c, d_ = st.columns([2.4, 1, 1, 1.1])
if style == "pair":
    zones = a.multiselect("Zone", list(EIC), default=["DE-LU", "FR", "AT", "HU"], key="zp")
    only_borders = a.checkbox("Doar granite care exista fizic", True)
    targets = [(x, y) for x in zones for y in zones if x != y
               and (not only_borders or (x, y) in BORDERS or (y, x) in BORDERS)]
    a.caption(f"{len(targets)} direcții. ENTSO-E publica fiecare sens separat.")
else:
    zones = a.multiselect("Zone", list(EIC), default=DEFAULT)
    targets = zones
yr_now = date.today().year
y0 = b.number_input("Din", 2015, yr_now, max(2015, yr_now - 1), step=1)
y1 = c.number_input("Pana in", 2015, yr_now, yr_now, step=1)
out_res = d_.selectbox("Rezolutie iesire", ["60min", "15min", "D"], index=0)

if split:
    keep = st.multiselect("Tipuri de producție", psrfilt or list(PSR),
                          default=psrfilt or ["B16", "B18", "B19"],
                          format_func=lambda p: f"{p} {PSR.get(p, p)}")
else:
    keep = None

years = list(range(int(y0), int(y1) + 1))
n = len(targets) * len(years)
st.caption(f"{len(targets)} ținte x {len(years)} ani = **{n} cereri**, "
           f"~{max(1, round(n * MIN_GAP / 60))}-{n} minute. Limita ENTSO-E e 60 cereri / 60s. "
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
                if df.empty:
                    notes.append(f"{name} {y}: document fara puncte")
                else:
                    chunks.append(df)
                    if df["seq"].max() > 1 and not split:
                        notes.append(f"{name} {y}: {int(df['seq'].max())} secvente, prima pastrata")
            except Exception as exc:                   # noqa: BLE001
                notes.append(f"**{name} {y}**: {str(exc)[:220]}")
            box.markdown("  \n".join(notes[-10:]) if notes else "")
        if not chunks:
            continue
        allc = pd.concat(chunks, ignore_index=True)
        if split:
            for (psr, fl), g in allc.groupby(["psr", "flow"], dropna=False):
                if keep and psr not in keep:
                    continue
                suffix = PSR.get(psr, psr or "na")
                if fl == "cons":
                    suffix += "_cons"      # pumped-storage consumption, not generation
                cols[f"{name}_{suffix}"] = to_series(g, out_res)
        else:
            cols[name] = to_series(allc, out_res)

    if not cols:
        st.error("Nu am obtinut nimic. Mesajele de mai sus sunt exact ce a raspuns "
                 "ENTSO-E — citeste-le, nu sunt generice.")
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
                   ". O medie lunara pe date incomplete arata normala si e parțiala.")
    for nt in st.session_state.get("dl_notes", []):
        st.caption(nt)
    st.download_button(
        "CSV", wide.round(2).to_csv().encode(),
        f"{st.session_state.get('dl_label','entsoe')}_{wide.index.min():%Y%m%d}_"
        f"{wide.index.max():%Y%m%d}_{out_res}.csv", "text/csv", type="primary")
    st.caption("Format lat, ora locala CET/CEST cu DST tratat. Preturile sunt EUR/MWh; "
               "load si generare sunt MW — la agregare, energia e MW mediu x orele "
               "perioadei, nu suma valorilor.")
