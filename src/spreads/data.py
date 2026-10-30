"""Download + cache public input data.

* Energy-Charts API (Fraunhofer ISE), CC BY 4.0: DE-LU day-ahead price and public net generation by fuel.
* World Bank Commodity Price Data (Pink Sheet), monthly: Natural gas Europe (TTF) and Coal South Africa.
* ECB Data Portal, monthly USD per EUR reference rate (series EXR.M.USD.EUR.SP00.A).

Raw files are cached in ``data/raw`` (git-ignored) and never committed.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
TZ = "Europe/Berlin"

EC_BASE = "https://api.energy-charts.info"
WB_PINK_SHEET = (
    "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/"
    "related/CMO-Historical-Data-Monthly.xlsx"
)
ECB_FX = "https://data-api.ecb.europa.eu/service/data/EXR/M.USD.EUR.SP00.A?format=csvdata&startPeriod=2018-01"

FUEL_COLS = {
    "Fossil gas": "gas",
    "Fossil hard coal": "hard_coal",
    "Fossil brown coal / lignite": "lignite",
}


def _get(url: str, dest: Path, refresh: bool = False, binary: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not refresh:
        return dest
    for attempt in range(5):
        r = requests.get(url, timeout=120, headers={"User-Agent": "clean-spreads-fuel-switching-de (research)"})
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 10)))
            continue
        if r.status_code >= 500 and attempt < 4:     # transient server error -> back off and retry
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        dest.write_bytes(r.content) if binary else dest.write_text(r.text, encoding="utf-8")
        return dest
    raise RuntimeError(f"Download failed after retries: {url}")


def _years(start: str, end: str) -> list[int]:
    return list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))


def load_day_ahead(start: str, end: str, bzn: str = "DE-LU", refresh: bool = False) -> pd.Series:
    """Hourly day-ahead price (EUR/MWh), Europe/Berlin index. 15-min data (from Oct 2025) -> hourly mean."""
    parts = []
    for y in _years(start, end):
        s, e = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01")), min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31"))
        f = _get(f"{EC_BASE}/price?bzn={bzn}&start={s:%Y-%m-%d}&end={e:%Y-%m-%d}", RAW / f"price_{bzn}_{y}.json", refresh)
        d = json.loads(f.read_text())
        idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
        parts.append(pd.Series(d["price"], index=idx, dtype="float64"))
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()]
    return s.resample("1h").mean().loc[start:end].rename("price")


def load_generation(start: str, end: str, country: str = "de", refresh: bool = False) -> pd.DataFrame:
    """Hourly average public net generation (MW) for gas, hard coal and lignite."""
    parts = []
    for y in _years(start, end):
        s, e = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01")), min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31"))
        f = _get(f"{EC_BASE}/public_power?country={country}&start={s:%Y-%m-%d}&end={e:%Y-%m-%d}",
                 RAW / f"public_power_{country}_{y}.json", refresh)
        d = json.loads(f.read_text())
        idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
        cols = {FUEL_COLS[p["name"]]: p["data"] for p in d["production_types"] if p["name"] in FUEL_COLS}
        parts.append(pd.DataFrame(cols, index=idx, dtype="float64"))
    g = pd.concat(parts).sort_index()
    g = g[~g.index.duplicated()]
    return g.resample("1h").mean().loc[start:end]


def load_fuels(refresh: bool = False) -> pd.DataFrame:
    """Monthly World Bank fuel prices in USD: gas_usd_mmbtu (Europe/TTF) and coal_usd_t (South Africa, FOB)."""
    f = _get(WB_PINK_SHEET, RAW / "CMO-Historical-Data-Monthly.xlsx", refresh, binary=True)
    raw = pd.read_excel(f, sheet_name="Monthly Prices", header=None)
    header_row = raw.index[raw.iloc[:, 1].map(lambda x: "Crude oil, average" in str(x))][0]
    names = pd.Series([str(x).strip() if pd.notna(x) else "" for x in raw.iloc[header_row]])
    data = raw.iloc[header_row + 2:].copy()
    data.columns = ["month"] + list(names.iloc[1:])
    data = data[data["month"].astype(str).str.match(r"^\d{4}M\d{2}$")]
    gas_col = [c for c in names if c.startswith("Natural gas, Europe")][0]
    coal_col = [c for c in names if c.startswith("Coal, South African")][0]
    out = pd.DataFrame({
        "gas_usd_mmbtu": pd.to_numeric(data[gas_col], errors="coerce").values,
        "coal_usd_t": pd.to_numeric(data[coal_col], errors="coerce").values,
    }, index=pd.PeriodIndex([m.replace("M", "-") for m in data["month"]], freq="M"))
    return out.dropna()


def load_fx(refresh: bool = False) -> pd.Series:
    """Monthly average USD per 1 EUR (ECB reference rate)."""
    f = _get(ECB_FX, RAW / "ecb_usd_per_eur_monthly.csv", refresh)
    d = pd.read_csv(io.StringIO(f.read_text()))
    return pd.Series(d["OBS_VALUE"].values, index=pd.PeriodIndex(d["TIME_PERIOD"], freq="M"), name="usd_per_eur")
