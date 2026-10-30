"""Clean spark / dark spreads, SRMC and coal-to-gas fuel-switching CO2 price.

Notation (per MWh of electricity, MWh_e):
    fuel cost   F_x = p_fuel,x / eta_x + VOM_x                       [EUR/MWh_e]
    CO2 inten.  c_x = EF_x / eta_x                                    [tCO2/MWh_e]
    SRMC        S_x = F_x + EUA * c_x                                 [EUR/MWh_e]
    clean spark spread   CSS = P - S_gas      clean dark spread  CDS = P - S_coal
    switching price      EUA* = (F_gas - F_coal) / (c_coal - c_gas)  -> gas cheaper than coal when EUA > EUA*
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MMBTU_PER_MWH = 3.412142          # 1 MWh = 3.412142 MMBtu
KJ_PER_KCAL = 4.1868


def coal_mwh_th_per_t(kcal_per_kg: float) -> float:
    """Energy content of 1 t coal in MWh_th (kcal/kg -> GJ/t -> MWh/t)."""
    return kcal_per_kg * KJ_PER_KCAL / 3600.0       # kJ/kg == MJ/t; /3600 -> MWh/t


def fuel_prices_eur(fuels_usd: pd.DataFrame, fx_usd_per_eur: pd.Series, cfg: dict) -> pd.DataFrame:
    """Convert World Bank USD prices to EUR/MWh_th."""
    df = fuels_usd.join(fx_usd_per_eur, how="inner")
    c = cfg["coal_plant"]
    gas = df["gas_usd_mmbtu"] * MMBTU_PER_MWH / df["usd_per_eur"] + cfg["gas_plant"]["hub_to_plant_eur_per_mwh_th"]
    coal = (df["coal_usd_t"] + c["freight_usd_per_t"]) / df["usd_per_eur"] / coal_mwh_th_per_t(c["calorific_kcal_per_kg"])
    return pd.DataFrame({"gas_eur_mwh_th": gas, "coal_eur_mwh_th": coal, "usd_per_eur": df["usd_per_eur"]})


def fuel_cost(p_fuel_th, eta, vom):
    return p_fuel_th / eta + vom


def carbon_intensity(ef_th, eta):
    return ef_th / eta


def switching_price(p_gas, p_coal, eta_g, eta_c, ef_g, ef_c, vom_g=0.0, vom_c=0.0):
    """CO2 price (EUR/t) at which gas and coal SRMC are equal. NaN if gas is not less carbon-intensive."""
    num = fuel_cost(p_gas, eta_g, vom_g) - fuel_cost(p_coal, eta_c, vom_c)
    den = carbon_intensity(ef_c, eta_c) - carbon_intensity(ef_g, eta_g)
    return np.where(np.asarray(den) > 0, num / den, np.nan)


def eua_monthly(index: pd.PeriodIndex, eua_by_year: dict) -> pd.Series:
    eua = {int(k): float(v) for k, v in eua_by_year.items()}
    return pd.Series([eua[p.year] for p in index], index=index, name="eua")


def monthly_table(fuels_eur: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    g, c = cfg["gas_plant"], cfg["coal_plant"]
    m = fuels_eur.copy()
    m["eua"] = eua_monthly(m.index, cfg["eua_eur_per_t"])
    m["fuel_gas"] = fuel_cost(m["gas_eur_mwh_th"], g["efficiency"], g["vom_eur_per_mwh_e"])
    m["fuel_coal"] = fuel_cost(m["coal_eur_mwh_th"], c["efficiency"], c["vom_eur_per_mwh_e"])
    m["ci_gas"] = carbon_intensity(g["emission_factor_t_per_mwh_th"], g["efficiency"])
    m["ci_coal"] = carbon_intensity(c["emission_factor_t_per_mwh_th"], c["efficiency"])
    m["srmc_gas"] = m["fuel_gas"] + m["eua"] * m["ci_gas"]
    m["srmc_coal"] = m["fuel_coal"] + m["eua"] * m["ci_coal"]
    m["carbon_gas"] = m["eua"] * m["ci_gas"]
    m["carbon_coal"] = m["eua"] * m["ci_coal"]
    args = dict(ef_g=g["emission_factor_t_per_mwh_th"], ef_c=c["emission_factor_t_per_mwh_th"],
                vom_g=g["vom_eur_per_mwh_e"], vom_c=c["vom_eur_per_mwh_e"])
    m["switch_price"] = switching_price(m["gas_eur_mwh_th"], m["coal_eur_mwh_th"], g["efficiency"], c["efficiency"], **args)
    # band: easiest switch (efficient CCGT vs old coal) .. hardest switch (old CCGT vs efficient coal)
    m["switch_low"] = switching_price(m["gas_eur_mwh_th"], m["coal_eur_mwh_th"], g["efficiency_high"], c["efficiency_low"], **args)
    m["switch_high"] = switching_price(m["gas_eur_mwh_th"], m["coal_eur_mwh_th"], g["efficiency_low"], c["efficiency_high"], **args)
    m["gas_cheaper"] = m["srmc_gas"] < m["srmc_coal"]
    return m


def hourly_spreads(price: pd.Series, monthly: pd.DataFrame) -> pd.DataFrame:
    h = price.to_frame()
    per = h.index.tz_localize(None).to_period("M")
    for col in ["srmc_gas", "srmc_coal", "fuel_gas", "fuel_coal"]:
        h[col] = monthly[col].reindex(per).values
    h["css"] = h["price"] - h["srmc_gas"]
    h["cds"] = h["price"] - h["srmc_coal"]
    h["spark_spread"] = h["price"] - h["fuel_gas"]     # "dirty" spreads (before carbon)
    h["dark_spread"] = h["price"] - h["fuel_coal"]
    return h.dropna(subset=["price", "srmc_gas"])


def annual_summary(h: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    y = h.index.year
    out = pd.DataFrame({
        "hours": h.groupby(y).size(),
        "baseload_price": h.groupby(y)["price"].mean(),
        "mean_css": h.groupby(y)["css"].mean(),
        "mean_cds": h.groupby(y)["cds"].mean(),
        "share_hours_css_pos": h.assign(x=h["css"] > 0).groupby(y)["x"].mean(),
        "share_hours_cds_pos": h.assign(x=h["cds"] > 0).groupby(y)["x"].mean(),
        # gross margin of a fully flexible 1 MW unit running only when its clean spread > 0 (no start costs/constraints)
        "gm_gas_eur_per_mw": h["css"].clip(lower=0).groupby(y).sum(),
        "gm_coal_eur_per_mw": h["cds"].clip(lower=0).groupby(y).sum(),
    })
    my = monthly.index.year
    out["eua"] = monthly.groupby(my)["eua"].first()
    out["mean_switch_price"] = monthly.groupby(my)["switch_price"].mean()
    out["months_gas_cheaper"] = monthly.groupby(my)["gas_cheaper"].sum()
    out["months"] = monthly.groupby(my).size()
    out["mean_gas_eur_mwh_th"] = monthly.groupby(my)["gas_eur_mwh_th"].mean()
    out["mean_coal_eur_mwh_th"] = monthly.groupby(my)["coal_eur_mwh_th"].mean()
    out.index.name = "year"
    return out


def hour_of_day_profile(h: pd.DataFrame, col: str = "css") -> pd.DataFrame:
    return h.groupby([h.index.year, h.index.hour])[col].mean().unstack(0)


def eua_sweep(monthly: pd.DataFrame, eua_grid: np.ndarray) -> pd.DataFrame:
    """Share of months in each year where gas SRMC < coal SRMC, for a hypothetical flat EUA price."""
    res = {}
    for yr, m in monthly.groupby(monthly.index.year):
        gap = (m["fuel_gas"].values[None, :] - m["fuel_coal"].values[None, :]
               + eua_grid[:, None] * (m["ci_gas"].values[None, :] - m["ci_coal"].values[None, :]))
        res[yr] = (gap < 0).mean(axis=1)
    return pd.DataFrame(res, index=pd.Index(eua_grid, name="eua"))


def generation_monthly(gen: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    g = gen.copy()
    g.index = g.index.tz_localize(None)
    gm = g.resample("MS").mean()
    gm.index = gm.index.to_period("M")
    gm["gas_share_gas_coal"] = gm["gas"] / (gm["gas"] + gm["hard_coal"])
    gm = gm.join(monthly[["srmc_gas", "srmc_coal", "switch_price", "eua"]], how="inner")
    gm["srmc_gap_coal_minus_gas"] = gm["srmc_coal"] - gm["srmc_gas"]
    gm["eua_minus_switch"] = gm["eua"] - gm["switch_price"]
    return gm


def spearman(a: pd.Series, b: pd.Series) -> float:
    d = pd.concat([a, b], axis=1).dropna()
    return float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()))
