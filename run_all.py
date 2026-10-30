"""End-to-end pipeline: download/cache data -> spreads & switching price -> results/ + figures/.

Usage:  python run_all.py [--refresh]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from spreads import data, model, plots  # noqa: E402

FIG, RES = ROOT / "figures", ROOT / "results"


def main(refresh: bool = False) -> dict:
    cfg = yaml.safe_load((ROOT / "config" / "assumptions.yaml").read_text())
    start, end = cfg["period"]["start"], cfg["period"]["end"]
    FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    log = []

    def say(msg):
        print(msg); log.append(msg)

    # --- data
    fuels = data.load_fuels(refresh)
    fx = data.load_fx(refresh)
    fuels_eur = model.fuel_prices_eur(fuels, fx, cfg).loc[start[:7]:end[:7]]
    price = data.load_day_ahead(start, end, cfg["market"]["bidding_zone"], refresh)
    gen = data.load_generation(start, end, cfg["market"]["country"], refresh)
    say(f"Fuel months: {fuels_eur.index[0]} .. {fuels_eur.index[-1]} ({len(fuels_eur)})")
    say(f"Hourly prices: {len(price)} h, missing {int(price.isna().sum())}; generation: {len(gen)} h")

    # --- model
    m = model.monthly_table(fuels_eur, cfg)
    h = model.hourly_spreads(price, m)
    ann = model.annual_summary(h, m)
    css_prof = model.hour_of_day_profile(h, "css")
    cds_prof = model.hour_of_day_profile(h, "cds")
    grid = np.arange(0, cfg["analysis"]["eua_sweep_max"] + 1e-9, cfg["analysis"]["eua_sweep_step"])
    sweep = model.eua_sweep(m, grid)
    gm = model.generation_monthly(gen, m)
    rho = model.spearman(gm["gas_share_gas_coal"], gm["srmc_gap_coal_minus_gas"])
    rho_switch = model.spearman(gm["gas_share_gas_coal"], gm["eua_minus_switch"])
    # within-year: remove each year's mean (controls for slow structural change, e.g. coal-plant closures)
    yr = gm.index.year
    rho_within = model.spearman(gm["gas_share_gas_coal"] - gm.groupby(yr)["gas_share_gas_coal"].transform("mean"),
                                gm["srmc_gap_coal_minus_gas"] - gm.groupby(yr)["srmc_gap_coal_minus_gas"].transform("mean"))

    # gas share when gas is cheaper vs. when coal is cheaper
    share_gas_cheaper = float(gm.loc[gm["srmc_gap_coal_minus_gas"] > 0, "gas_share_gas_coal"].mean())
    share_coal_cheaper = float(gm.loc[gm["srmc_gap_coal_minus_gas"] <= 0, "gas_share_gas_coal"].mean())

    # EUA needed for gas to be cheaper in >= 50 % of months, per year
    eua_needed_50 = {int(y): (float(sweep.index[np.argmax(sweep[y].values >= 0.5)]) if (sweep[y] >= 0.5).any() else None)
                     for y in sweep.columns}

    # --- outputs
    m_out = m.copy(); m_out.index = m_out.index.astype(str)
    m_out.round(4).to_csv(RES / "monthly_srmc_switching.csv", index_label="month")
    ann.round(3).to_csv(RES / "annual_summary.csv")
    gm_out = gm.copy(); gm_out.index = gm_out.index.astype(str)
    gm_out.round(4).to_csv(RES / "monthly_generation_vs_spread.csv", index_label="month")
    css_prof.round(3).to_csv(RES / "css_hour_of_day.csv", index_label="hour")
    cds_prof.round(3).to_csv(RES / "cds_hour_of_day.csv", index_label="hour")

    price_m = h["price"].copy(); price_m.index = price_m.index.tz_localize(None)
    price_m = price_m.resample("MS").mean(); price_m.index = price_m.index.to_period("M")
    price_m = price_m.reindex(m.index)

    plots.switching_price(m, FIG / "fuel_switching_price.png", cfg)
    plots.srmc_stack(m, price_m, FIG / "srmc_gas_vs_coal.png")
    plots.clean_spreads_monthly(h, FIG / "clean_spreads_monthly.png")
    plots.hour_profile(css_prof, cds_prof, FIG / "spreads_hour_of_day.png")
    plots.gas_share_scatter(gm, rho, FIG / "gas_share_vs_srmc_gap.png")
    plots.eua_sweep(sweep, cfg["eua_eur_per_t"], FIG / "eua_sweep_gas_vs_coal.png")
    plots.gross_margin(ann, FIG / "spread_option_gross_margin.png")

    peak = m["switch_price"].idxmax()
    summary = {
        "period": f"{start} .. {end}",
        "assumptions": {"eta_gas": cfg["gas_plant"]["efficiency"], "eta_coal": cfg["coal_plant"]["efficiency"]},
        "carbon_intensity_t_per_mwh_e": {"gas": round(float(m["ci_gas"].iloc[0]), 4), "coal": round(float(m["ci_coal"].iloc[0]), 4)},
        "switch_price_peak": {"month": str(peak), "eur_per_t": round(float(m.loc[peak, "switch_price"]), 1)},
        "switch_price_min": {"month": str(m["switch_price"].idxmin()), "eur_per_t": round(float(m["switch_price"].min()), 1)},
        "months_gas_cheaper": {"n": int(m["gas_cheaper"].sum()), "of": int(len(m))},
        "spearman_gas_share_vs_srmc_gap": round(rho, 3),
        "spearman_gas_share_vs_eua_minus_switch": round(rho_switch, 3),
        "spearman_within_year_demeaned": round(rho_within, 3),
        "gas_share_when_gas_cheaper": round(share_gas_cheaper, 3),
        "gas_share_when_coal_cheaper": round(share_coal_cheaper, 3),
        "eua_needed_for_gas_cheaper_in_50pct_of_months": eua_needed_50,
        "annual": json.loads(ann.round(3).to_json(orient="index")),
        "notes": ["2026 covers Jan-Sep; Q3 2026 uses the H1 2026 DEHSt EUA average (no published Q3 figure yet).",
                  "Spreads use monthly fuel prices and an annual-average EUA: intra-month fuel/carbon moves are ignored."],
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    say(ann.round(2).to_string())
    say(f"Spearman(gas share, coal-gas SRMC gap) = {rho:.3f}; vs (EUA - switch price) = {rho_switch:.3f}; within-year = {rho_within:.3f}")
    say(f"Gas share of gas+hard coal: {share_gas_cheaper:.1%} when gas cheaper vs {share_coal_cheaper:.1%} when coal cheaper")
    say(f"EUA needed for gas cheaper in >=50% of months: {eua_needed_50}")
    (RES / "run_log.txt").write_text("\n".join(log) + "\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download all cached raw data")
    main(ap.parse_args().refresh)
