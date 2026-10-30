"""Figures (matplotlib, PNG, 150 dpi)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GAS, COAL, CO2, PWR = "#1f77b4", "#4d4d4d", "#2ca02c", "#d62728"
SRC = "Data: Energy-Charts.info (Fraunhofer ISE; SMARD/BNetzA), World Bank Pink Sheet, ECB, DEHSt/EEX. CC BY 4.0 where stated."

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def _ts(idx: pd.PeriodIndex):
    return idx.to_timestamp(how="start")


def _footer(fig, extra=""):
    fig.text(0.01, 0.005, SRC + (" " + extra if extra else ""), fontsize=6.5, color="grey", ha="left", va="bottom")


def switching_price(m: pd.DataFrame, out: Path, cfg: dict):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    t = _ts(m.index)
    cap = 400
    ax.fill_between(t, m["switch_low"].clip(upper=cap), m["switch_high"].clip(upper=cap), color=GAS, alpha=0.15,
                    step="mid", label="Switching band (CCGT 58%/coal 35% .. CCGT 45%/coal 45%)")
    ax.step(t, m["switch_price"].clip(upper=cap), where="mid", color=GAS, lw=1.8,
            label=f"Switching CO2 price, base (CCGT {cfg['gas_plant']['efficiency']:.0%} vs coal {cfg['coal_plant']['efficiency']:.0%})")
    ax.step(t, m["eua"], where="mid", color=CO2, lw=2.2, label="EUA, German annual avg. auction price (DEHSt)")
    ax.fill_between(t, -60, cap, where=m["gas_cheaper"].values, color=CO2, alpha=0.07, step="mid",
                    label="Months where gas SRMC < coal SRMC (base case)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-60, cap)
    ax.set_ylabel("EUR / tCO2")
    ax.set_title("Coal-to-gas fuel-switching carbon price vs. EUA, Germany 2019 - Sep 2026 (monthly)")
    peak = m["switch_price"].idxmax()
    ax.annotate(f"peak {m['switch_price'].max():.0f} EUR/t ({peak})", xy=(_ts(pd.PeriodIndex([peak]))[0], cap),
                xytext=(10, -18), textcoords="offset points", fontsize=8)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    _footer(fig, "Values above 400 EUR/t clipped.")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out); plt.close(fig)


def srmc_stack(m: pd.DataFrame, price_monthly: pd.Series, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    t = _ts(m.index)
    for ax, fuel, col, lab in [(axes[0], "gas", GAS, "CCGT"), (axes[1], "coal", COAL, "Hard coal")]:
        fuel_part = m[f"fuel_{fuel}"]
        ax.stackplot(t, fuel_part, m[f"carbon_{fuel}"], colors=[col, CO2], alpha=0.75,
                     labels=["Fuel + VOM", "Carbon (EUA x intensity)"], step="mid")
        ax.step(t, price_monthly.values, where="mid", color=PWR, lw=1.4, label="DE-LU day-ahead, monthly baseload avg.")
        ax.set_title(f"{lab} short-run marginal cost vs. power price")
        ax.set_ylabel("EUR / MWh_e")
        ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out); plt.close(fig)


def clean_spreads_monthly(h: pd.DataFrame, out: Path):
    hm = h[["css", "cds"]].copy()
    hm.index = hm.index.tz_localize(None)
    mm = hm.resample("MS").mean()
    pk = h[(h.index.dayofweek < 5) & (h.index.hour >= 8) & (h.index.hour < 20)][["css", "cds"]].copy()
    pk.index = pk.index.tz_localize(None)
    pm = pk.resample("MS").mean()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(mm.index, mm["css"], color=GAS, lw=1.8, label="Clean spark spread, baseload")
    ax.plot(mm.index, mm["cds"], color=COAL, lw=1.8, label="Clean dark spread, baseload")
    ax.plot(pm.index, pm["css"], color=GAS, lw=1, ls="--", label="Clean spark spread, peak (Mon-Fri 08-20)")
    ax.plot(pm.index, pm["cds"], color=COAL, lw=1, ls="--", label="Clean dark spread, peak (Mon-Fri 08-20)")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("EUR / MWh_e")
    ax.set_title("Monthly clean spark and clean dark spreads, Germany (DE-LU day-ahead)")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out); plt.close(fig)


def hour_profile(prof_css: pd.DataFrame, prof_cds: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cmap = plt.get_cmap("viridis")
    prof_css = prof_css.drop(columns=[2022], errors="ignore")
    prof_cds = prof_cds.drop(columns=[2022], errors="ignore")
    years = list(prof_css.columns)
    for ax, prof, title in [(axes[0], prof_css, "Clean spark spread (CCGT)"), (axes[1], prof_cds, "Clean dark spread (hard coal)")]:
        for i, y in enumerate(years):
            ax.plot(prof.index, prof[y], color=cmap(i / max(len(years) - 1, 1)), lw=1.6 if y >= 2023 else 1,
                    label=str(y) + (" (Jan-Sep)" if y == years[-1] else ""))
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("Hour of day (Europe/Berlin, hour starting)")
        ax.set_title(title + " by hour of day")
        ax.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("Mean spread, EUR / MWh_e")
    axes[1].legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    _footer(fig, "2022 (energy crisis, spreads off-scale) omitted; see results/*_hour_of_day.csv.")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out); plt.close(fig)


def gas_share_scatter(gm: pd.DataFrame, rho: float, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    yrs = gm.index.year
    sc = ax.scatter(gm["srmc_gap_coal_minus_gas"], 100 * gm["gas_share_gas_coal"], c=yrs, cmap="viridis", s=28,
                    edgecolor="k", lw=0.3)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("Year")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("Coal SRMC - gas SRMC (EUR/MWh_e)   ->  gas cheaper to the right")
    ax.set_ylabel("Gas share of gas + hard-coal generation (%)")
    ax.set_title(f"Does the merit order follow the spread? Germany, monthly\nSpearman rho = {rho:.2f} (n = {len(gm)} months)")
    _footer(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out); plt.close(fig)


def eua_sweep(sweep: pd.DataFrame, eua_actual: dict, out: Path):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    cmap = plt.get_cmap("viridis")
    years = list(sweep.columns)
    for i, y in enumerate(years):
        c = cmap(i / max(len(years) - 1, 1))
        ax.plot(sweep.index, 100 * sweep[y], color=c, lw=1.6, label=str(y) + (" (Jan-Sep)" if y == years[-1] else ""))
        ea = float(eua_actual[y])
        ax.plot([ea], [100 * np.interp(ea, sweep.index, sweep[y])], "o", color=c, mec="k", ms=5)
    ax.set_xlabel("Flat EUA price assumed for the whole year (EUR/tCO2)")
    ax.set_ylabel("Months with gas SRMC < coal SRMC (%)")
    ax.set_title("How high must the carbon price be for gas to beat coal? (base efficiencies)\n"
                 "dots = actual German average auction price that year")
    ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _footer(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out); plt.close(fig)


def gross_margin(ann: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(ann))
    ax.bar(x - 0.2, ann["gm_gas_eur_per_mw"] / 1000, 0.4, color=GAS, label="CCGT (hours with CSS > 0)")
    ax.bar(x + 0.2, ann["gm_coal_eur_per_mw"] / 1000, 0.4, color=COAL, label="Hard coal (hours with CDS > 0)")
    ax.set_xticks(x, [f"{y}" + ("\n(Jan-Sep)" if y == ann.index[-1] else "") for y in ann.index])
    ax.set_ylabel("k EUR / MW / year")
    ax.set_title("Hourly spread-option gross margin of a perfectly flexible 1 MW unit (no start costs, no constraints)")
    for i, (g, c) in enumerate(zip(ann["gm_gas_eur_per_mw"], ann["gm_coal_eur_per_mw"])):
        ax.text(i - 0.2, g / 1000, f"{g/1000:.0f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + 0.2, c / 1000, f"{c/1000:.0f}", ha="center", va="bottom", fontsize=7)
    ax.legend(frameon=False, fontsize=8)
    _footer(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out); plt.close(fig)
