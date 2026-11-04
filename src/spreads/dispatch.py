"""Dispatch-constrained value of a CCGT and a hard-coal unit on hourly DE-LU day-ahead prices.

The "flexible gross margin" in `model.annual_summary` sums every positive hourly clean spread, as if the
unit could switch on and off each hour for free. Real thermal units cannot: they have a minimum stable
generation, a worse efficiency at part load, minimum up/down times and a cost per start (wear and tear
plus start-up fuel and its CO2). This module values a 1 MW unit with those constraints by exact dynamic
programming over its commitment state, so the gap between the free-switching bound and a technically
feasible schedule can be measured.

Two information settings are compared:
  * ``horizon="year"``  perfect foresight over the whole year (intrinsic value of the DA price path);
  * ``horizon="day"``   the unit is committed one delivery day at a time, seeing only that day's 24 DA
                        prices (as in the day-ahead auction) and carrying its on/off state and up/down
                        counters over midnight, but placing no value on what happens after midnight.

Everything is per MW of capacity, net output, prices in EUR/MWh_e. Plant parameters are illustrative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Unit:
    name: str
    eta_full: float            # net efficiency at full load
    eta_min: float             # net efficiency at minimum stable generation
    p_min: float               # minimum stable generation, share of capacity
    min_up: int                # hours
    min_down: int              # hours
    start_cm_eur_per_mw: float  # wear-and-tear (capital & maintenance) cost per start
    start_fuel_mwh_th_per_mw: float  # start-up fuel burned per start (also emits CO2)
    ef_t_per_mwh_th: float     # fuel emission factor
    vom: float                 # EUR/MWh_e

    def __post_init__(self):
        if not (0 < self.p_min < 1 and 0 < self.eta_min <= self.eta_full < 1):
            raise ValueError("need 0 < p_min < 1 and 0 < eta_min <= eta_full < 1")
        if self.min_up < 1 or self.min_down < 1:
            raise ValueError("min up/down times must be >= 1 h")
        if self.incremental_heat_rate <= 0:
            raise ValueError("part-load curve implies non-positive incremental heat rate")

    @property
    def heat_at_min(self) -> float:      # MWh_th per hour at p_min (per MW capacity)
        return self.p_min / self.eta_min

    @property
    def incremental_heat_rate(self) -> float:  # MWh_th per extra MWh_e between p_min and full load
        return (1.0 / self.eta_full - self.heat_at_min) / (1.0 - self.p_min)


def unit_from_cfg(cfg_plant: dict, cfg_dispatch: dict, name: str) -> Unit:
    d = cfg_dispatch
    return Unit(name=name, eta_full=cfg_plant["efficiency"],
                eta_min=cfg_plant["efficiency"] - d["part_load_efficiency_drop"],
                p_min=d["min_stable_generation"], min_up=int(d["min_up_h"]), min_down=int(d["min_down_h"]),
                start_cm_eur_per_mw=d["start_cost_cm_eur_per_mw"],
                start_fuel_mwh_th_per_mw=d["start_fuel_mwh_th_per_mw"],
                ef_t_per_mwh_th=cfg_plant["emission_factor_t_per_mwh_th"], vom=cfg_plant["vom_eur_per_mwh_e"])


def hourly_economics(unit: Unit, price, fuel, eua):
    """Per-hour margin if the unit is on (optimal output between p_min and 1) and the cost of a start.

    fuel = delivered fuel price EUR/MWh_th, eua = EUR/t. Returns (on_margin, output_if_on, start_cost).
    """
    price, fuel, eua = (np.asarray(x, dtype=float) for x in (price, fuel, eua))
    fuel_all_in = fuel + unit.ef_t_per_mwh_th * eua                   # EUR/MWh_th incl. carbon
    cost_min = unit.heat_at_min * fuel_all_in + unit.p_min * unit.vom  # EUR/h at p_min
    inc_cost = unit.incremental_heat_rate * fuel_all_in + unit.vom      # EUR per extra MWh_e
    upper = np.maximum(price - inc_cost, 0.0) * (1.0 - unit.p_min)
    on_margin = unit.p_min * price - cost_min + upper
    output = unit.p_min + (1.0 - unit.p_min) * (price > inc_cost)
    start_cost = unit.start_cm_eur_per_mw + unit.start_fuel_mwh_th_per_mw * fuel_all_in
    return on_margin, output, start_cost


def _solve(on_margin, start_cost, U, D, init_state):
    """Exact DP over commitment states. State index: 0..U-1 = on for k+1 h (U-1 means >= U);
    U..U+D-1 = off for k+1 h (last means >= D). Returns on/off schedule, starts, final state."""
    T, S = len(on_margin), U + D
    nxt = np.empty((T, S), dtype=np.int8)
    V = np.zeros(S)
    on_idx = np.arange(U)
    for t in range(T - 1, -1, -1):
        # value of *entering* hour t in each state, given V = value of entering t+1
        Vn = np.empty(S)
        choice = np.empty(S, dtype=np.int8)
        r = on_margin[t]
        # on states: get the hour's margin, then continue on or (if min up met) switch off
        for k in range(U):
            stay = min(k + 1, U - 1)
            best, ch = r + V[stay], stay
            if k == U - 1 and r + V[U] > best:   # after this hour go to off_1
                best, ch = r + V[U], U
            Vn[k], choice[k] = best, ch
        # off states: earn nothing; continue off or (if min down met) start for the next hour
        for k in range(D):
            stay = U + min(k + 1, D - 1)
            best, ch = V[stay], stay
            if k == D - 1:
                start_val = V[0] - (start_cost[t + 1] if t + 1 < T else 0.0)
                if start_val > best:
                    best, ch = start_val, 0
            Vn[U + k], choice[U + k] = best, ch
        V, nxt[t] = Vn, choice
    # forward pass. Entering hour 0 from init_state; a start in hour 0 is allowed from off_D (pays start_cost[0]).
    s = init_state
    if s == U + D - 1 and V[0] - start_cost[0] > V[s]:
        s, starts = 0, 1
    else:
        starts = 0
    on = np.zeros(T, dtype=bool)
    for t in range(T):
        on[t] = s < U
        s2 = int(nxt[t, s])
        if t + 1 < T and s >= U and s2 == 0:
            starts += 1
        s = s2
    return on, starts, s


def dispatch(unit: Unit, price, fuel, eua, horizon: str = "year", day_labels=None):
    """Value one unit over a price path. Returns dict with schedule and P&L totals (EUR/MW).

    For ``horizon="day"`` the path is split into delivery days by ``day_labels`` (one label per hour, e.g. the
    local calendar date, so 23/25-hour DST days are handled); without labels, fixed 24-hour blocks are used.
    """
    on_margin, output, start_cost = hourly_economics(unit, price, fuel, eua)
    T = len(on_margin)
    U, D = unit.min_up, unit.min_down
    state = U + D - 1                       # start "off and available"
    if horizon == "year":
        blocks = [(0, T)]
    elif horizon == "day":
        if day_labels is None:
            cuts = list(range(0, T, 24))
        else:
            lab = np.asarray(day_labels)
            if len(lab) != T:
                raise ValueError("day_labels must have one entry per hour")
            cuts = [0] + [i for i in range(1, T) if lab[i] != lab[i - 1]]
        blocks = list(zip(cuts, cuts[1:] + [T]))
    else:
        raise ValueError("horizon must be 'year' or 'day'")
    on = np.zeros(T, dtype=bool)
    starts = 0
    for a, b in blocks:
        o, n, state = _solve(on_margin[a:b], start_cost[a:b], U, D, state)
        on[a:b] = o
        starts += n
    # recompute start costs from the realised schedule (exact, independent of DP bookkeeping)
    start_flags = on & ~np.r_[False, on[:-1]]
    margin = float(on_margin[on].sum())
    start_total = float(start_cost[start_flags].sum())
    return {"on": on, "output": np.where(on, output, 0.0), "starts": int(start_flags.sum()),
            "run_hours": int(on.sum()), "generation_mwh": float(np.where(on, output, 0.0).sum()),
            "energy_margin": margin, "start_costs": start_total, "gross_margin": margin - start_total}


def free_switching_bound(unit: Unit, price, fuel, eua) -> float:
    """Free-switching full-load bound with the same plant (no p_min, no start cost): sum of positive spreads."""
    price, fuel, eua = (np.asarray(x, dtype=float) for x in (price, fuel, eua))
    srmc = (fuel + unit.ef_t_per_mwh_th * eua) / unit.eta_full + unit.vom
    return float(np.clip(price - srmc, 0, None).sum())


def annual_dispatch_table(h: pd.DataFrame, monthly: pd.DataFrame, units: dict[str, tuple[Unit, str]]) -> pd.DataFrame:
    """For each year and unit: free-switching bound vs constrained perfect-foresight vs day-by-day commitment."""
    per = h.index.tz_localize(None).to_period("M")
    rows = []
    for name, (unit, fuel_col) in units.items():
        fuel_h = monthly[fuel_col].reindex(per).to_numpy()
        eua_h = monthly["eua"].reindex(per).to_numpy()
        for y in sorted(set(h.index.year)):
            m = h.index.year == y
            p, f, e = h["price"].to_numpy()[m], fuel_h[m], eua_h[m]
            bound = free_switching_bound(unit, p, f, e)
            pf = dispatch(unit, p, f, e, "year")
            da = dispatch(unit, p, f, e, "day", day_labels=np.asarray(h.index.date)[m])
            rows.append({"unit": name, "year": y, "hours": int(m.sum()), "free_switching_bound": bound,
                         "pf_gross_margin": pf["gross_margin"], "pf_energy_margin": pf["energy_margin"],
                         "pf_start_costs": pf["start_costs"], "pf_starts": pf["starts"], "pf_run_hours": pf["run_hours"],
                         "pf_capacity_factor": pf["generation_mwh"] / m.sum(),
                         "day_gross_margin": da["gross_margin"], "day_starts": da["starts"],
                         "day_run_hours": da["run_hours"],
                         "pf_capture": pf["gross_margin"] / bound if bound > 0 else np.nan,
                         "day_capture": da["gross_margin"] / bound if bound > 0 else np.nan})
    return pd.DataFrame(rows)


def start_cost_sensitivity(h, monthly, unit: Unit, fuel_col: str, factors=(0.0, 0.5, 1.0, 2.0, 4.0)) -> pd.DataFrame:
    """Perfect-foresight gross margin and starts per year when the start C&M cost is scaled."""
    from dataclasses import replace
    per = h.index.tz_localize(None).to_period("M")
    fuel_h = monthly[fuel_col].reindex(per).to_numpy()
    eua_h = monthly["eua"].reindex(per).to_numpy()
    rows = []
    for fct in factors:
        u = replace(unit, start_cm_eur_per_mw=unit.start_cm_eur_per_mw * fct)
        for y in sorted(set(h.index.year)):
            m = h.index.year == y
            r = dispatch(u, h["price"].to_numpy()[m], fuel_h[m], eua_h[m], "year")
            rows.append({"unit": unit.name, "start_cost_factor": fct, "year": y,
                         "gross_margin": r["gross_margin"], "starts": r["starts"], "run_hours": r["run_hours"]})
    return pd.DataFrame(rows)
