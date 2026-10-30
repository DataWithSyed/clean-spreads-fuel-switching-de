"""Statistical model of the monthly gas share of (gas + hard-coal) generation vs. the modelled SRMC gap.

Two questions, kept separate:

1. **Explanation** (full sample, in-sample): how many percentage points of gas share does a cheaper-gas month buy,
   once seasonality and slow structural change are controlled for? OLS with Newey-West (HAC) standard errors,
   because monthly residuals are autocorrelated.
2. **Forecast / nowcast** (out-of-sample): given this month's fuel and carbon prices (hence the SRMC gap, which is
   known once the month's prices are), does the gap improve on naive forecasts of the gas share? Expanding window,
   refitted every month on past data only; the tanh scale is re-selected inside every training window.

The SRMC gap enters as ``tanh(gap / s)``: switching saturates (in 2022 the gap reached -324 EUR/MWh, but the gas share
cannot fall below what must-run CHP and system needs keep online), so a linear term would be dominated by the crisis months.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

S_GRID = (10.0, 20.0, 30.0, 50.0, 100.0)   # EUR/MWh_e, candidate tanh scales


def hac_ols(X: np.ndarray, y: np.ndarray, lags: int = 3) -> dict:
    """OLS with Newey-West (Bartlett kernel) covariance. ``lags=0`` gives White (HC0) standard errors."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    xe = X * e[:, None]
    s = xe.T @ xe
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        g = xe[lag:].T @ xe[:-lag]
        s += w * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    return {"beta": beta, "se": np.sqrt(np.diag(cov)), "resid": e, "r2": 1.0 - e.var() / y.var(), "n": len(y)}


def _frame(gm: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"share": 100.0 * gm["gas_share_gas_coal"].to_numpy(),
                      "gap": gm["srmc_gap_coal_minus_gas"].to_numpy(),
                      "month": np.asarray(gm.index.month), "year": np.asarray(gm.index.year)}, index=gm.index)
    d["t"] = np.arange(len(d)) / 12.0
    d["share_lag"] = d["share"].shift(1)
    return d


def design(d: pd.DataFrame, spec: str, s: float) -> tuple[np.ndarray, list[str]]:
    """Regressor matrix for a named specification (``spec`` in SPECS)."""
    cols, names = [np.ones(len(d))], ["const"]
    if spec != "ar1":
        cols.append(np.tanh(d["gap"].to_numpy() / s)); names.append("tanh_gap")
    if spec in ("arx", "ar1"):
        cols.append(d["share_lag"].to_numpy()); names.append("share_lag")
    if spec in ("static_seasonal", "within_year"):
        for k in range(2, 13):
            cols.append((d["month"].to_numpy() == k).astype(float)); names.append(f"m{k}")
    if spec in ("static", "static_seasonal", "arx"):
        cols.append(d["t"].to_numpy()); names.append("trend")
    if spec == "within_year":
        for yv in sorted(d["year"].unique())[1:]:
            cols.append((d["year"].to_numpy() == yv).astype(float)); names.append(f"y{yv}")
    return np.column_stack(cols), names


SPECS = {
    "static": "gap + linear trend",
    "static_seasonal": "gap + month-of-year + trend",
    "within_year": "gap + month-of-year + year fixed effects (explanatory only)",
    "arx": "last month's share + gap + trend",
    "ar1": "last month's share only (no gap)",
}


def best_scale(d: pd.DataFrame, spec: str, grid=S_GRID) -> float:
    """tanh scale with the highest in-sample R^2 on ``d`` (rows with NaNs dropped)."""
    best, best_r2 = grid[0], -np.inf
    for s in grid:
        X, _ = design(d, spec, s)
        ok = ~np.isnan(X).any(axis=1)
        r2 = hac_ols(X[ok], d["share"].to_numpy()[ok], 0)["r2"]
        if r2 > best_r2:
            best, best_r2 = s, r2
    return best


def explanatory_table(gm: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """Full-sample fits. ``pp_per_eur_at_0`` = marginal effect d(share)/d(gap) at gap = 0 [pp per EUR/MWh];
    ``swing_pp_-20_+20`` = modelled share change when the gap moves from -20 to +20 EUR/MWh."""
    d = _frame(gm)
    rows = []
    for spec in ("static", "static_seasonal", "within_year", "arx"):
        s = best_scale(d, spec)
        X, names = design(d, spec, s)
        ok = ~np.isnan(X).any(axis=1)
        f = hac_ols(X[ok], d["share"].to_numpy()[ok], lags)
        b, se = f["beta"][names.index("tanh_gap")], f["se"][names.index("tanh_gap")]
        rows.append({"spec": spec, "description": SPECS[spec], "tanh_scale_eur_mwh": s, "n": f["n"],
                     "coef_tanh_gap_pp": b, "hac_se": se, "t_stat": b / se, "r2": f["r2"],
                     "pp_per_eur_at_0": b / s, "swing_pp_-20_+20": b * (np.tanh(20 / s) - np.tanh(-20 / s)),
                     "coef_share_lag": f["beta"][names.index("share_lag")] if "share_lag" in names else np.nan})
    return pd.DataFrame(rows)


def expanding_forecast(gm: pd.DataFrame, start: int = 24, specs=("static", "static_seasonal", "arx", "ar1")) -> pd.DataFrame:
    """One-step-ahead predictions for months ``start``..end, each from a model fitted on earlier months only
    (tanh scale re-selected on the training window). Also the persistence (last month) and same-calendar-month
    mean baselines. Returns one row per predicted month."""
    d = _frame(gm)
    y = d["share"].to_numpy()
    out = {"month": d.index[start:].astype(str), "actual": y[start:],
           "persistence": y[start - 1:-1],
           "calendar_mean": [y[:k][d["month"].to_numpy()[:k] == d["month"].iloc[k]].mean() for k in range(start, len(d))]}
    for spec in specs:
        pred = []
        for k in range(start, len(d)):
            tr = d.iloc[:k]
            s = best_scale(tr, spec) if spec != "ar1" else 1.0
            X, _ = design(tr, spec, s)
            ok = ~np.isnan(X).any(axis=1)
            beta = hac_ols(X[ok], tr["share"].to_numpy()[ok], 0)["beta"]
            xk, _ = design(d.iloc[: k + 1], spec, s)
            pred.append(float(xk[-1] @ beta))
        out[spec] = pred
    return pd.DataFrame(out)


def forecast_scores(fc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in fc.columns.drop(["month", "actual"]):
        e = fc[c].to_numpy() - fc["actual"].to_numpy()
        rows.append({"model": c, "rmse_pp": float(np.sqrt(np.mean(e ** 2))), "mae_pp": float(np.mean(np.abs(e))),
                     "bias_pp": float(e.mean())})
    s = pd.DataFrame(rows).sort_values("rmse_pp").reset_index(drop=True)
    s["rmse_vs_persistence"] = s["rmse_pp"] / float(s.loc[s.model == "persistence", "rmse_pp"].iloc[0]) - 1.0
    return s


def leave_one_year_out(gm: pd.DataFrame, spec: str = "static") -> pd.DataFrame:
    """Predict every month of year Y from a model fitted on all other years (no lagged terms allowed)."""
    if spec in ("arx", "ar1", "within_year"):
        raise ValueError("leave-one-year-out needs a spec without lags or year effects")
    d = _frame(gm)
    rows = []
    for yv in sorted(d["year"].unique()):
        tr, te = d[d["year"] != yv], d[d["year"] == yv]
        s = best_scale(tr, spec)
        beta = hac_ols(design(tr, spec, s)[0], tr["share"].to_numpy(), 0)["beta"]
        # the trend regressor must use the global time index, so build the test design on the full frame
        X_all, _ = design(d, spec, s)
        p = X_all[(d["year"] == yv).to_numpy()] @ beta
        e = p - te["share"].to_numpy()
        rows.append({"year": int(yv), "n": len(te), "rmse_pp": float(np.sqrt(np.mean(e ** 2))), "bias_pp": float(e.mean()),
                     "tanh_scale": s})
    return pd.DataFrame(rows)
