import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spreads import model  # noqa: E402

CFG = {
    "gas_plant": {"efficiency": 0.5, "efficiency_low": 0.45, "efficiency_high": 0.58,
                  "emission_factor_t_per_mwh_th": 0.202, "vom_eur_per_mwh_e": 2.5, "hub_to_plant_eur_per_mwh_th": 0.0},
    "coal_plant": {"efficiency": 0.38, "efficiency_low": 0.35, "efficiency_high": 0.45,
                   "emission_factor_t_per_mwh_th": 0.341, "calorific_kcal_per_kg": 6000,
                   "freight_usd_per_t": 12.0, "vom_eur_per_mwh_e": 4.0},
    "eua_eur_per_t": {2024: 65.0},
}


def _monthly(gas=35.0, coal=15.0):
    idx = pd.period_range("2024-01", "2024-03", freq="M")
    fe = pd.DataFrame({"gas_eur_mwh_th": gas, "coal_eur_mwh_th": coal, "usd_per_eur": 1.1}, index=idx)
    return model.monthly_table(fe, CFG)


def test_coal_energy_content():
    # 6000 kcal/kg = 25.12 GJ/t = 6.978 MWh/t
    assert model.coal_mwh_th_per_t(6000) == pytest.approx(6.978, abs=1e-3)


def test_usd_to_eur_conversion():
    fu = pd.DataFrame({"gas_usd_mmbtu": [10.0], "coal_usd_t": [100.0]}, index=pd.PeriodIndex(["2024-01"], freq="M"))
    fx = pd.Series([1.25], index=fu.index, name="usd_per_eur")
    out = model.fuel_prices_eur(fu, fx, CFG)
    assert out["gas_eur_mwh_th"].iloc[0] == pytest.approx(10 * 3.412142 / 1.25)
    assert out["coal_eur_mwh_th"].iloc[0] == pytest.approx(112 / 1.25 / model.coal_mwh_th_per_t(6000))


def test_srmc_equal_at_switching_price():
    m = _monthly()
    eua_star = m["switch_price"].iloc[0]
    sg = m["fuel_gas"].iloc[0] + eua_star * m["ci_gas"].iloc[0]
    sc = m["fuel_coal"].iloc[0] + eua_star * m["ci_coal"].iloc[0]
    assert sg == pytest.approx(sc)


def test_gas_cheaper_iff_eua_above_switch():
    m = _monthly()
    assert bool(m["gas_cheaper"].iloc[0]) == bool(m["eua"].iloc[0] > m["switch_price"].iloc[0])


def test_switch_band_ordering():
    m = _monthly()
    assert (m["switch_low"] <= m["switch_price"]).all() and (m["switch_price"] <= m["switch_high"]).all()


def test_switching_price_nan_when_gas_not_cleaner():
    # absurdly inefficient gas unit -> gas more carbon-intensive than coal -> undefined
    v = model.switching_price(30.0, 15.0, eta_g=0.2, eta_c=0.4, ef_g=0.202, ef_c=0.341)
    assert np.isnan(v)


def test_hourly_spreads_and_sweep():
    m = _monthly()
    idx = pd.date_range("2024-01-01", "2024-03-31 23:00", freq="1h", tz="Europe/Berlin")
    price = pd.Series(100.0, index=idx, name="price")
    h = model.hourly_spreads(price, m)
    assert (h["css"] == 100 - m["srmc_gas"].iloc[0]).all()
    sweep = model.eua_sweep(m, np.array([0.0, m["switch_price"].iloc[0] + 1, 500.0]))
    assert sweep[2024].tolist() == [0.0, 1.0, 1.0]


# ---- gas-share regression / forecast ------------------------------------------------------------
from spreads import regression  # noqa: E402


def _synthetic_gm(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2019-01", periods=n, freq="M")
    gap = rng.normal(0, 25, n)
    share = 0.55 + 0.10 * np.tanh(gap / 30) + rng.normal(0, 0.01, n)
    return pd.DataFrame({"gas_share_gas_coal": share, "srmc_gap_coal_minus_gas": gap}, index=idx)


def test_hac_ols_recovers_coefficients_and_hc0():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(500), rng.normal(size=500)])
    y = 2.0 + 3.0 * X[:, 1] + rng.normal(0, 0.1, 500)
    f = regression.hac_ols(X, y, lags=0)
    assert np.allclose(f["beta"], [2.0, 3.0], atol=0.02)
    e = f["resid"]; xtx_inv = np.linalg.inv(X.T @ X)
    hc0 = xtx_inv @ (X * e[:, None]).T @ (X * e[:, None]) @ xtx_inv
    assert np.allclose(f["se"], np.sqrt(np.diag(hc0)))
    assert regression.hac_ols(X, y, lags=4)["se"][1] > 0


def test_explanatory_fit_finds_synthetic_effect():
    tab = regression.explanatory_table(_synthetic_gm()).set_index("spec")
    assert tab.loc["static", "tanh_scale_eur_mwh"] == 30.0
    assert tab.loc["static", "coef_tanh_gap_pp"] == pytest.approx(10.0, abs=1.0)
    assert tab.loc["static", "t_stat"] > 10


def test_expanding_forecast_uses_only_past_data():
    gm = _synthetic_gm()
    fc = regression.expanding_forecast(gm, start=24)
    gm2 = gm.copy(); gm2.iloc[40:, 0] += 0.3          # change the future only
    fc2 = regression.expanding_forecast(gm2, start=24)
    k = 40 - 24
    cols = ["persistence", "static", "static_seasonal", "arx", "ar1", "calendar_mean"]
    assert np.allclose(fc.loc[:k - 1, cols], fc2.loc[:k - 1, cols])   # predictions for months < 40 unchanged
    assert np.allclose(fc["persistence"].iloc[1:].to_numpy(), fc["actual"].iloc[:-1].to_numpy())
    s = regression.forecast_scores(fc)
    assert s.loc[s.model == "persistence", "rmse_vs_persistence"].iloc[0] == 0.0


def test_leave_one_year_out_rejects_lagged_specs():
    with pytest.raises(ValueError):
        regression.leave_one_year_out(_synthetic_gm(), "arx")
    assert len(regression.leave_one_year_out(_synthetic_gm(), "static")) == 5
