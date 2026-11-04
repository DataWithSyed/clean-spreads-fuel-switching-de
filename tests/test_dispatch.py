import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spreads import dispatch  # noqa: E402

GAS = dispatch.Unit("CCGT", eta_full=0.5, eta_min=0.44, p_min=0.45, min_up=3, min_down=2,
                    start_cm_eur_per_mw=55, start_fuel_mwh_th_per_mw=1.5, ef_t_per_mwh_th=0.20196, vom=2.5)


def _feasible(on, U, D):
    """Every completed on-run lasts >= U h and every off-run between two on-runs lasts >= D h.
    The unit starts off and available; a run cut off by the end of the horizon may be shorter."""
    runs = [(k, len(list(g))) for k, g in itertools.groupby(on)]
    for i, (k, n) in enumerate(runs):
        last = i == len(runs) - 1
        if k and n < U and not last:
            return False
        if (not k) and 0 < i and not last and n < D:
            return False
    return True


def _brute(unit, price, fuel, eua):
    om, _, sc = dispatch.hourly_economics(unit, price, fuel, eua)
    best = 0.0
    for on in itertools.product([False, True], repeat=len(price)):
        on = np.array(on)
        if not _feasible(on, unit.min_up, unit.min_down):
            continue
        starts = on & ~np.r_[False, on[:-1]]
        best = max(best, om[on].sum() - sc[starts].sum())
    return best


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_dp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    T = 12
    price = rng.normal(95, 40, T)
    fuel, eua = np.full(T, 35.0), np.full(T, 70.0)
    r = dispatch.dispatch(GAS, price, fuel, eua, "year")
    assert r["gross_margin"] == pytest.approx(_brute(GAS, price, fuel, eua), abs=1e-6)
    assert _feasible(r["on"], GAS.min_up, GAS.min_down)


def test_constrained_below_free_switching_bound_and_day_below_year():
    rng = np.random.default_rng(7)
    T = 24 * 20
    hours = np.arange(T) % 24
    price = 90 + 45 * np.sin((hours - 9) / 24 * 2 * np.pi) + rng.normal(0, 25, T)
    fuel, eua = np.full(T, 35.0), np.full(T, 70.0)
    bound = dispatch.free_switching_bound(GAS, price, fuel, eua)
    pf = dispatch.dispatch(GAS, price, fuel, eua, "year")
    day = dispatch.dispatch(GAS, price, fuel, eua, "day")
    assert 0 < day["gross_margin"] <= pf["gross_margin"] + 1e-9 <= bound + 1e-9
    assert _feasible(pf["on"], GAS.min_up, GAS.min_down) and _feasible(day["on"], GAS.min_up, GAS.min_down)


def test_part_load_curve_and_on_margin():
    # heat input at full load must equal 1/eta_full
    assert GAS.heat_at_min + GAS.incremental_heat_rate * (1 - GAS.p_min) == pytest.approx(1 / GAS.eta_full)
    fuel_all_in = 35 + 0.20196 * 70
    srmc_full = fuel_all_in / 0.5 + 2.5
    om, out, _ = dispatch.hourly_economics(GAS, [srmc_full + 30], [35.0], [70.0])
    assert out[0] == 1.0 and om[0] == pytest.approx(30.0)      # full load earns exactly the clean spread
    om, out, _ = dispatch.hourly_economics(GAS, [0.0], [35.0], [70.0])
    assert out[0] == GAS.p_min and om[0] < 0                   # at a zero price it sits at min load and loses money


def test_free_start_and_no_minimums_recovers_bound():
    u = dispatch.Unit("flex", 0.5, 0.5, 1e-9, 1, 1, 0.0, 0.0, 0.20196, 2.5)
    rng = np.random.default_rng(3)
    price = rng.normal(100, 50, 200)
    fuel, eua = np.full(200, 35.0), np.full(200, 70.0)
    assert dispatch.dispatch(u, price, fuel, eua)["gross_margin"] == pytest.approx(
        dispatch.free_switching_bound(u, price, fuel, eua), rel=1e-6)


def test_day_blocks_follow_labels_and_never_bind_start_state():
    # 23-hour DST day followed by a 25-hour day: labels, not fixed 24-h blocks, set the commitment days
    labels = np.array([0] * 23 + [1] * 25)
    price = np.full(48, 300.0)                                 # always profitable: stays on from hour 0
    r = dispatch.dispatch(GAS, price, np.full(48, 35.0), np.full(48, 70.0), "day", day_labels=labels)
    assert r["on"].all() and r["starts"] == 1
    with pytest.raises(ValueError):
        dispatch.dispatch(GAS, price, np.full(48, 35.0), np.full(48, 70.0), "day", day_labels=labels[:-1])


def test_invalid_unit_rejected():
    with pytest.raises(ValueError):
        dispatch.Unit("bad", 0.5, 0.6, 0.4, 4, 4, 50, 1, 0.2, 2.5)
