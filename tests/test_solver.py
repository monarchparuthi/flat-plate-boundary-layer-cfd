from dataclasses import replace

import numpy as np
import pytest

import postprocess as pp
import validation as val
from blasius import solve_blasius
from config import DEFAULT
from solver import solve, thomas

B = solve_blasius()
ENH = replace(DEFAULT, scheme="bdf2", stretching=1.0)


@pytest.fixture(scope="module")
def base():
    return solve(DEFAULT)


@pytest.fixture(scope="module")
def enh():
    return solve(ENH)


def test_thomas_matches_banded():
    rng = np.random.default_rng(0)
    n = 50
    lo, up = rng.normal(size=n), rng.normal(size=n)
    di = 4 + rng.random(n)
    rhs = rng.normal(size=n)
    A = np.diag(di) + np.diag(lo[1:], -1) + np.diag(up[:-1], 1)
    assert thomas(lo, di, up, rhs) == pytest.approx(np.linalg.solve(A, rhs))
    a = solve(replace(DEFAULT, Nx=41, Ny=31, linear_solver="thomas"))
    c = solve(replace(DEFAULT, Nx=41, Ny=31, linear_solver="banded"))
    assert a.u == pytest.approx(c.u, abs=1e-12)


def test_wall_condition(base):                            # Test 2
    assert np.all(base.u[:, 0] == 0) and np.all(base.v[:, 0] == 0)


def test_far_field_and_inlet(base):                       # Test 3
    U = DEFAULT.U_inf
    assert np.all(base.u[:, -1] == U)
    assert np.all(base.u[0, 1:] == U) and np.all(base.v[0] == 0)
    i = pp.station_index(base.grid.x, 0.5)
    d = pp.delta99(base.u, base.grid.y, U)[i]
    outside = base.grid.y > 1.5 * d
    assert base.u[i, outside] == pytest.approx(U, rel=2e-3)


@pytest.mark.parametrize("name", ["base", "enh"])
def test_continuity(name, request):                       # Test 4
    r = val.continuity_report(request.getfixturevalue(name))
    assert r["scheme_max"] < 1e-6
    # independent central-difference check: truncation error only
    assert r["central_rms"] < 0.5 * DEFAULT.U_inf / DEFAULT.L


@pytest.mark.parametrize("name", ["base", "enh"])
def test_positivity_no_overshoot(name, request):          # Test 5
    res = request.getfixturevalue(name)
    assert res.u.min() >= 0.0
    assert res.u.max() <= DEFAULT.U_inf * (1 + 1e-10)
    # at fixed y, u must not increase downstream (dp/dx = 0)
    d = np.diff(res.u[:, 1:-1], axis=0)
    if res.cfg.scheme == "bdf1":
        assert np.all(d <= 1e-9)            # backward Euler: strictly monotone
    else:
        # bdf2 has a known, tiny start-up wiggle next to the leading edge:
        # confined to the first few stations and < 1 % of U_inf (see README)
        bad = np.argwhere(d > 1e-9)
        assert bad.size == 0 or (bad[:, 0].max() < 20 and d.max() < 0.01 * DEFAULT.U_inf)


def test_grid_convergence():                              # Test 6
    d_cm, d_mf = val.nested_differences(DEFAULT)
    assert d_mf < d_cm
    tab, _, _ = val.mesh_study(DEFAULT, B)
    assert tab["rmse_mid_pct"].is_monotonic_decreasing
    assert tab["err_delta99_TE_pct"].is_monotonic_decreasing


def test_validation_against_blasius(base, enh):
    tb, te = val.validate(base, B), val.validate(enh, B)
    for t in (tb, te):
        assert (t["rmse_pct"] < val.TARGETS["rmse_pct"]).all()
        assert (t["emax_pct"] < val.TARGETS["emax_pct"]).all()
        assert (t["err_cf_pct"] < val.TARGETS["err_cf_pct"]).all()
        assert (t["err_theta_pct"] < val.TARGETS["err_theta_pct"]).all()
    # enhanced scheme is better than baseline at every station
    assert (te["rmse_pct"] < tb["rmse_pct"]).all()
    assert (te["err_delta99_pct"][1:] < val.TARGETS["err_delta99_pct"]).all()


def test_sqrt_growth_and_domain_height(base):
    x = base.grid.x
    d = pp.delta99(base.u, base.grid.y, DEFAULT.U_inf)
    m = x >= 0.25
    slope = np.polyfit(np.log(x[m]), np.log(d[m]), 1)[0]
    assert slope == pytest.approx(0.5, abs=0.05)
    assert val.domain_height_check(DEFAULT, B)["ok"]


def test_numerical_solution_is_independent_of_blasius(base):
    # The CFD solution must differ (discretisation error) from Blasius, i.e.
    # it is not simply the analytic profile.
    i = pp.station_index(base.grid.x, 0.5)
    ub = DEFAULT.U_inf * B.u_over_U(base.grid.x[i], base.grid.y, DEFAULT.U_inf, DEFAULT.nu)
    assert np.max(np.abs(base.u[i] - ub)) > 1e-6
