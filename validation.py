"""Quantitative validation of the CFD solution against the Blasius benchmark."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

import postprocess as pp
from blasius import BlasiusSolution
from config import Config
from solver import Result, solve

# Target engineering tolerances from the brief (section 10), in percent of the
# reference value (profile errors: percent of U_inf).
TARGETS = {
    "rmse_pct": 2.0,
    "emax_pct": 5.0,
    "err_delta99_pct": 3.0,
    "err_cf_pct": 5.0,
    "err_delta_star_pct": 3.0,
    "err_theta_pct": 3.0,
}


def profile_errors(u_row, y, x, cfg: Config, b: BlasiusSolution, ymax=None):
    """Normalised RMSE and max-abs error of u/U along one wall-normal line.

    ymax restricts the comparison to y <= ymax (None = full column, as in the
    brief).  Returns percentages of U_inf.
    """
    U, nu = cfg.U_inf, cfg.nu
    sel = slice(None) if ymax is None else (y <= ymax)
    d = (u_row[sel] - U * b.u_over_U(x, y[sel], U, nu)) / U
    return 100.0 * np.sqrt(np.mean(d ** 2)), 100.0 * np.max(np.abs(d))


def validate(res: Result, b: BlasiusSolution, stations=None) -> pd.DataFrame:
    """One row per streamwise station with all validation metrics."""
    cfg, g = res.cfg, res.grid
    U, nu = cfg.U_inf, cfg.nu
    stations = cfg.profile_stations if stations is None else stations
    cur = pp.summary_curves(res)
    rows = []
    for s in stations:
        i = pp.station_index(g.x, s * cfg.L)
        x = g.x[i]
        db99 = float(b.delta99(x, U, nu))
        rmse, emax = profile_errors(res.u[i], g.y, x, cfg, b)
        rmse_bl, emax_bl = profile_errors(res.u[i], g.y, x, cfg, b, ymax=2 * db99)
        ref = {"delta99": db99, "cf": float(b.cf(x, U, nu)),
               "delta_star": float(b.delta_star(x, U, nu)),
               "theta": float(b.theta(x, U, nu))}
        row = {"x_over_L": x / cfg.L, "Re_x": U * x / nu,
               "rmse_pct": rmse, "emax_pct": emax,
               "rmse_bl_pct": rmse_bl, "emax_bl_pct": emax_bl}
        for k, rv in ref.items():
            cv = float(cur[k][i])
            row[f"{k}_cfd"] = cv
            row[f"{k}_blasius"] = rv
            row[f"err_{k}_pct"] = 100.0 * abs(cv - rv) / rv
        rows.append(row)
    return pd.DataFrame(rows)


def check_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Pass/fail of the brief's tolerances at each station (True = met)."""
    out = pd.DataFrame({"x_over_L": df["x_over_L"]})
    for k, tol in TARGETS.items():
        out[k] = df[k] < tol
    return out


def error_curves(res: Result, b: BlasiusSolution, x_min_over_L=0.0):
    """Relative errors (percent) of the integral/wall quantities along x."""
    cfg, g = res.cfg, res.grid
    U, nu = cfg.U_inf, cfg.nu
    cur = pp.summary_curves(res)
    x = g.x
    m = x >= x_min_over_L * cfg.L
    m[0] = False
    refs = {"delta99": b.delta99(x, U, nu), "delta_star": b.delta_star(x, U, nu),
            "theta": b.theta(x, U, nu), "cf": b.cf(x[m], U, nu)}
    out = {"x": x[m]}
    for k in ("delta99", "delta_star", "theta"):
        out[k] = 100 * (cur[k][m] - refs[k][m]) / refs[k][m]
    out["cf"] = 100 * (cur["cf"][m] - refs["cf"]) / refs["cf"]
    return out


# ---------------------------------------------------------------------------
# Domain-height check and discrete-conservation diagnostics
# ---------------------------------------------------------------------------
def domain_height_check(cfg: Config, b: BlasiusSolution) -> dict:
    d_te = float(b.delta99(cfg.L, cfg.U_inf, cfg.nu))
    return {"delta99_trailing_edge_m": d_te, "H_over_delta99": cfg.H / d_te,
            "ok": cfg.H / d_te > 5.0}


def continuity_report(res: Result) -> dict:
    r = pp.continuity_residual(res)
    out = {}
    for k, a in r.items():
        out[f"{k}_max"] = float(np.max(np.abs(a)))
        out[f"{k}_rms"] = float(np.sqrt(np.mean(a ** 2)))
    return out


# ---------------------------------------------------------------------------
# Mesh independence / Grid Convergence Index
# ---------------------------------------------------------------------------
MESHES = {"coarse": (101, 51), "medium": (201, 101), "fine": (401, 201)}


def gci(f_c, f_m, f_f, r=2.0):
    """Richardson/GCI (Roache) for three solutions on grids refined by r.

    Returns dict with observed order p, extrapolated value and GCI of the fine
    grid in percent.  If the sequence is not monotonically convergent
    (oscillatory or diverging) p / GCI are NaN and ``monotone`` is False.
    """
    e21, e32 = f_m - f_c, f_f - f_m
    out = {"monotone": False, "order": np.nan, "order_plausible": False,
           "f_extrapolated": np.nan, "gci_fine_pct": np.nan}
    if e21 == 0 or e32 == 0:
        return out
    R = e32 / e21
    if not (0.0 < R < 1.0):
        return out
    p = np.log(1.0 / R) / np.log(r)
    # an observed order far outside the formal range [~0.5, ~3] means the grids
    # are not in the asymptotic range; the GCI is then not trustworthy
    out.update(monotone=True, order=float(p), order_plausible=bool(0.5 <= p <= 3.0),
               f_extrapolated=float(f_f + e32 / (r ** p - 1.0)),
               gci_fine_pct=float(1.25 * abs(e32 / f_f) / (r ** p - 1.0) * 100.0))
    return out


def mesh_study(cfg: Config, b: BlasiusSolution, meshes=MESHES):
    """Run the coarse/medium/fine meshes.

    Returns (table, profiles, gci_table):
      table      - per-mesh scalars at x/L = 0.5 and at the trailing edge
      profiles   - dict mesh -> (y, u) at x/L = 0.5
      gci_table  - Richardson/GCI analysis of each scalar
    """
    U, nu = cfg.U_inf, cfg.nu
    rows, profiles = [], {}
    for name, (nx, ny) in meshes.items():
        res = solve(replace(cfg, Nx=nx, Ny=ny))
        g = res.grid
        cur = pp.summary_curves(res)
        i_mid = pp.station_index(g.x, 0.5 * cfg.L)
        te = -1
        rmse, emax = profile_errors(res.u[i_mid], g.y, g.x[i_mid], res.cfg, b)
        row = {"mesh": name, "Nx": nx, "Ny": ny, "dx_m": g.dx,
               "dy_m": float(g.dy[0]),
               "rmse_mid_pct": rmse, "emax_mid_pct": emax,
               "delta99_mid_m": cur["delta99"][i_mid],
               "delta99_TE_m": cur["delta99"][te], "cf_TE": cur["cf"][te],
               "delta_star_TE_m": cur["delta_star"][te],
               "theta_TE_m": cur["theta"][te], "wall_time_s": res.wall_time}
        row["err_delta99_TE_pct"] = 100 * abs(row["delta99_TE_m"] / b.delta99(cfg.L, U, nu) - 1)
        row["err_cf_TE_pct"] = 100 * abs(row["cf_TE"] / b.cf(cfg.L, U, nu) - 1)
        row["err_delta_star_TE_pct"] = 100 * abs(row["delta_star_TE_m"] / b.delta_star(cfg.L, U, nu) - 1)
        rows.append(row)
        profiles[name] = (g.y.copy(), res.u[i_mid].copy())
    table = pd.DataFrame(rows)
    gci_rows = []
    for col in ("rmse_mid_pct", "delta99_TE_m", "cf_TE", "delta_star_TE_m"):
        d = {"quantity": col}
        d.update(gci(*table[col].iloc[:3].to_numpy()))
        gci_rows.append(d)
    return table, profiles, pd.DataFrame(gci_rows)


def nested_differences(cfg: Config, meshes=MESHES):
    """Test 6 helper: solution-to-solution differences on the nodes shared by
    the nested coarse/medium/fine meshes.  Returns (|u_m-u_c|, |u_f-u_m|) as
    max-norm of u/U over all shared nodes; the second must be smaller."""
    sols = {}
    for name, (nx, ny) in meshes.items():
        sols[name] = solve(replace(cfg, Nx=nx, Ny=ny)).u / cfg.U_inf
    (nxc, nyc), (nxm, nym), (nxf, nyf) = meshes["coarse"], meshes["medium"], meshes["fine"]
    sx_m, sy_m = (nxm - 1) // (nxc - 1), (nym - 1) // (nyc - 1)
    sx_f, sy_f = (nxf - 1) // (nxc - 1), (nyf - 1) // (nyc - 1)
    c = sols["coarse"]
    m = sols["medium"][::sx_m, ::sy_m]
    f = sols["fine"][::sx_f, ::sy_f]
    return float(np.max(np.abs(m - c))), float(np.max(np.abs(f - m)))
