"""Additional numerical investigations (brief, section 12)."""
from __future__ import annotations

import tracemalloc
from dataclasses import replace

import numpy as np
import pandas as pd

import postprocess as pp
import validation as val
from blasius import BlasiusSolution
from config import Config
from solver import solve


def _mean_max(curve):
    a = np.abs(curve)
    return float(np.mean(a)), float(np.max(a))


def _error_summary(res, b, x_min=0.25):
    """Mean |error| (%) over x/L >= x_min for the main scalar outputs."""
    c = val.error_curves(res, b, x_min_over_L=x_min)
    out = {}
    for k in ("delta99", "delta_star", "theta", "cf"):
        out[f"mean_err_{k}_pct"], out[f"max_err_{k}_pct"] = _mean_max(c[k])
    return out


# 1 ------------------------------------------------------------------------
def reynolds_variation(cfg: Config, b: BlasiusSolution, Re_list=(1e4, 1e5, 1e6)):
    rows = []
    for Re in Re_list:
        c = cfg.with_Re_L(Re)
        res = solve(c)
        cur = pp.summary_curves(res)
        U, nu, L = c.U_inf, c.nu, c.L
        row = {"Re_L": Re, "U_inf": U, "H_m": c.H,
               "delta99_over_L_cfd": cur["delta99"][-1] / L,
               "delta99_over_L_blasius": float(b.delta99(L, U, nu)) / L,
               "err_delta99_pct": 100 * abs(cur["delta99"][-1] / b.delta99(L, U, nu) - 1),
               "err_cf_pct": 100 * abs(cur["cf"][-1] / b.cf(L, U, nu) - 1),
               "err_delta_star_pct": 100 * abs(cur["delta_star"][-1] / b.delta_star(L, U, nu) - 1),
               "laminar_assumption_ok": Re < 5e5}
        rows.append(row)
    return pd.DataFrame(rows)


# 2 ------------------------------------------------------------------------
def wall_clustering(cfg: Config, b: BlasiusSolution, betas=(0.0, 1.0, 2.0, 3.0),
                    Ny_list=(51, 201)):
    rows, curves = [], {}
    for Ny in Ny_list:
        for beta in betas:
            res = solve(replace(cfg, Ny=Ny, stretching=beta, scheme="bdf1"))
            cur = pp.summary_curves(res)
            i = -1
            row = {"Ny": Ny, "beta": beta,
                   "dy_wall_m": float(res.grid.dy[0]),
                   "dudy_wall_TE_1_per_s": float(pp.wall_gradient(res.u[i], res.grid.y)),
                   "dudy_wall_TE_blasius": float(b.fpp0 * cfg.U_inf * np.sqrt(cfg.U_inf / (cfg.nu * cfg.L))),
                   "tau_w_TE_Pa": float(cur["tau_w"][i]),
                   "tau_w_TE_blasius_Pa": float(b.tau_wall(cfg.L, cfg.U_inf, cfg.nu, cfg.rho))}
            row["err_dudy_wall_pct"] = 100 * abs(row["dudy_wall_TE_1_per_s"] / row["dudy_wall_TE_blasius"] - 1)
            row.update(_error_summary(res, b))
            rows.append(row)
            if Ny == Ny_list[0]:
                curves[f"β={beta:g}"] = val.error_curves(res, b, x_min_over_L=0.05)
    return pd.DataFrame(rows), curves


# 3 ------------------------------------------------------------------------
SCHEMES = {
    "1st-order backward (bdf1)": dict(scheme="bdf1", n_picard=1),
    "2nd-order backward (bdf2)": dict(scheme="bdf2", n_picard=1),
    "bdf1 + 3 Picard sweeps": dict(scheme="bdf1", n_picard=3),
    "bdf2 + 3 Picard sweeps": dict(scheme="bdf2", n_picard=3),
}


def scheme_comparison(cfg: Config, b: BlasiusSolution):
    rows, curves = [], {}
    for name, kw in SCHEMES.items():
        res = solve(replace(cfg, **kw))
        row = {"scheme": name, "time_s": res.wall_time,
               "picard_last_change_mps": res.picard_change}
        row.update(_error_summary(res, b))
        i = pp.station_index(res.grid.x, 0.5 * cfg.L)
        row["profile_rmse_mid_pct"], row["profile_emax_mid_pct"] = val.profile_errors(
            res.u[i], res.grid.y, res.grid.x[i], res.cfg, b)
        row["u_min"] = float(res.u.min()); row["u_over_Uinf_max"] = float(res.u.max() / cfg.U_inf)
        rows.append(row)
        curves[name] = val.error_curves(res, b, x_min_over_L=0.05)
    # observed streamwise order: refine Nx with Ny fixed fine, uniform grid
    ord_rows = []
    for sch in ("bdf1", "bdf2"):
        errs = []
        for nx in (101, 201, 401, 801):
            res = solve(replace(cfg, Nx=nx, Ny=801, scheme=sch))
            c = pp.summary_curves(res)
            errs.append(abs(c["theta"][-1] / b.theta(cfg.L, cfg.U_inf, cfg.nu) - 1))
        for k in range(1, len(errs)):
            ord_rows.append({"scheme": sch, "Nx_coarse": (101, 201, 401, 801)[k - 1],
                             "err_theta_TE_coarse_pct": 100 * errs[k - 1],
                             "err_theta_TE_fine_pct": 100 * errs[k],
                             "observed_order": float(np.log2(errs[k - 1] / errs[k]))})
    return pd.DataFrame(rows), curves, pd.DataFrame(ord_rows)


# 4 ------------------------------------------------------------------------
def performance(cfg: Config, repeats=3):
    rows = []
    for name, (nx, ny) in val.MESHES.items():
        for ls in ("thomas", "banded"):
            c = replace(cfg, Nx=nx, Ny=ny, linear_solver=ls, scheme="bdf1")
            times = []
            for _ in range(repeats):
                times.append(solve(c).wall_time)
            tracemalloc.start()
            solve(c)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append({"mesh": name, "Nx": nx, "Ny": ny, "grid_points": nx * ny,
                         "linear_solver": ls, "time_s": float(np.median(times)),
                         "peak_memory_MB": peak / 1e6,
                         "stations_marched": nx - 1,
                         "tridiagonal_solves": nx - 1})
    return pd.DataFrame(rows)
