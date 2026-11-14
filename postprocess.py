"""Derived boundary-layer quantities computed from the CFD velocity field."""
from __future__ import annotations

import numpy as np

from solver import Result


def wall_gradient(u: np.ndarray, y: np.ndarray) -> np.ndarray:
    """du/dy at the wall, second-order one-sided (non-uniform-grid) stencil.

    u has shape (..., Ny); returns shape (...).
    """
    h1, h2 = y[1] - y[0], y[2] - y[1]
    w0 = -(2.0 * h1 + h2) / (h1 * (h1 + h2))
    w1 = (h1 + h2) / (h1 * h2)
    w2 = -h1 / (h2 * (h1 + h2))
    return w0 * u[..., 0] + w1 * u[..., 1] + w2 * u[..., 2]


def delta99(u: np.ndarray, y: np.ndarray, U: float) -> np.ndarray:
    """99 % thickness for every row of u, by linear interpolation of the first
    crossing of 0.99 U.  Rows that never reach 0.99 U give NaN."""
    target = 0.99 * U
    out = np.full(u.shape[0], np.nan)
    for i in range(u.shape[0]):
        above = np.nonzero(u[i] >= target)[0]
        if above.size == 0:
            continue
        k = above[0]
        if k == 0:
            out[i] = 0.0
            continue
        u0, u1 = u[i, k - 1], u[i, k]
        out[i] = y[k - 1] + (target - u0) / (u1 - u0) * (y[k] - y[k - 1])
    return out


def displacement_thickness(u, y, U):
    return np.trapezoid(1.0 - u / U, y, axis=-1)


def momentum_thickness(u, y, U):
    r = u / U
    return np.trapezoid(r * (1.0 - r), y, axis=-1)


def skin_friction(res: Result):
    """Return (cf, tau_w) arrays over x (cf[0] is NaN at the leading edge)."""
    cfg = res.cfg
    tau = cfg.mu * wall_gradient(res.u, res.grid.y)
    cf = tau / (0.5 * cfg.rho * cfg.U_inf ** 2)
    cf[0] = np.nan
    tau[0] = np.nan
    return cf, tau


def continuity_residual(res: Result):
    """Continuity residual du/dx + dv/dy on interior nodes.

    Two measures are returned:
      * 'scheme'  : with the discretisation used by the solver (backward x,
                    trapezoid y) -- zero up to round-off by construction;
      * 'central' : independent second-order central differences in x and y
                    (non-zero by truncation error -- a genuine accuracy check).
    Both are evaluated on nodes 1..Nx-2 (x) and 1..Ny-2 (y).
    """
    u, v, g = res.u, res.v, res.grid
    y, dx = g.y, g.dx
    dudx_c = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)
    hm = (y[1:-1] - y[:-2])
    hp = (y[2:] - y[1:-1])
    dvdy_c = (-hp / (hm * (hm + hp)) * v[1:-1, :-2]
              + (hp - hm) / (hm * hp) * v[1:-1, 1:-1]
              + hm / (hp * (hm + hp)) * v[1:-1, 2:])
    central = dudx_c + dvdy_c
    # scheme-consistent residual (trapezoid in y, scheme's own d/dx)
    cfg = res.cfg
    out = np.zeros((g.Nx - 1, g.Ny - 1))
    for i in range(1, g.Nx):
        if cfg.scheme == "bdf2" and i >= 2:
            dudx = (1.5 * u[i] - 2 * u[i - 1] + 0.5 * u[i - 2]) / dx
        else:
            dudx = (u[i] - u[i - 1]) / dx
        dv = (v[i, 1:] - v[i, :-1]) / np.diff(y)
        out[i - 1] = 0.5 * (dudx[1:] + dudx[:-1]) + dv
    return {"scheme": out, "central": central}


def station_index(x: np.ndarray, x_target: float) -> int:
    return int(np.argmin(np.abs(x - x_target)))


def summary_curves(res: Result) -> dict:
    """Everything needed to plot/export streamwise development (no x = 0)."""
    cfg, g = res.cfg, res.grid
    U, nu = cfg.U_inf, cfg.nu
    cf, tau = skin_friction(res)
    d99 = delta99(res.u, g.y, U)
    ds = displacement_thickness(res.u, g.y, U)
    th = momentum_thickness(res.u, g.y, U)
    d99[0] = ds[0] = th[0] = np.nan
    return {"x": g.x, "Re_x": U * g.x / nu, "delta99": d99, "delta_star": ds,
            "theta": th, "cf": cf, "tau_w": tau}
