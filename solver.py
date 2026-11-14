"""Finite-difference marching solver for the 2-D laminar boundary-layer equations.

    du/dx + dv/dy = 0                         (continuity)
    u du/dx + v du/dy = nu d2u/dy2            (x-momentum, dp/dx = 0)

NOTE: this is a *boundary-layer* solver, not a full Navier-Stokes
pressure-velocity solver.  The equations are parabolic in x, so the solution
is marched from the inlet to the outlet one streamwise station at a time.

Scheme at station i (unknowns u_{i,j}, j = 1..Ny-2)
---------------------------------------------------
Streamwise derivative (backward, order 1 or 2):

    du/dx |_i ~ (c0 u_i + c1 u_{i-1} + c2 u_{i-2}) / dx,
        bdf1: (c0,c1,c2) = (1, -1, 0)
        bdf2: (c0,c1,c2) = (3/2, -2, 1/2)

Wall-normal derivatives: second-order central (non-uniform-grid stencils from
grid.y_stencil_weights), diffusion implicit at the new station.

The convection velocities multiplying the derivatives are *frozen* coefficients
(u~, v~).  Freezing them makes the x-momentum equation linear in the unknown
u_i, giving a tridiagonal system:

    u~_j c0/dx u_j + v~_j (du/dy)_j - nu (d2u/dy2)_j = -u~_j (c1 u_{i-1,j} + c2 u_{i-2,j}) / dx

Choice of (u~, v~):
  * first Picard sweep: values lagged from upstream (u_{i-1}, v_{i-1}) for bdf1,
    linear extrapolation 2 u_{i-1} - u_{i-2} for bdf2;
  * further sweeps (n_picard > 1): values from the previous sweep at the
    *current* station, so the converged result solves the full nonlinear
    discrete equation  u du/dx + v du/dy = nu d2u/dy2.

Wall-normal velocity: after u_i is known, continuity is integrated from the
wall (v = 0) with the trapezoid rule, using the same streamwise difference
operator as the momentum equation:

    v_{i,j} = v_{i,j-1} - 0.5 (y_j - y_{j-1}) (dudx_{i,j} + dudx_{i,j-1}).

Boundary conditions
-------------------
Inlet   u = U_inf, v = 0 (the wall node of the inlet column is set to u = 0)
Wall    u = v = 0
Top     u = U_inf (Dirichlet); v is free, determined by continuity
Outlet  zero streamwise gradient.  For a marching scheme this is automatic:
        no outlet condition is imposed, the last station is simply the last
        step of the march, so no reflection can travel upstream.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from config import Config
from grid import Grid, make_grid, y_stencil_weights

_COEFFS = {"bdf1": (1.0, -1.0, 0.0), "bdf2": (1.5, -2.0, 0.5)}


def thomas(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray,
           rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm for a tridiagonal system.

    lower[k] multiplies x[k-1] (lower[0] unused), upper[k] multiplies x[k+1]
    (upper[-1] unused).  Inputs are not modified.
    """
    n = diag.size
    c = np.empty(n)
    d = np.empty(n)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for k in range(1, n):
        denom = diag[k] - lower[k] * c[k - 1]
        c[k] = upper[k] / denom
        d[k] = (rhs[k] - lower[k] * d[k - 1]) / denom
    x = np.empty(n)
    x[-1] = d[-1]
    for k in range(n - 2, -1, -1):
        x[k] = d[k] - c[k] * x[k + 1]
    return x


def _tridiag_solve(lower, diag, upper, rhs, method: str) -> np.ndarray:
    if method == "thomas":
        return thomas(lower, diag, upper, rhs)
    if method == "banded":
        ab = np.empty((3, diag.size))
        ab[0, 1:] = upper[:-1]
        ab[1] = diag
        ab[2, :-1] = lower[1:]
        return solve_banded((1, 1), ab, rhs, check_finite=False)
    raise ValueError(f"unknown linear solver '{method}'")


@dataclass
class Result:
    cfg: Config
    grid: Grid
    u: np.ndarray            # (Nx, Ny)
    v: np.ndarray            # (Nx, Ny)
    wall_time: float         # seconds spent marching
    picard_change: float     # largest last-sweep change of u (0 if n_picard == 1)


def _dudx(c, u_i, u_im1, u_im2, dx):
    return (c[0] * u_i + c[1] * u_im1 + c[2] * u_im2) / dx


def solve(cfg: Config) -> Result:
    if cfg.scheme not in _COEFFS:
        raise ValueError(f"unknown scheme '{cfg.scheme}'")
    g = make_grid(cfg)
    Nx, Ny, dx = g.Nx, g.Ny, g.dx
    U, nu = cfg.U_inf, cfg.nu
    y = g.y
    h = np.diff(y)
    a_m, a_0, a_p, b_m, b_0, b_p = y_stencil_weights(y)

    u = np.zeros((Nx, Ny))
    v = np.zeros((Nx, Ny))
    u[0, :] = U                       # uniform inlet ...
    u[0, 0] = 0.0                     # ... except the wall node
    # v[0, :] = 0 already

    max_change = 0.0
    t0 = time.perf_counter()
    for i in range(1, Nx):
        c = _COEFFS["bdf1"] if (cfg.scheme == "bdf1" or i == 1) else _COEFFS["bdf2"]
        u_im1 = u[i - 1]
        u_im2 = u[i - 2] if i >= 2 else u[i - 1]

        # frozen convection velocities for the first sweep
        if c[2] != 0.0:               # bdf2: extrapolate
            ut = 2.0 * u[i - 1] - u[i - 2]
            vt = 2.0 * v[i - 1] - v[i - 2]
        else:                         # bdf1: lag
            ut = u[i - 1].copy()
            vt = v[i - 1].copy()

        for sweep in range(max(cfg.n_picard, 1)):
            ut_in, vt_in = ut[1:-1], vt[1:-1]
            lower = vt_in * a_m - nu * b_m
            diag = ut_in * c[0] / dx + vt_in * a_0 - nu * b_0
            upper = vt_in * a_p - nu * b_p
            rhs = -ut_in * (c[1] * u_im1[1:-1] + c[2] * u_im2[1:-1]) / dx
            rhs[-1] -= upper[-1] * U          # top Dirichlet node u = U
            # (the wall node u = 0 contributes nothing to the first row)

            u_new = np.empty(Ny)
            u_new[0] = 0.0
            u_new[-1] = U
            u_new[1:-1] = _tridiag_solve(lower, diag, upper, rhs, cfg.linear_solver)

            # continuity -> v at this station, same x-operator as momentum
            dudx = _dudx(c, u_new, u_im1, u_im2, dx)
            v_new = np.zeros(Ny)
            v_new[1:] = -np.cumsum(0.5 * h * (dudx[1:] + dudx[:-1]))

            if sweep > 0:           # change between the last two sweeps
                max_change = max(max_change, float(np.max(np.abs(u_new - ut))))
            ut, vt = u_new, v_new
        u[i] = u_new
        v[i] = v_new
    wall = time.perf_counter() - t0
    return Result(cfg=cfg, grid=g, u=u, v=v, wall_time=wall,
                  picard_change=float(max_change) if cfg.n_picard > 1 else 0.0)
