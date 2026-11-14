"""Structured 2-D grid generation.

x is always uniform.  y is either uniform (``stretching == 0``) or clustered
towards the wall with a tanh mapping

    y(s) = H * (1 - tanh(beta (1 - s)) / tanh(beta)),   s in [0, 1]

which has dy/ds -> small at s = 0 (wall) and dy/ds -> large at s = 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Config


@dataclass(frozen=True)
class Grid:
    x: np.ndarray      # (Nx,)
    y: np.ndarray      # (Ny,)
    dx: float
    Nx: int
    Ny: int

    @property
    def dy(self) -> np.ndarray:
        """Cell heights, shape (Ny-1,)."""
        return np.diff(self.y)

    @property
    def uniform_y(self) -> bool:
        d = self.dy
        return bool(np.allclose(d, d[0], rtol=1e-12, atol=0))


def make_grid(cfg: Config) -> Grid:
    if cfg.Nx < 3 or cfg.Ny < 3:
        raise ValueError("need at least 3 points in each direction")
    x = np.linspace(0.0, cfg.L, cfg.Nx)
    s = np.linspace(0.0, 1.0, cfg.Ny)
    if cfg.stretching > 0.0:
        b = cfg.stretching
        y = cfg.H * (1.0 - np.tanh(b * (1.0 - s)) / np.tanh(b))
    else:
        y = cfg.H * s
    y[0], y[-1] = 0.0, cfg.H            # kill round-off at the end points
    return Grid(x=x, y=y, dx=cfg.L / (cfg.Nx - 1), Nx=cfg.Nx, Ny=cfg.Ny)


def y_stencil_weights(y: np.ndarray):
    """Second-order central stencils on a (possibly non-uniform) y grid.

    For interior node j with hm = y_j - y_{j-1}, hp = y_{j+1} - y_j::

        du/dy   ~ a_m u_{j-1} + a_0 u_j + a_p u_{j+1}
        d2u/dy2 ~ b_m u_{j-1} + b_0 u_j + b_p u_{j+1}

    Returns (a_m, a_0, a_p, b_m, b_0, b_p), each of length Ny-2 (nodes 1..Ny-2).
    On a uniform grid these reduce to (-1, 0, 1)/(2 dy) and (1, -2, 1)/dy^2.
    """
    hm = y[1:-1] - y[:-2]
    hp = y[2:] - y[1:-1]
    a_m = -hp / (hm * (hm + hp))
    a_0 = (hp - hm) / (hm * hp)
    a_p = hm / (hp * (hm + hp))
    b_m = 2.0 / (hm * (hm + hp))
    b_0 = -2.0 / (hm * hp)
    b_p = 2.0 / (hp * (hm + hp))
    return a_m, a_0, a_p, b_m, b_0, b_p
