"""Physical and numerical parameters for the flat-plate boundary-layer solver.

Everything is stored in one immutable-ish dataclass so a run is fully described
by a single object.  Use ``dataclasses.replace(cfg, ...)`` to derive variants
(different Reynolds number, mesh, scheme, ...).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Config:
    # ---- physical parameters (brief, section 5) ---------------------------
    U_inf: float = 10.0          # free-stream velocity            [m/s]
    rho: float = 1.225           # density                         [kg/m^3]
    mu: float = 1.81e-5          # dynamic viscosity               [Pa s]
    L: float = 1.0               # plate length                    [m]
    H: float = 0.05              # domain height                   [m]

    # ---- grid -------------------------------------------------------------
    Nx: int = 401                # streamwise grid points
    Ny: int = 201                # wall-normal grid points
    stretching: float = 0.0      # wall-clustering strength beta (0 -> uniform)

    # ---- numerics ---------------------------------------------------------
    # "bdf1": first-order backward difference in x
    # "bdf2": second-order backward difference in x (bdf1 on the first step)
    scheme: str = "bdf1"
    n_picard: int = 1            # Picard sweeps per station (1 = fully lagged
                                 # coefficients, >1 = coefficients re-evaluated
                                 # at the current station -> consistent nonlinear)
    linear_solver: str = "banded"  # "banded" (scipy) or "thomas" (hand-written)

    # ---- post-processing --------------------------------------------------
    profile_stations: tuple = (0.1, 0.25, 0.5, 0.75, 1.0)   # x/L

    @property
    def nu(self) -> float:
        return self.mu / self.rho

    @property
    def Re_L(self) -> float:
        return self.U_inf * self.L / self.nu

    def with_Re_L(self, Re_L: float, h_over_delta: float = 10.0) -> "Config":
        """Variant with the requested plate Reynolds number.

        U_inf is varied at fixed nu and L.  The domain height is rescaled to
        ``h_over_delta`` times the trailing-edge thickness so the boundary
        layer always occupies the same fraction of the domain.
        """
        U = Re_L * self.nu / self.L
        delta_L = 4.91 * self.L / Re_L ** 0.5
        return replace(self, U_inf=U, H=h_over_delta * delta_L)


DEFAULT = Config()
