"""Independent Blasius similarity solution.

Solves  f''' + 0.5 f f'' = 0,  f(0) = f'(0) = 0,  f'(inf) = 1
with scipy.integrate.solve_ivp and a shooting method on f''(0).  Nothing in
here is used to build or prescribe the CFD velocity field; it is only the
reference used for validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ETA_MAX = 12.0          # "infinity" for the shooting problem
F2_WALL_REF = 0.332057336215


def _rhs(eta, s):
    f, fp, fpp = s
    return [fp, fpp, -0.5 * f * fpp]


def _integrate(fpp0: float, eta_max: float = ETA_MAX, dense: bool = False):
    return solve_ivp(_rhs, (0.0, eta_max), [0.0, 0.0, fpp0], method="DOP853",
                     rtol=1e-12, atol=1e-13, dense_output=dense)


def _residual(fpp0: float) -> float:
    return _integrate(fpp0).y[1, -1] - 1.0


@dataclass(frozen=True)
class BlasiusSolution:
    fpp0: float                 # f''(0), wall-shear parameter
    eta99: float                # f'(eta99) = 0.99
    delta_star_coeff: float     # delta*  / sqrt(nu x / U)
    theta_coeff: float          # theta   / sqrt(nu x / U)
    _sol: object

    # --- similarity-space accessors ---------------------------------------
    def f(self, eta):
        return self._eval(eta)[0]

    def fprime(self, eta):
        """u/U_inf as a function of eta (clipped to 1 beyond ETA_MAX)."""
        return self._eval(eta)[1]

    def _eval(self, eta):
        eta = np.asarray(eta, dtype=float)
        e = np.clip(eta, 0.0, ETA_MAX)
        out = self._sol.sol(e)
        f, fp = out[0].copy(), out[1].copy()
        beyond = eta > ETA_MAX          # linear extension of f, f' = 1
        f = np.where(beyond, out[0] + (eta - ETA_MAX), f)
        fp = np.where(beyond, 1.0, fp)
        return f, fp

    # --- physical-space reference quantities ------------------------------
    @staticmethod
    def eta(x, y, U, nu):
        return y * np.sqrt(U / (nu * np.maximum(x, 1e-300)))

    def u_over_U(self, x, y, U, nu):
        return self.fprime(self.eta(x, y, U, nu))

    def v_over_U(self, x, y, U, nu):
        """v/U = 0.5 sqrt(nu/(U x)) (eta f' - f)."""
        eta = self.eta(x, y, U, nu)
        f, fp = self._eval(eta)
        return 0.5 * np.sqrt(nu / (U * np.maximum(x, 1e-300))) * (eta * fp - f)

    def delta99(self, x, U, nu):
        return self.eta99 * np.sqrt(nu * x / U)

    def delta_star(self, x, U, nu):
        return self.delta_star_coeff * np.sqrt(nu * x / U)

    def theta(self, x, U, nu):
        return self.theta_coeff * np.sqrt(nu * x / U)

    @staticmethod
    def Re_x(x, U, nu):
        return U * x / nu

    def cf(self, x, U, nu):
        """C_f = 2 f''(0) / sqrt(Re_x)  (= 0.664 / sqrt(Re_x))."""
        return 2.0 * self.fpp0 / np.sqrt(self.Re_x(x, U, nu))

    def tau_wall(self, x, U, nu, rho):
        return 0.5 * rho * U ** 2 * self.cf(x, U, nu)


@lru_cache(maxsize=1)
def solve_blasius() -> BlasiusSolution:
    # Bracket the shooting parameter: too small -> f'(inf) < 1, too big -> > 1.
    fpp0 = brentq(_residual, 0.2, 0.5, xtol=1e-14, rtol=1e-14)
    sol = _integrate(fpp0, dense=True)
    # eta99: root of f'(eta) - 0.99
    eta99 = brentq(lambda e: sol.sol(e)[1] - 0.99, 1.0, 10.0, xtol=1e-14)
    f_inf = sol.y[0, -1]
    # delta* = int (1 - f') deta = eta - f(eta) at large eta
    ds = ETA_MAX - f_inf
    # theta = int f'(1 - f') deta = 2 f''(0)  (exact momentum-integral identity)
    # but evaluate by quadrature so it is an independent check, not an assumption
    eta = np.linspace(0.0, ETA_MAX, 200001)
    fp = sol.sol(eta)[1]
    th = np.trapezoid(fp * (1.0 - fp), eta)
    return BlasiusSolution(fpp0=fpp0, eta99=eta99, delta_star_coeff=ds,
                           theta_coeff=th, _sol=sol)


if __name__ == "__main__":
    b = solve_blasius()
    print(f"f''(0)        = {b.fpp0:.9f}  (reference {F2_WALL_REF:.9f})")
    print(f"eta99         = {b.eta99:.5f}  (classical 4.91)")
    print(f"delta*/sqrt() = {b.delta_star_coeff:.5f}  (classical 1.7208)")
    print(f"theta/sqrt()  = {b.theta_coeff:.5f}  (classical 0.664)")
