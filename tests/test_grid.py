import numpy as np
import pytest
from dataclasses import replace

from config import DEFAULT
from grid import make_grid, y_stencil_weights


def test_dimensions_and_spacing():
    g = make_grid(DEFAULT)
    assert g.x.size == 401 and g.y.size == 201
    assert g.dx == pytest.approx(1.0 / 400)
    assert g.dy == pytest.approx(0.05 / 200)
    assert g.x[0] == 0 and g.x[-1] == pytest.approx(DEFAULT.L)
    assert g.y[0] == 0 and g.y[-1] == pytest.approx(DEFAULT.H)
    assert g.uniform_y


def test_clustered_grid_monotone_and_finer_at_wall():
    g = make_grid(replace(DEFAULT, stretching=2.0))
    d = g.dy
    assert np.all(d > 0) and g.y[-1] == DEFAULT.H
    assert d[0] < d[-1] and not g.uniform_y


@pytest.mark.parametrize("beta", [0.0, 2.0])
def test_stencils_exact_for_quadratics(beta):
    y = make_grid(replace(DEFAULT, stretching=beta)).y
    a_m, a_0, a_p, b_m, b_0, b_p = y_stencil_weights(y)
    f = 3 * y ** 2 - 2 * y + 1
    d1 = a_m * f[:-2] + a_0 * f[1:-1] + a_p * f[2:]
    d2 = b_m * f[:-2] + b_0 * f[1:-1] + b_p * f[2:]
    assert d1 == pytest.approx(6 * y[1:-1] - 2, abs=1e-6)
    assert d2 == pytest.approx(6.0, abs=1e-5)   # round-off ~ eps/h^2


def test_parameters_configurable():
    c = DEFAULT.with_Re_L(1e5)
    assert c.Re_L == pytest.approx(1e5)
