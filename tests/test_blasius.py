import pytest

from blasius import F2_WALL_REF, solve_blasius


def test_wall_shear_parameter():                          # Test 1
    assert solve_blasius().fpp0 == pytest.approx(F2_WALL_REF, abs=1e-8)


def test_classical_coefficients():
    b = solve_blasius()
    assert b.eta99 == pytest.approx(4.91, abs=1e-2)
    assert b.delta_star_coeff == pytest.approx(1.7208, abs=1e-3)
    assert b.theta_coeff == pytest.approx(0.664, abs=1e-3)
    assert 2 * b.fpp0 == pytest.approx(0.664, abs=1e-3)


def test_profile_limits():
    b = solve_blasius()
    assert float(b.fprime(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(b.fprime(15.0)) == pytest.approx(1.0, abs=1e-8)
    assert float(b.fprime(b.eta99)) == pytest.approx(0.99, abs=1e-8)
