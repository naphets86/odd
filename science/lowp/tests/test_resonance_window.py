import math

import pytest

from src import resonance_window as rw


def test_golden_ratio_and_fibonacci_approximations():
    golden = (1 + math.sqrt(5)) / 2
    assert rw.phi() == pytest.approx(golden)
    assert rw.phi_inv() == pytest.approx(1 / golden)
    assert all(rw.golden_identities().values())
    assert rw.golden_quadratic_roots() == pytest.approx((golden, -1 / golden))
    assert rw.continued_fraction_phi(12) == pytest.approx(golden, abs=1e-4)
    assert [rw.fibonacci(n) for n in range(7)] == [0, 1, 1, 2, 3, 5, 8]
    assert rw.fibonacci_ratios(10)[-1] == pytest.approx(golden, abs=1e-3)
    assert rw.fibonacci_ratio_error(10) < rw.fibonacci_ratio_error(5)
    assert rw.golden_powers(3) == pytest.approx([1, 1 / golden, 1 / golden**2, 1 / golden**3])


def test_trigonometric_point_and_resonance_window():
    assert rw.theta_sym() == pytest.approx(math.pi / 4)
    assert rw.theta_sym_numeric() == pytest.approx(math.pi / 4, abs=1e-12)
    assert rw.s_c() == pytest.approx(math.sqrt(2) / 2)
    assert rw.isosceles_legs_equal()
    assert rw.harmonic_difference() == pytest.approx(rw.harmonic_difference_closed_form())
    lo, hi = rw.window_bounds()
    assert (lo + hi) / 2 == pytest.approx(rw.s_c())
    assert hi == pytest.approx(rw.window_upper_closed_form())
    assert rw.window_center() == pytest.approx(rw.s_c())
    assert rw.window_half_width() == pytest.approx(rw.harmonic_difference())
    assert rw.window_is_symmetric()
    assert rw.delta_h_is_irrational_witness()


def test_window_zones_distances_and_disturbance_tolerance():
    lo, hi = rw.window_bounds()
    assert rw.in_window(lo) and rw.in_window(hi)
    assert not rw.in_window(lo - 0.01)
    assert rw.window_position(lo - 0.01) == "below"
    assert rw.window_position((lo + hi) / 2) == "inside"
    assert rw.window_position(hi + 0.01) == "above"
    assert rw.economic_zone(lo - 0.01) == "stagnation"
    assert rw.economic_zone(hi + 0.01) == "crisis"
    assert rw.distance_to_window(lo - 0.1) == pytest.approx(0.1)
    assert rw.distance_to_window((lo + hi) / 2) == 0.0
    assert rw.tolerance_to_bounds((lo + hi) / 2) == pytest.approx(
        ((hi - lo) / 2, (hi - lo) / 2)
    )
    assert rw.absorbs_disturbance((lo + hi) / 2, 0.0)
    with pytest.raises(ValueError):
        rw.tolerance_to_bounds(lo - 0.1)


def test_productivity_predictability_and_economic_models():
    lo, hi = rw.window_bounds()
    center = rw.window_center()
    assert rw.freedom_range_in_window() == pytest.approx(
        (-math.log(1 - lo), -math.log(1 - hi))
    )
    assert rw.productivity(center) == pytest.approx(rw.freedom(center))
    assert rw.predictability(center) == 1.0
    assert rw.min_predictability_in_window() > 0.8
    assert rw.freedom_times_predictability(center) == pytest.approx(rw.freedom(center))
    assert rw.lyapunov_proxy(center) < 0
    assert rw.innovation_rate(center) == pytest.approx(rw.freedom(center))
    assert rw.price_variance(0.5) == pytest.approx(0.25)
    assert rw.biodiversity(0.5) == pytest.approx(0.25)
    assert rw.productivity_convex_in_window()
    assert rw.corn_law_left_window()
    assert 0.0 <= rw.switzerland_in_window_fraction() <= 1.0
    assert rw.ussr_below_window()
    assert rw.policy_target_window() == pytest.approx((lo, hi))


def test_weighted_covariance_and_case_study_helpers():
    value = rw.weighted_cov_index([[1, 3]], [1.0], [2.0])
    assert value == pytest.approx(0.25)
    assert rw.fraction_in_window([0.5, 0.7, 0.9]) == pytest.approx(1 / 3)
    assert rw.excursions([0.1, 0.7, 0.9]) == [(0, "below"), (2, "above")]
    assert rw.crises_above_window(rw.ARGENTINA)


@pytest.mark.parametrize(
    "call",
    [
        lambda: rw.fibonacci(-1),
        lambda: rw.fibonacci_ratios(0),
        lambda: rw._check_omega(1.0),
        lambda: rw.tolerance_to_bounds(0.1),
        lambda: rw.productivity(0.5, lam=0.0),
        lambda: rw.biodiversity(1.1),
        lambda: rw.weighted_cov_index([], [], []),
        lambda: rw.fraction_in_window([]),
    ],
)
def test_invalid_resonance_window_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()
