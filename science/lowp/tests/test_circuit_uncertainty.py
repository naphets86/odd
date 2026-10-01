import math

import numpy as np
import pytest

from src import circuit_uncertainty as cu


def test_uncertainty_index_and_entropy():
    omega = cu.omega_circuit([2.0, -1.0, 0.0])
    assert omega == pytest.approx([0.0, 0.5, 1.0])
    assert cu.omega_circuit_from_signal([0.0, 1.0, 2.0], 1.0) == pytest.approx([0, 0, 0])
    assert cu.normalized_entropy([1.0] * 10) == pytest.approx(0.0)
    assert cu.normalized_entropy(np.arange(100), bins=10) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cu.omega_circuit([0.0, 0.0])
    with pytest.raises(ValueError):
        cu.normalized_entropy([], bins=4)


def test_freedom_and_pareto_functions():
    omega = 0.4
    assert cu.freedom(omega) == pytest.approx(-math.log(0.6))
    assert cu.freedom_derivative(omega) == pytest.approx(1 / 0.6)
    assert cu.freedom_second_derivative(omega) == pytest.approx(1 / 0.6**2)
    assert cu.deterministic_has_freedom() is False
    assert cu.freedom_growth_peak([0.1, 0.4, 0.8]) == pytest.approx(0.8)
    assert cu.pareto_objective(0.5) == pytest.approx(cu.freedom(0.5) * 0.5)
    assert cu.stationary_points() == []
    assert cu.golden_omega() == pytest.approx((math.sqrt(5) - 1) / 2)
    assert cu.golden_freedom() == pytest.approx(-math.log(1 - cu.golden_omega()))
    assert [cu.omega_zone(o) for o in (0.1, 0.5, 0.9)] == ["stagnant", "viable", "chaotic"]
    with pytest.raises(ValueError):
        cu.pareto_objective(0.5, alpha=1.0)


def test_deterministic_and_robust_stability():
    assert cu.critical_resistance(1.0, 1.0) == pytest.approx(2.0)
    assert cu.damping_ratio(4.0, 1.0, 1.0) == pytest.approx(2.0)
    assert cu.hurwitz_stable(0.1, 1.0, 1.0)
    assert cu.nominal_condition(4.0, 1.0, 1.0)
    assert not cu.nominal_condition(1.0, 1.0, 1.0)
    delta_max = cu.delta_max_corners(4.0, 1.0, 1.0)
    assert 0.3 < delta_max < 0.32
    assert cu.corner_overdamped(4.0, 1.0, 1.0, delta_max - 1e-9)
    assert not cu.corner_overdamped(4.0, 1.0, 1.0, delta_max + 1e-9)
    assert cu.corner_overdamped(4.0, 1.0, 1.0, 0.2)
    assert not cu.corner_overdamped(4.0, 1.0, 1.0, 0.4)
    assert cu.deterministic_margin(4.0, 1.0, 1.0) == pytest.approx(2.0)
    assert cu.stability_region_width(4.0, 1.0, 1.0, 0.25) == pytest.approx(2.0)
    assert cu.stable_fraction(4, 1, 1, 0.1, 100, rng=42) == 1.0


def test_resonance_and_uncertain_cascade_reproducibility():
    assert cu.resonance_ratio(0.0, 0.0) == 1.0
    assert cu.resonance_shift_extremes(0.2) == pytest.approx((1 / 1.2 - 1, 1 / 0.8 - 1))
    assert cu.resonance_shift_ok(0.2)
    assert not cu.resonance_shift_ok(0.4)
    cascade = cu.perturbed_cascade(9.0, 4.0, 3.0, 5, 0.1, rng=123)
    repeat = cu.perturbed_cascade(9.0, 4.0, 3.0, 5, 0.1, rng=123)
    assert cascade.keys() == {"L", "C", "R"}
    for key in cascade:
        assert cascade[key].shape == (5,)
        assert cascade[key] == pytest.approx(repeat[key])
    deviations = cu.resonance_deviation(9.0, 4.0, 5, 0.1, rng=123)
    assert np.all(np.abs(deviations) <= cu.resonance_deviation_bound(0.1) + 1e-12)
    assert [cu.leibniz_regime(x) for x in (0.1, 0.2, 0.4)] == ["robust", "degraded", "lost"]
    assert cu.leibniz_weight_error(4, 9.0, 4.0, 3.0, 0.0) == pytest.approx(0.0)


def test_rlc_frequency_response_and_gain():
    l, c, r = 2.0, 0.5, 4.0
    w0 = cu.resonance_omega(l, c)
    assert w0 == pytest.approx(1.0)
    assert abs(cu.rlc_admittance(w0, l, r, c)) == pytest.approx(1 / r)
    assert cu.gain_at_resonance(r) == pytest.approx(0.25)
    assert cu.gain_deviation(l, r, c, 0, 0, 0) == pytest.approx(0.0)
    assert cu.phase_deviation(l, r, c, 0, 0, 0) == pytest.approx(0.0)
    assert cu.worst_case_gain_deviation(l, r, c, 0.0) == pytest.approx(0.0)
    assert cu.worst_case_phase_deviation(l, r, c, 0.0) == pytest.approx(0.0)
    assert cu.mean_gain_deviation(l, r, c, 0.0, n_samples=10) == pytest.approx(0.0)
    assert cu.frequency_response_tex(w0, l, r, c) == pytest.approx(cu.rlc_admittance(w0, l, r, c) / c)
    assert cu.gain_deviation_model(0.2) == pytest.approx(0.2)
    assert cu.gain_deviation_model(0.4) == pytest.approx(0.56)


def test_design_guidance_and_sustainability():
    assert cu.scenario_label(0.25) == "herausfordernd"
    assert cu.component_tolerance(0.3) == pytest.approx(0.1)
    assert not cu.needs_maintenance(0.6)
    assert cu.needs_maintenance(0.61)
    conditions = cu.sustainability_conditions(2.0, 10.0, 5.0, 4.0, 1.0, True, 0.5, 0.01, 0.1)
    assert all(conditions.values())
    assert cu.omega_rate([0.0, 0.2, 0.5], 0.5) == pytest.approx([0.4, 0.6])
    assert cu.bipolar_jump(0.1, 0.5)
    assert not cu.bipolar_jump(0.1, 0.4, threshold=0.4)
    assert cu.circuit_spec()["omega0"] == pytest.approx(1 / math.sqrt(1e-9))
    assert cu.corn_law_delta_max() < 0


@pytest.mark.parametrize(
    "call",
    [
        lambda: cu.omega_circuit([]),
        lambda: cu.omega_circuit_from_signal([1.0], 0.1),
        lambda: cu.freedom(1.0),
        lambda: cu.critical_resistance(0.0, 1.0),
        lambda: cu.resonance_shift_extremes(1.0),
        lambda: cu.stable_fraction(1.0, 1.0, 1.0, 1.0, n_samples=0),
        lambda: cu.perturbed_cascade(1.0, 1.0, 1.0, 0, 0.1),
        lambda: cu.gain_deviation_model(0.7),
        lambda: cu.system_uncertainty([]),
        lambda: cu.omega_rate([0.1], 1.0),
    ],
)
def test_invalid_uncertainty_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()
