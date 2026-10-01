import math

import numpy as np
import pytest

from src import economic_dynamics as ed


def test_lowpass_response_and_frequency_characteristics():
    assert ed.adjustment_time(2.0, 3.0) == pytest.approx(6.0)
    assert ed.sensitivity(2.0) == pytest.approx(0.5)
    assert ed.lowpass_response(0.0, 2.0, 3.0, lambda _: 99.0) == 3.0
    assert ed.lowpass_response(2.0, 2.0, 4.0, lambda _: 4.0) == pytest.approx(4.0)
    assert ed.shock_influence(2.0, 1.0, 1.0) == pytest.approx(math.exp(-1))
    assert ed.max_shock_influence(2.0) == 0.5
    assert ed.magnitude(1.0, 1.0) == pytest.approx(1 / math.sqrt(2))
    assert ed.phase(1.0, 1.0) == pytest.approx(-math.pi / 4)
    assert ed.cutoff_omega(2.0) == 0.5
    assert ed.cutoff_frequency(2.0) == pytest.approx(1 / (4 * math.pi))
    assert ed.period_damping(2 * math.pi, 1.0) == pytest.approx(1 / math.sqrt(2))
    assert ed.shock_classification(2.0, 1.0) == "damped"
    assert ed.shock_classification(2 * math.pi, 1.0) == "critical"
    assert ed.shock_classification(10.0, 1.0) == "follows"
    assert ed.periodic_shock_absorbed(10.0, 1.0)
    assert ed.phase_at_cutoff_degrees() == pytest.approx(-45.0)
    assert ed.rc_quality_factor() == 0.5


def test_harmonics_supply_and_market_damping():
    assert ed.heaviside_square_series(math.pi / 2, 1.0, 500) == pytest.approx(1.0, abs=0.002)
    assert ed.saturate(3.0, 2.0) == 2.0
    assert ed.saturate(1.0, 2.0) == 1.0
    amplitudes = ed.clipped_sine_harmonics(1.0, n_harmonics=4, n_samples=2048)
    assert amplitudes[0] == pytest.approx(1.0, abs=1e-3)
    assert amplitudes[1:] == pytest.approx([0.0] * 3, abs=1e-3)
    assert ed.supply_function(0.0, 10.0, [2.0, 3.0], 1.0) == pytest.approx(10.0)
    assert ed.supply_function(0.2, 10.0, [2.0], 1.0, [math.pi / 2]) == pytest.approx(
        10 + 2 * math.sin(0.2 + math.pi / 2)
    )
    assert ed.l_odd(0.5, 1.0, 100) == pytest.approx(ed.l_odd_closed_form(0.5, 1.0), abs=1e-12)
    assert ed.market_damping_rate(4.0) == 0.25
    with pytest.raises(ValueError):
        ed.supply_function(0.0, 0.0, [1.0], 1.0, [])


def test_forced_cycle_and_rk4_simulation():
    alpha, period, eta = 0.5, 8.0, 2.0
    amplitude = eta / math.sqrt(alpha**2 + (2 * math.pi / period)**2)
    phase = -math.atan((2 * math.pi / period) / alpha)
    assert ed.forced_amplitude(alpha, period, eta) == pytest.approx(amplitude)
    assert ed.forced_phase(alpha, period) == pytest.approx(phase)
    assert ed.limit_cycle(1.0, 10.0, eta, alpha, period) == pytest.approx(
        10 + amplitude * math.sin(2 * math.pi / period + phase)
    )
    t, values = ed.simulate_price(5.0, 2.0, 0.0, alpha, period, 3.0, steps=300)
    assert t[-1] == 3.0
    assert values[-1] == pytest.approx(2 + 3 * math.exp(-alpha * 3), abs=1e-8)
    assert ed.stability_condition(1.0, 10.0)
    assert ed.critical_period(1.0) == pytest.approx(2 * math.pi)
    assert ed.is_critical_resonance(1.0, 2 * math.pi)
    assert ed.minimal_alpha(2 * math.pi) == pytest.approx(1.0)


def test_resource_and_collapse_calculations():
    demand, supply = lambda _: 3.0, lambda _: 1.0
    assert ed.resource_level(10.0, demand, supply, 2.0) == pytest.approx(6.0)
    assert ed.resources_exhausted(1.0, demand, supply, 1.0)
    assert ed.overdemand(3.0, 2.0)
    assert ed.high_frequency_shock(1.0, 2.0)
    assert ed.too_sluggish(0.5, 4.0)
    assert ed.collapse_flags(0.5, 4.0, 1.0, 3.0, 2.0, -1.0) == {
        "high_frequency_shock": True,
        "overdemand": True,
        "too_sluggish": True,
        "resource_exhaustion": True,
    }
    assert ed.collapse_timescale_resonance(1.0, 2 * math.pi) == math.inf
    assert ed.collapse_time_resources(10.0, 2.0) == 5.0
    assert ed.shock_growth_rate(1.0, 2.0) > 0.0
    assert ed.shock_growth_rate(0.1, 2.0) == 0.0
    assert ed.will_collapse(1.0, 10.0, 0.1, 10.0, 2.0, 6.0)
    assert not ed.will_collapse(1.0, 10.0, 0.1, 10.0, 2.0, 4.0)


def test_examples_and_unit_conversion():
    medieval = ed.medieval_example()
    modern = ed.modern_example()
    assert medieval["critical_period"] == pytest.approx(2 * math.pi / 0.03)
    assert medieval["stable"] == 0.0
    assert modern["stable"] == 1.0
    assert modern["amplitude"] > 0
    assert ed.alpha_ratio() == pytest.approx(10 / (0.03 / 365))


@pytest.mark.parametrize(
    "call",
    [
        lambda: ed.adjustment_time(0.0, 1.0),
        lambda: ed.sensitivity(0.0),
        lambda: ed.lowpass_response(-1.0, 1.0, 0.0, lambda _: 0.0),
        lambda: ed.shock_influence(0.0, 1.0, 1.0),
        lambda: ed.clipped_sine_harmonics(0.0),
        lambda: ed.l_odd(-1.0, 1.0),
        lambda: ed.l_odd(1.0, 0.0),
        lambda: ed.simulate_price(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, steps=1),
        lambda: ed.resource_level(1.0, lambda _: 1.0, lambda _: 0.0, -1.0),
        lambda: ed.collapse_time_resources(1.0, 0.0),
    ],
)
def test_invalid_economic_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()
