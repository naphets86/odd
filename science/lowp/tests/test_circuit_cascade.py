import math

import pytest

from src import circuit_cascade as cc


def test_circuit_type_catalog_and_selection():
    assert cc.circuit_type("rc_lowpass")["name"] == "RC-Tiefpass"
    assert cc.leibniz_suitability("parallel_rlc_cascade") == "ideal"
    assert cc.best_circuit_type() == "parallel_rlc_cascade"
    result = cc.circuit_type("rc_lowpass")
    result["name"] = "changed"
    assert cc.circuit_type("rc_lowpass")["name"] == "RC-Tiefpass"
    with pytest.raises(ValueError):
        cc.circuit_type("unknown")


def test_square_wave_harmonics_and_leibniz_link():
    assert cc.odd_harmonic_numbers(5) == [1, 3, 5, 7, 9]
    assert cc.square_wave(0.0, 1.0) == 0.0
    assert cc.square_wave(math.pi / 2, 1.0) == 1.0
    assert cc.square_wave(3 * math.pi / 2, 1.0) == -1.0
    assert cc.harmonic_amplitudes(1.0, 3) == pytest.approx(
        [4 / math.pi, 4 / (3 * math.pi), 4 / (5 * math.pi)]
    )
    assert cc.leibniz_partial_sum(1000) == pytest.approx(math.pi / 4, abs=0.001)
    assert cc.leibniz_from_square_wave(1000) == pytest.approx(math.pi / 4, abs=0.001)
    assert cc.square_wave_series(math.pi / 2, 1.0, n_terms=1000) == pytest.approx(1.0, abs=0.002)


def test_cascade_component_and_resonance_scaling():
    cascade = cc.LeibnizCascade(L1=9.0, C=4.0, R=3.0, n_stages=4)
    assert cascade.inductances() == pytest.approx([9.0, 1.0, 9 / 25, 9 / 49])
    assert cascade.capacitances() == [4.0] * 4
    assert cascade.resistances() == [3.0] * 4
    assert cascade.omega0 == pytest.approx(1 / 6)
    assert cascade.resonance_ratios() == pytest.approx([1, 3, 5, 7])
    assert cascade.verify_resonances()
    assert cascade.alpha == pytest.approx(1 / 6)
    assert cascade.stage_damping(1) == pytest.approx(1.5)
    assert cascade.quality_factor(1) == pytest.approx(cascade.quality_factor(0) / 3)
    assert cc.inductance_ratios(4) == pytest.approx([1, 1 / 9, 1 / 25, 1 / 49])


def test_impulse_response_matches_finite_sum_and_closed_forms():
    t, alpha, n_terms = 0.7, 0.8, 40
    unsigned = cc.impulse_response(t, alpha, n_terms)
    signed = cc.impulse_response(t, alpha, n_terms, signed=True)
    assert unsigned == pytest.approx(cc.impulse_response_closed_form(t, alpha), abs=1e-7)
    assert signed == pytest.approx(cc.impulse_response_closed_form(t, alpha, signed=True), abs=1e-12)
    assert cc.LeibnizCascade(1.0, 1.0, 2.0, 20).impulse_response(
        t, signed=True
    ) == pytest.approx(cc.impulse_response(t, 1.0, 20, signed=True))


def test_cascade_design_helpers_and_synthesis():
    taus = cc.standard_rc_taus(10.0, 0.2, 4)
    leibniz_taus = cc.leibniz_rc_taus(10.0, 0.2, 4)
    assert len(taus) == len(leibniz_taus) == 4
    assert taus == pytest.approx([2.0, 2.0, 2.0, 2.0])
    assert leibniz_taus[0] == pytest.approx(2.0)
    assert leibniz_taus[1] == pytest.approx(2.0 / 9)
    assert cc.rc_decay(0.0, 2.0) == 1.0
    assert cc.weighted_decay_sum(0.0, [1.0, 2.0]) == pytest.approx(4 / 3)
    assert cc.synthesizer_harmonics(440.0, 4) == [440.0, 1320.0, 2200.0, 3080.0]
    assert cc.synthesizer_weights(3) == pytest.approx([1.0, 1 / 3, 1 / 5])
    assert cc.design_cascade(50.0, 1e-6, 100.0, 3).verify_resonances()
    assert cc.resonance_frequencies_hz(50.0, 4) == pytest.approx([50, 150, 250, 350])


@pytest.mark.parametrize(
    "call",
    [
        lambda: cc.odd_harmonic_numbers(0),
        lambda: cc.leibniz_partial_sum(0),
        lambda: cc.resonance_frequency(0, 1),
        lambda: cc.stage_inductance(1, -1),
        lambda: cc.impulse_response(-0.1, 1, 3),
        lambda: cc.impulse_response_closed_form(0, 1),
        lambda: cc.LeibnizCascade(0, 1, 1),
    ],
)
def test_invalid_cascade_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()
