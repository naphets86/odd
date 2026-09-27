"""tests/test_basilar.py – Tests for digi.basilar"""
import numpy as np
import pytest
from digi.basilar import (
    tonotopic_frequency,
    tonotopic_position,
    tonotopic_gradient,
    bandwidth_efficiency,
    q10db,
    active_quality_factor,
    cochlear_amplifier_gain,
    snr_gain_db,
    gammatone_impulse_response,
    gammatone_envelope,
)


class TestTonotopicFrequency:
    def test_at_apex_x0(self):
        """x=0 → f = f_apex"""
        f_apex, beta = 200.0, 2.1
        f = tonotopic_frequency(0.0, f_apex, beta)
        assert pytest.approx(float(f)) == f_apex

    def test_increases_with_x(self):
        f_apex, beta = 200.0, 2.1
        x = np.linspace(0, 10, 50)
        f = tonotopic_frequency(x, f_apex, beta)
        assert np.all(np.diff(f) > 0)

    def test_formula(self):
        x, f_apex, beta = 5.0, 100.0, 1.5
        f = tonotopic_frequency(x, f_apex, beta)
        assert pytest.approx(float(f)) == f_apex * np.exp(beta * x)

    def test_array_input(self):
        x = np.array([0.0, 5.0, 10.0])
        f = tonotopic_frequency(x, 200.0, 2.1)
        assert f.shape == (3,)

    def test_cat_parameters(self):
        """Cat: f_apex=200 Hz, L=25 mm, f_base=85 kHz → β≈0.242 mm⁻¹"""
        import math
        beta = math.log(85_000 / 200.0) / 25.0  # ≈ 0.242 mm⁻¹
        f_base = tonotopic_frequency(25.0, 200.0, beta)
        assert 50_000 < float(f_base) < 200_000


class TestTonotopicPosition:
    def test_round_trip(self):
        """f → position → f should recover original frequency"""
        f_orig = np.array([1000.0, 5000.0, 20_000.0])
        f_apex, beta = 200.0, 2.1
        x = tonotopic_position(f_orig, f_apex, beta)
        f_recovered = tonotopic_frequency(x, f_apex, beta)
        np.testing.assert_allclose(f_recovered, f_orig, rtol=1e-10)

    def test_apex_gives_zero(self):
        f_apex, beta = 200.0, 2.1
        x = tonotopic_position(f_apex, f_apex, beta)
        assert pytest.approx(float(x), abs=1e-10) == 0.0

    def test_positive_for_f_above_apex(self):
        x = tonotopic_position(1000.0, 200.0, 2.1)
        assert float(x) > 0

    def test_zero_frequency_raises(self):
        with pytest.raises(ValueError):
            tonotopic_position(0.0, 200.0, 2.1)

    def test_negative_frequency_raises(self):
        with pytest.raises(ValueError):
            tonotopic_position(-100.0, 200.0, 2.1)


class TestTonotopicGradient:
    def test_formula(self):
        """β = ln(f_base/f_apex) / L"""
        f_apex, f_base, L = 200.0, 85_000.0, 25.0
        beta = tonotopic_gradient(f_apex, f_base, L)
        assert pytest.approx(beta) == np.log(f_base / f_apex) / L

    def test_consistent_with_mapping(self):
        """β computed from endpoints should reproduce f_base at x=L"""
        f_apex, f_base, L = 200.0, 85_000.0, 25.0
        beta = tonotopic_gradient(f_apex, f_base, L)
        f_computed = tonotopic_frequency(L, f_apex, beta)
        assert pytest.approx(float(f_computed), rel=1e-9) == f_base

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            tonotopic_gradient(200.0, 100.0, 25.0)  # f_base <= f_apex


class TestBandwidthEfficiency:
    def test_cat_value(self):
        """Cat: η_B ≈ 3.4 kHz/mm"""
        eta = bandwidth_efficiency(85_000.0, 25.0)
        assert pytest.approx(eta, rel=1e-3) == 3400.0

    def test_dog_value(self):
        """Dog: η_B ≈ 2.0 kHz/mm"""
        eta = bandwidth_efficiency(65_000.0, 32.0)
        assert pytest.approx(eta, abs=100) == 65_000 / 32

    def test_cat_higher_than_dog(self):
        eta_cat = bandwidth_efficiency(85_000.0, 25.0)
        eta_dog = bandwidth_efficiency(65_000.0, 32.0)
        assert eta_cat > eta_dog

    def test_zero_length_raises(self):
        with pytest.raises(ValueError):
            bandwidth_efficiency(1000.0, 0.0)


class TestQ10dB:
    def test_formula(self):
        assert pytest.approx(q10db(1000.0, 100.0)) == 10.0

    def test_cat_range(self):
        """Q10 for cat should be in range 8.5–30"""
        Q = q10db(10_000.0, 500.0)
        assert Q == pytest.approx(20.0)

    def test_zero_bandwidth_raises(self):
        with pytest.raises(ValueError):
            q10db(1000.0, 0.0)

    def test_narrow_bandwidth_gives_high_Q(self):
        assert q10db(10_000.0, 100.0) > q10db(10_000.0, 1000.0)


class TestActiveCochlearQ:
    def test_formula(self):
        """Q_active = Q_passive * R_passive / (R_passive - g_motor)"""
        Q_p, g, R = 5.0, 0.5, 1.0
        Q_a = active_quality_factor(Q_p, g, R)
        assert pytest.approx(Q_a) == Q_p * R / (R - g)

    def test_higher_than_passive(self):
        Q_a = active_quality_factor(5.0, 0.5, 1.0)
        assert Q_a > 5.0

    def test_unstable_raises(self):
        """g_motor >= R_passive should raise ValueError"""
        with pytest.raises(ValueError):
            active_quality_factor(5.0, 1.0, 1.0)  # g = R

    def test_cat_prestin(self):
        """Cat: g_motor ≈ 0.85 R_passive → Q × 6.7"""
        Q_p, R = 5.0, 1.0
        g = 0.85 * R
        Q_a = active_quality_factor(Q_p, g, R)
        assert pytest.approx(Q_a / Q_p, rel=0.05) == 1.0 / (1 - 0.85)


class TestCochlearAmplifierGain:
    def test_formula(self):
        G = cochlear_amplifier_gain(R_passive=1.0, g_motor=0.5)
        assert pytest.approx(G) == 2.0

    def test_unity_at_zero_motor(self):
        G = cochlear_amplifier_gain(R_passive=1.0, g_motor=0.0)
        assert pytest.approx(G) == 1.0

    def test_unstable_raises(self):
        with pytest.raises(ValueError):
            cochlear_amplifier_gain(1.0, 1.0)


class TestSNRGain:
    def test_cat_vs_dog(self):
        """Cat Q ~17, Dog Q ~7 → ΔSNRdB ≈ 7.7 according to 20log10 ratio"""
        delta = snr_gain_db(17.0, 7.0)
        assert pytest.approx(delta, rel=0.01) == 20 * np.log10(17.0 / 7.0)

    def test_equal_Q_gives_zero(self):
        assert pytest.approx(snr_gain_db(10.0, 10.0)) == 0.0

    def test_positive_when_cat_better(self):
        assert snr_gain_db(20.0, 5.0) > 0


class TestGammatone:
    def test_zero_for_negative_t(self):
        """Gammatone is causal: zero for t < 0"""
        t = np.array([-0.01, -0.001])
        g = gammatone_impulse_response(t, f_center=1000.0)
        np.testing.assert_array_equal(g, 0.0)

    def test_envelope_non_negative(self):
        t = np.linspace(0, 0.1, 1000)
        e = gammatone_envelope(t, f_center=1000.0)
        assert np.all(e >= 0)

    def test_envelope_decays(self):
        t = np.linspace(0.001, 0.1, 500)
        e = gammatone_envelope(t, f_center=1000.0)
        # Envelope decays after its peak
        peak_idx = np.argmax(e)
        if peak_idx < len(e) - 10:
            assert e[-1] < e[peak_idx]

    def test_shape_matches_t(self):
        t = np.linspace(0, 0.05, 200)
        g = gammatone_impulse_response(t, f_center=4000.0)
        assert g.shape == t.shape

    def test_explicit_bandwidth(self):
        t = np.linspace(0, 0.1, 1000)
        g = gammatone_impulse_response(t, f_center=1000.0, b=100.0)
        assert g.shape == t.shape
