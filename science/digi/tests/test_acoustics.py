"""tests/test_acoustics.py – Tests for digi.acoustics"""
import numpy as np
import pytest
from digi.acoustics import (
    pressure_wave,
    wave_number,
    wavelength,
    sound_intensity,
    spl,
    spl_to_pressure,
    amplitude_to_rms,
    P_REF,
    C_SOUND_AIR,
    RHO_AIR,
)


class TestPressureWave:
    def test_stationary_at_x0_t0_zero_ambient(self):
        """p(0, 0) = p_ambient when kx=0, ωt=0"""
        p = pressure_wave(
            x=0.0, t=0.0, p0=1.0, frequency=100.0, p_ambient=0.0
        )
        assert pytest.approx(float(p), abs=1e-10) == 0.0

    def test_peak_at_t_quarter_period_x0(self):
        """p0 · sin(ω · T/4) = p0  (x=0, ambient=0)"""
        f = 100.0
        T = 1.0 / f
        p = pressure_wave(
            x=0.0, t=T / 4, p0=2.0, frequency=f, p_ambient=0.0
        )
        assert pytest.approx(float(p), rel=1e-6) == 2.0

    def test_ambient_adds_correctly(self):
        """Ambient pressure is added as offset"""
        p_amb = 101_325.0
        p = pressure_wave(x=0.0, t=0.0, p0=0.0, frequency=100.0, p_ambient=p_amb)
        assert pytest.approx(float(p)) == p_amb

    def test_array_broadcast(self):
        """x and t can be arrays"""
        x = np.array([0.0, 1.0, 2.0])
        t = np.array([0.0])
        result = pressure_wave(x, t, p0=1.0, frequency=10.0, p_ambient=0.0)
        assert result.shape == (3, 1) or result.shape == (3,)


class TestWaveNumber:
    def test_basic(self):
        f = 343.0  # Hz — wavelength ≈ 1 m
        k = wave_number(f)
        lam = 2 * np.pi / k
        assert pytest.approx(lam, rel=1e-6) == C_SOUND_AIR / f

    def test_k_positive(self):
        assert wave_number(1000.0) > 0


class TestWavelength:
    def test_zero_frequency_raises(self):
        with pytest.raises(ValueError):
            wavelength(0.0)

    def test_negative_frequency_raises(self):
        with pytest.raises(ValueError):
            wavelength(-100.0)

    def test_middle_a(self):
        """440 Hz in air: λ ≈ 0.7795 m"""
        lam = wavelength(440.0)
        assert pytest.approx(lam, rel=1e-3) == C_SOUND_AIR / 440.0

    def test_ultrasound(self):
        """20 kHz → λ ≈ 17 mm"""
        lam = wavelength(20_000.0)
        assert pytest.approx(lam * 1000, rel=1e-3) == C_SOUND_AIR / 20.0


class TestSoundIntensity:
    def test_proportional_to_p0_squared(self):
        I1 = sound_intensity(1.0)
        I2 = sound_intensity(2.0)
        assert pytest.approx(I2 / I1) == 4.0

    def test_formula(self):
        p0 = 1.0
        expected = p0**2 / (2 * RHO_AIR * C_SOUND_AIR)
        assert pytest.approx(sound_intensity(p0)) == expected

    def test_positive(self):
        assert sound_intensity(0.001) > 0


class TestSPL:
    def test_reference_gives_zero_db(self):
        L = spl(P_REF)
        assert pytest.approx(float(L)) == 0.0

    def test_double_pressure_gives_6db(self):
        """Doubling RMS pressure → +6.02 dB"""
        L = spl(2 * P_REF)
        assert pytest.approx(float(L), abs=0.1) == 6.02

    def test_roundtrip(self):
        p_orig = np.array([0.001, 0.1, 1.0, 10.0])
        p_rt = spl_to_pressure(spl(p_orig))
        np.testing.assert_allclose(p_rt, p_orig, rtol=1e-9)

    def test_array_input(self):
        pressures = np.array([P_REF, 2 * P_REF, 10 * P_REF])
        levels = spl(pressures)
        assert levels.shape == (3,)
        assert levels[0] == pytest.approx(0.0)

    def test_130db_threshold_pressure(self):
        """130 dB SPL ≈ 63.2 Pa"""
        p = spl_to_pressure(130.0)
        assert pytest.approx(p, rel=1e-3) == P_REF * 10 ** (130 / 20)


class TestAmplitudeToRMS:
    def test_sine_rms(self):
        """RMS of A·sin = A/√2"""
        A = np.sqrt(2.0)
        assert pytest.approx(amplitude_to_rms(A)) == 1.0

    def test_zero(self):
        assert amplitude_to_rms(0.0) == 0.0
