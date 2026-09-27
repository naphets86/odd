"""tests/test_signals.py – Tests for digi.signals"""
import numpy as np
import pytest
from digi.signals import (
    harmonic,
    harmonic_derivative,
    signal_energy,
    signal_power,
    rms,
    fourier_transform,
    spectrum_magnitude,
    angular_frequency,
    frequency_from_angular,
)


class TestHarmonic:
    def test_zero_phase_peak(self):
        """A=1, f=1 Hz, φ=0 → peak at t=0.25 s"""
        t = np.array([0.25])
        x = harmonic(t, amplitude=1.0, frequency=1.0, phase=0.0)
        assert pytest.approx(x[0], abs=1e-9) == 1.0

    def test_amplitude_scale(self):
        # Max of a discretely-sampled sinusoid is ≤ amplitude; use relaxed tolerance.
        t = np.linspace(0, 1, 1000)
        x = harmonic(t, amplitude=3.0, frequency=5.0)
        assert pytest.approx(np.max(np.abs(x)), abs=1e-2) == 3.0

    def test_phase_shift(self):
        """Phase of π/2 converts sin → cos"""
        t = np.array([0.0])
        x = harmonic(t, amplitude=1.0, frequency=1.0, phase=np.pi / 2)
        # sin(0 + π/2) = 1.0
        assert pytest.approx(x[0], abs=1e-9) == 1.0

    def test_zero_amplitude(self):
        t = np.linspace(0, 1, 100)
        x = harmonic(t, amplitude=0.0)
        np.testing.assert_array_equal(x, 0.0)

    def test_returns_ndarray(self):
        assert isinstance(harmonic(np.array([0.0])), np.ndarray)

    def test_shape_preserved(self):
        t = np.zeros((3, 4))
        x = harmonic(t)
        assert x.shape == (3, 4)


class TestHarmonicDerivative:
    def test_derivative_is_cos(self):
        """d/dt[A sin(ωt)] = Aω cos(ωt)"""
        A, f = 2.0, 3.0
        omega = 2 * np.pi * f
        t = np.linspace(0, 1, 500)
        dx = harmonic_derivative(t, amplitude=A, frequency=f)
        expected = A * omega * np.cos(omega * t)
        np.testing.assert_allclose(dx, expected, atol=1e-12)

    def test_derivative_zero_crossing_where_sin_peaks(self):
        """Derivative is zero at t = 1/(4f) (peak of sin)"""
        t_peak = np.array([0.25])  # f=1 Hz peak
        dx = harmonic_derivative(t_peak, frequency=1.0)
        # cos(π/2) = 0
        assert pytest.approx(dx[0], abs=1e-9) == 0.0


class TestEnergyPower:
    def test_energy_pure_sine(self):
        """Energy of A·sin over N full periods ≈ A²N/(2f)"""
        A, f, N = 1.0, 1.0, 10
        dt = 1e-5
        t = np.arange(0, N / f, dt)
        x = harmonic(t, amplitude=A, frequency=f)
        E = signal_energy(x, dt)
        expected = A**2 * N / (2 * f)
        assert pytest.approx(E, rel=1e-3) == expected

    def test_power_equals_half_amplitude_squared(self):
        """Mean power of A·sin = A²/2"""
        A = 3.0
        t = np.linspace(0, 100, 1_000_000)
        x = harmonic(t, amplitude=A)
        P = signal_power(x, t[1] - t[0])
        assert pytest.approx(P, rel=1e-4) == A**2 / 2

    def test_rms_of_sine(self):
        """RMS of A·sin(t) ≈ A/√2"""
        A = np.sqrt(2)
        t = np.linspace(0, 1000, 10_000_000)
        x = harmonic(t, amplitude=A)
        assert pytest.approx(rms(x), rel=1e-4) == 1.0

    def test_energy_of_zeros(self):
        assert signal_energy(np.zeros(100), dt=1e-3) == 0.0

    def test_power_of_constant(self):
        x = np.ones(1000) * 5.0
        assert pytest.approx(signal_power(x, dt=1e-3), abs=1e-10) == 25.0


class TestFourierTransform:
    def test_dominant_frequency_detected(self):
        """FFT peak should match the input frequency"""
        fs = 1000.0
        f0 = 50.0
        t = np.arange(0, 1.0, 1 / fs)
        x = harmonic(t, amplitude=1.0, frequency=f0)
        freqs, X = fourier_transform(x, dt=1 / fs)
        peak_idx = np.argmax(np.abs(X))
        assert pytest.approx(freqs[peak_idx], abs=1.0) == f0

    def test_magnitude_spectrum_positive(self):
        t = np.linspace(0, 1, 512, endpoint=False)
        x = harmonic(t)
        freqs, mag = spectrum_magnitude(x, dt=t[1] - t[0])
        assert np.all(mag >= 0)

    def test_fourier_returns_tuple(self):
        t = np.linspace(0, 1, 128, endpoint=False)
        result = fourier_transform(harmonic(t), dt=t[1] - t[0])
        assert len(result) == 2


class TestAngularFrequency:
    def test_angular_frequency(self):
        assert pytest.approx(angular_frequency(1.0)) == 2 * np.pi

    def test_round_trip(self):
        f = 440.0
        assert pytest.approx(frequency_from_angular(angular_frequency(f))) == f

    def test_zero_frequency(self):
        assert angular_frequency(0.0) == 0.0
