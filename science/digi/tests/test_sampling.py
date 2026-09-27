"""tests/test_sampling.py – Tests for digi.sampling"""
import numpy as np
import pytest
from digi.sampling import (
    nyquist_rate,
    nyquist_satisfied,
    sample,
    sample_array,
    alias_frequency,
    sinc_reconstruct,
    sampling_interval,
    max_representable_frequency,
)


class TestNyquistRate:
    def test_basic(self):
        assert nyquist_rate(1000.0) == 2000.0

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            nyquist_rate(0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            nyquist_rate(-5.0)

    def test_float(self):
        assert nyquist_rate(22050.0) == 44100.0

    def test_exact_threshold(self):
        """Nyquist rate for 20 kHz (audio CD lower bound)"""
        assert nyquist_rate(20_000.0) == 40_000.0


class TestNyquistSatisfied:
    def test_satisfied(self):
        assert nyquist_satisfied(f_max=10_000.0, fs=44_100.0) is True

    def test_not_satisfied(self):
        assert nyquist_satisfied(f_max=10_000.0, fs=15_000.0) is False

    def test_exactly_satisfied(self):
        assert nyquist_satisfied(f_max=5_000.0, fs=10_000.0) is True


class TestSampleFunction:
    def test_correct_number_of_samples(self):
        fs = 100.0
        t_s, x_s = sample(np.sin, 0.0, 1.0, fs)
        assert len(t_s) == pytest.approx(fs + 1, abs=1)

    def test_values_match_function(self):
        fs = 10.0
        t_s, x_s = sample(lambda t: 2.0 * t, 0.0, 1.0, fs)
        expected = 2.0 * t_s
        np.testing.assert_allclose(x_s, expected, atol=1e-12)

    def test_constant_signal(self):
        _, x_s = sample(lambda t: 5.0, 0.0, 1.0, 50.0)
        np.testing.assert_array_equal(x_s, 5.0)


class TestSampleArray:
    def test_decimation(self):
        fs_fine = 1000.0
        dt = 1.0 / fs_fine
        t = np.arange(0, 1.0, dt)
        x = np.sin(2 * np.pi * t)
        fs_target = 50.0
        t_s, x_s = sample_array(x, t, fs_target)
        expected_step = int(round(fs_fine / fs_target))
        assert len(t_s) == len(t[::expected_step])

    def test_values_preserved(self):
        t = np.arange(0, 1.0, 0.001)
        x = np.ones_like(t) * 3.14
        _, x_s = sample_array(x, t, fs=100.0)
        np.testing.assert_allclose(x_s, 3.14, atol=1e-12)


class TestAliasFrequency:
    def test_below_nyquist_unchanged(self):
        """Signal below fs/2 – no aliasing"""
        assert pytest.approx(alias_frequency(100.0, 1000.0)) == 100.0

    def test_above_nyquist_folds(self):
        """f=600 Hz, fs=1000 Hz → alias = 1000-600 = 400 Hz"""
        assert pytest.approx(alias_frequency(600.0, 1000.0)) == 400.0

    def test_exactly_nyquist(self):
        """f = fs/2 → no aliasing (boundary)"""
        assert pytest.approx(alias_frequency(500.0, 1000.0)) == 500.0

    def test_integer_multiple_of_fs(self):
        """f = fs → folds back to 0"""
        assert pytest.approx(alias_frequency(1000.0, 1000.0), abs=1e-9) == 0.0

    def test_3x_nyquist(self):
        """f = 1.5*fs = 1500, fs = 1000 → alias = 500"""
        assert pytest.approx(alias_frequency(1500.0, 1000.0)) == 500.0


class TestSincReconstruct:
    def test_perfect_reconstruction_below_nyquist(self):
        """Sinc interpolation should perfectly reconstruct a bandlimited signal"""
        fs = 100.0
        f0 = 5.0
        t_coarse = np.arange(0, 1.0, 1.0 / fs)
        x_coarse = np.sin(2 * np.pi * f0 * t_coarse)

        t_fine = np.linspace(0.1, 0.9, 50)  # interior to avoid edge effects
        x_fine = sinc_reconstruct(x_coarse, t_coarse, t_fine)
        expected = np.sin(2 * np.pi * f0 * t_fine)

        np.testing.assert_allclose(x_fine, expected, atol=3e-3)

    def test_shape(self):
        t_s = np.linspace(0, 1, 20)
        x_s = np.zeros(20)
        t_out = np.linspace(0, 1, 100)
        out = sinc_reconstruct(x_s, t_s, t_out)
        assert out.shape == (100,)

    def test_zero_signal_stays_zero(self):
        t_s = np.linspace(0, 1, 50)
        x_s = np.zeros(50)
        t_out = np.linspace(0, 1, 200)
        out = sinc_reconstruct(x_s, t_s, t_out)
        np.testing.assert_allclose(out, 0.0, atol=1e-12)


class TestSamplingHelpers:
    def test_sampling_interval(self):
        assert pytest.approx(sampling_interval(1000.0)) == 1e-3

    def test_sampling_interval_negative_raises(self):
        with pytest.raises(ValueError):
            sampling_interval(-1.0)

    def test_sampling_interval_zero_raises(self):
        with pytest.raises(ValueError):
            sampling_interval(0.0)

    def test_max_representable_frequency(self):
        assert max_representable_frequency(44100.0) == 22050.0
