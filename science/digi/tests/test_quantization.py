"""tests/test_quantization.py – Tests for digi.quantization"""
import numpy as np
import pytest
from digi.quantization import (
    quantization_step,
    num_levels,
    quantize,
    quantization_error,
    snr_max_db,
    snr_db,
    snr_linear_to_db,
    snr_db_to_linear,
)


class TestQuantizationStep:
    def test_1bit_full_range(self):
        """1-bit over [-1, 1] → 2 levels → step = 1.0"""
        assert pytest.approx(quantization_step(1, -1.0, 1.0)) == 1.0

    def test_2bits(self):
        """2-bit over [0, 4] → 4 levels → step = 1.0"""
        assert pytest.approx(quantization_step(2, 0.0, 4.0)) == 1.0

    def test_8bits(self):
        delta = quantization_step(8, 0.0, 1.0)
        assert pytest.approx(delta) == 1.0 / 256

    def test_invalid_n_bits(self):
        with pytest.raises(ValueError):
            quantization_step(0)

    def test_invalid_range(self):
        with pytest.raises(ValueError):
            quantization_step(4, 1.0, 0.0)

    def test_equal_range_raises(self):
        with pytest.raises(ValueError):
            quantization_step(4, 0.0, 0.0)


class TestNumLevels:
    def test_1bit(self):
        assert num_levels(1) == 2

    def test_8bit(self):
        assert num_levels(8) == 256

    def test_16bit(self):
        assert num_levels(16) == 65_536

    def test_invalid(self):
        with pytest.raises(ValueError):
            num_levels(0)


class TestQuantize:
    def test_clipping_at_upper(self):
        x = np.array([2.0])  # outside [-1, 1]
        x_q = quantize(x, n_bits=3, x_min=-1.0, x_max=1.0)
        assert x_q[0] <= 1.0

    def test_clipping_at_lower(self):
        x = np.array([-2.0])
        x_q = quantize(x, n_bits=3, x_min=-1.0, x_max=1.0)
        assert x_q[0] >= -1.0

    def test_monotone(self):
        """Quantised output must be non-decreasing for increasing input"""
        x = np.linspace(-1, 1, 200)
        x_q = quantize(x, n_bits=4)
        assert np.all(np.diff(x_q) >= 0)

    def test_correct_number_of_levels_used(self):
        x = np.linspace(-1, 0.9999, 10_000)
        x_q = quantize(x, n_bits=3)
        assert len(np.unique(x_q)) <= num_levels(3)

    def test_error_max(self):
        """Quantisation error must be at most Δu/2"""
        n_bits = 4
        delta = quantization_step(n_bits)
        x = np.linspace(-1, 1, 5000)
        x_q = quantize(x, n_bits=n_bits)
        e = np.abs(quantization_error(x, x_q))
        assert np.all(e <= delta / 2.0 + 1e-12)

    def test_output_shape(self):
        x = np.ones((3, 5))
        x_q = quantize(x, n_bits=8)
        assert x_q.shape == (3, 5)


class TestQuantizationError:
    def test_perfect_quantization(self):
        """If input = quantised, error is zero"""
        x = np.zeros(10)
        e = quantization_error(x, x)
        np.testing.assert_array_equal(e, 0.0)

    def test_sign_of_error(self):
        x = np.array([0.0])
        x_q = np.array([0.1])
        e = quantization_error(x, x_q)
        assert e[0] == pytest.approx(0.1)


class TestSNR:
    def test_snr_max_formula(self):
        """SNR = 6.02*N + 1.76"""
        assert pytest.approx(snr_max_db(1)) == 7.78
        assert pytest.approx(snr_max_db(8)) == 49.92
        assert pytest.approx(snr_max_db(16)) == 98.08

    def test_snr_max_monotone(self):
        values = [snr_max_db(n) for n in range(1, 25)]
        assert all(v2 > v1 for v1, v2 in zip(values, values[1:]))

    def test_snr_max_invalid(self):
        with pytest.raises(ValueError):
            snr_max_db(0)

    def test_snr_db_perfect_signal(self):
        signal = np.ones(100) * 5.0
        noise = np.zeros(100)
        assert snr_db(signal, noise) == float("inf")

    def test_snr_db_equal_power(self):
        x = np.ones(100)
        assert pytest.approx(snr_db(x, x)) == 0.0

    def test_snr_round_trip(self):
        snr = 42.7
        assert pytest.approx(snr_linear_to_db(snr_db_to_linear(snr))) == snr

    def test_snr_24bit_exceeds_140db(self):
        """24-bit audio ≈ 144 dB SNR"""
        assert snr_max_db(24) > 140.0
