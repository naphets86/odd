"""tests/test_information.py – Tests for digi.information"""
import numpy as np
import pytest
from digi.information import (
    information_content,
    shannon_entropy,
    shannon_capacity,
    shannon_capacity_db,
    capacity_over_bandwidth,
    thermal_noise_voltage,
    thermal_noise_power,
    sql_displacement_psd,
    sql_displacement_rms,
    noise_figure_passive_min_db,
    noise_figure_sql_db,
    noise_figure_margin_db,
    bekenstein_bound_bits,
    K_B,
    H_BAR,
    T_BODY,
)


class TestInformationContent:
    def test_certain_event_zero_bits(self):
        """p=1 → I=0"""
        I = information_content(np.array([1.0]))
        assert pytest.approx(float(I[0])) == 0.0

    def test_half_probability_one_bit(self):
        """p=0.5 → I=1 bit"""
        I = information_content(np.array([0.5]))
        assert pytest.approx(float(I[0])) == 1.0

    def test_quarter_probability_two_bits(self):
        I = information_content(np.array([0.25]))
        assert pytest.approx(float(I[0])) == 2.0

    def test_array_input(self):
        I = information_content(np.array([0.5, 0.25, 0.125]))
        np.testing.assert_allclose(I, [1.0, 2.0, 3.0])

    def test_zero_probability_raises(self):
        with pytest.raises(ValueError):
            information_content(np.array([0.0]))

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError):
            information_content(np.array([-0.1]))

    def test_above_one_raises(self):
        with pytest.raises(ValueError):
            information_content(np.array([1.1]))


class TestShannonEntropy:
    def test_binary_max_entropy(self):
        """H({0.5, 0.5}) = 1 bit"""
        H = shannon_entropy([0.5, 0.5])
        assert pytest.approx(H) == 1.0

    def test_uniform_four_symbols(self):
        """H({1/4, 1/4, 1/4, 1/4}) = 2 bit"""
        H = shannon_entropy([0.25, 0.25, 0.25, 0.25])
        assert pytest.approx(H) == 2.0

    def test_deterministic_zero_entropy(self):
        """H([1.0]) = 0 bits"""
        H = shannon_entropy([1.0])
        assert pytest.approx(H) == 0.0

    def test_not_sum_to_one_raises(self):
        with pytest.raises(ValueError):
            shannon_entropy([0.3, 0.3, 0.3])

    def test_zero_prob_handled(self):
        """p=0 terms contribute 0 (0 log 0 = 0 convention)"""
        H = shannon_entropy([0.5, 0.5, 0.0])
        assert pytest.approx(H) == 1.0


class TestShannonCapacity:
    def test_formula(self):
        """C = B log2(1 + SNR)"""
        B, snr = 10_000.0, 100.0
        C = shannon_capacity(B, snr)
        assert pytest.approx(C) == B * np.log2(1 + snr)

    def test_zero_snr(self):
        """Zero SNR → zero capacity"""
        assert pytest.approx(shannon_capacity(10_000.0, 0.0)) == 0.0

    def test_negative_B_raises(self):
        with pytest.raises(ValueError):
            shannon_capacity(-1.0, 10.0)

    def test_negative_snr_raises(self):
        with pytest.raises(ValueError):
            shannon_capacity(1000.0, -1.0)

    def test_db_version_matches_linear(self):
        B, snr_db = 1000.0, 20.0
        C_db = shannon_capacity_db(B, snr_db)
        C_lin = shannon_capacity(B, 10 ** (snr_db / 10))
        assert pytest.approx(C_db) == C_lin

    def test_spectral_efficiency(self):
        """C/B = log2(1+SNR)"""
        eta = capacity_over_bandwidth(np.array([0.0]))
        assert pytest.approx(float(eta[0])) == 1.0  # log2(2) at 0 dB

    def test_spectral_efficiency_increases(self):
        snr_dbs = np.array([0.0, 10.0, 20.0, 30.0])
        eta = capacity_over_bandwidth(snr_dbs)
        assert np.all(np.diff(eta) > 0)


class TestThermalNoise:
    def test_formula(self):
        """U_N = sqrt(4 k_B T R Δf)"""
        T, R, df = 300.0, 1e6, 1.0
        U_N = thermal_noise_voltage(T, R, df)
        expected = np.sqrt(4 * K_B * T * R * df)
        assert pytest.approx(U_N) == expected

    def test_zero_resistance(self):
        assert thermal_noise_voltage(300.0, 0.0, 1.0) == 0.0

    def test_zero_temperature(self):
        assert thermal_noise_voltage(0.0, 1e6, 1.0) == 0.0

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            thermal_noise_voltage(-1.0, 1000.0, 1.0)

    def test_negative_bandwidth_raises(self):
        with pytest.raises(ValueError):
            thermal_noise_voltage(300.0, 1000.0, -1.0)

    def test_increases_with_temperature(self):
        U1 = thermal_noise_voltage(300.0, 1000.0, 1.0)
        U2 = thermal_noise_voltage(600.0, 1000.0, 1.0)
        assert U2 > U1

    def test_thermal_noise_power(self):
        """P = k_B T Δf"""
        T, df = 300.0, 1e9
        P = thermal_noise_power(T, df)
        assert pytest.approx(P) == K_B * T * df


class TestSQL:
    def test_psd_formula(self):
        """S_xx = ℏ/(m*ω₀)"""
        m, omega_0 = 1e-11, 2 * np.pi * 10_000.0
        S = sql_displacement_psd(m, omega_0)
        assert pytest.approx(S) == H_BAR / (m * omega_0)

    def test_rms_is_sqrt_psd(self):
        m, omega_0 = 1e-11, 2 * np.pi * 10_000.0
        assert pytest.approx(sql_displacement_rms(m, omega_0)) == np.sqrt(sql_displacement_psd(m, omega_0))

    def test_paper_value_order(self):
        """Paper: x_SQL(cat, 10 kHz) ≈ 4 fm = 4e-15 m"""
        m, f0 = 1e-11, 10_000.0
        x_sql = sql_displacement_rms(m, 2 * np.pi * f0)
        assert 1e-15 < x_sql < 1e-13  # correct order of magnitude

    def test_zero_mass_raises(self):
        with pytest.raises(ValueError):
            sql_displacement_psd(0.0, 1.0)

    def test_zero_omega_raises(self):
        with pytest.raises(ValueError):
            sql_displacement_psd(1e-11, 0.0)


class TestNoiseFigure:
    def test_passive_minimum_is_zero_db(self):
        assert noise_figure_passive_min_db() == 0.0

    def test_sql_nf_is_negative_db(self):
        """For biological frequencies, quantum NF is hugely negative"""
        nf = noise_figure_sql_db(f0=10_000.0, T=T_BODY)
        assert nf < -70.0  # paper: −81 dB at 10 kHz, 37 °C

    def test_sql_nf_lower_at_lower_frequency(self):
        nf_hi = noise_figure_sql_db(100_000.0, T_BODY)
        nf_lo = noise_figure_sql_db(1_000.0, T_BODY)
        assert nf_lo < nf_hi

    def test_margin_for_cat(self):
        """Cat NF ≈ −18 dB → margin ≈ +63 dB above SQL"""
        margin = noise_figure_margin_db(-18.0, f0=10_000.0, T=T_BODY)
        assert 50.0 < margin < 80.0

    def test_passive_margin_is_large(self):
        """A passive system (NF = 0 dB) is far above quantum SQL"""
        margin = noise_figure_margin_db(0.0, f0=10_000.0, T=T_BODY)
        assert margin > 60.0


class TestBekensteinBound:
    def test_positive(self):
        bits = bekenstein_bound_bits(1.0, 1.0)
        assert bits > 0

    def test_scales_with_energy(self):
        b1 = bekenstein_bound_bits(1.0, 1.0)
        b2 = bekenstein_bound_bits(2.0, 1.0)
        assert pytest.approx(b2 / b1) == 2.0

    def test_scales_with_radius(self):
        b1 = bekenstein_bound_bits(1.0, 1.0)
        b2 = bekenstein_bound_bits(1.0, 2.0)
        assert pytest.approx(b2 / b1) == 2.0
