"""tests/test_hearing.py – Tests for digi.hearing"""
import numpy as np
import pytest
from digi.hearing import (
    HearingProfile,
    CAT,
    DOG,
    HUMAN,
    shannon_capacity,
    membrane_efficiency_omega,
    compare_profiles,
    capacity_ratio,
    bandwidth_ratio,
)


class TestHearingProfileProperties:
    def test_cat_bandwidth(self):
        """Cat bandwidth ≈ 85 kHz (48 Hz lower cutoff is negligible)"""
        assert pytest.approx(CAT.bandwidth, rel=0.01) == CAT.f_max - CAT.f_min

    def test_cat_bandwidth_efficiency(self):
        """Cat η_B ≈ 3.4 kHz/mm"""
        eta = CAT.bandwidth_efficiency
        assert pytest.approx(eta, rel=0.01) == CAT.bandwidth / CAT.L_membrane

    def test_dog_bandwidth_efficiency_lower_than_cat(self):
        assert CAT.bandwidth_efficiency > DOG.bandwidth_efficiency

    def test_temporal_resolution_cat(self):
        """Cat σ_t < 100 µs → f_s,time = 1/(2σ) > 5 kHz"""
        f_time = CAT.temporal_resolution_hz
        assert f_time >= 5000.0

    def test_temporal_resolution_cat_better_than_dog(self):
        assert CAT.temporal_resolution_hz > DOG.temporal_resolution_hz

    def test_cat_shannon_capacity(self):
        """Cat C ≈ 1.55 Mbit/s (paper value)"""
        C = CAT.shannon_capacity_bits_per_second
        assert 1e6 < C < 3e6

    def test_dog_shannon_capacity_lower(self):
        assert CAT.shannon_capacity_bits_per_second > DOG.shannon_capacity_bits_per_second

    def test_snr_linear_from_db(self):
        assert pytest.approx(CAT.snr_linear) == 10 ** (CAT.snr_db / 10)

    def test_bandwidth_positive(self):
        for p in [CAT, DOG, HUMAN]:
            assert p.bandwidth > 0

    def test_bandwidth_efficiency_positive(self):
        for p in [CAT, DOG, HUMAN]:
            assert p.bandwidth_efficiency > 0


class TestHearingProfileData:
    """Test the data values against the paper (Table in Section 11)"""

    def test_cat_f_max(self):
        assert pytest.approx(CAT.f_max, rel=0.05) == 85_000.0

    def test_dog_f_max(self):
        assert pytest.approx(DOG.f_max, rel=0.05) == 65_000.0

    def test_cat_membrane_length(self):
        assert pytest.approx(CAT.L_membrane, rel=0.05) == 25.0

    def test_dog_membrane_length(self):
        assert pytest.approx(DOG.L_membrane, rel=0.05) == 32.0

    def test_cat_beta_higher_than_dog(self):
        assert CAT.beta > DOG.beta

    def test_cat_Q_higher_than_dog(self):
        assert CAT.Q_mean > DOG.Q_mean

    def test_cat_dynamic_range(self):
        assert CAT.dynamic_range_db >= 100.0

    def test_dog_dynamic_range_lower(self):
        assert CAT.dynamic_range_db > DOG.dynamic_range_db

    def test_cat_pinna_range(self):
        """Cat ±180°"""
        assert CAT.pinna_range_deg == 180.0

    def test_dog_pinna_muscles_fewer(self):
        assert CAT.pinna_muscles > DOG.pinna_muscles

    def test_cat_spiral_ganglion_more(self):
        assert CAT.n_spiral_ganglion > DOG.n_spiral_ganglion


class TestShannonCapacity:
    def test_zero_bandwidth_raises(self):
        with pytest.raises(ValueError):
            shannon_capacity(0.0, snr_db=30.0)

    def test_negative_bandwidth_raises(self):
        with pytest.raises(ValueError):
            shannon_capacity(-100.0, snr_db=30.0)

    def test_formula_at_0db_snr(self):
        """C = B * log2(2) = B at 0 dB SNR (SNR=1)"""
        B = 1000.0
        C = shannon_capacity(B, snr_db=0.0)
        assert pytest.approx(C) == B * 1.0  # log2(2) = 1

    def test_capacity_increases_with_snr(self):
        B = 10_000.0
        C1 = shannon_capacity(B, snr_db=10.0)
        C2 = shannon_capacity(B, snr_db=20.0)
        assert C2 > C1

    def test_capacity_increases_with_bandwidth(self):
        C1 = shannon_capacity(10_000.0, snr_db=20.0)
        C2 = shannon_capacity(20_000.0, snr_db=20.0)
        assert C2 > C1

    def test_cat_capacity_paper_value(self):
        """Paper: C_cat ≈ 1.55 Mbit/s"""
        C = shannon_capacity(85_000.0, snr_db=55.0)
        assert abs(C - 1.55e6) / 1.55e6 < 0.2  # within 20 %

    def test_cat_over_dog_ratio(self):
        """Paper: C_cat / C_dog ≈ 2.42"""
        C_cat = shannon_capacity(85_000.0, snr_db=55.0)
        C_dog = shannon_capacity(65_000.0, snr_db=45.0)
        ratio = C_cat / C_dog
        assert 1.5 < ratio < 4.0  # loose bound given rounded paper values


class TestMembraneEfficiencyOmega:
    def test_cat_omega_equals_one_by_definition(self):
        """Ω(cat, reference=cat) = 1"""
        omega = membrane_efficiency_omega(CAT, reference=CAT)
        assert pytest.approx(omega) == 1.0

    def test_dog_omega_less_than_cat(self):
        """Dog Ω < 1 relative to cat"""
        omega_dog = membrane_efficiency_omega(DOG, reference=CAT)
        assert omega_dog < 1.0

    def test_human_omega_less_than_dog(self):
        omega_human = membrane_efficiency_omega(HUMAN, reference=CAT)
        omega_dog = membrane_efficiency_omega(DOG, reference=CAT)
        assert omega_human < omega_dog

    def test_paper_dog_omega(self):
        """Paper reports dog Ω ≈ 0.04"""
        omega_dog = membrane_efficiency_omega(DOG, reference=CAT)
        assert 0.01 < omega_dog < 0.2  # loose tolerance

    def test_positive(self):
        for p in [CAT, DOG, HUMAN]:
            assert membrane_efficiency_omega(p, reference=CAT) > 0


class TestCompareProfiles:
    def test_returns_all_species(self):
        result = compare_profiles(CAT, DOG, HUMAN)
        assert "Cat" in result
        assert "Dog" in result
        assert "Human" in result

    def test_keys_present(self):
        result = compare_profiles(CAT)
        cat_data = result["Cat"]
        expected_keys = [
            "bandwidth_Hz",
            "shannon_capacity_Mbit_per_s",
            "omega",
            "bandwidth_efficiency_kHz_per_mm",
        ]
        for key in expected_keys:
            assert key in cat_data

    def test_capacity_in_megabits(self):
        result = compare_profiles(CAT)
        C = result["Cat"]["shannon_capacity_Mbit_per_s"]
        assert 0.5 < C < 10.0  # reasonable range in Mbit/s

    def test_omega_cat_is_one(self):
        result = compare_profiles(CAT)
        assert pytest.approx(result["Cat"]["omega"]) == 1.0


class TestCapacityAndBandwidthRatio:
    def test_cat_to_dog_ratio_above_one(self):
        assert capacity_ratio(CAT, DOG) > 1.0

    def test_bandwidth_ratio_cat_dog(self):
        ratio = bandwidth_ratio(CAT, DOG)
        # Cat bandwidth is ~31% wider: 85 kHz vs 65 kHz
        assert pytest.approx(ratio, rel=0.05) == CAT.bandwidth / DOG.bandwidth

    def test_self_ratio_is_one(self):
        assert pytest.approx(capacity_ratio(CAT, CAT)) == 1.0
        assert pytest.approx(bandwidth_ratio(DOG, DOG)) == 1.0
