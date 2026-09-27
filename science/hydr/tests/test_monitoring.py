"""Tests für hydr.monitoring – 100 % Abdeckung."""

import pytest

from hydr.monitoring import (
    ultrasonic_bulk_modulus,
    sound_speed_from_transit,
    recuperation_efficiency,
    OilCondition,
)


class TestUltrasonicBulkModulus:
    def test_formula(self):
        rho = 870.0
        L = 0.05     # 5 cm Messstrecke
        c_s = 1350.0
        delta_t = L / c_s
        expected = rho * c_s**2
        assert ultrasonic_bulk_modulus(rho, L, delta_t) == pytest.approx(expected)

    def test_typical_mineral_oil(self):
        """c_s = 1350 m/s, rho = 870 kg/m³ → K ≈ 1.584 GPa."""
        rho = 870.0
        c_s = 1350.0
        L = 0.1
        delta_t = L / c_s
        result = ultrasonic_bulk_modulus(rho, L, delta_t)
        assert result == pytest.approx(rho * c_s**2, rel=1e-9)

    def test_zero_rho_raises(self):
        with pytest.raises(ValueError):
            ultrasonic_bulk_modulus(0.0, 0.05, 3.7e-5)

    def test_negative_rho_raises(self):
        with pytest.raises(ValueError):
            ultrasonic_bulk_modulus(-1.0, 0.05, 3.7e-5)

    def test_zero_L_raises(self):
        with pytest.raises(ValueError):
            ultrasonic_bulk_modulus(870.0, 0.0, 3.7e-5)

    def test_negative_L_raises(self):
        with pytest.raises(ValueError):
            ultrasonic_bulk_modulus(870.0, -0.05, 3.7e-5)

    def test_zero_delta_t_raises(self):
        with pytest.raises(ValueError):
            ultrasonic_bulk_modulus(870.0, 0.05, 0.0)

    def test_negative_delta_t_raises(self):
        with pytest.raises(ValueError):
            ultrasonic_bulk_modulus(870.0, 0.05, -1e-5)


class TestSoundSpeedFromTransit:
    def test_formula(self):
        L = 0.1
        delta_t = 7.4e-5
        expected = L / delta_t
        assert sound_speed_from_transit(L, delta_t) == pytest.approx(expected)

    def test_zero_L_raises(self):
        with pytest.raises(ValueError):
            sound_speed_from_transit(0.0, 7.4e-5)

    def test_negative_L_raises(self):
        with pytest.raises(ValueError):
            sound_speed_from_transit(-0.1, 7.4e-5)

    def test_zero_delta_t_raises(self):
        with pytest.raises(ValueError):
            sound_speed_from_transit(0.1, 0.0)

    def test_negative_delta_t_raises(self):
        with pytest.raises(ValueError):
            sound_speed_from_transit(0.1, -1e-5)


class TestRecuperationEfficiency:
    def test_zero_loss_gives_one(self):
        result = recuperation_efficiency(1.78e9, 1e-3, 0.0)
        assert result == pytest.approx(1.0)

    def test_large_loss_gives_near_zero(self):
        result = recuperation_efficiency(1.78e9, 1e-3, 1e15)
        assert result < 0.01

    def test_bounded_0_to_1(self):
        result = recuperation_efficiency(1.78e9, 1e-3, 500.0)
        assert 0.0 <= result <= 1.0

    def test_formula(self):
        K = 1.78e9
        V = 1e-3
        E_loss = 1000.0
        stored = K * V
        expected = stored / (stored + E_loss)
        assert recuperation_efficiency(K, V, E_loss) == pytest.approx(expected)

    def test_zero_K_eff_raises(self):
        with pytest.raises(ValueError):
            recuperation_efficiency(0.0, 1e-3, 500.0)

    def test_zero_V_acc_raises(self):
        with pytest.raises(ValueError):
            recuperation_efficiency(1.78e9, 0.0, 500.0)

    def test_negative_E_loss_raises(self):
        with pytest.raises(ValueError):
            recuperation_efficiency(1.78e9, 1e-3, -1.0)


class TestOilCondition:
    """Tests für das Ampel-Konditionsmodell."""

    # Frischöl-Referenzwerte
    K_FRESH = 1.58e9
    ETA_FRESH = 0.046
    TAN_FRESH = 0.2

    def test_green_all_ok(self):
        result = OilCondition.evaluate(
            K_eff=self.K_FRESH * 0.95,
            K_eff_fresh=self.K_FRESH,
            eta=self.ETA_FRESH * 1.10,
            eta_fresh=self.ETA_FRESH,
            tan=0.5,
        )
        assert result == OilCondition.GREEN

    def test_yellow_low_K_eff(self):
        """K_eff < 90 % von K_fresh → YELLOW."""
        result = OilCondition.evaluate(
            K_eff=self.K_FRESH * 0.88,
            K_eff_fresh=self.K_FRESH,
            eta=self.ETA_FRESH,
            eta_fresh=self.ETA_FRESH,
            tan=0.5,
        )
        assert result == OilCondition.YELLOW

    def test_yellow_high_tan(self):
        """TAN ≥ 1.5 → YELLOW."""
        result = OilCondition.evaluate(
            K_eff=self.K_FRESH,
            K_eff_fresh=self.K_FRESH,
            eta=self.ETA_FRESH,
            eta_fresh=self.ETA_FRESH,
            tan=1.5,
        )
        assert result == OilCondition.YELLOW

    def test_yellow_viscosity_change(self):
        """Δη/η_0 > 15 % → YELLOW."""
        result = OilCondition.evaluate(
            K_eff=self.K_FRESH,
            K_eff_fresh=self.K_FRESH,
            eta=self.ETA_FRESH * 1.20,
            eta_fresh=self.ETA_FRESH,
            tan=0.5,
        )
        assert result == OilCondition.YELLOW

    def test_red_very_low_K_eff(self):
        """K_eff < 80 % → RED."""
        result = OilCondition.evaluate(
            K_eff=self.K_FRESH * 0.75,
            K_eff_fresh=self.K_FRESH,
            eta=self.ETA_FRESH,
            eta_fresh=self.ETA_FRESH,
            tan=0.5,
        )
        assert result == OilCondition.RED

    def test_red_tan_above_2(self):
        """TAN ≥ 2.0 → RED."""
        result = OilCondition.evaluate(
            K_eff=self.K_FRESH,
            K_eff_fresh=self.K_FRESH,
            eta=self.ETA_FRESH,
            eta_fresh=self.ETA_FRESH,
            tan=2.1,
        )
        assert result == OilCondition.RED

    def test_red_eta_change_above_25(self):
        """Δη > 25 % → RED."""
        result = OilCondition.evaluate(
            K_eff=self.K_FRESH,
            K_eff_fresh=self.K_FRESH,
            eta=self.ETA_FRESH * 1.30,
            eta_fresh=self.ETA_FRESH,
            tan=0.5,
        )
        assert result == OilCondition.RED

    def test_zero_K_fresh_raises(self):
        with pytest.raises(ValueError):
            OilCondition.evaluate(
                K_eff=self.K_FRESH,
                K_eff_fresh=0.0,
                eta=self.ETA_FRESH,
                eta_fresh=self.ETA_FRESH,
                tan=0.5,
            )

    def test_zero_eta_fresh_raises(self):
        with pytest.raises(ValueError):
            OilCondition.evaluate(
                K_eff=self.K_FRESH,
                K_eff_fresh=self.K_FRESH,
                eta=self.ETA_FRESH,
                eta_fresh=0.0,
                tan=0.5,
            )

    def test_negative_K_fresh_raises(self):
        with pytest.raises(ValueError):
            OilCondition.evaluate(
                K_eff=self.K_FRESH,
                K_eff_fresh=-1.0,
                eta=self.ETA_FRESH,
                eta_fresh=self.ETA_FRESH,
                tan=0.5,
            )

    def test_red_beats_yellow(self):
        """Wenn RED-Kriterium erfüllt, soll RED zurückgegeben werden, nicht YELLOW."""
        # K_eff/K_fresh < 0.80 → RED (überschreibt auch YELLOW via TAN)
        result = OilCondition.evaluate(
            K_eff=self.K_FRESH * 0.70,
            K_eff_fresh=self.K_FRESH,
            eta=self.ETA_FRESH,
            eta_fresh=self.ETA_FRESH,
            tan=1.6,   # würde nur YELLOW ergeben
        )
        assert result == OilCondition.RED
