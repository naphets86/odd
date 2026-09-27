"""Tests for liionp.thermal."""
import pytest

from liionp.thermal import ThermalModel


class TestHeatGeneration:
    def test_joule_heating_only(self):
        # dE_dT = 0 → only Joule heating
        model = ThermalModel(dE_dT=0.0)
        q = model.heat_generation(I=2.0, R_int=0.05, T=298.0)
        assert q == pytest.approx(2.0**2 * 0.05)  # 0.2 W

    def test_entropic_term_present(self):
        model = ThermalModel(dE_dT=-1e-4)
        q_with = model.heat_generation(I=2.0, R_int=0.05, T=298.0)
        q_without = ThermalModel(dE_dT=0.0).heat_generation(I=2.0, R_int=0.05, T=298.0)
        assert q_with != pytest.approx(q_without)

    def test_zero_current_zero_heating(self):
        model = ThermalModel(dE_dT=0.0)
        assert model.heat_generation(I=0.0, R_int=0.05, T=298.0) == pytest.approx(0.0)

    def test_joule_heating_scales_with_R(self):
        model = ThermalModel(dE_dT=0.0)
        q1 = model.heat_generation(I=1.0, R_int=0.05, T=298.0)
        q2 = model.heat_generation(I=1.0, R_int=0.10, T=298.0)
        assert q2 == pytest.approx(2.0 * q1)


class TestTemperatureDerivative:
    def test_positive_current_heats_cell(self):
        model = ThermalModel(dE_dT=0.0)
        dT = model.temperature_derivative(T=298.0, T_amb=298.0, I=5.0, R_int=0.05)
        assert dT > 0.0

    def test_equilibrium_at_ambient(self):
        # At equilibrium dT/dt = 0 → heat generated = heat lost
        model = ThermalModel(m_cell_cp=45.0, R_th=3.5, dE_dT=0.0)
        # With no current and T == T_amb, dT/dt = 0
        dT = model.temperature_derivative(T=298.0, T_amb=298.0, I=0.0, R_int=0.05)
        assert dT == pytest.approx(0.0)

    def test_cooling_when_above_ambient(self):
        model = ThermalModel(dE_dT=0.0)
        dT = model.temperature_derivative(T=320.0, T_amb=298.0, I=0.0, R_int=0.05)
        assert dT < 0.0


class TestThermalModelStep:
    def test_temperature_increases_under_load(self):
        model = ThermalModel(dE_dT=0.0)
        T_new = model.step(T=298.0, T_amb=298.0, I=5.0, R_int=0.05, dt=10.0)
        assert T_new > 298.0

    def test_temperature_decays_to_ambient(self):
        model = ThermalModel(dE_dT=0.0)
        # Hot cell, no current → should cool
        T_new = model.step(T=330.0, T_amb=298.0, I=0.0, R_int=0.05, dt=10.0)
        assert T_new < 330.0

    def test_euler_step_correctness(self):
        model = ThermalModel(m_cell_cp=45.0, R_th=3.5, dE_dT=0.0)
        T = 298.0
        T_amb = 298.0
        I = 2.0
        R_int = 0.05
        dt = 1.0
        dT_expected = model.temperature_derivative(T, T_amb, I, R_int)
        T_expected = T + dT_expected * dt
        assert model.step(T, T_amb, I, R_int, dt) == pytest.approx(T_expected)

    def test_default_parameters(self):
        model = ThermalModel()
        assert model.m_cell_cp == 45.0
        assert model.R_th == 3.5
        assert model.dE_dT == pytest.approx(-1e-4)
