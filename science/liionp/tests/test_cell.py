"""Tests for liionp.cell."""
import pytest

from liionp.cell import CellModel, CellParameters, CellState
from liionp.constants import D_EOL, V_MAX_DEFAULT, V_MIN_DEFAULT, V_NOM_DEFAULT


class TestCellParameters:
    def test_defaults(self):
        p = CellParameters()
        assert p.Q0 == 3.0
        assert p.V_max == V_MAX_DEFAULT
        assert p.V_min == V_MIN_DEFAULT
        assert p.D_EoL == D_EOL

    def test_custom(self):
        p = CellParameters(Q0=5.0, V_max=4.35)
        assert p.Q0 == 5.0
        assert p.V_max == 4.35


class TestCellState:
    def test_initial_fresh(self):
        p = CellParameters()
        s = CellState.initial(p, T0=298.0)
        assert s.Q == p.Q0
        assert s.T == 298.0
        assert s.V == p.V_nom
        assert s.SoC == 0.5
        assert s.degradation == 0.0
        assert s.delta_SEI > 0.0

    def test_initial_custom_temperature(self):
        p = CellParameters()
        s = CellState.initial(p, T0=310.0)
        assert s.T == 310.0

    def test_is_eol_false_when_fresh(self):
        p = CellParameters()
        s = CellState.initial(p)
        assert not s.is_eol

    def test_is_eol_true_at_threshold(self):
        p = CellParameters()
        s = CellState.initial(p)
        s.degradation = D_EOL
        assert s.is_eol

    def test_is_eol_true_above_threshold(self):
        p = CellParameters()
        s = CellState.initial(p)
        s.degradation = 0.25
        assert s.is_eol


class TestCellModelVoltage:
    def test_ocv_at_zero_soc(self):
        model = CellModel()
        ocv = model.open_circuit_voltage(0.0)
        assert ocv == pytest.approx(model.params.V_min)

    def test_ocv_at_full_soc(self):
        model = CellModel()
        ocv = model.open_circuit_voltage(1.0)
        assert ocv == pytest.approx(model.params.V_max)

    def test_ocv_midpoint(self):
        model = CellModel()
        ocv = model.open_circuit_voltage(0.5)
        expected = (model.params.V_min + model.params.V_max) / 2.0
        assert ocv == pytest.approx(expected)

    def test_terminal_voltage_discharge_lower(self):
        model = CellModel()
        V_term = model.terminal_voltage(0.5, I=1.0)  # discharge
        OCV = model.open_circuit_voltage(0.5)
        assert V_term < OCV

    def test_terminal_voltage_charge_higher(self):
        model = CellModel()
        V_term = model.terminal_voltage(0.5, I=-1.0)  # charge
        OCV = model.open_circuit_voltage(0.5)
        assert V_term > OCV

    def test_default_params(self):
        model = CellModel()
        assert model.params.Q0 == 3.0

    def test_custom_params(self):
        p = CellParameters(Q0=5.0)
        model = CellModel(p)
        assert model.params.Q0 == 5.0


class TestCellModelStep:
    def test_returns_state_and_rate(self):
        model = CellModel()
        state = CellState.initial(model.params)
        new_state, rate = model.step(state, I=1.0, T_amb=298.0, dt=1.0)
        assert isinstance(new_state, CellState)
        assert isinstance(rate, float)

    def test_degradation_accumulates(self):
        model = CellModel()
        state = CellState.initial(model.params)
        state, _ = model.step(state, I=1.0, T_amb=298.0, dt=3600.0)
        assert state.degradation > 0.0

    def test_temperature_increases_under_load(self):
        model = CellModel()
        state = CellState.initial(model.params, T0=298.0)
        new_state, _ = model.step(state, I=5.0, T_amb=298.0, dt=60.0)
        assert new_state.T > 298.0

    def test_soc_decreases_on_discharge(self):
        model = CellModel()
        state = CellState.initial(model.params)
        new_state, _ = model.step(state, I=1.0, T_amb=298.0, dt=60.0)
        assert new_state.SoC < state.SoC

    def test_soc_increases_on_charge(self):
        model = CellModel()
        state = CellState.initial(model.params)
        new_state, _ = model.step(state, I=-1.0, T_amb=298.0, dt=60.0)
        assert new_state.SoC > state.SoC

    def test_soc_clamped_at_zero(self):
        model = CellModel()
        state = CellState.initial(model.params)
        state.SoC = 0.001
        # Large discharge
        new_state, _ = model.step(state, I=100.0, T_amb=298.0, dt=3600.0)
        assert new_state.SoC >= 0.0

    def test_soc_clamped_at_one(self):
        model = CellModel()
        state = CellState.initial(model.params)
        state.SoC = 0.999
        # Large charge
        new_state, _ = model.step(state, I=-100.0, T_amb=298.0, dt=3600.0)
        assert new_state.SoC <= 1.0

    def test_capacity_decreases_over_time(self):
        model = CellModel()
        state = CellState.initial(model.params)
        for _ in range(10):
            state, _ = model.step(state, I=1.0, T_amb=298.0, dt=3600.0)
        assert state.Q <= model.params.Q0

    def test_internal_resistance_grows_with_degradation(self):
        model = CellModel()
        state = CellState.initial(model.params)
        state.degradation = 0.10
        new_state, _ = model.step(state, I=1.0, T_amb=298.0, dt=1.0)
        assert new_state.R_int > model.params.R_int
