"""Tests für src.logic, src.matrix, src.droplet, src.memory,
src.architecture, src.simulation, src.fabrication"""

import math
import pytest

# ============================================================
# LOGIC
# ============================================================
from src.logic import DropletGate, GateType, LogicNetwork, TruthTable
from src.utils import ValidationError


class TestDropletGate:
    def test_or_gate(self):
        g = DropletGate(GateType.OR, "g1", ["A", "B"], "Y")
        assert g.evaluate({"A": False, "B": False}) is False
        assert g.evaluate({"A": True,  "B": False}) is True
        assert g.evaluate({"A": False, "B": True})  is True
        assert g.evaluate({"A": True,  "B": True})  is True

    def test_and_gate(self):
        g = DropletGate(GateType.AND, "g2", ["A", "B"], "Y")
        assert g.evaluate({"A": False, "B": False}) is False
        assert g.evaluate({"A": True,  "B": False}) is False
        assert g.evaluate({"A": False, "B": True})  is False
        assert g.evaluate({"A": True,  "B": True})  is True

    def test_not_gate(self):
        g = DropletGate(GateType.NOT, "g3", ["A"], "nA")
        assert g.evaluate({"A": False}) is True
        assert g.evaluate({"A": True})  is False

    def test_buffer_gate(self):
        g = DropletGate(GateType.BUFFER, "g4", ["A"], "Y")
        assert g.evaluate({"A": True})  is True
        assert g.evaluate({"A": False}) is False

    def test_nand_gate(self):
        g = DropletGate(GateType.NAND, "g5", ["A", "B"], "Y")
        assert g.evaluate({"A": True,  "B": True})  is False
        assert g.evaluate({"A": True,  "B": False}) is True
        assert g.evaluate({"A": False, "B": True})  is True
        assert g.evaluate({"A": False, "B": False}) is True

    def test_nor_gate(self):
        g = DropletGate(GateType.NOR, "g6", ["A", "B"], "Y")
        assert g.evaluate({"A": False, "B": False}) is True
        assert g.evaluate({"A": True,  "B": False}) is False

    def test_xor_gate(self):
        g = DropletGate(GateType.XOR, "g7", ["A", "B"], "Y")
        assert g.evaluate({"A": False, "B": False}) is False
        assert g.evaluate({"A": True,  "B": False}) is True
        assert g.evaluate({"A": True,  "B": True})  is False

    def test_xnor_gate(self):
        g = DropletGate(GateType.XNOR, "g8", ["A", "B"], "Y")
        assert g.evaluate({"A": True,  "B": True})  is True
        assert g.evaluate({"A": True,  "B": False}) is False

    def test_not_requires_exactly_one_input(self):
        with pytest.raises(ValidationError):
            DropletGate(GateType.NOT, "g", ["A", "B"], "Y")

    def test_buffer_requires_exactly_one_input(self):
        with pytest.raises(ValidationError):
            DropletGate(GateType.BUFFER, "g", ["A", "B"], "Y")

    def test_binary_requires_two_inputs(self):
        with pytest.raises(ValidationError):
            DropletGate(GateType.AND, "g", ["A"], "Y")

    def test_num_inputs(self):
        g = DropletGate(GateType.OR, "g", ["A", "B", "C"], "Y")
        assert g.num_inputs == 3

    def test_is_universal_nand(self):
        g = DropletGate(GateType.NAND, "g", ["A", "B"], "Y")
        assert g.is_universal is True

    def test_is_universal_nor(self):
        g = DropletGate(GateType.NOR, "g", ["A", "B"], "Y")
        assert g.is_universal is True

    def test_is_not_universal_and(self):
        g = DropletGate(GateType.AND, "g", ["A", "B"], "Y")
        assert g.is_universal is False

    def test_energy_model_or_passive(self):
        g = DropletGate(GateType.OR, "g", ["A", "B"], "Y")
        assert "passiv" in g.energy_model.lower()

    def test_energy_model_and_active(self):
        g = DropletGate(GateType.AND, "g", ["A", "B"], "Y")
        assert "aktiv" in g.energy_model.lower()

    def test_three_input_or(self):
        g = DropletGate(GateType.OR, "g", ["A", "B", "C"], "Y")
        assert g.evaluate({"A": False, "B": False, "C": True}) is True
        assert g.evaluate({"A": False, "B": False, "C": False}) is False

    def test_propagation_delay_default(self):
        g = DropletGate(GateType.AND, "g", ["A", "B"], "Y")
        assert g.propagation_delay_s == pytest.approx(0.01)


class TestLogicNetwork:
    @pytest.fixture
    def nand_network(self):
        net = LogicNetwork("NAND-Test")
        net.declare_primary_input("A")
        net.declare_primary_input("B")
        g = DropletGate(GateType.NAND, "nand1", ["A", "B"], "Y")
        net.add_gate(g)
        return net

    def test_add_gate(self, nand_network):
        assert nand_network.num_gates == 1

    def test_duplicate_gate_raises(self, nand_network):
        g2 = DropletGate(GateType.AND, "nand1", ["A", "B"], "Z")
        with pytest.raises(ValueError):
            nand_network.add_gate(g2)

    def test_primary_inputs(self, nand_network):
        assert "A" in nand_network.primary_inputs
        assert "B" in nand_network.primary_inputs

    def test_evaluate_nand(self, nand_network):
        result = nand_network.evaluate({"A": True, "B": True})
        assert result["Y"] is False
        result = nand_network.evaluate({"A": True, "B": False})
        assert result["Y"] is True

    def test_evaluate_chain(self):
        """A → NOT → BUFFER → Y"""
        net = LogicNetwork()
        net.add_gate(DropletGate(GateType.NOT, "not1", ["A"], "nA"))
        net.add_gate(DropletGate(GateType.BUFFER, "buf1", ["nA"], "Y"))
        result = net.evaluate({"A": True})
        assert result["Y"] is False
        result = net.evaluate({"A": False})
        assert result["Y"] is True

    def test_evaluate_missing_input_raises(self, nand_network):
        with pytest.raises(ValueError, match="Unaufgelöste Signale"):
            nand_network.evaluate({"A": True})

    def test_cycle_detection(self):
        net = LogicNetwork()
        # Zyklus: A → g1 → B → g2 → A
        net.add_gate(DropletGate(GateType.NOT, "g1", ["X"], "Y"))
        net.add_gate(DropletGate(GateType.NOT, "g2", ["Y"], "X"))
        with pytest.raises(ValueError):
            net.evaluate({"Z": True})

    def test_is_turing_complete(self, nand_network):
        assert nand_network.is_turing_complete() is True

    def test_not_turing_complete(self):
        net = LogicNetwork()
        net.add_gate(DropletGate(GateType.AND, "g", ["A", "B"], "Y"))
        assert net.is_turing_complete() is False

    def test_total_propagation_delay(self, nand_network):
        assert nand_network.total_propagation_delay_s > 0

    def test_empty_network_delay(self):
        net = LogicNetwork()
        assert net.total_propagation_delay_s == 0.0

    def test_gates_property(self, nand_network):
        gates = nand_network.gates
        assert "nand1" in gates

    def test_or_gate_count(self):
        net = LogicNetwork()
        net.add_gate(DropletGate(GateType.OR, "o1", ["A", "B"], "Y1"))
        net.add_gate(DropletGate(GateType.OR, "o2", ["A", "B"], "Y2"))
        net.add_gate(DropletGate(GateType.AND, "a1", ["A", "B"], "Y3"))
        assert net.or_gate_count == 2


class TestTruthTable:
    @pytest.fixture
    def nand_table(self):
        net = LogicNetwork()
        net.add_gate(DropletGate(GateType.NAND, "n", ["A", "B"], "Y"))
        return TruthTable.generate(net, ["A", "B"], ["Y"])

    def test_num_rows(self, nand_table):
        assert nand_table.num_rows == 4

    def test_nand_all_true(self, nand_table):
        # Letzte Zeile: A=1, B=1 → Y=0
        last_inp, last_out = nand_table.rows[-1]
        assert last_inp == {"A": True, "B": True}
        assert last_out["Y"] is False

    def test_format_contains_header(self, nand_table):
        s = nand_table.format()
        assert "A" in s
        assert "B" in s
        assert "Y" in s

    def test_format_has_correct_number_of_rows(self, nand_table):
        lines = nand_table.format().split("\n")
        # header + sep + 4 data rows
        assert len(lines) == 6


# ============================================================
# MATRIX
# ============================================================
from src.matrix import ElectrodeMatrix, ElectrodeState, MatrixConfig


class TestMatrixConfig:
    def test_basic(self):
        cfg = MatrixConfig(4, 4, 5e-6, 4e-6, 50e-6)
        assert cfg.num_electrodes == 16
        assert cfg.chip_area_m2 == pytest.approx((4 * 5e-6) ** 2)

    def test_3d(self):
        cfg = MatrixConfig(4, 4, 5e-6, 4e-6, 50e-6, num_layers=3)
        assert cfg.num_electrodes == 48

    def test_electrode_density(self):
        cfg = MatrixConfig(10, 10, 5e-6, 4e-6, 50e-6)
        assert cfg.electrode_density_per_m2 > 0

    def test_equivalent_transistor_density(self):
        cfg = MatrixConfig(10, 10, 5e-6, 4e-6, 50e-6)
        assert cfg.equivalent_transistor_density_per_m2 == pytest.approx(
            2 * cfg.electrode_density_per_m2
        )

    def test_invalid_rows(self):
        with pytest.raises(ValidationError):
            MatrixConfig(0, 4, 5e-6, 4e-6, 50e-6)

    def test_invalid_pitch(self):
        with pytest.raises(ValidationError):
            MatrixConfig(4, 4, 0.0, 4e-6, 50e-6)

    def test_size_equals_pitch_raises(self):
        with pytest.raises(ValidationError):
            MatrixConfig(4, 4, 5e-6, 5e-6, 50e-6)

    def test_invalid_layers(self):
        with pytest.raises(ValidationError):
            MatrixConfig(4, 4, 5e-6, 4e-6, 50e-6, num_layers=0)

    def test_invalid_channel_height(self):
        with pytest.raises(ValidationError):
            MatrixConfig(4, 4, 5e-6, 4e-6, 0.0)


class TestElectrodeMatrix:
    @pytest.fixture
    def matrix(self):
        cfg = MatrixConfig(4, 4, 5e-6, 4e-6, 50e-6)
        return ElectrodeMatrix(cfg)

    def test_initial_state_dry(self, matrix):
        assert matrix.get_state(0, 0, 0) == ElectrodeState.DRY

    def test_set_get_state(self, matrix):
        matrix.set_state(0, 1, 2, ElectrodeState.ON)
        assert matrix.get_state(0, 1, 2) == ElectrodeState.ON

    def test_set_get_voltage(self, matrix):
        matrix.set_voltage(0, 2, 3, 0.5)
        assert matrix.get_voltage(0, 2, 3) == pytest.approx(0.5)

    def test_invalid_coords(self, matrix):
        with pytest.raises(ValidationError):
            matrix.get_state(0, 99, 0)
        with pytest.raises(ValidationError):
            matrix.get_state(0, 0, 99)
        with pytest.raises(ValidationError):
            matrix.get_state(5, 0, 0)

    def test_apply_voltage_pattern(self, matrix):
        pattern = [[0.5] * 4 for _ in range(4)]
        matrix.apply_voltage_pattern(pattern)
        assert matrix.get_voltage(0, 0, 0) == pytest.approx(0.5)

    def test_apply_voltage_pattern_wrong_rows(self, matrix):
        with pytest.raises(ValidationError):
            matrix.apply_voltage_pattern([[0.1] * 4] * 3)

    def test_apply_voltage_pattern_wrong_cols(self, matrix):
        with pytest.raises(ValidationError):
            matrix.apply_voltage_pattern([[0.1] * 3] * 4)

    def test_reset_layer(self, matrix):
        matrix.set_voltage(0, 1, 1, 1.0)
        matrix.set_state(0, 1, 1, ElectrodeState.ON)
        matrix.reset_layer(0)
        assert matrix.get_voltage(0, 1, 1) == pytest.approx(0.0)
        assert matrix.get_state(0, 1, 1) == ElectrodeState.DRY

    def test_reset_all(self, matrix):
        matrix.set_voltage(0, 0, 0, 1.0)
        matrix.reset_all()
        assert matrix.get_voltage(0, 0, 0) == pytest.approx(0.0)

    def test_active_electrodes(self, matrix):
        matrix.set_state(0, 0, 0, ElectrodeState.ON)
        matrix.set_state(0, 1, 1, ElectrodeState.FLOODED)
        active = matrix.active_electrodes(0)
        assert (0, 0) in active
        assert (1, 1) in active

    def test_neighbors(self, matrix):
        n = matrix.neighbors(0, 1, 1)
        assert len(n) == 4

    def test_neighbors_corner(self, matrix):
        n = matrix.neighbors(0, 0, 0)
        assert len(n) == 2

    def test_count_state(self, matrix):
        matrix.set_state(0, 0, 0, ElectrodeState.ON)
        matrix.set_state(0, 1, 1, ElectrodeState.ON)
        assert matrix.count_state(ElectrodeState.ON) == 2

    def test_utilization_zero(self, matrix):
        assert matrix.utilization(0) == pytest.approx(0.0)

    def test_utilization_partial(self, matrix):
        matrix.set_state(0, 0, 0, ElectrodeState.ON)
        u = matrix.utilization(0)
        assert 0.0 < u <= 1.0

    def test_all_states(self, matrix):
        states = matrix.all_states(0)
        assert len(states) == 4
        assert len(states[0]) == 4


# ============================================================
# DROPLET
# ============================================================
from src.droplet import Droplet, DropletMerger, SelfOrganizer


class TestDroplet:
    @pytest.fixture
    def d(self):
        return Droplet(volume_m3=1e-15, position=(0, 0))

    def test_radius_positive(self, d):
        assert d.radius_m > 0

    def test_surface_area_positive(self, d):
        assert d.surface_area_m2 > 0

    def test_surface_energy_positive(self, d):
        assert d.surface_energy_J > 0

    def test_total_ion_content(self, d):
        assert d.total_ion_content_mol == pytest.approx(
            d.ion_concentration_mol_m3 * d.volume_m3
        )

    def test_laplace_pressure_positive(self, d):
        assert d.laplace_pressure_Pa() > 0

    def test_conductance_positive(self, d):
        g = d.conductance_S(1e-7, (20e-9) ** 2)
        assert g > 0

    def test_conductance_invalid_length(self, d):
        with pytest.raises(ValidationError):
            d.conductance_S(0.0, 1e-16)

    def test_conductance_invalid_area(self, d):
        with pytest.raises(ValidationError):
            d.conductance_S(1e-7, 0.0)

    def test_move_to(self, d):
        d2 = d.move_to((3, 5))
        assert d2.position == (3, 5)
        assert d2.volume_m3 == d.volume_m3

    def test_invalid_volume(self):
        with pytest.raises(ValidationError):
            Droplet(volume_m3=0.0, position=(0, 0))

    def test_invalid_concentration(self):
        with pytest.raises(ValidationError):
            Droplet(volume_m3=1e-15, position=(0, 0), ion_concentration_mol_m3=-1.0)

    def test_invalid_temperature(self):
        with pytest.raises(ValidationError):
            Droplet(volume_m3=1e-15, position=(0, 0), temperature_K=0.0)


class TestDropletMerger:
    @pytest.fixture
    def d1(self):
        return Droplet(1e-15, (0, 0), label="A")

    @pytest.fixture
    def d2(self):
        return Droplet(2e-15, (0, 1), label="B")

    def test_can_merge_adjacent(self, d1, d2):
        assert DropletMerger.can_merge(d1, d2)

    def test_cannot_merge_distant(self, d1):
        d_far = Droplet(1e-15, (5, 5))
        assert not DropletMerger.can_merge(d1, d_far)

    def test_merge_volume_conservation(self, d1, d2):
        m = DropletMerger.merge(d1, d2)
        assert m.volume_m3 == pytest.approx(3e-15, rel=1e-9)

    def test_merge_ion_conservation(self, d1, d2):
        m = DropletMerger.merge(d1, d2)
        expected_ions = d1.total_ion_content_mol + d2.total_ion_content_mol
        assert m.total_ion_content_mol == pytest.approx(expected_ions, rel=1e-9)

    def test_merge_label_combined(self, d1, d2):
        m = DropletMerger.merge(d1, d2)
        assert "A" in m.label
        assert "B" in m.label

    def test_merge_label_one_none(self, d1):
        d_no_label = Droplet(1e-15, (0, 1))
        m = DropletMerger.merge(d1, d_no_label)
        assert m.label == "A"

    def test_merge_label_both_none(self):
        d1 = Droplet(1e-15, (0, 0))
        d2 = Droplet(1e-15, (0, 1))
        m = DropletMerger.merge(d1, d2)
        assert m.label is None

    def test_merge_energy_released_positive(self, d1, d2):
        e = DropletMerger.merge_energy_released_J(d1, d2)
        assert e > 0


class TestSelfOrganizer:
    @pytest.fixture
    def org(self):
        return SelfOrganizer(threshold_voltage_V=0.12)

    def test_add_droplet(self, org):
        d = Droplet(1e-15, (0, 0))
        org.add_droplet(d)
        assert org.num_droplets == 1

    def test_remove_droplet(self, org):
        d = Droplet(1e-15, (0, 0))
        org.add_droplet(d)
        org.remove_droplet(d)
        assert org.num_droplets == 0

    def test_step_merges_adjacent(self, org):
        org.add_droplet(Droplet(1e-15, (0, 0)))
        org.add_droplet(Droplet(1e-15, (0, 1)))
        voltages = {(0, 0): 0.5, (0, 1): 0.5}
        result = org.step(voltages)
        # Beide nah beieinander → sollten zu einem verschmelzen
        assert len(result) <= 2

    def test_step_no_merge_distant(self, org):
        org.add_droplet(Droplet(1e-15, (0, 0)))
        org.add_droplet(Droplet(1e-15, (5, 5)))
        voltages = {(0, 0): 0.5, (5, 5): 0.5}
        result = org.step(voltages)
        assert len(result) == 2

    def test_relaxation_time(self, org):
        tau = org.relaxation_time_s(1e-6)
        assert tau == pytest.approx((1e-6) ** 2 / 1e-10)

    def test_relaxation_time_invalid(self, org):
        with pytest.raises(ValidationError):
            org.relaxation_time_s(0.0)

    def test_invalid_threshold(self):
        with pytest.raises(ValidationError):
            SelfOrganizer(threshold_voltage_V=0.0)

    def test_droplets_property(self, org):
        d = Droplet(1e-15, (0, 0))
        org.add_droplet(d)
        assert len(org.droplets) == 1


# ============================================================
# MEMORY
# ============================================================
from src.memory import IonicMemoryCell, MemoryArray, PinningWell


class TestPinningWell:
    @pytest.fixture
    def well(self):
        return PinningWell((0, 0), well_area_m2=25e-12, pinning_energy_J=1e-15)

    def test_initially_unoccupied(self, well):
        assert not well.is_occupied
        assert well.logical_value is False

    def test_write_and_read(self, well):
        d = Droplet(1e-15, (0, 0))
        well.write(d)
        assert well.is_occupied
        assert well.read() is d

    def test_erase(self, well):
        d = Droplet(1e-15, (0, 0))
        well.write(d)
        erased = well.erase()
        assert erased is d
        assert not well.is_occupied

    def test_write_none_clears(self, well):
        d = Droplet(1e-15, (0, 0))
        well.write(d)
        well.write(None)
        assert not well.is_occupied

    def test_can_hold_low_energy(self, well):
        d = Droplet(1e-15, (0, 0))
        assert well.can_hold(d, applied_energy_J=0.0)

    def test_cannot_hold_high_energy(self, well):
        d = Droplet(1e-15, (0, 0))
        assert not well.can_hold(d, applied_energy_J=1e-14)

    def test_invalid_area(self):
        with pytest.raises(ValidationError):
            PinningWell((0, 0), well_area_m2=0.0, pinning_energy_J=1e-15)

    def test_invalid_pinning_energy(self):
        with pytest.raises(ValidationError):
            PinningWell((0, 0), well_area_m2=1e-12, pinning_energy_J=-1e-15)

    def test_logical_value_true_when_occupied(self, well):
        well.write(Droplet(1e-15, (0, 0)))
        assert well.logical_value is True


class TestIonicMemoryCell:
    @pytest.fixture
    def cell(self):
        return IonicMemoryCell((0, 0), volume_m3=1e-15)

    def test_initial_concentration(self, cell):
        assert cell.concentration_mol_m3 == pytest.approx(100.0)

    def test_activate_increases_concentration(self, cell):
        c0 = cell.concentration_mol_m3
        c1 = cell.activate()
        assert c1 > c0

    def test_activation_count(self, cell):
        cell.activate()
        cell.activate()
        assert cell.activation_count == 2

    def test_decay_approaches_base(self, cell):
        cell.activate()
        c_before = cell.concentration_mol_m3
        for _ in range(100):
            cell.decay(0.1)
        assert cell.concentration_mol_m3 < c_before

    def test_decay_invalid_factor(self, cell):
        with pytest.raises(ValidationError):
            cell.decay(1.5)
        with pytest.raises(ValidationError):
            cell.decay(-0.1)

    def test_reset(self, cell):
        cell.activate()
        cell.reset()
        assert cell.concentration_mol_m3 == pytest.approx(100.0)
        assert cell.activation_count == 0

    def test_normalized_strength_zero(self, cell):
        assert cell.normalized_strength == pytest.approx(0.0)

    def test_normalized_strength_after_activation(self, cell):
        cell.activate()
        assert cell.normalized_strength > 0

    def test_write_analog(self, cell):
        cell.write_analog(500.0)
        assert cell.concentration_mol_m3 == pytest.approx(500.0)

    def test_write_analog_invalid(self, cell):
        with pytest.raises(ValidationError):
            cell.write_analog(-1.0)

    def test_invalid_volume(self):
        with pytest.raises(ValidationError):
            IonicMemoryCell((0, 0), volume_m3=0.0)

    def test_invalid_concentration(self):
        with pytest.raises(ValidationError):
            IonicMemoryCell((0, 0), volume_m3=1e-15, base_concentration_mol_m3=-1.0)

    def test_invalid_plasticity_rate(self):
        with pytest.raises(ValidationError):
            IonicMemoryCell((0, 0), volume_m3=1e-15, plasticity_rate=0.0)
        with pytest.raises(ValidationError):
            IonicMemoryCell((0, 0), volume_m3=1e-15, plasticity_rate=1.1)

    def test_normalized_strength_zero_base(self):
        cell = IonicMemoryCell((0, 0), volume_m3=1e-15, base_concentration_mol_m3=0.0,
                                plasticity_rate=0.5)
        assert cell.normalized_strength == pytest.approx(0.0)


class TestMemoryArray:
    @pytest.fixture
    def arr(self):
        return MemoryArray(rows=4, cols=8)

    def test_capacity(self, arr):
        assert arr.capacity_bits == 32

    def test_write_read_bit(self, arr):
        arr.write_bit(0, 0, True)
        assert arr.read_bit(0, 0) is True
        arr.write_bit(0, 0, False)
        assert arr.read_bit(0, 0) is False

    def test_write_read_word(self, arr):
        arr.write_word(0, 42)
        assert arr.read_word(0) == 42

    def test_write_read_word_zero(self, arr):
        arr.write_word(1, 0)
        assert arr.read_word(1) == 0

    def test_clear(self, arr):
        arr.write_word(0, 255)
        arr.clear()
        assert arr.read_word(0) == 0

    def test_count_ones(self, arr):
        arr.write_bit(0, 0, True)
        arr.write_bit(1, 1, True)
        assert arr.count_ones() == 2

    def test_invalid_row(self, arr):
        with pytest.raises(ValidationError):
            arr.read_bit(99, 0)

    def test_invalid_col(self, arr):
        with pytest.raises(ValidationError):
            arr.read_bit(0, 99)

    def test_invalid_write_row(self, arr):
        with pytest.raises(ValidationError):
            arr.write_word(-1, 0)

    def test_invalid_word_value(self, arr):
        with pytest.raises(ValidationError):
            arr.write_word(0, -1)

    def test_invalid_read_word_row(self, arr):
        with pytest.raises(ValidationError):
            arr.read_word(-1)

    def test_invalid_rows(self):
        with pytest.raises(ValidationError):
            MemoryArray(rows=0, cols=8)

    def test_invalid_cols(self):
        with pytest.raises(ValidationError):
            MemoryArray(rows=4, cols=0)


# ============================================================
# ARCHITECTURE
# ============================================================
from src.architecture import (
    HybridController,
    IRFSystem,
    ThroughputAnalyzer,
    VonNeumannBottleneck,
)
from src.matrix import ElectrodeMatrix, MatrixConfig


class TestVonNeumannBottleneck:
    @pytest.fixture
    def vn(self):
        return VonNeumannBottleneck(
            cpu_frequency_Hz=3e9,
            memory_bandwidth_bytes_s=50e9,
            word_size_bytes=8,
        )

    def test_max_accesses_per_cycle(self, vn):
        expected = (50e9 / 8) / 3e9
        assert vn.max_memory_accesses_per_cycle == pytest.approx(expected, rel=1e-6)

    def test_bottleneck_ratio(self, vn):
        assert vn.bottleneck_ratio > 0

    def test_cycles_stalled_no_stall(self, vn):
        # Wenige Zugriffe → kein Stall
        stall = vn.cycles_stalled_per_operation(1)
        assert stall == pytest.approx(0.0)

    def test_cycles_stalled_with_stall(self, vn):
        stall = vn.cycles_stalled_per_operation(1000)
        assert stall > 0

    def test_cycles_stalled_invalid(self, vn):
        with pytest.raises(ValidationError):
            vn.cycles_stalled_per_operation(-1)

    def test_invalid_cpu_freq(self):
        with pytest.raises(ValidationError):
            VonNeumannBottleneck(0.0, 50e9)

    def test_invalid_bandwidth(self):
        with pytest.raises(ValidationError):
            VonNeumannBottleneck(3e9, 0.0)

    def test_invalid_word_size(self):
        with pytest.raises(ValidationError):
            VonNeumannBottleneck(3e9, 50e9, word_size_bytes=0)


class TestHybridController:
    @pytest.fixture
    def controller(self):
        cfg = MatrixConfig(4, 4, 5e-6, 4e-6, 50e-6)
        mat = ElectrodeMatrix(cfg)
        return HybridController(
            cmos_frequency_Hz=1e9,
            electrode_matrix=mat,
            configuration_time_s=0.01,
        )

    def test_config_rate(self, controller):
        assert controller.config_rate_Hz == pytest.approx(100.0)

    def test_num_electrodes(self, controller):
        assert controller.num_electrodes == 16

    def test_parallel_ops(self, controller):
        assert controller.parallel_operations_per_config == 16

    def test_throughput_positive(self, controller):
        tp = controller.throughput_ops_per_second()
        assert tp > 0

    def test_throughput_invalid_utilization(self, controller):
        with pytest.raises(ValidationError):
            controller.throughput_ops_per_second(utilization=1.5)

    def test_energy_per_config(self, controller):
        e = controller.energy_per_config_J(0.12)
        assert e >= 0

    def test_energy_per_config_invalid(self, controller):
        with pytest.raises(ValidationError):
            controller.energy_per_config_J(0.12, switching_fraction=2.0)

    def test_invalid_cmos_freq(self):
        cfg = MatrixConfig(4, 4, 5e-6, 4e-6, 50e-6)
        with pytest.raises(ValidationError):
            HybridController(0.0, ElectrodeMatrix(cfg))

    def test_invalid_config_time(self):
        cfg = MatrixConfig(4, 4, 5e-6, 4e-6, 50e-6)
        with pytest.raises(ValidationError):
            HybridController(1e9, ElectrodeMatrix(cfg), configuration_time_s=0.0)


class TestThroughputAnalyzer:
    @pytest.fixture
    def analyzer(self):
        cfg = MatrixConfig(10, 10, 5e-6, 4e-6, 50e-6, num_layers=5)
        return ThroughputAnalyzer(cfg)

    def test_electrode_density_cm2(self, analyzer):
        assert analyzer.electrode_density_cm2 > 0

    def test_equivalent_transistor_density_cm2(self, analyzer):
        assert analyzer.equivalent_transistor_density_cm2 > analyzer.electrode_density_cm2

    def test_compare_to_cmos(self, analyzer):
        ratio = analyzer.compare_to_cmos_density(1e9)
        assert ratio > 0

    def test_compare_to_cmos_invalid(self, analyzer):
        with pytest.raises(ValidationError):
            analyzer.compare_to_cmos_density(0.0)

    def test_total_throughput(self, analyzer):
        tp = analyzer.total_throughput_ops_s(1000.0)
        assert tp > 0

    def test_total_throughput_invalid_freq(self, analyzer):
        with pytest.raises(ValidationError):
            analyzer.total_throughput_ops_s(0.0)

    def test_total_throughput_invalid_configs(self, analyzer):
        with pytest.raises(ValidationError):
            analyzer.total_throughput_ops_s(1000.0, configs_per_computation=0)

    def test_scaling_exponent(self, analyzer):
        assert analyzer.irf_scaling_exponent() == pytest.approx(3.0)

    def test_energy_efficiency(self, analyzer):
        eff = analyzer.energy_efficiency_ops_J(1000.0, 0.12)
        assert eff > 0

    def test_energy_efficiency_invalid_voltage(self, analyzer):
        with pytest.raises(ValidationError):
            analyzer.energy_efficiency_ops_J(1000.0, 0.0)


class TestIRFSystem:
    @pytest.fixture
    def system(self):
        cfg = MatrixConfig(4, 4, 5e-6, 4e-6, 50e-6)
        return IRFSystem(cfg, cmos_frequency_Hz=1e9, configuration_time_s=0.01)

    def test_initial_cycles(self, system):
        assert system.cycle_count == 0

    def test_execute_configuration(self, system):
        pattern = [[0.5] * 4 for _ in range(4)]
        metrics = system.execute_configuration(pattern)
        assert metrics["cycle"] == 1
        assert system.cycle_count == 1

    def test_reset(self, system):
        pattern = [[0.5] * 4 for _ in range(4)]
        system.execute_configuration(pattern)
        system.reset()
        assert system.cycle_count == 0

    def test_status_report(self, system):
        report = system.status_report()
        assert "matrix_rows" in report
        assert "total_electrodes" in report
        assert report["total_electrodes"] == 16

    def test_config_property(self, system):
        assert system.config.rows == 4


# ============================================================
# SIMULATION
# ============================================================
from src.simulation import SimulationConfig, SimulationResult, Simulator, TimeStep


class TestSimulationConfig:
    def test_num_steps(self):
        cfg = SimulationConfig(total_time_s=1.0, time_step_s=0.1)
        assert cfg.num_steps == 10

    def test_invalid_total_time(self):
        with pytest.raises(ValidationError):
            SimulationConfig(total_time_s=0.0, time_step_s=0.1)

    def test_invalid_time_step(self):
        with pytest.raises(ValidationError):
            SimulationConfig(total_time_s=1.0, time_step_s=0.0)

    def test_time_step_exceeds_total(self):
        with pytest.raises(ValidationError):
            SimulationConfig(total_time_s=0.1, time_step_s=1.0)

    def test_invalid_threshold_voltage(self):
        with pytest.raises(ValidationError):
            SimulationConfig(total_time_s=1.0, time_step_s=0.1, threshold_voltage_V=0.0)


class TestSimulationResult:
    @pytest.fixture
    def result(self):
        cfg = SimulationConfig(total_time_s=0.3, time_step_s=0.1)
        r = SimulationResult(config=cfg)
        r.steps.append(TimeStep(0, 0.0, 3, 10, 1e-16, merges=1, utilization=0.5))
        r.steps.append(TimeStep(1, 0.1, 2, 8, 2e-16, merges=0, utilization=0.4))
        r.steps.append(TimeStep(2, 0.2, 2, 8, 1e-16, merges=0, utilization=0.3))
        return r

    def test_total_energy(self, result):
        assert result.total_energy_J == pytest.approx(4e-16)

    def test_total_merges(self, result):
        assert result.total_merges == 1

    def test_mean_utilization(self, result):
        assert result.mean_utilization == pytest.approx((0.5 + 0.4 + 0.3) / 3)

    def test_num_steps_recorded(self, result):
        assert result.num_steps_recorded == 3

    def test_peak_droplet_count(self, result):
        assert result.peak_droplet_count() == 3

    def test_time_series(self, result):
        ts = result.time_series("num_droplets")
        assert len(ts) == 3
        assert ts[0] == (0.0, 3.0)

    def test_time_series_empty(self):
        cfg = SimulationConfig(total_time_s=0.1, time_step_s=0.01)
        r = SimulationResult(config=cfg)
        assert r.mean_utilization == pytest.approx(0.0)
        assert r.peak_droplet_count() == 0

    def test_time_series_unknown_metric(self, result):
        ts = result.time_series("does_not_exist")
        assert ts == []


class TestSimulator:
    @pytest.fixture
    def sim(self):
        cfg = MatrixConfig(4, 4, 5e-6, 4e-6, 50e-6)
        system = IRFSystem(cfg)
        sim_cfg = SimulationConfig(total_time_s=0.03, time_step_s=0.01)
        return Simulator(system, sim_cfg)

    def test_run_returns_result(self, sim):
        schedule = lambda t: {(0, 0): 0.5, (1, 1): 0.5}
        result = sim.run(schedule)
        assert result is not None
        assert result.num_steps_recorded == 3

    def test_add_droplet(self, sim):
        from src.droplet import Droplet
        sim.add_droplet(Droplet(1e-15, (0, 0)))
        schedule = lambda t: {}
        result = sim.run(schedule)
        assert len(result.final_droplets) >= 0

    def test_result_property(self, sim):
        assert sim.result is None
        schedule = lambda t: {}
        sim.run(schedule)
        assert sim.result is not None

    def test_reset(self, sim):
        schedule = lambda t: {}
        sim.run(schedule)
        sim.reset()
        assert sim.result is None


# ============================================================
# FABRICATION
# ============================================================
from src.fabrication import (
    FabricationProcess,
    ProcessStep,
    ProcessValidator,
    TECHNOLOGY_PARAMETERS,
    TechnologyNode,
)


class TestTechnologyNode:
    def test_all_nodes_have_parameters(self):
        for node in TechnologyNode:
            assert node in TECHNOLOGY_PARAMETERS
            params = TECHNOLOGY_PARAMETERS[node]
            assert "electrode_pitch_m" in params
            assert "switching_voltage_V" in params


class TestProcessStep:
    def test_valid_step(self):
        s = ProcessStep(1, "Test", "Beschreibung", 298.15, 3600.0)
        assert s.step_number == 1

    def test_invalid_step_number(self):
        with pytest.raises(ValidationError):
            ProcessStep(0, "X", "X", 298.15, 3600.0)

    def test_invalid_temperature(self):
        with pytest.raises(ValidationError):
            ProcessStep(1, "X", "X", 0.0, 3600.0)

    def test_invalid_duration(self):
        with pytest.raises(ValidationError):
            ProcessStep(1, "X", "X", 298.15, 0.0)

    def test_invalid_yield_zero(self):
        with pytest.raises(ValidationError):
            ProcessStep(1, "X", "X", 298.15, 3600.0, yield_factor=0.0)

    def test_invalid_yield_above_one(self):
        with pytest.raises(ValidationError):
            ProcessStep(1, "X", "X", 298.15, 3600.0, yield_factor=1.1)


class TestProcessValidator:
    def test_valid_dielectric(self):
        w = ProcessValidator.validate_dielectric(10e-9, 9.0)
        assert isinstance(w, list)

    def test_thin_dielectric_warning(self):
        w = ProcessValidator.validate_dielectric(1e-9, 9.0)
        assert len(w) > 0

    def test_thick_dielectric_warning(self):
        w = ProcessValidator.validate_dielectric(200e-9, 9.0)
        assert len(w) > 0

    def test_invalid_epsilon_r(self):
        w = ProcessValidator.validate_dielectric(10e-9, 0.5)
        assert len(w) > 0

    def test_high_epsilon_r_warning(self):
        w = ProcessValidator.validate_dielectric(10e-9, 150.0)
        assert len(w) > 0

    def test_valid_electrode(self):
        w = ProcessValidator.validate_electrode(5e-6, 4e-6)
        assert w == []

    def test_electrode_too_small_pitch(self):
        w = ProcessValidator.validate_electrode(10e-9, 5e-9)
        assert len(w) > 0

    def test_electrode_size_ge_pitch(self):
        w = ProcessValidator.validate_electrode(5e-6, 5e-6)
        assert len(w) > 0

    def test_electrode_size_zero(self):
        w = ProcessValidator.validate_electrode(5e-6, 0.0)
        assert len(w) > 0

    def test_valid_voltage(self):
        w = ProcessValidator.validate_voltage(0.12)
        assert w == []

    def test_high_voltage_warning(self):
        w = ProcessValidator.validate_voltage(10.0)
        assert len(w) > 0

    def test_low_voltage_warning(self):
        w = ProcessValidator.validate_voltage(0.001)
        assert len(w) > 0

    def test_validate_technology_node(self):
        w = ProcessValidator.validate_technology_node(TechnologyNode.IRF_1)
        assert isinstance(w, list)


class TestFabricationProcess:
    @pytest.fixture
    def proc(self):
        return FabricationProcess(TechnologyNode.IRF_1)

    def test_num_steps(self, proc):
        assert proc.num_steps == 7

    def test_total_duration(self, proc):
        assert proc.total_duration_s > 0

    def test_total_yield(self, proc):
        assert 0.9 < proc.total_yield <= 1.0

    def test_execute_step(self, proc):
        result = proc.execute_step(1)
        assert result["step_number"] == 1
        assert result["status"] == "completed"

    def test_execute_all(self, proc):
        results = proc.execute_all()
        assert len(results) == 7

    def test_invalid_step(self, proc):
        with pytest.raises(ValidationError):
            proc.execute_step(99)

    def test_validate_returns_list(self, proc):
        assert isinstance(proc.validate(), list)

    def test_technology_parameters(self, proc):
        params = proc.technology_parameters
        assert "electrode_pitch_m" in params

    def test_is_step_completed(self, proc):
        assert not proc.is_step_completed(1)
        proc.execute_step(1)
        assert proc.is_step_completed(1)

    def test_all_technology_nodes(self):
        for node in TechnologyNode:
            p = FabricationProcess(node)
            assert p.num_steps > 0


# ============================================================
# PACKAGE __init__ imports
# ============================================================
import src


class TestPackageImports:
    def test_all_symbols_importable(self):
        for sym in src.__all__:
            assert hasattr(src, sym), f"Symbol '{sym}' fehlt im Package"

    def test_version(self):
        assert src.__version__ == "1.0.0"

    def test_author(self):
        assert src.__author__ == "Stephan Epp"
