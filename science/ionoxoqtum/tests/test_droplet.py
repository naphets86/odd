"""
tests/test_droplet.py
=====================
Tests für core/droplet.py – Tropfenlogik und boolesche Gatter
"""

import pytest
import numpy as np
from src.droplet import (
    Droplet,
    NANDGate, NORGate, NOTGate, ANDGate, ORGate, XORGate,
    HalfAdder, FullAdder, RippleCarryAdder,
    IRFNode, IRFNetwork,
    LogicState,
)


# ── Droplet ───────────────────────────────────────────────────────────────────

class TestDroplet:

    def test_volume_formula(self):
        d = Droplet((0, 0), radius=1e-6)
        expected = (4.0 / 3.0) * np.pi * (1e-6)**3
        assert d.volume == pytest.approx(expected, rel=1e-10)

    def test_merge_radius_formula(self):
        """r_f = (r1³ + r2³)^(1/3)."""
        d1 = Droplet((0, 0), radius=1e-6)
        d2 = Droplet((5e-6, 0), radius=2e-6)
        r_f = d1.merge_radius(d2)
        expected = ((1e-6)**3 + (2e-6)**3) ** (1.0 / 3.0)
        assert r_f == pytest.approx(expected, rel=1e-10)

    def test_can_merge_always_true(self):
        """Satz 3.1: Verschmelzung immer thermodynamisch begünstigt."""
        d1 = Droplet((0, 0), radius=1e-6)
        d2 = Droplet((0, 0), radius=1e-6)
        assert d1.can_merge(d2) is True

    def test_can_merge_unequal_radii(self):
        d1 = Droplet((0, 0), radius=1e-7)
        d2 = Droplet((0, 0), radius=5e-6)
        assert d1.can_merge(d2) is True

    def test_energy_released_positive(self):
        d1 = Droplet((0, 0), radius=1e-6)
        d2 = Droplet((0, 0), radius=1e-6)
        E = d1.energy_released_on_merge(d2)
        assert E > 0

    def test_merged_radius_smaller_than_sum(self):
        """r_f < r1 + r2 (Kugeloberfläche wird reduziert)."""
        d1 = Droplet((0, 0), radius=1e-6)
        d2 = Droplet((0, 0), radius=1e-6)
        r_f = d1.merge_radius(d2)
        assert r_f < d1.radius + d2.radius

    def test_merged_volume_conserved(self):
        """Volumen bleibt erhalten: V_f = V1 + V2."""
        d1 = Droplet((0, 0), radius=1e-6)
        d2 = Droplet((0, 0), radius=2e-6)
        r_f = d1.merge_radius(d2)
        V_f = (4.0 / 3.0) * np.pi * r_f**3
        V_expected = d1.volume + d2.volume
        assert V_f == pytest.approx(V_expected, rel=1e-6)


# ── NAND-Gate ─────────────────────────────────────────────────────────────────

class TestNANDGate:

    def setup_method(self):
        self.gate = NANDGate()

    def test_00(self): assert self.gate.evaluate(0, 0) == 1
    def test_01(self): assert self.gate.evaluate(0, 1) == 1
    def test_10(self): assert self.gate.evaluate(1, 0) == 1
    def test_11(self): assert self.gate.evaluate(1, 1) == 0

    def test_truth_table_complete(self):
        tt = self.gate.truth_table()
        assert len(tt) == 4

    def test_functional_completeness_not(self):
        """NOT lässt sich aus NAND konstruieren."""
        nand = NANDGate()
        assert nand.evaluate(0, 0) == 1  # NOT 0 = 1
        assert nand.evaluate(1, 1) == 0  # NOT 1 = 0


# ── NOR-Gate ──────────────────────────────────────────────────────────────────

class TestNORGate:

    def setup_method(self):
        self.gate = NORGate()

    def test_00(self): assert self.gate.evaluate(0, 0) == 1
    def test_01(self): assert self.gate.evaluate(0, 1) == 0
    def test_10(self): assert self.gate.evaluate(1, 0) == 0
    def test_11(self): assert self.gate.evaluate(1, 1) == 0


# ── NOT-Gate ──────────────────────────────────────────────────────────────────

class TestNOTGate:

    def setup_method(self):
        self.gate = NOTGate()

    def test_not_0(self): assert self.gate.evaluate(0) == 1
    def test_not_1(self): assert self.gate.evaluate(1) == 0


# ── AND-Gate ──────────────────────────────────────────────────────────────────

class TestANDGate:

    def setup_method(self):
        self.gate = ANDGate()

    def test_00(self): assert self.gate.evaluate(0, 0) == 0
    def test_01(self): assert self.gate.evaluate(0, 1) == 0
    def test_10(self): assert self.gate.evaluate(1, 0) == 0
    def test_11(self): assert self.gate.evaluate(1, 1) == 1

    def test_commutative(self):
        assert self.gate.evaluate(1, 0) == self.gate.evaluate(0, 1)


# ── OR-Gate ───────────────────────────────────────────────────────────────────

class TestORGate:

    def setup_method(self):
        self.gate = ORGate()

    def test_00(self): assert self.gate.evaluate(0, 0) == 0
    def test_01(self): assert self.gate.evaluate(0, 1) == 1
    def test_10(self): assert self.gate.evaluate(1, 0) == 1
    def test_11(self): assert self.gate.evaluate(1, 1) == 1


# ── XOR-Gate ──────────────────────────────────────────────────────────────────

class TestXORGate:

    def setup_method(self):
        self.gate = XORGate()

    def test_00(self): assert self.gate.evaluate(0, 0) == 0
    def test_01(self): assert self.gate.evaluate(0, 1) == 1
    def test_10(self): assert self.gate.evaluate(1, 0) == 1
    def test_11(self): assert self.gate.evaluate(1, 1) == 0

    def test_commutative(self):
        assert self.gate.evaluate(1, 0) == self.gate.evaluate(0, 1)


# ── Halbaddierer ──────────────────────────────────────────────────────────────

class TestHalfAdder:

    def setup_method(self):
        self.ha = HalfAdder()

    def test_0_plus_0(self):
        s, c = self.ha.compute(0, 0)
        assert s == 0 and c == 0

    def test_0_plus_1(self):
        s, c = self.ha.compute(0, 1)
        assert s == 1 and c == 0

    def test_1_plus_0(self):
        s, c = self.ha.compute(1, 0)
        assert s == 1 and c == 0

    def test_1_plus_1(self):
        s, c = self.ha.compute(1, 1)
        assert s == 0 and c == 1

    def test_commutative(self):
        s1, c1 = self.ha.compute(1, 0)
        s2, c2 = self.ha.compute(0, 1)
        assert s1 == s2 and c1 == c2


# ── Volladdierer ──────────────────────────────────────────────────────────────

class TestFullAdder:

    def setup_method(self):
        self.fa = FullAdder()

    def test_all_zeros(self):
        s, c = self.fa.compute(0, 0, 0)
        assert s == 0 and c == 0

    def test_one_plus_zero_carry(self):
        s, c = self.fa.compute(1, 0, 0)
        assert s == 1 and c == 0

    def test_one_plus_one(self):
        s, c = self.fa.compute(1, 1, 0)
        assert s == 0 and c == 1

    def test_all_ones(self):
        s, c = self.fa.compute(1, 1, 1)
        assert s == 1 and c == 1

    def test_carry_propagation(self):
        s, c = self.fa.compute(0, 0, 1)
        assert s == 1 and c == 0


# ── RippleCarryAdder ──────────────────────────────────────────────────────────

class TestRippleCarryAdder:

    def test_4bit_add(self):
        adder = RippleCarryAdder(4)
        result, overflow = adder.add(5, 3)
        assert result == 8
        assert overflow == 0

    def test_4bit_overflow(self):
        adder = RippleCarryAdder(4)
        result, overflow = adder.add(15, 1)
        assert overflow == 1

    def test_zero_plus_zero(self):
        adder = RippleCarryAdder(8)
        result, overflow = adder.add(0, 0)
        assert result == 0
        assert overflow == 0

    def test_8bit_add(self):
        adder = RippleCarryAdder(8)
        result, overflow = adder.add(100, 55)
        assert result == 155
        assert overflow == 0

    def test_commutative(self):
        adder = RippleCarryAdder(8)
        r1, _ = adder.add(37, 42)
        r2, _ = adder.add(42, 37)
        assert r1 == r2


# ── IRFNetwork ────────────────────────────────────────────────────────────────

class TestIRFNetwork:

    def test_empty_network(self):
        net = IRFNetwork()
        assert net.node_count() == 0
        assert net.edge_count() == 0

    def test_add_node(self):
        net = IRFNetwork()
        node = IRFNode("A", (0, 0, 0))
        net.add_node(node)
        assert net.node_count() == 1

    def test_add_edge(self):
        net = IRFNetwork()
        net.add_node(IRFNode("A", (0, 0, 0)))
        net.add_node(IRFNode("B", (1e-6, 0, 0)))
        net.add_edge("A", "B")
        assert net.edge_count() == 1

    def test_set_get_state(self):
        net = IRFNetwork()
        net.add_node(IRFNode("X", (0, 0, 0)))
        net.set_state("X", 1)
        assert net.get_state("X") == 1

    def test_set_state_invalid_node(self):
        net = IRFNetwork()
        with pytest.raises(KeyError):
            net.set_state("NONEXISTENT", 1)

    def test_storage_capacity(self):
        net = IRFNetwork()
        for i in range(10):
            net.add_node(IRFNode(f"N{i}", (i * 1e-6, 0, 0)))
        assert net.storage_capacity_bits() == 10

    def test_adjacency_list(self):
        net = IRFNetwork()
        net.add_node(IRFNode("A", (0, 0, 0)))
        net.add_node(IRFNode("B", (1e-6, 0, 0)))
        net.add_edge("A", "B")
        adj = net.adjacency_list()
        assert "B" in adj["A"]

    def test_repr(self):
        net = IRFNetwork()
        s = repr(net)
        assert "IRFNetwork" in s
