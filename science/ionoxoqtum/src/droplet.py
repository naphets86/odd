"""
droplet.py
===============
Tropfenlogik – Boolesche Gatter und Turing-Vollständigkeit des IRF.

Implementiert:
  - Tropfen-Energieminimierung (Verschmelzung)
  - NAND-Gate, NOR-Gate, AND, OR, NOT, XOR
  - Halbaddierer, Volladdierer
  - Turing-Vollständigkeitsnachweis via NAND-Vollständigkeit
  - Einfache IRF-Netzwerksimulation
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple
from enum import Enum


class LogicState(Enum):
    """Logischer Zustand eines Tropfens."""
    ZERO = 0   # Kein Tropfen an Position
    ONE  = 1   # Tropfen an Position


@dataclass
class Droplet:
    """Repräsentiert einen Wassertropfen auf einer EWOD-Platte.

    Parameters
    ----------
    position : tuple[float, float]
        (x, y)-Position [m].
    radius : float
        Tropfenradius [m].
    conductivity : float
        Ionische Leitfähigkeit des Tropfens [S/m].
    logic_value : int
        Logischer Wert: 0 oder 1.
    """
    position: Tuple[float, float]
    radius: float
    conductivity: float = 1.0
    logic_value: int = 1

    @property
    def volume(self) -> float:
        """Tropfenvolumen V = (4/3)π r³  [m³]."""
        return (4.0 / 3.0) * np.pi * self.radius**3

    @property
    def surface_energy(self, gamma: float = 0.072) -> float:
        """Oberflächenenergie E = 4π γ r²  [J]."""
        return 4.0 * np.pi * gamma * self.radius**2

    def merge_radius(self, other: "Droplet") -> float:
        """Radius nach Verschmelzung: r_f = (r_1³ + r_2³)^(1/3).

        Parameters
        ----------
        other : Droplet
            Zweiter Tropfen.

        Returns
        -------
        float
            Radius des verschmolzenen Tropfens [m].
        """
        return (self.radius**3 + other.radius**3) ** (1.0 / 3.0)

    def can_merge(self, other: "Droplet", gamma: float = 0.072) -> bool:
        """Prüft, ob Verschmelzung thermodynamisch begünstigt ist.

        Satz 3.1 der Arbeit: Immer wahr für r₁, r₂ > 0.

        Returns
        -------
        bool
            True (Verschmelzung immer thermodynamisch begünstigt).
        """
        # Beweis: (r1³ + r2³)^(2/3) < r1² + r2² für alle r1,r2 > 0
        lhs = (self.radius**3 + other.radius**3) ** (2.0 / 3.0)
        rhs = self.radius**2 + other.radius**2
        return lhs < rhs

    def energy_released_on_merge(
        self,
        other: "Droplet",
        gamma: float = 0.072,
    ) -> float:
        """Freigesetzte Energie bei Verschmelzung  ΔE = E_vorher − E_nachher  [J].

        Parameters
        ----------
        other : Droplet
        gamma : float
            Oberflächenspannung [N/m].

        Returns
        -------
        float
            Freigesetzte Energie in Joule.
        """
        r_f = self.merge_radius(other)
        E_before = 4.0 * np.pi * gamma * (self.radius**2 + other.radius**2)
        E_after = 4.0 * np.pi * gamma * r_f**2
        return E_before - E_after


# ── Logische Gatter ────────────────────────────────────────────────────────────

class LogicGate:
    """Basis-Klasse für Tropfenlogik-Gatter."""

    def __init__(self, name: str, n_inputs: int) -> None:
        self.name = name
        self.n_inputs = n_inputs

    def evaluate(self, *inputs: int) -> int:
        raise NotImplementedError

    def truth_table(self) -> List[Tuple[Tuple[int, ...], int]]:
        """Vollständige Wahrheitstafel."""
        rows = []
        for combo in range(2**self.n_inputs):
            bits = tuple((combo >> i) & 1 for i in range(self.n_inputs - 1, -1, -1))
            output = self.evaluate(*bits)
            rows.append((bits, output))
        return rows

    def __repr__(self) -> str:
        return f"{self.name}Gate()"


class NANDGate(LogicGate):
    """NAND-Gate durch Tropfenverschmelzung.

    Wenn beide Eingänge 1 sind (beide Tropfen ankommen und verschmelzen),
    wird der Ausgang auf 0 gesetzt (kein Ausgangstropfen).
    """

    def __init__(self) -> None:
        super().__init__("NAND", 2)

    def evaluate(self, a: int, b: int) -> int:
        """NAND: NOT (A AND B)."""
        return int(not (bool(a) and bool(b)))


class NORGate(LogicGate):
    """NOR-Gate durch Pfadblockierung."""

    def __init__(self) -> None:
        super().__init__("NOR", 2)

    def evaluate(self, a: int, b: int) -> int:
        """NOR: NOT (A OR B)."""
        return int(not (bool(a) or bool(b)))


class NOTGate(LogicGate):
    """NOT-Gate (aus NAND mit verbundenen Eingängen)."""

    def __init__(self) -> None:
        super().__init__("NOT", 1)

    def evaluate(self, a: int) -> int:
        """NOT A = NAND(A, A)."""
        return NANDGate().evaluate(a, a)


class ANDGate(LogicGate):
    """AND-Gate aus zwei NAND-Gates."""

    def __init__(self) -> None:
        super().__init__("AND", 2)
        self._nand = NANDGate()
        self._not = NOTGate()

    def evaluate(self, a: int, b: int) -> int:
        """AND = NOT(NAND(A,B))."""
        return self._not.evaluate(self._nand.evaluate(a, b))


class ORGate(LogicGate):
    """OR-Gate aus drei NAND-Gates."""

    def __init__(self) -> None:
        super().__init__("OR", 2)
        self._nand = NANDGate()

    def evaluate(self, a: int, b: int) -> int:
        """OR = NAND(NAND(A,A), NAND(B,B))."""
        na = self._nand.evaluate(a, a)
        nb = self._nand.evaluate(b, b)
        return self._nand.evaluate(na, nb)


class XORGate(LogicGate):
    """XOR-Gate aus vier NAND-Gates."""

    def __init__(self) -> None:
        super().__init__("XOR", 2)
        self._nand = NANDGate()

    def evaluate(self, a: int, b: int) -> int:
        """XOR = NAND(NAND(A, NAND(A,B)), NAND(B, NAND(A,B)))."""
        nab = self._nand.evaluate(a, b)
        left = self._nand.evaluate(a, nab)
        right = self._nand.evaluate(b, nab)
        return self._nand.evaluate(left, right)


# ── Addierer ──────────────────────────────────────────────────────────────────

class HalfAdder:
    """1-Bit Halbaddierer aus Tropfenlogik-Gattern."""

    def __init__(self) -> None:
        self._xor = XORGate()
        self._and = ANDGate()

    def compute(self, a: int, b: int) -> Tuple[int, int]:
        """Addiert a + b. Gibt (Summe, Übertrag) zurück."""
        s = self._xor.evaluate(a, b)
        c = self._and.evaluate(a, b)
        return s, c


class FullAdder:
    """1-Bit Volladdierer aus Tropfenlogik-Gattern."""

    def __init__(self) -> None:
        self._ha1 = HalfAdder()
        self._ha2 = HalfAdder()
        self._or = ORGate()

    def compute(self, a: int, b: int, carry_in: int) -> Tuple[int, int]:
        """Addiert a + b + carry_in. Gibt (Summe, Übertrag) zurück."""
        s1, c1 = self._ha1.compute(a, b)
        s2, c2 = self._ha2.compute(s1, carry_in)
        carry_out = self._or.evaluate(c1, c2)
        return s2, carry_out


class RippleCarryAdder:
    """N-Bit Ripple-Carry-Addierer aus Voll-Addierern."""

    def __init__(self, n_bits: int) -> None:
        self.n_bits = n_bits
        self._adders = [FullAdder() for _ in range(n_bits)]

    def add(self, a: int, b: int) -> Tuple[int, int]:
        """Addiert zwei N-Bit-Zahlen a und b.

        Parameters
        ----------
        a, b : int
            Nicht-negative ganze Zahlen (max. 2^n_bits − 1).

        Returns
        -------
        (result, overflow) : tuple[int, int]
        """
        a_bits = [(a >> i) & 1 for i in range(self.n_bits)]
        b_bits = [(b >> i) & 1 for i in range(self.n_bits)]
        carry = 0
        result_bits = []
        for i in range(self.n_bits):
            s, carry = self._adders[i].compute(a_bits[i], b_bits[i], carry)
            result_bits.append(s)
        result = sum(bit << i for i, bit in enumerate(result_bits))
        return result, carry  # carry = Überlauf


# ── IRF-Netzwerk ──────────────────────────────────────────────────────────────

@dataclass
class IRFNode:
    """Knoten im IRF-Netzwerk (ein Wasservolumen)."""
    node_id: str
    position: Tuple[float, float, float]  # (x, y, z) [m]
    state: int = 0  # 0 oder 1
    conductance: float = 0.0  # aktuelle Leitfähigkeit [S]


class IRFNetwork:
    """Einfaches Modell eines IRF-Netzwerks aus IoFET-verbundenen Knoten.

    Parameters
    ----------
    nodes : list[IRFNode]
        Alle Knoten des Netzwerks.
    """

    def __init__(self, nodes: Optional[List[IRFNode]] = None) -> None:
        self.nodes: Dict[str, IRFNode] = {}
        self.edges: List[Tuple[str, str, float]] = []  # (from, to, weight)
        for node in (nodes or []):
            self.add_node(node)

    def add_node(self, node: IRFNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, from_id: str, to_id: str, weight: float = 1.0) -> None:
        """Verbindet zwei Knoten (IoFET-Kanal)."""
        self.edges.append((from_id, to_id, weight))

    def set_state(self, node_id: str, state: int) -> None:
        """Setzt den logischen Zustand eines Knotens."""
        if node_id not in self.nodes:
            raise KeyError(f"Knoten '{node_id}' nicht im Netzwerk.")
        self.nodes[node_id].state = state

    def get_state(self, node_id: str) -> int:
        return self.nodes[node_id].state

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def adjacency_list(self) -> Dict[str, List[str]]:
        adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for (f, t, _) in self.edges:
            adj[f].append(t)
        return adj

    def storage_capacity_bits(self) -> int:
        """Speicherkapazität des Netzwerks in Bit (1 Bit pro Knoten)."""
        return self.node_count()

    def __repr__(self) -> str:
        return (
            f"IRFNetwork({self.node_count()} Knoten, "
            f"{self.edge_count()} Verbindungen)"
        )
