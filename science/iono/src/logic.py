"""
irf.logic - Tropfenlogik: Gatter, Netzwerke, Wahrheitstabellen
===============================================================

Implementiert:
  - GateType      Enum der unterstützten Logikgatter-Typen
  - DropletGate   Einzelnes Logikgatter (physikalisch realisiert)
  - LogicNetwork  Netzwerk aus mehreren Gattern
  - TruthTable    Automatische Wahrheitstabellenberechnung
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from src.utils import ValidationError


# ---------------------------------------------------------------------------
# Gatter-Typen
# ---------------------------------------------------------------------------

class GateType(Enum):
    """Unterstützte logische Gatter-Typen in der Tropfenlogik."""
    AND = auto()
    OR = auto()
    NOT = auto()
    NAND = auto()
    NOR = auto()
    XOR = auto()
    XNOR = auto()
    BUFFER = auto()


# ---------------------------------------------------------------------------
# Einzelnes Logikgatter
# ---------------------------------------------------------------------------

@dataclass
class DropletGate:
    """
    Einzelnes Logikgatter, realisiert durch Tropfendynamik.

    In der physikalischen Implementierung entspricht:
      OR   → Selbstverbindung zweier Wasserkanäle (passiv, null Energie)
      AND  → Geometrische Engstelle (beide Kanäle müssen Druck liefern)
      NOT  → Kapillarer Siphon (invertierender Gegendruck)
      NAND → AND gefolgt von NOT (vollständiges Gatter-System)

    Attributes:
        gate_type:    Art des Gatters
        name:         Bezeichner des Gatters
        inputs:       Liste der Eingangssignalbezeichner
        output:       Ausgangssignalbezeichner
        propagation_delay_s: Signallaufzeit [s] (abhängig von Benetzungszeit)
    """

    gate_type: GateType
    name: str
    inputs: list[str]
    output: str
    propagation_delay_s: float = 0.01  # Standard: 10 ms (Benetzungszeit)

    def __post_init__(self) -> None:
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        expected = {
            GateType.NOT: 1,
            GateType.BUFFER: 1,
        }
        if self.gate_type in expected:
            if len(self.inputs) != expected[self.gate_type]:
                raise ValidationError(
                    "inputs",
                    len(self.inputs),
                    f"== {expected[self.gate_type]} für {self.gate_type.name}",
                )
        elif len(self.inputs) < 2:
            raise ValidationError(
                "inputs",
                len(self.inputs),
                f">= 2 für {self.gate_type.name}",
            )

    def evaluate(self, signal_values: dict[str, bool]) -> bool:
        """
        Berechnet den logischen Ausgabewert des Gatters.

        Args:
            signal_values: Mapping von Signalbezeichner → boolescher Wert

        Returns:
            Ausgabewert des Gatters

        Raises:
            KeyError: Wenn ein Eingangsignal nicht in signal_values enthalten ist
        """
        in_vals = [signal_values[s] for s in self.inputs]

        if self.gate_type == GateType.BUFFER:
            return in_vals[0]
        if self.gate_type == GateType.NOT:
            return not in_vals[0]
        if self.gate_type == GateType.AND:
            return all(in_vals)
        if self.gate_type == GateType.OR:
            return any(in_vals)
        if self.gate_type == GateType.NAND:
            return not all(in_vals)
        if self.gate_type == GateType.NOR:
            return not any(in_vals)
        if self.gate_type == GateType.XOR:
            return sum(in_vals) % 2 == 1
        if self.gate_type == GateType.XNOR:
            return sum(in_vals) % 2 == 0
        raise ValueError(f"Unbekannter GateType: {self.gate_type}")

    @property
    def num_inputs(self) -> int:
        """Anzahl der Eingänge."""
        return len(self.inputs)

    @property
    def is_universal(self) -> bool:
        """NAND und NOR sind funktional vollständige (universelle) Gatter."""
        return self.gate_type in (GateType.NAND, GateType.NOR)

    @property
    def energy_model(self) -> str:
        """Physikalischer Energiemechanismus des Gatters."""
        _energy = {
            GateType.OR: "passiv (Selbstverbindung durch Oberflächenspannung)",
            GateType.AND: "aktiv (Laplace-Druck durch Engstelle)",
            GateType.NOT: "aktiv (Siphon-Gegendruck)",
            GateType.NAND: "aktiv (AND + Siphon)",
            GateType.NOR: "aktiv (OR + Siphon)",
            GateType.XOR: "aktiv (kombinierte Engstellen)",
            GateType.XNOR: "aktiv (XOR + Siphon)",
            GateType.BUFFER: "passiv (Kanalweiterleitung)",
        }
        return _energy[self.gate_type]


# ---------------------------------------------------------------------------
# Logisches Netzwerk
# ---------------------------------------------------------------------------

class LogicNetwork:
    """
    Netzwerk aus mehreren DropletGates.

    Realisiert einen vollständigen Schaltkreis aus der Tropfenlogik.
    Die Auswertung erfolgt topologisch sortiert (kein Feedback-Support).
    """

    def __init__(self, name: str = "IRF-Netzwerk") -> None:
        self.name = name
        self._gates: dict[str, DropletGate] = {}
        self._primary_inputs: set[str] = set()

    # -----------------------------------------------------------------------
    # Aufbau
    # -----------------------------------------------------------------------

    def add_gate(self, gate: DropletGate) -> None:
        """Fügt ein Gatter zum Netzwerk hinzu."""
        if gate.name in self._gates:
            raise ValueError(f"Gatter '{gate.name}' existiert bereits.")
        self._gates[gate.name] = gate

    def declare_primary_input(self, signal: str) -> None:
        """Markiert ein Signal als primären Eingang."""
        self._primary_inputs.add(signal)

    # -----------------------------------------------------------------------
    # Eigenschaften
    # -----------------------------------------------------------------------

    @property
    def gates(self) -> dict[str, DropletGate]:
        """Alle Gatter im Netzwerk."""
        return dict(self._gates)

    @property
    def num_gates(self) -> int:
        """Anzahl der Gatter."""
        return len(self._gates)

    @property
    def primary_inputs(self) -> set[str]:
        """Primäre Eingangssignale."""
        return set(self._primary_inputs)

    @property
    def total_propagation_delay_s(self) -> float:
        """Maximale Signallaufzeit durch alle Gatter [s]."""
        if not self._gates:
            return 0.0
        return max(g.propagation_delay_s for g in self._gates.values())

    @property
    def or_gate_count(self) -> int:
        """Anzahl passiver OR-Gatter (null Energie)."""
        return sum(1 for g in self._gates.values() if g.gate_type == GateType.OR)

    # -----------------------------------------------------------------------
    # Auswertung
    # -----------------------------------------------------------------------

    def evaluate(self, input_values: dict[str, bool]) -> dict[str, bool]:
        """
        Wertet das gesamte Netzwerk für gegebene Eingangsbelegungen aus.

        Args:
            input_values: Primäre Eingänge mit booleschen Werten

        Returns:
            Mapping aller Signale (Eingänge + Gatterausgänge) → boolescher Wert

        Raises:
            ValueError: Wenn das Netzwerk Zyklen enthält
        """
        signals = dict(input_values)
        # Topologische Abarbeitung: wiederholt Gatter auswerten,
        # sobald alle Eingänge bekannt sind
        remaining = list(self._gates.values())
        max_iter = len(remaining) + 1
        iterations = 0

        while remaining:
            iterations += 1
            if iterations > max_iter:
                names = [g.name for g in remaining]
                raise ValueError(
                    f"Netzwerk enthält Zyklen oder unbefriedigte Abhängigkeiten: {names}"
                )
            progress = False
            still_remaining = []
            for gate in remaining:
                if all(s in signals for s in gate.inputs):
                    signals[gate.output] = gate.evaluate(signals)
                    progress = True
                else:
                    still_remaining.append(gate)
            if not progress and still_remaining:
                names = [g.name for g in still_remaining]
                raise ValueError(
                    f"Unaufgelöste Signale für Gatter: {names}"
                )
            remaining = still_remaining

        return signals

    def is_turing_complete(self) -> bool:
        """
        Gibt True zurück, wenn das Netzwerk mind. ein NAND- oder NOR-Gatter enthält
        (notwendige Bedingung für Turing-Vollständigkeit als Gatter-Set).
        """
        return any(
            g.gate_type in (GateType.NAND, GateType.NOR)
            for g in self._gates.values()
        )


# ---------------------------------------------------------------------------
# Wahrheitstabelle
# ---------------------------------------------------------------------------

@dataclass
class TruthTable:
    """
    Automatisch generierte Wahrheitstabelle für ein LogicNetwork.

    Attributes:
        network:    Das zugrundeliegende LogicNetwork
        inputs:     Geordnete Liste der Eingangssignale
        outputs:    Geordnete Liste der Ausgangssignale
        rows:       Alle Zeilen: (input_dict, output_dict)
    """

    network: LogicNetwork
    inputs: list[str]
    outputs: list[str]
    rows: list[tuple[dict[str, bool], dict[str, bool]]] = field(
        default_factory=list
    )

    @classmethod
    def generate(
        cls,
        network: LogicNetwork,
        primary_inputs: list[str],
        primary_outputs: list[str],
    ) -> "TruthTable":
        """
        Erzeugt die vollständige Wahrheitstabelle durch Erschöpfungssuche.

        Args:
            network:         LogicNetwork
            primary_inputs:  Liste der Eingangssignalbezeichner
            primary_outputs: Liste der zu beobachtenden Ausgangssignale

        Returns:
            Vollständige TruthTable
        """
        n = len(primary_inputs)
        table = cls(
            network=network,
            inputs=primary_inputs,
            outputs=primary_outputs,
        )
        for i in range(2 ** n):
            input_assignment = {
                primary_inputs[j]: bool((i >> (n - 1 - j)) & 1)
                for j in range(n)
            }
            all_signals = network.evaluate(input_assignment)
            output_assignment = {
                o: all_signals[o] for o in primary_outputs if o in all_signals
            }
            table.rows.append((input_assignment, output_assignment))
        return table

    @property
    def num_rows(self) -> int:
        """Anzahl der Zeilen (= 2^n für n Eingänge)."""
        return len(self.rows)

    def format(self) -> str:
        """Formatiert die Wahrheitstabelle als lesbare ASCII-Tabelle."""
        header = " | ".join(self.inputs + self.outputs)
        sep = "-" * len(header)
        lines = [header, sep]
        for inp, out in self.rows:
            row_vals = [str(int(inp[s])) for s in self.inputs]
            row_vals += [str(int(out.get(s, False))) for s in self.outputs]
            lines.append(" | ".join(row_vals))
        return "\n".join(lines)
