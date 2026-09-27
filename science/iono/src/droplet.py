"""
irf.droplet - Tropfendynamik und Selbstorganisation
====================================================

Implementiert:
  - Droplet        Einzelner Wassertropfen mit Volumen, Position, Ionengehalt
  - DropletMerger  Verschmelzungslogik zweier Tropfen
  - SelfOrganizer  Massenhafte Selbstorganisation auf der Elektroden-Matrix
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.physics import SurfaceTension
from src.utils import CONST, ValidationError


# ---------------------------------------------------------------------------
# Tropfen
# ---------------------------------------------------------------------------

@dataclass
class Droplet:
    """
    Einzelner Wassertropfen auf der IRF-Elektroden-Matrix.

    Attributes:
        volume_m3:          Volumen [m³]
        position:           (row, col) auf der Elektrodenmatrix
        ion_concentration_mol_m3: Ionenkonzentration [mol/m³]
        label:              Optionaler logischer Bezeichner
        temperature_K:      Temperatur des Tropfens [K]
    """

    volume_m3: float
    position: tuple[int, int]
    ion_concentration_mol_m3: float = 100.0  # 0.1 mol/L KCl
    label: Optional[str] = None
    temperature_K: float = 298.15

    def __post_init__(self) -> None:
        if self.volume_m3 <= 0:
            raise ValidationError("volume_m3", self.volume_m3, "> 0")
        if self.ion_concentration_mol_m3 < 0:
            raise ValidationError(
                "ion_concentration_mol_m3", self.ion_concentration_mol_m3, ">= 0"
            )
        if self.temperature_K <= 0:
            raise ValidationError("temperature_K", self.temperature_K, "> 0")

    @property
    def radius_m(self) -> float:
        """Äquivalenter Kugelradius r = (3V/4π)^(1/3) [m]."""
        return (3.0 * self.volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0)

    @property
    def surface_area_m2(self) -> float:
        """Oberfläche der äquivalenten Kugel [m²]."""
        return 4.0 * math.pi * self.radius_m ** 2

    @property
    def surface_energy_J(self) -> float:
        """Oberflächenenergie E = γ · A [J]."""
        gamma = SurfaceTension.at_temperature(self.temperature_K)
        return gamma * self.surface_area_m2

    @property
    def total_ion_content_mol(self) -> float:
        """Gesamter Ionengehalt n = c · V [mol]."""
        return self.ion_concentration_mol_m3 * self.volume_m3

    def laplace_pressure_Pa(self) -> float:
        """
        Laplace-Druck ΔP = 2γ/r [Pa].

        Treibt die Tropfenbewegung durch Kapillareffekte.
        """
        gamma = SurfaceTension.at_temperature(self.temperature_K)
        return 2.0 * gamma / self.radius_m

    def conductance_S(self, length_m: float, cross_section_m2: float) -> float:
        """
        Elektrische Leitfähigkeit des Tropfens als Kanalabschnitt [S].

        Args:
            length_m:         Pfadlänge durch den Tropfen [m]
            cross_section_m2: Effektiver Querschnitt [m²]

        Returns:
            Leitfähigkeit [S]
        """
        if length_m <= 0:
            raise ValidationError("length_m", length_m, "> 0")
        if cross_section_m2 <= 0:
            raise ValidationError("cross_section_m2", cross_section_m2, "> 0")
        # Vereinfachte Leitfähigkeit: σ = |z| μ c F (nur K+ und Cl-)
        mobility_avg = 7.76e-8  # Mittlere Beweglichkeit KCl [m²/(V·s)]
        sigma = abs(1) * mobility_avg * self.ion_concentration_mol_m3 * CONST.F
        return sigma * cross_section_m2 / length_m

    def move_to(self, new_position: tuple[int, int]) -> "Droplet":
        """
        Gibt eine Kopie des Tropfens an neuer Position zurück.

        Args:
            new_position: (row, col) Zielposition

        Returns:
            Neues Droplet-Objekt an Zielposition
        """
        return Droplet(
            volume_m3=self.volume_m3,
            position=new_position,
            ion_concentration_mol_m3=self.ion_concentration_mol_m3,
            label=self.label,
            temperature_K=self.temperature_K,
        )


# ---------------------------------------------------------------------------
# Tropfen-Verschmelzung
# ---------------------------------------------------------------------------

class DropletMerger:
    """
    Verschmelzungslogik für Wassertropfen.

    Realisiert das OR-Gatter passiv (durch Oberflächenspannungsminimierung).
    """

    @staticmethod
    def can_merge(d1: Droplet, d2: Droplet, max_distance_cells: int = 1) -> bool:
        """
        Prüft, ob zwei Tropfen verschmelzen können.

        Tropfen können verschmelzen, wenn sie benachbart sind (Manhattan-Distanz ≤ Schwelle).

        Args:
            d1, d2:              Zwei Tropfen
            max_distance_cells:  Maximale Manhattan-Distanz in Zellen

        Returns:
            True wenn Verschmelzung physikalisch möglich
        """
        r1, c1 = d1.position
        r2, c2 = d2.position
        return abs(r1 - r2) + abs(c1 - c2) <= max_distance_cells

    @staticmethod
    def merge(d1: Droplet, d2: Droplet) -> Droplet:
        """
        Verschmilzt zwei Tropfen zu einem neuen, größeren Tropfen.

        Volumenerhaltung:      V_f = V_1 + V_2
        Ionenerhaltung:        n_f = n_1 + n_2
        Neue Konzentration:    c_f = (n_1 + n_2) / V_f
        Position:              Schwerpunkt gewichtet nach Volumen

        Args:
            d1, d2: Tropfen zum Verschmelzen

        Returns:
            Neuer zusammengeführter Tropfen
        """
        v_f = d1.volume_m3 + d2.volume_m3
        n_f = d1.total_ion_content_mol + d2.total_ion_content_mol
        c_f = n_f / v_f

        # Volumengewichteter Schwerpunkt (auf Integer-Gitterpositionen gerundet)
        r1, c1 = d1.position
        r2, c2 = d2.position
        r_f = round((r1 * d1.volume_m3 + r2 * d2.volume_m3) / v_f)
        c_f_pos = round((c1 * d1.volume_m3 + c2 * d2.volume_m3) / v_f)

        # Temperatur: volumengewichtetes Mittel
        t_f = (d1.temperature_K * d1.volume_m3 + d2.temperature_K * d2.volume_m3) / v_f

        merged_label = None
        if d1.label and d2.label:
            merged_label = f"{d1.label}+{d2.label}"
        elif d1.label or d2.label:
            merged_label = d1.label or d2.label

        return Droplet(
            volume_m3=v_f,
            position=(r_f, c_f_pos),
            ion_concentration_mol_m3=c_f,
            label=merged_label,
            temperature_K=t_f,
        )

    @staticmethod
    def merge_energy_released_J(d1: Droplet, d2: Droplet) -> float:
        """
        Berechnet die bei der Verschmelzung freigesetzte Oberflächenenergie [J].

        ΔE = γ · 4π · (r₁² + r₂² - r_f²)
        """
        return SurfaceTension.droplet_merge_energy_J(
            d1.radius_m, d2.radius_m, d1.temperature_K
        )


# ---------------------------------------------------------------------------
# Selbst-Organisator
# ---------------------------------------------------------------------------

class SelfOrganizer:
    """
    Koordiniert die massenhafte Selbstorganisation von Tropfen auf der Matrix.

    Realisiert das Kernelement der IRF-Technologie:
    Gleichzeitiges Einpendeln aller Tropfen in ihre Energieminima
    durch gezielte EWOD-Aktivierung.
    """

    def __init__(self, threshold_voltage_V: float = 0.12) -> None:
        if threshold_voltage_V <= 0:
            raise ValidationError("threshold_voltage_V", threshold_voltage_V, "> 0")
        self.threshold_voltage_V = threshold_voltage_V
        self._droplets: list[Droplet] = []

    def add_droplet(self, droplet: Droplet) -> None:
        """Fügt einen Tropfen zur Simulation hinzu."""
        self._droplets.append(droplet)

    def remove_droplet(self, droplet: Droplet) -> None:
        """Entfernt einen Tropfen."""
        self._droplets.remove(droplet)

    @property
    def droplets(self) -> list[Droplet]:
        """Aktuelle Liste aller Tropfen."""
        return list(self._droplets)

    @property
    def num_droplets(self) -> int:
        """Anzahl der aktuellen Tropfen."""
        return len(self._droplets)

    def step(self, voltage_pattern: dict[tuple[int, int], float]) -> list[Droplet]:
        """
        Führt einen Simulationsschritt durch.

        Tropfen bewegen sich zu aktiven (Spannung > Schwelle) Elektroden.
        Benachbarte Tropfen auf aktiven Elektroden verschmelzen spontan.

        Args:
            voltage_pattern: Mapping (row, col) → Spannung [V]

        Returns:
            Aktualisierte Tropfenliste nach dem Schritt
        """
        active_cells = {
            pos for pos, v in voltage_pattern.items()
            if v >= self.threshold_voltage_V
        }

        # Tropfen zu aktiven Elektroden bewegen
        moved = []
        for d in self._droplets:
            if d.position in active_cells:
                moved.append(d)
                continue
            # Finde nächste aktive Elektrode
            best_pos = None
            best_dist = float("inf")
            for pos in active_cells:
                dist = abs(d.position[0] - pos[0]) + abs(d.position[1] - pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best_pos = pos
            if best_pos is not None and best_dist <= 1:
                moved.append(d.move_to(best_pos))
            else:
                moved.append(d)

        # Verschmelzung benachbarter Tropfen
        self._droplets = self._merge_all(moved)
        return list(self._droplets)

    def _merge_all(self, droplets: list[Droplet]) -> list[Droplet]:
        """Wiederholt Verschmelzungen, bis keine benachbarten Tropfen mehr übrig sind."""
        changed = True
        result = list(droplets)
        while changed:
            changed = False
            merged: list[Droplet] = []
            used = [False] * len(result)
            for i, d1 in enumerate(result):
                if used[i]:
                    continue
                for j, d2 in enumerate(result):
                    if i >= j or used[j]:
                        continue
                    if DropletMerger.can_merge(d1, d2):
                        merged.append(DropletMerger.merge(d1, d2))
                        used[i] = used[j] = True
                        changed = True
                        break
                if not used[i]:
                    merged.append(d1)
            result = merged
        return result

    def relaxation_time_s(self, max_path_length_m: float) -> float:
        """
        Schätzt die Relaxationszeit τ = L² / D_eff [s].

        Args:
            max_path_length_m: Maximale Wegstrecke eines Tropfens [m]

        Returns:
            Geschätzte Relaxationszeit [s]
        """
        if max_path_length_m <= 0:
            raise ValidationError("max_path_length_m", max_path_length_m, "> 0")
        d_eff = 1e-10  # m²/s (effektiver Benetzungskoeffizient)
        return max_path_length_m ** 2 / d_eff
