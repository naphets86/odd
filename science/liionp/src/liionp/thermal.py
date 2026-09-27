"""Thermisches Netzwerkmodell für eine einzelne LIB-Zelle (Gl. 19–20 aus liionp.tex).

  m_cell · c_p · dT/dt = Q̇_gen(t) − (T(t) − T_amb(t)) / R_th

Wärmegeneration:
  Q̇_gen = I² · R_int  +  I · T · ∂E_cell/∂T
"""
from __future__ import annotations

from dataclasses import dataclass

from .constants import M_CELL_CP_DEFAULT, R_TH_DEFAULT


@dataclass
class ThermalModel:
    """Vereinfachtes thermisches Einmassen-Modell einer einzelnen Zelle.

    Attributes:
        m_cell_cp: Effektive Wärmekapazität m_cell · c_p [J/K].
        R_th:      Thermischer Widerstand zur Umgebung [K/W].
        dE_dT:     Entropischer Wärmekoeffizient ∂E_cell/∂T [V/K].
    """

    m_cell_cp: float = M_CELL_CP_DEFAULT
    R_th: float = R_TH_DEFAULT
    dE_dT: float = -1e-4  # V/K  (typical NMC/Graphit value)

    def heat_generation(self, I: float, R_int: float, T: float) -> float:
        """Wärmegenerationsrate Q̇_gen (Gl. 20).

        Q̇_gen = I² · R_int  +  I · T · dE/dT

        Der erste Term entspricht der irreversiblen Jouleschen Wärme,
        der zweite der reversiblen (entropischen) Komponente.

        Args:
            I:     Angelegter Strom [A].
            R_int: Innenwiderstand [Ω].
            T:     Zelltemperatur [K].

        Returns:
            Wärmegenerationsrate [W].
        """
        return I**2 * R_int + I * T * self.dE_dT

    def temperature_derivative(
        self, T: float, T_amb: float, I: float, R_int: float
    ) -> float:
        """Berechnet dT/dt (Gl. 19).

        Args:
            T:     Aktuelle Zelltemperatur [K].
            T_amb: Umgebungstemperatur [K].
            I:     Angelegter Strom [A].
            R_int: Innenwiderstand [Ω].

        Returns:
            Temperaturänderungsrate [K/s].
        """
        Q_gen = self.heat_generation(I, R_int, T)
        Q_loss = (T - T_amb) / self.R_th
        return (Q_gen - Q_loss) / self.m_cell_cp

    def step(
        self, T: float, T_amb: float, I: float, R_int: float, dt: float
    ) -> float:
        """Euler-Integrationsschritt.

        Args:
            T:     Aktuelle Temperatur [K].
            T_amb: Umgebungstemperatur [K].
            I:     Angelegter Strom [A].
            R_int: Innenwiderstand [Ω].
            dt:    Zeitschritt [s].

        Returns:
            Aktualisierte Temperatur [K].
        """
        return T + self.temperature_derivative(T, T_amb, I, R_int) * dt
