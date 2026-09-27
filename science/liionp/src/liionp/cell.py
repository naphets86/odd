"""Zustandsraummodell einer Zelle (Gl. 11–12 aus liionp.tex).

Zustandsvektor:  x = [Q, T, V, δ_SEI, R_int]ᵀ  ergänzt um SoC und D(t).

Das Modell koppelt:
  - ein lineares OCV- und resistives Spannungsmodell,
  - das thermische Netzwerk erster Ordnung (siehe thermal.py),
  - SEI-Wachstum (siehe degradation.py) sowie
  - den Gesamt-Degradationsakkumulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import (
    ALPHA_SEI_DEFAULT,
    BETA_OV_DEFAULT,
    BETA_UV_DEFAULT,
    D_EOL,
    E_A_SEI_DEFAULT,
    GAMMA_C_DEFAULT,
    GAMMA_T_DEFAULT,
    GAMMA_V_DEFAULT,
    K_OV_DEFAULT,
    K_SEI_DEFAULT,
    K_UV_DEFAULT,
    M_CELL_CP_DEFAULT,
    Q0_DEFAULT,
    R_INT_DEFAULT,
    R_TH_DEFAULT,
    V_MAX_DEFAULT,
    V_MIN_DEFAULT,
    V_NOM_DEFAULT,
)
from .degradation import (
    capacity_loss_sei,
    sei_growth_rate,
    total_degradation_rate,
)
from .thermal import ThermalModel


@dataclass
class CellParameters:
    """Physikalische Parameter einer NMC/Graphit-LIB-Zelle (Tabelle 1).

    Alle Werte beziehen sich auf eine einzelne Zelle.
    """

    Q0: float = Q0_DEFAULT           # Ah   nominal capacity
    V_nom: float = V_NOM_DEFAULT     # V    nominal (average) open-circuit voltage
    V_max: float = V_MAX_DEFAULT     # V    upper voltage limit
    V_min: float = V_MIN_DEFAULT     # V    lower voltage limit
    R_int: float = R_INT_DEFAULT     # Ω    initial internal resistance
    Ea_SEI: float = E_A_SEI_DEFAULT  # J/mol
    k_SEI: float = K_SEI_DEFAULT     # m²/s
    alpha_SEI: float = ALPHA_SEI_DEFAULT  # Ah/m
    m_cell_cp: float = M_CELL_CP_DEFAULT  # J/K
    R_th: float = R_TH_DEFAULT       # K/W
    D_EoL: float = D_EOL
    gamma_T: float = GAMMA_T_DEFAULT
    gamma_V: float = GAMMA_V_DEFAULT
    gamma_C: float = GAMMA_C_DEFAULT
    k_OV: float = K_OV_DEFAULT
    beta_OV: float = BETA_OV_DEFAULT
    k_UV: float = K_UV_DEFAULT
    beta_UV: float = BETA_UV_DEFAULT


@dataclass
class CellState:
    """Dynamischer Zustand einer einzelnen LIB-Zelle.

    Entspricht dem erweiterten Zustandsvektor (Gl. 11) ergänzt um SoC
    und den kumulativen Degradationswert D(t).
    """

    Q: float           # Current capacity [Ah]
    T: float           # Cell temperature [K]
    V: float           # Terminal voltage [V]
    delta_SEI: float   # SEI thickness [m]
    R_int: float       # Internal resistance [Ω]
    SoC: float         # State of charge ∈ [0, 1]
    degradation: float = 0.0  # Cumulative D(t) ∈ [0, 1]

    @classmethod
    def initial(cls, params: CellParameters, T0: float = 298.0) -> "CellState":
        """Erzeugt einen frischen Zellzustand (Zyklus 0).

        Args:
            params: Zellparameter.
            T0:     Anfangstemperatur [K].
        """
        return cls(
            Q=params.Q0,
            T=T0,
            V=params.V_nom,
            delta_SEI=1e-9,  # tiny non-zero initial SEI layer
            R_int=params.R_int,
            SoC=0.5,
        )

    @property
    def is_eol(self) -> bool:
        """True, wenn die kumulative Degradation den EoL-Schwellwert erreicht hat."""
        return self.degradation >= D_EOL


class CellModel:
    """Gekoppeltes elektrochemisches, thermisches und Degradationsmodell einer Zelle.

    Verwendung::

        params = CellParameters()
        model  = CellModel(params)
        state  = CellState.initial(params)
        state, rate = model.step(state, I=1.0, T_amb=298.0, dt=1.0)
    """

    def __init__(self, params: CellParameters | None = None) -> None:
        self.params = params or CellParameters()
        self.thermal = ThermalModel(
            m_cell_cp=self.params.m_cell_cp,
            R_th=self.params.R_th,
        )

    def open_circuit_voltage(self, SoC: float) -> float:
        """Lineares OCV-Modell.

        OCV(SoC) = V_min + SoC · (V_max − V_min)

        Args:
            SoC: Ladezustand ∈ [0, 1].

        Returns:
            Leerlaufspannung [V].
        """
        p = self.params
        return p.V_min + SoC * (p.V_max - p.V_min)

    def terminal_voltage(self, SoC: float, I: float) -> float:
        """Klemmenspannung einschließlich resistivem Spannungsabfall.

        V = OCV(SoC) − I · R_int   (positiver I = Entladung)

        Args:
            SoC:   Ladezustand ∈ [0, 1].
            I:     Angelegter Strom [A].

        Returns:
            Klemmenspannung [V].
        """
        return self.open_circuit_voltage(SoC) - I * self.params.R_int

    def step(
        self,
        state: CellState,
        I: float,
        T_amb: float,
        dt: float = 1.0,
    ) -> tuple[CellState, float]:
        """Führt den Zellzustand um *dt* Sekunden vorwärts.

        Berechnet einen Vorwärts-Euler-Integrationsschritt für alle Zustandsgrößen.

        Args:
            state: Aktueller Zellzustand.
            I:     Angelegter Strom [A]  (positiv = Entladung, negativ = Ladung).
            T_amb: Umgebungstemperatur [K].
            dt:    Zeitschritt [s].

        Returns:
            (neuer_zustand, momentane_degradationsrate)
        """
        p = self.params

        # 1. SoC  (dSoC/dt = −I / (Q0 · 3600))
        dSoC = -I / (p.Q0 * 3600.0) * dt
        new_SoC = float(np.clip(state.SoC + dSoC, 0.0, 1.0))

        # 2. Terminal voltage
        new_V = self.terminal_voltage(new_SoC, I)

        # 3. Temperature (Euler step via ThermalModel)
        new_T = self.thermal.step(state.T, T_amb, I, state.R_int, dt)

        # 4. SEI thickness (Euler step via growth rate)
        d_sei_rate = sei_growth_rate(p.k_SEI, p.Ea_SEI, new_T, state.delta_SEI)
        new_delta_SEI = state.delta_SEI + d_sei_rate * dt

        # 5. Capacity (decreases with SEI growth)
        delta_Q = capacity_loss_sei(p.alpha_SEI, new_delta_SEI - state.delta_SEI)
        new_Q = float(max(state.Q - delta_Q, 0.0))

        # 6. Internal resistance grows with degradation (empirical linear model)
        new_R_int = p.R_int * (1.0 + 0.5 * state.degradation)

        # 7. Instantaneous degradation rate
        deg_rate = total_degradation_rate(
            new_T, new_V, I,
            p.V_min, p.V_max,
            p.k_SEI, p.Ea_SEI,
            p.gamma_T, p.gamma_V, p.gamma_C,
            p.k_OV, p.beta_OV, p.k_UV, p.beta_UV,
        )

        # 8. Cumulative degradation
        new_degradation = state.degradation + deg_rate * dt

        new_state = CellState(
            Q=new_Q,
            T=new_T,
            V=new_V,
            delta_SEI=new_delta_SEI,
            R_int=new_R_int,
            SoC=new_SoC,
            degradation=new_degradation,
        )
        return new_state, deg_rate
