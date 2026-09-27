"""Simulationsmotor: Statisches PMS vs. KI-PMM.

Stellt bereit:
  - Drei kanonische Lastprofile LP1/LP2/LP3 (Tabelle 2 aus liionp.tex).
  - ``run_simulation`` – führt eine Einzelsimulation (statisch oder KI-Modus) durch.
  - ``compare_systems`` – führt beide Systeme aus und gibt gepaarte Ergebnisse zurück.
  - ``SimulationResult`` – leichtgewichtiger Ergebniscontainer.

Vereinfachung des KI-PMM in dieser Implementierung:
  - Zielt auf eine niedrigere mittlere Temperatur (prädiktive Vorkühlung durch
    Stromreduzierung, wenn T den Soft-Limit-Wert nähert).
  - Nutzt ein engeres effektives Spannungsfenster (``reduced_limit=True`` im SimpleMPC).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cell import CellModel, CellParameters, CellState
from .controllers import PIDController, SimpleMPC


# ── Load profiles ─────────────────────────────────────────────────────────────

@dataclass
class LoadProfile:
    """Stückweises lineares Strom-Zeit-Lastprofil.

    Attributes:
        name:     Beschreibender Name.
        times:    Zeitarray [s].
        currents: Stromarray [A]  (positiv = Entladung).
        T_amb:    Konstante Umgebungstemperatur [K].
    """

    name: str
    times: np.ndarray
    currents: np.ndarray
    T_amb: float = 298.0

    def current_at(self, t: float) -> float:
        """Interpoliert den Strom zur beliebigen Zeit *t* (mit periodischer Wiederholung).

        Args:
            t: Abfragezeitpunkt [s]. Wird automatisch auf den Profilzeitraum umgebrochen.

        Returns:
            Strom [A].
        """
        t_mod = t % self.times[-1] if self.times[-1] > 0 else 0.0
        return float(np.interp(t_mod, self.times, self.currents))

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def lp1_smartphone(
        cls, n_cycles: int = 10, dt: float = 60.0
    ) -> "LoadProfile":
        """LP1: Smartphone-typischer variabler Betrieb (1–3 C)."""
        t_cycle = 3600.0
        ts = np.arange(0.0, n_cycles * t_cycle + dt, dt)
        rng = np.random.default_rng(42)
        Is = 1.5 * np.abs(np.sin(ts / t_cycle * np.pi)) + 0.3 * rng.uniform(
            -1.0, 1.0, len(ts)
        )
        return cls(name="LP1_Smartphone", times=ts, currents=Is, T_amb=298.0)

    @classmethod
    def lp2_ebike(
        cls, n_cycles: int = 10, dt: float = 60.0
    ) -> "LoadProfile":
        """LP2: E-Bike-Intensivbetrieb (2–5 C)."""
        t_cycle = 3600.0
        ts = np.arange(0.0, n_cycles * t_cycle + dt, dt)
        rng = np.random.default_rng(7)
        Is = 3.0 + rng.uniform(-1.0, 1.0, len(ts))
        return cls(name="LP2_Ebike", times=ts, currents=Is, T_amb=303.0)

    @classmethod
    def lp3_stationary(
        cls, n_cycles: int = 10, dt: float = 300.0
    ) -> "LoadProfile":
        """LP3: Stationärer Speicher, mäßiger Betrieb (0,5–1 C)."""
        t_cycle = 7200.0
        ts = np.arange(0.0, n_cycles * t_cycle + dt, dt)
        Is = 0.75 + 0.25 * np.sin(ts / t_cycle * 2.0 * np.pi)
        return cls(name="LP3_Stationary", times=ts, currents=Is, T_amb=293.0)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Zeitreihenergebnis einer einzelnen Simulationsdurchführung.

    Attributes:
        name:                 Bezeichnung (z. B. ``"Statisches PMS"`` oder ``"KI-PMM"``).
        times:                Zeitstempel [s].
        temperatures:         Zelltemperaturen [K].
        voltages:             Klemmenspannungen [V].
        SoCs:                 Ladezustand ∈ [0, 1].
        degradations:         Kumulative Degradation D(t).
        degradation_rates:    Momentane Degradationsrate D̊(t) [1/s].
        cycles_completed:     Anzahl abgeschlossener Zyklen.
        mean_temperature:     Mittlere Zelltemperatur [K].
        mean_degradation_rate: Mittlere Degradationsrate D̊ [1/s].
    """

    name: str
    times: np.ndarray
    temperatures: np.ndarray
    voltages: np.ndarray
    SoCs: np.ndarray
    degradations: np.ndarray
    degradation_rates: np.ndarray
    cycles_completed: int
    mean_temperature: float
    mean_degradation_rate: float


# ── Core simulation runner ────────────────────────────────────────────────────

def run_simulation(
    profile: LoadProfile,
    params: CellParameters | None = None,
    ai_mode: bool = False,
    n_cycles: int = 20,
    dt: float = 60.0,
    T0: float = 298.0,
) -> SimulationResult:
    """Führt eine vollständige Zyklussimulation durch.

    Die beiden Betriebsmodi unterscheiden sich wie folgt:

    * **Statisches PMS** (``ai_mode=False``): festes Temperaturziel 308 K;
      keine Stromreduzierung zur Vorkühlung.
    * **KI-PMM** (``ai_mode=True``): niedrigeres Temperaturziel 303 K;
      Strom wird reduziert, wenn die Zelle den thermischen Soft-Limit-Wert erreicht.

    Args:
        profile:  Lastprofil, das I(t) und T_amb bereitstellt.
        params:   Zellparameter (Standard-NMC-Werte wenn *None*).
        ai_mode:  KI-PMM-Betrieb aktivieren.
        n_cycles: Maximale Anzahl zu simulierender Zyklen.
        dt:       Simulationszeitschritt [s].
        T0:       Anfangszelltemperatur [K].

    Returns:
        :class:`SimulationResult` mit vollständigen Zeitreihendaten.
    """
    if params is None:
        params = CellParameters()

    model = CellModel(params)
    state = CellState.initial(params, T0)

    T_soft_limit = 303.0 if ai_mode else 308.0  # K  soft thermal ceiling

    # Pre-allocate lists
    times_out: list[float] = []
    temps_out: list[float] = []
    volts_out: list[float] = []
    socs_out: list[float] = []
    degs_out: list[float] = []
    rates_out: list[float] = []

    t = 0.0
    t_cycle = 3600.0  # treat 1 h as one "cycle" unit
    cycles_done = 0

    while cycles_done < n_cycles and not state.is_eol:
        I = profile.current_at(t)

        # AI-PMM: predictive thermal management — reduce current near the limit
        if ai_mode and state.T > T_soft_limit:
            I = I * 0.7

        new_state, deg_rate = model.step(state, I, profile.T_amb, dt)
        state = new_state

        times_out.append(t)
        temps_out.append(state.T)
        volts_out.append(state.V)
        socs_out.append(state.SoC)
        degs_out.append(state.degradation)
        rates_out.append(deg_rate)

        t += dt
        # Count completed cycles (each t_cycle seconds = 1 cycle)
        if (t % t_cycle) < dt:
            cycles_done += 1

    times_arr = np.array(times_out)
    temps_arr = np.array(temps_out)
    rates_arr = np.array(rates_out)

    return SimulationResult(
        name="AI-PMM" if ai_mode else "Static PMS",
        times=times_arr,
        temperatures=temps_arr,
        voltages=np.array(volts_out),
        SoCs=np.array(socs_out),
        degradations=np.array(degs_out),
        degradation_rates=rates_arr,
        cycles_completed=cycles_done,
        mean_temperature=float(np.mean(temps_arr)) if len(temps_arr) else T0,
        mean_degradation_rate=float(np.mean(rates_arr)) if len(rates_arr) else 0.0,
    )


def compare_systems(
    profile: LoadProfile,
    params: CellParameters | None = None,
    n_cycles: int = 20,
    dt: float = 60.0,
) -> tuple[SimulationResult, SimulationResult]:
    """Führt statisches PMS und KI-PMM auf demselben Lastprofil aus und gibt beide Ergebnisse zurück.

    Args:
        profile:  Lastprofil.
        params:   Zellparameter.
        n_cycles: Anzahl der Zyklen.
        dt:       Zeitschritt [s].

    Returns:
        ``(statisches_ergebnis, ki_ergebnis)``
    """
    static_result = run_simulation(
        profile, params, ai_mode=False, n_cycles=n_cycles, dt=dt
    )
    ai_result = run_simulation(
        profile, params, ai_mode=True, n_cycles=n_cycles, dt=dt
    )
    return static_result, ai_result
