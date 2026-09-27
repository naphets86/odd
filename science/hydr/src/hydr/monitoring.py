"""
monitoring.py
=============
Inline-Überwachung elastischer Parameter hydraulischer Flüssigkeiten.

Gleichungen aus Epp (2026):
  Gl. (31)  Ultraschall-Kompressionsmodul  K_eff = ρ · (L / Δt)²
  Satz      Grenzkontaktkraft für Cobot-Sicherheit
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Ultraschall-Laufzeitmessung
# ---------------------------------------------------------------------------

def ultrasonic_bulk_modulus(rho: float, L: float, delta_t: float) -> float:
    """Effektiver Kompressionsmodul aus Ultraschall-Laufzeitmessung.

       K_eff = ρ · (L / Δt)²   (Gl. 31)

    Parameters
    ----------
    rho:     Dichte der Flüssigkeit [kg/m³]
    L:       Messstrecke (Wandabstand) [m]
    delta_t: Laufzeit des Ultraschallpulses [s]

    Returns
    -------
    K_eff [Pa]
    """
    if rho <= 0:
        raise ValueError("rho muss positiv sein.")
    if L <= 0:
        raise ValueError("L muss positiv sein.")
    if delta_t <= 0:
        raise ValueError("delta_t muss positiv sein.")
    return rho * (L / delta_t) ** 2


# ---------------------------------------------------------------------------
# Schallgeschwindigkeit aus Laufzeitmessung
# ---------------------------------------------------------------------------

def sound_speed_from_transit(L: float, delta_t: float) -> float:
    """Schallgeschwindigkeit c_s = L / Δt aus Laufzeitmessung.

    Parameters
    ----------
    L:       Messstrecke [m]
    delta_t: Laufzeit [s]

    Returns
    -------
    c_s [m/s]
    """
    if L <= 0:
        raise ValueError("L muss positiv sein.")
    if delta_t <= 0:
        raise ValueError("delta_t muss positiv sein.")
    return L / delta_t


# ---------------------------------------------------------------------------
# Rekuperationsgrad
# ---------------------------------------------------------------------------

def recuperation_efficiency(
    K_eff: float,
    V_acc: float,
    E_loss: float,
) -> float:
    """Energierückgewinnungsgrad mit Druckakkumulator.

       η_Rek = K_eff · V_Acc / (K_eff · V_Acc + E_Verlust)   (Gl. Rekuperation)

    Parameters
    ----------
    K_eff:  Effektiver Kompressionsmodul der Flüssigkeit [Pa]
    V_acc:  Akkumulatorvolumen [m³]
    E_loss: Verlustenergie pro Hub [J]

    Returns
    -------
    η_Rek [dimensionslos, 0–1]
    """
    if K_eff <= 0 or V_acc <= 0:
        raise ValueError("K_eff und V_acc müssen positiv sein.")
    if E_loss < 0:
        raise ValueError("E_loss muss nicht-negativ sein.")
    stored = K_eff * V_acc
    return stored / (stored + E_loss)


# ---------------------------------------------------------------------------
# Konditionsindex  (einfaches Ampel-Kriterium)
# ---------------------------------------------------------------------------

class OilCondition:
    """Konditionsindex eines Hydrauliköls im Betrieb.

    Umfasst die drei Leitgrößen:
      - Effektiver Kompressionsmodul K_eff(t)
      - Viskosität eta(t)
      - Säurezahl TAN(t)

    und gibt eine Ampel-Bewertung (GREEN / YELLOW / RED) zurück.
    """

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"

    # Grenzwerte (Vorgabewerte gem. Tab. 9.1 des Papers)
    K_EFF_LIMIT_YELLOW: float = 0.90   # relativer Rückgang gegenüber Frischöl
    K_EFF_LIMIT_RED: float = 0.80
    ETA_CHANGE_YELLOW: float = 0.15    # ±15 % Viskositätsänderung
    ETA_CHANGE_RED: float = 0.25       # ±25 %
    TAN_YELLOW: float = 1.5            # mgKOH/g
    TAN_RED: float = 2.0

    @classmethod
    def evaluate(
        cls,
        K_eff: float,
        K_eff_fresh: float,
        eta: float,
        eta_fresh: float,
        tan: float,
    ) -> str:
        """Bewertet den Ölzustand anhand dreier Kenngrößen.

        Parameters
        ----------
        K_eff:       Aktueller effektiver Kompressionsmodul [Pa]
        K_eff_fresh: Frischöl-Kompressionsmodul [Pa]
        eta:         Aktuelle Viskosität [Pa·s]
        eta_fresh:   Frischöl-Viskosität [Pa·s]
        tan:         Aktuelle Säurezahl [mgKOH/g]

        Returns
        -------
        "GREEN", "YELLOW" oder "RED"
        """
        if K_eff_fresh <= 0 or eta_fresh <= 0:
            raise ValueError("Frischöl-Referenzwerte müssen positiv sein.")

        k_ratio = K_eff / K_eff_fresh
        eta_change = abs(eta - eta_fresh) / eta_fresh

        is_red = (
            k_ratio < cls.K_EFF_LIMIT_RED
            or eta_change > cls.ETA_CHANGE_RED
            or tan >= cls.TAN_RED
        )
        is_yellow = (
            k_ratio < cls.K_EFF_LIMIT_YELLOW
            or eta_change > cls.ETA_CHANGE_YELLOW
            or tan >= cls.TAN_YELLOW
        )

        if is_red:
            return cls.RED
        if is_yellow:
            return cls.YELLOW
        return cls.GREEN
