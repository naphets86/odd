"""Aerodynamisches Modell des Wanderfalken-Sturzfluges (*Falco peregrinus*).

Dieses Modul implementiert:

- **Widerstandsmodell** (Gl. cd_model): zeitvarianter Widerstandsbeiwert
  abhängig von Anstellwinkel und Flügelfaltungsgrad.
- **Auftriebsmodell** (Gl. lift): modifizierte Dünnprofiltheorie.
- **Zeitvariante Referenzfläche** (Gl. aref): Flügelfalte erzeugt
  veränderliche Querschnittsfläche.
- **Aerodynamische Kräfte**: Widerstands- und Auftriebskraft.
- **Grenzgeschwindigkeit** (Satz 3.1): analytische Obergrenze der
  Sturzfluggeschwindigkeit.

Physikalische Konstanten
------------------------
- `RHO_0`   — Luftdichte auf Meereshöhe (ISA) [kg/m³]
- `H_SCALE` — Atmosphärische Skalenhöhe [m]
- `G`       — Erdbeschleunigung [m/s²]
"""

import math

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO_0: float = 1.225        # kg/m³  — ISA sea-level air density
H_SCALE: float = 8500.0     # m      — atmospheric scale height
G: float = 9.81             # m/s²   — gravitational acceleration

# ---------------------------------------------------------------------------
# Drag model coefficients  (eq. cd_model)
# ---------------------------------------------------------------------------
C_D0: float = 0.028         # body drag (fuselage / rump)
K_ALPHA: float = 0.18       # angle-of-attack factor
K_BETA: float = 0.42        # wing-fold coupling factor
C_DW: float = 0.15          # wing additional drag

# ---------------------------------------------------------------------------
# Lift model coefficients  (thin-airfoil model)
# ---------------------------------------------------------------------------
C_L_ALPHA: float = 4.7      # lift-curve slope
C_L0: float = 0.05          # zero-alpha lift offset

# ---------------------------------------------------------------------------
# Reference area parameters for the drone  (eq. aref)
# ---------------------------------------------------------------------------
A_CS_MIN: float = 5.0e-3    # m²  — minimum cross-section (wings fully folded)
A_CS_MAX: float = 0.18      # m²  — maximum cross-section (wings fully open)


# ---------------------------------------------------------------------------
# Atmosphere
# ---------------------------------------------------------------------------

def air_density(h: float) -> float:
    """Berechne die Luftdichte [kg/m³] in Höhe *h* [m] über NN.

    Verwendet das exponentielle Atmosphärenmodell:

        ρ(h) = ρ₀ · exp(−h / H_s)

    Parameters
    ----------
    h : float
        Höhe über dem Meeresspiegel [m].

    Returns
    -------
    float
        Luftdichte ρ(h) [kg/m³].
    """
    return RHO_0 * math.exp(-h / H_SCALE)


# ---------------------------------------------------------------------------
# Aerodynamic coefficients
# ---------------------------------------------------------------------------

def drag_coefficient(alpha: float, beta_fold: float) -> float:
    """Berechne den aerodynamischen Widerstandsbeiwert C_D.

    Das Modell setzt sich zusammen aus Rumpfwiderstand, anstellwinkelabhängigem
    Term und einem Strafterm für ausgefahrene Flügel (Gl. cd_model):

        C_D = C_D0 + k_α·α² + k_β·(1 − β_fold)²·C_Dw

    Parameters
    ----------
    alpha : float
        Anstellwinkel [rad].
    beta_fold : float
        Flügelfaltungsgrad β_fold ∈ [0, 1]
        (0 = vollständig gefaltet, 1 = vollständig geöffnet).

    Returns
    -------
    float
        Widerstandsbeiwert C_D [dimensionslos].
    """
    return C_D0 + K_ALPHA * alpha ** 2 + K_BETA * (1.0 - beta_fold) ** 2 * C_DW


def lift_coefficient(alpha: float, beta_fold: float) -> float:
    """Berechne den aerodynamischen Auftriebsbeiwert C_L.

    Basiert auf der modifizierten Dünnprofiltheorie:

        C_L = C_Lα · α · β_fold + C_L0

    Bei vollständig gefalteten Flügeln (β_fold = 0) reduziert sich der Auftrieb
    auf den Restterm C_L0.

    Parameters
    ----------
    alpha : float
        Anstellwinkel [rad].
    beta_fold : float
        Flügelfaltungsgrad β_fold ∈ [0, 1].

    Returns
    -------
    float
        Auftriebsbeiwert C_L [dimensionslos].
    """
    return C_L_ALPHA * alpha * beta_fold + C_L0


# ---------------------------------------------------------------------------
# Reference area
# ---------------------------------------------------------------------------

def reference_area(
    beta_fold: float,
    a_cs_min: float = A_CS_MIN,
    a_cs_max: float = A_CS_MAX,
) -> float:
    """Berechne die zeitvariante aerodynamische Referenzfläche [m²] (Gl. aref).

    Die Flügelfalte erzeugt eine veränderliche Querschnittsfläche:

        A_ref = A_cs_min + (A_cs_max − A_cs_min) · β_fold^(2/3)

    Parameters
    ----------
    beta_fold : float
        Flügelfaltungsgrad β_fold ∈ [0, 1].
    a_cs_min : float
        Minimale Querschnittsfläche (Flügel vollständig gefaltet) [m²].
    a_cs_max : float
        Maximale Querschnittsfläche (Flügel vollständig geöffnet) [m²].

    Returns
    -------
    float
        Aerodynamische Referenzfläche A_ref [m²].
    """
    return a_cs_min + (a_cs_max - a_cs_min) * beta_fold ** (2.0 / 3.0)


# ---------------------------------------------------------------------------
# Aerodynamic forces
# ---------------------------------------------------------------------------

def drag_force(
    v: float,
    h: float,
    alpha: float,
    beta_fold: float,
    a_cs_min: float = A_CS_MIN,
    a_cs_max: float = A_CS_MAX,
) -> float:
    """Berechne die aerodynamische Widerstandskraft [N] (Gl. drag).

        D = ½ · ρ(h) · v² · C_D(α, β_fold) · A_ref(β_fold)

    Parameters
    ----------
    v : float
        Fluggeschwindigkeit [m/s].
    h : float
        Höhe über dem Meeresspiegel [m].
    alpha : float
        Anstellwinkel [rad].
    beta_fold : float
        Flügelfaltungsgrad β_fold ∈ [0, 1].
    a_cs_min : float
        Minimale Querschnittsfläche [m²].
    a_cs_max : float
        Maximale Querschnittsfläche [m²].

    Returns
    -------
    float
        Widerstandskraft D [N].
    """
    rho = air_density(h)
    cd = drag_coefficient(alpha, beta_fold)
    a_ref = reference_area(beta_fold, a_cs_min, a_cs_max)
    return 0.5 * rho * v ** 2 * cd * a_ref


def lift_force(
    v: float,
    h: float,
    alpha: float,
    beta_fold: float,
    s_w: float = 0.12,
) -> float:
    """Berechne die aerodynamische Auftriebskraft [N] (Gl. lift).

        L = ½ · ρ(h) · v² · C_L(α, β_fold) · S_w

    Parameters
    ----------
    v : float
        Fluggeschwindigkeit [m/s].
    h : float
        Höhe über dem Meeresspiegel [m].
    alpha : float
        Anstellwinkel [rad].
    beta_fold : float
        Flügelfaltungsgrad β_fold ∈ [0, 1].
    s_w : float
        Flügelreferenzfläche S_w [m²].

    Returns
    -------
    float
        Auftriebskraft L [N].
    """
    rho = air_density(h)
    cl = lift_coefficient(alpha, beta_fold)
    return 0.5 * rho * v ** 2 * cl * s_w


# ---------------------------------------------------------------------------
# Terminal velocity  (Theorem 3.1)
# ---------------------------------------------------------------------------

def terminal_velocity(
    mass: float,
    rho: float = RHO_0,
    c_d_min: float = C_D0,
    a_cs_min: float = 0.004,
) -> float:
    """Berechne die Grenzgeschwindigkeit des vertikalen Sturzfluges [m/s].

    Satz 3.1 — Grenzgeschwindigkeit des Sturzfluges:
    Im vertikalen Sturzflug (γ = 90°) mit vollständig gefalteten Flügeln
    existiert eine eindeutige Grenzgeschwindigkeit:

        v_max = sqrt(2 · m · g / (ρ · C_D_min · A_cs_min))

    Parameters
    ----------
    mass : float
        Körpermasse [kg].
    rho : float
        Luftdichte [kg/m³] (Standard: Meereshöhe ISA).
    c_d_min : float
        Minimaler Widerstandsbeiwert (gefaltete Flügel) [dimensionslos].
    a_cs_min : float
        Minimale Querschnittsfläche (gefaltete Flügel) [m²].

    Returns
    -------
    float
        Theoretische Grenzgeschwindigkeit v_max [m/s].
    """
    return math.sqrt(2.0 * mass * G / (rho * c_d_min * a_cs_min))
