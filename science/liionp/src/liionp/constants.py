"""Physikalische Konstanten und Standard-Zellparameter (NMC/Graphit, Tabelle 1 aus liionp.tex).

Dieses Modul stellt alle global genutzten Konstanten bereit:

* **Physikalische Grundkonstanten**: universelle Gaskonstante, Faraday-Konstante.
* **Standard-NMC/Graphit-Zellparameter**: Kapazität, Spannungsgrenzen, Widerstand,
  SEI-Parameter, Wärmeparameter und Degradationsgewichte (vgl. Tabelle 1).
* **Mechanikparameter**: Separatorgeometrie, Tortuosität, Quellkoeffizienten.
* **E-Fahrzeug-Modellparameter**: Kälteleistungskoeffizient, Referenztemperatur.
"""

# ── Physical constants ────────────────────────────────────────────────────────
R_GAS: float = 8.314       # J/(mol·K)  universal gas constant
F_FARADAY: float = 96485.0  # C/mol     Faraday constant

# ── Default NMC/Graphit cell parameters (Table 1) ────────────────────────────
Q0_DEFAULT: float = 3.0        # Ah   nominal capacity
V_NOM_DEFAULT: float = 3.7     # V    nominal voltage
V_MAX_DEFAULT: float = 4.2     # V    upper voltage limit
V_MIN_DEFAULT: float = 2.5     # V    lower voltage limit
R_INT_DEFAULT: float = 0.050   # Ohm  internal resistance (50 mΩ)
E_A_SEI_DEFAULT: float = 50_000.0  # J/mol  SEI activation energy
M_CELL_CP_DEFAULT: float = 45.0    # J/K    thermal mass (m_cell * c_p)
R_TH_DEFAULT: float = 3.5          # K/W    thermal resistance
D_EOL: float = 0.20                # –      end-of-life degradation threshold

# ── SEI model defaults ────────────────────────────────────────────────────────
K_SEI_DEFAULT: float = 1e-17   # m²/s   SEI growth pre-exponential factor
ALPHA_SEI_DEFAULT: float = 1e-4  # Ah/m  capacity-loss coefficient per SEI metre

# ── Overvoltage / undervoltage degradation ────────────────────────────────────
K_OV_DEFAULT: float = 0.01   # a.u.
BETA_OV_DEFAULT: float = 10.0  # 1/V
K_UV_DEFAULT: float = 0.01
BETA_UV_DEFAULT: float = 10.0  # 1/V

# ── Degradation weighting coefficients ───────────────────────────────────────
GAMMA_T_DEFAULT: float = 0.65   # thermal weight
GAMMA_V_DEFAULT: float = 0.35   # voltage weight  (γ_V* in the paper)
GAMMA_C_DEFAULT: float = 0.10   # cyclic weight
GAMMA_M_DEFAULT: float = 0.15   # mechanical weight  (ρ_M = 15 %)

# ── Mechanical / plate-distance model ────────────────────────────────────────
D_SEP_0: float = 25e-6    # m   nominal PE-separator thickness (25 µm)
D_SEP_MIN: float = 20e-6  # m
D_SEP_MAX: float = 35e-6  # m
TAU_0: float = 2.5        # –   reference tortuosity
GAMMA_TAU: float = 1.8    # –   tortuosity exponent
EPS_SEP: float = 0.40     # –   separator porosity
SIGMA_EL: float = 1.0     # S/m electrolyte conductivity
K_MECH_DEFAULT: float = 0.5   # a.u.
M_MECH_DEFAULT: float = 2.5   # –    exponent in mechanical degradation law
SIGMA_YIELD_DEFAULT: float = 1.0   # a.u.  yield stress

# SoC-dependent swelling polynomial coefficients (eq. 47, NMC/Graphit)
# ε_vol(SoC) = a0 + a1·SoC + a2·SoC² + a3·SoC³
EPS_VOL_COEFFS: tuple[float, float, float, float] = (0.0, 0.062, 0.015, 0.003)

# ── EV / cold-weather model ──────────────────────────────────────────────────
KAPPA_T: float = 0.007      # K⁻¹  cold-capacity coefficient
T_REF_COLD: float = 293.0   # K    reference temperature (20 °C)
