"""
hearing.py – HearingProfile: cat, dog and comparison framework
===============================================================
Based on Sections 10–12 ("Das Gehör der Katze" & comparison) of
"Von der Diskretion".

Provides
--------
• ``HearingProfile``  – dataclass capturing all physiologically relevant
  parameters of a mammalian auditory system.
• ``CAT``             – Felis catus profile (literature values).
• ``DOG``             – Canis lupus familiaris profile (literature values).
• Shannon capacity calculation.
• Membrane efficiency criterion Ω.
• Side-by-side comparison helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


# ---------------------------------------------------------------------------
# HearingProfile dataclass
# ---------------------------------------------------------------------------


@dataclass
class HearingProfile:
    """Container for physiological parameters of a mammalian hearing system.

    Attributes
    ----------
    species:
        Common name of the species.
    latin_name:
        Scientific (Latin) name.
    f_min:
        Lower hearing threshold frequency [Hz].
    f_max:
        Upper hearing threshold frequency [Hz].
    L_membrane:
        Basilar membrane length [mm].
    beta:
        Tonotopic gradient [mm⁻¹].
    f_apex:
        Characteristic frequency at the cochlear apex [Hz].
    Q_mean:
        Representative (geometric) mean Q₁₀dB selectivity factor.
    Q_range:
        (min, max) range of observed Q₁₀dB values.
    sigma_t:
        Neural timing jitter (standard deviation) [s].  Sets temporal resolution.
    dynamic_range_db:
        Usable dynamic range of the auditory system [dB].
    n_outer_hair_cells:
        Number of outer hair cells in one cochlea.
    n_spiral_ganglion:
        Number of auditory nerve fibres (spiral ganglion).
    pinna_range_deg:
        Maximum pinna rotation per ear [deg].
    pinna_muscles:
        Number of pinna muscles.
    snr_db:
        Effective SNR of the cochlear system [dB] (used for Shannon capacity).
    medium:
        Propagation medium (``'air'`` or ``'water'``).
    notes:
        Free-text notes.
    """

    species: str
    latin_name: str
    f_min: float            # Hz
    f_max: float            # Hz
    L_membrane: float       # mm
    beta: float             # mm⁻¹
    f_apex: float           # Hz
    Q_mean: float           # dimensionless
    Q_range: tuple[float, float] = field(default_factory=lambda: (1.0, 1.0))
    sigma_t: float = 100e-6         # s
    dynamic_range_db: float = 100.0  # dB
    n_outer_hair_cells: int = 0
    n_spiral_ganglion: int = 0
    pinna_range_deg: float = 0.0    # deg
    pinna_muscles: int = 0
    snr_db: float = 50.0
    medium: str = "air"
    notes: str = ""

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def bandwidth(self) -> float:
        """Usable hearing bandwidth B = f_max - f_min [Hz]."""
        return self.f_max - self.f_min

    @property
    def bandwidth_efficiency(self) -> float:
        """Bandwidth per unit membrane length η_B = B / L [Hz/mm]."""
        return self.bandwidth / self.L_membrane

    @property
    def shannon_capacity_bits_per_second(self) -> float:
        """Shannon capacity C = B · log₂(1 + SNR) [bit/s]."""
        snr_linear = 10.0 ** (self.snr_db / 10.0)
        return self.bandwidth * np.log2(1.0 + snr_linear)

    @property
    def temporal_resolution_hz(self) -> float:
        """Equivalent Nyquist sampling rate of the temporal channel [Hz].

        .. math::
            f_{s,\\text{Zeit}} = \\frac{1}{2\\sigma_t}
        """
        return 1.0 / (2.0 * self.sigma_t)

    @property
    def snr_linear(self) -> float:
        """SNR as a linear (power) ratio."""
        return float(10.0 ** (self.snr_db / 10.0))


# ---------------------------------------------------------------------------
# Predefined species profiles
# ---------------------------------------------------------------------------

#: Domestic cat (*Felis catus*) – Section 10 of "Von der Diskretion"
CAT = HearingProfile(
    species="Cat",
    latin_name="Felis catus",
    f_min=48.0,
    f_max=85_000.0,
    L_membrane=25.0,
    beta=2.1,
    f_apex=200.0,
    Q_mean=17.0,           # geometric mean of 8.5 … 30
    Q_range=(8.5, 30.0),
    sigma_t=100e-6,        # < 100 µs
    dynamic_range_db=100.0,
    n_outer_hair_cells=10_000,
    n_spiral_ganglion=50_000,
    pinna_range_deg=180.0,
    pinna_muscles=32,
    snr_db=55.0,
    medium="air",
    notes=(
        "Best-studied mammalian auditory system. "
        "Active cochlear amplifier (Prestin) maximised. "
        "Operates near thermal noise floor."
    ),
)

#: Domestic dog (*Canis lupus familiaris*) – Section 11 of "Von der Diskretion"
DOG = HearingProfile(
    species="Dog",
    latin_name="Canis lupus familiaris",
    f_min=40.0,
    f_max=65_000.0,
    L_membrane=32.0,
    beta=1.7,
    f_apex=150.0,
    Q_mean=7.0,            # geometric mean of 4 … 12
    Q_range=(4.0, 12.0),
    sigma_t=400e-6,        # 300 … 500 µs
    dynamic_range_db=80.0,
    n_outer_hair_cells=10_500,
    n_spiral_ganglion=30_000,
    pinna_range_deg=90.0,
    pinna_muscles=18,
    snr_db=45.0,
    medium="air",
    notes=(
        "Broader membrane, lower tonotopic gradient. "
        "Prestin expression ~30-40 % lower than cat. "
        "Lower Q and timing resolution than cat."
    ),
)

#: Human (*Homo sapiens*) – for reference
HUMAN = HearingProfile(
    species="Human",
    latin_name="Homo sapiens",
    f_min=20.0,
    f_max=20_000.0,
    L_membrane=33.0,
    beta=0.9,
    f_apex=20.0,
    Q_mean=6.0,
    Q_range=(3.0, 10.0),
    sigma_t=600e-6,
    dynamic_range_db=120.0,
    n_outer_hair_cells=12_000,
    n_spiral_ganglion=30_000,
    pinna_range_deg=5.0,
    pinna_muscles=3,
    snr_db=40.0,
    medium="air",
    notes="Reference: ISO 226 equal-loudness contours. PLT ~0 dB SPL threshold.",
)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def shannon_capacity(B: float, snr_db: float) -> float:
    """Shannon channel capacity C = B · log₂(1 + S/N) [bit/s].

    Parameters
    ----------
    B:
        Bandwidth [Hz].  Must be positive.
    snr_db:
        Signal-to-noise ratio [dB].

    Raises
    ------
    ValueError
        If *B* ≤ 0.
    """
    if B <= 0:
        raise ValueError(f"Bandwidth must be positive, got B={B}")
    snr_linear = 10.0 ** (snr_db / 10.0)
    return float(B * np.log2(1.0 + snr_linear))


def membrane_efficiency_omega(
    profile: HearingProfile,
    reference: HearingProfile = CAT,
) -> float:
    """Composite membrane efficiency criterion Ω.

    .. math::
        \\Omega = \\frac{\\eta_B}{\\eta_{B,\\text{ref}}}
                  \\cdot \\frac{\\bar{Q}}{\\bar{Q}_{\\text{ref}}}
                  \\cdot \\frac{D}{D_{\\text{ref}}}
                  \\cdot \\frac{1/\\sigma_t}{1/\\sigma_{t,\\text{ref}}}

    All four terms are normalised to the reference profile (default: cat).

    Parameters
    ----------
    profile:
        Hearing profile of the species to evaluate.
    reference:
        Reference profile (Ω = 1 for the reference).

    Returns
    -------
    float
        Dimensionless Ω score.
    """
    eta_ratio = profile.bandwidth_efficiency / reference.bandwidth_efficiency
    q_ratio = profile.Q_mean / reference.Q_mean
    d_ratio = profile.dynamic_range_db / reference.dynamic_range_db
    t_ratio = reference.sigma_t / profile.sigma_t   # higher temporal res → larger ratio
    return float(eta_ratio * q_ratio * d_ratio * t_ratio)


def compare_profiles(*profiles: HearingProfile) -> Dict[str, dict]:
    """Return a structured comparison dictionary for one or more profiles.

    Parameters
    ----------
    *profiles:
        One or more :class:`HearingProfile` instances.

    Returns
    -------
    dict
        Keyed by ``profile.species``, containing all relevant metrics.
    """
    result: Dict[str, dict] = {}
    ref = CAT
    for p in profiles:
        result[p.species] = {
            "latin": p.latin_name,
            "f_min_Hz": p.f_min,
            "f_max_Hz": p.f_max,
            "bandwidth_Hz": p.bandwidth,
            "L_membrane_mm": p.L_membrane,
            "beta_per_mm": p.beta,
            "Q_mean": p.Q_mean,
            "Q_range": p.Q_range,
            "sigma_t_us": p.sigma_t * 1e6,
            "dynamic_range_dB": p.dynamic_range_db,
            "bandwidth_efficiency_kHz_per_mm": p.bandwidth_efficiency / 1e3,
            "temporal_resolution_kHz": p.temporal_resolution_hz / 1e3,
            "shannon_capacity_Mbit_per_s": p.shannon_capacity_bits_per_second / 1e6,
            "omega": membrane_efficiency_omega(p, reference=ref),
        }
    return result


def capacity_ratio(profile_a: HearingProfile, profile_b: HearingProfile) -> float:
    """Return Shannon capacity ratio C_a / C_b."""
    return profile_a.shannon_capacity_bits_per_second / profile_b.shannon_capacity_bits_per_second


def bandwidth_ratio(profile_a: HearingProfile, profile_b: HearingProfile) -> float:
    """Return bandwidth ratio B_a / B_b."""
    return profile_a.bandwidth / profile_b.bandwidth
