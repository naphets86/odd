"""
Spatial Localization: ITD, ILD, HRTFs and Localization Accuracy.

Implementiert:
- Interaurale Zeit-Differenz (ITD) (Eq. 8.1-8.2)
- Interaurale Pegel-Unterschied (ILD)
- Just Noticeable Difference (JND) und Fisher-Information (Eq. 8.3)
- Cramér-Rao-Schranke (Definition 9.1)
- Hörverlust und räumliche Desorientation (Satz 9.1)
"""

from typing import Tuple, Optional
import numpy as np
from .feline_auditory import FelineAuditory, Species


class SpatialLocalization:
    """
    Modell der auditiven Raumorientierung durch ITD und ILD.
    """
    
    def __init__(self, species: Species = Species.CAT):
        """
        Initialisiere das Lokalisierungs-Modell.
        
        Args:
            species: Tierspezies
        """
        self.auditory_system = FelineAuditory(species)
        self.species = species
        self.c_sound = 343.0  # m/s bei 20°C
    
    def itd_woodworth_schlosberg(
        self, azimuth_deg: float, elevation_deg: float = 0.0
    ) -> float:
        """
        Berechne die interaurale Zeit-Differenz nach Woodworth-Schlosberg.
        
        Formula (Eq. 8.1): Δt_ITD = (R_head / c) * (sin(θ) + θ_0)
        
        Args:
            azimuth_deg: Azimutwinkel in Grad
            elevation_deg: Elevationswinkel (optional)
        
        Returns:
            ITD in Sekunden
        """
        r_head_m = self.auditory_system.properties.head_radius_cm / 100.0
        theta_rad = np.radians(azimuth_deg)
        
        # Vereinfachte Form für Frontal-Ebene
        itd = (r_head_m / self.c_sound) * np.sin(theta_rad)
        
        return itd
    
    def itd_maximum(self) -> float:
        """
        Berechne die maximale ITD (bei θ = 90°).
        
        Returns:
            Maximale ITD in Sekunden
        """
        r_head_m = self.auditory_system.properties.head_radius_cm / 100.0
        itd_max = r_head_m / self.c_sound
        return itd_max
    
    def ild_frequency_dependent(
        self, freq_hz: float, azimuth_deg: float
    ) -> float:
        """
        Berechne den interauralen Pegel-Unterschied (ILD) frequenzabhängig.
        
        Formula (Eq. 8.2): ΔL_ILD = 20*log₁₀(p_R / p_L)
        
        Die ILD wird durch Kopfschatten und Pinna-Effekte bestimmt.
        Für hohe Frequenzen (f > 4 kHz) dominant.
        
        Args:
            freq_hz: Frequenz in Hz
            azimuth_deg: Azimutwinkel in Grad
        
        Returns:
            ILD in dB
        """
        # Vereinfachtes Modell: ILD nimmt mit Frequenz und Azimuth zu
        # Hohe Frequenzen: Kopfschatten prominent
        # Niedrige Frequenzen: schwach
        
        theta_rad = np.radians(azimuth_deg)
        
        # Frequenz-Abhängigkeit
        if freq_hz < 4000:
            # Niedrige Frequenzen: minimale ILD
            ild = 0.1 * freq_hz / 4000.0 * np.sin(theta_rad)
        else:
            # Hohe Frequenzen: starke ILD
            # Maximale ILD bei hohen Frequenzen: ~15 dB
            ild = 15.0 * np.sin(theta_rad) * np.log10(freq_hz / 4000.0)
        
        return float(np.clip(ild, -20.0, 20.0))
    
    def just_noticeable_difference_itd(self) -> float:
        """
        Berechne die Just Noticeable Difference (JND) für ITD.
        
        Basierend auf neuronaler zeitlicher Auflösung.
        
        Returns:
            JND für ITD in Sekunden
        """
        # JND ist durch die Phasenlocking-Frequenz begrenzt
        f_phase_lock = self.auditory_system.phase_locking_frequency()
        
        # JND ≈ 1 / (2 * f_phase_lock)
        jnd_itd = 1.0 / (50.0 * f_phase_lock)
        return jnd_itd
    
    def fisher_information_itd(
        self, signal_power: float = 1.0, snr_db: float = 10.0
    ) -> float:
        """
        Berechne die Fisher-Information für ITD-Schätzung.
        
        Formula (Eq. 8.3): I_θ = (4π² * σ_n²) / σ_s² * (d(HRTF)/dθ)²
        
        Args:
            signal_power: Signalleistung
            snr_db: Signal-Rausch-Verhältnis in dB
        
        Returns:
            Fisher-Information in bits/radian
        """
        snr_linear = 10 ** (snr_db / 10.0)
        
        # Approximation: I ∝ SNR * (dITD/dθ)²
        # dITD/dθ ≈ (R_head/c) * cos(θ), Maximum bei θ=0: ≈ R_head/c
        r_head_m = self.auditory_system.properties.head_radius_cm / 100.0
        ditd_dtheta = r_head_m / self.c_sound
        
        fisher_info = snr_linear * signal_power * ditd_dtheta**2
        return fisher_info
    
    def cramer_rao_bound(
        self, snr_db: float = 10.0
    ) -> float:
        """
        Berechne die Cramér-Rao-Schranke für Lokalisierungsfehler.
        
        Definition (Def. 9.1): σ²_θ ≥ 1 / I_θ
        
        Args:
            snr_db: Signal-Rausch-Verhältnis in dB
        
        Returns:
            Lokalisierungsfehler in Grad
        """
        fisher_info = self.fisher_information_itd(snr_db=snr_db)
        
        if fisher_info <= 0:
            return np.inf
        
        variance_rad = 1.0 / fisher_info
        variance_deg = np.degrees(np.sqrt(variance_rad))
        return variance_deg
    
    def localization_accuracy(self, snr_db: float = 10.0) -> float:
        """
        Berechne die erwartete Lokalisierungsgenauigkeit.
        
        Basierend auf Cramér-Rao-Schranke mit biologischen Modifikationen.
        
        Args:
            snr_db: Signal-Rausch-Verhältnis
        
        Returns:
            Lokalisierungsgenauigkeit in Grad
        """
        # Basis-Schranke
        crb = self.cramer_rao_bound(snr_db)
        
        # Biologische Effizienzen (< 100% bei realen Systemen)
        # Katzen: höhere Effizienz (~80%)
        # Hunde: mittlere Effizienz (~50%)
        # Menschen: niedrigere Effizienz (~30%)
        
        if self.species == Species.CAT:
            efficiency = 0.8
        elif self.species == Species.DOG:
            efficiency = 0.5
        else:  # HUMAN
            efficiency = 0.3
        
        return crb / efficiency
    
    def hearing_loss_effect(
        self, frequency_hz: float, hearing_loss_db: float
    ) -> Tuple[float, float]:
        """
        Modelliere die Auswirkung von Hörverlust auf ITD-Diskrimination.
        
        Satz 9.1: Hörverlust führt zu räumlicher Desorientation.
        
        Args:
            frequency_hz: Frequenz in Hz
            hearing_loss_db: Hörverlust in dB
        
        Returns:
            Tupel (JND_normal, JND_with_loss) für ITD
        """
        # Normale JND
        jnd_normal = self.just_noticeable_difference_itd()
        
        # Mit Hörverlust: JND verschlechtert sich
        # Für jedes dB Hörverlust: ~5% Verschlechterung
        jnd_with_loss = jnd_normal * hearing_loss_db / 10.0
        
        return jnd_normal, jnd_with_loss
    
    def informational_entropy_spatial(
        self, n_sources: int = 100, hearing_loss_db: float = 0.0
    ) -> float:
        """
        Berechne die Entropie der räumlich-orientierten Information.
        
        Satz 9.1: H_spatial = -Σ p(source_i) * log₂(p(source_i))
        
        Mit Hörverlust: H_spatial → H_max = log₂(n)
        
        Args:
            n_sources: Anzahl möglicher Quellen im Raum
            hearing_loss_db: Hörverlust in dB
        
        Returns:
            Entropie in Bits
        """
        # Mit normalem Gehör: kleine Entropie (hohe Gewissheit)
        # Mit Hörverlust: große Entropie (hohe Unsicherheit)
        
        if hearing_loss_db < 20:
            # Leichter bis moderater Hörverlust
            p_correct = 1.0 - 0.01 * hearing_loss_db
            p_error = 0.01 * hearing_loss_db / (n_sources - 1)
            
            p = np.concatenate([
                [p_correct],
                [p_error] * (n_sources - 1)
            ])
        else:
            # Schwerer Hörverlust: fast uniform distribution
            p = np.ones(n_sources) / n_sources
        
        # Shannon-Entropie
        p = p[p > 0]
        entropy = -np.sum(p * np.log2(p))
        
        return entropy
    
    def bilateral_hearing_loss_asymmetry(
        self, loss_left_db: float, loss_right_db: float
    ) -> float:
        """
        Berechne die Asymmetrie bei bilateralem Hörverlust.
        
        Einseitiger oder stark asymmetrischer Hörverlust zerstört
        die binaural verarbeitung (ITD, ILD).
        
        Args:
            loss_left_db: Hörverlust linkes Ohr
            loss_right_db: Hörverlust rechtes Ohr
        
        Returns:
            Asymmetrie-Index (0 = symmetrisch, 1 = völlig asymmetrisch)
        """
        asymmetry = np.abs(loss_left_db - loss_right_db) / 100.0
        return np.clip(asymmetry, 0.0, 1.0)
    
    def front_back_confusion(self) -> float:
        """
        Berechne das Risiko für Vorne-Hinten-Verwechselung.
        
        Wird durch spektrale Richtungsmerkmale (Pinna-HRTF) reduziert.
        Mit beweglichen Ohren: Fast vollständig gelöst.
        
        Returns:
            Verwechslungswahrscheinlichkeit (0 bis 1)
        """
        # Ohne Kopfbewegung: 20-30% Verwechslungsrate
        # Mit aktiven Ohren: < 5%
        
        if self.species == Species.CAT:
            # Katzen mit sehr beweglichen Ohren
            confusion_rate = 0.05
        elif self.species == Species.DOG:
            # Hunde mit moderaten Ohrenbewegung
            confusion_rate = 0.10
        else:  # HUMAN
            # Menschen mit begrenzter Ohrenbewegung
            confusion_rate = 0.20
        
        return confusion_rate
    
    def summary(self) -> str:
        """Gib eine Zusammenfassung aus."""
        lines = [
            f"Spatial Localization ({self.species.value}):",
            f"  Head Radius: {self.auditory_system.properties.head_radius_cm:.1f} cm",
            f"  ITD_max: {self.itd_maximum() * 1e6:.1f} µs",
            f"  JND_ITD: {self.just_noticeable_difference_itd() * 1e6:.2f} µs",
            f"  Localization Accuracy (0 dB SNR): {self.localization_accuracy(snr_db=0):.2f}°",
            f"  Front-Back Confusion Rate: {self.front_back_confusion():.1%}",
        ]
        return "\n".join(lines)
