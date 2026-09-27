"""
Hearing Comparison: Formale mathematische Vergleiche zwischen Spezies.

Implementiert alle Vergleichsfaktoren aus Satz 9.2 und Korollaren 9.1-9.2
der wissenschaftlichen Arbeit.
"""

from typing import Dict, Tuple
import numpy as np
from .feline_auditory import FelineAuditory, Species


class HearingComparison:
    """Vergleichende Analyse auditiver Systeme mehrerer Spezies."""
    
    def __init__(self, species_list: list[Species] | None = None):
        """
        Initialisiere den Vergleich.
        
        Args:
            species_list: Liste der zu vergleichenden Spezies
        """
        if species_list is None:
            species_list = [Species.CAT, Species.DOG, Species.HUMAN]
        
        self.species_list = species_list
        self.systems = {
            s: FelineAuditory(s) for s in species_list
        }
    
    def frequency_bandwidth_comparison(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Vergleiche die Frequenzbandbreiten.
        
        Returns:
            Dictionary mit (f_min, f_max, ratio) für jede Spezies
        """
        result = {}
        for species, system in self.systems.items():
            f_min, f_max, ratio = system.frequency_bandwidth()
            result[species.value] = (f_min, f_max, ratio)
        return result
    
    def upper_frequency_ratios(self) -> Dict[str, float]:
        """
        Berechne Verhältnisse der oberen Grenzfrequenzen (Tabelle 9.1, Eq. 9.8).
        
        Returns:
            Dictionary mit Verhältnissen (relativ zu Hund)
        """
        f_max_values = {
            species: system.properties.f_max_hz
            for species, system in self.systems.items()
        }
        
        f_max_dog = f_max_values[Species.DOG]
        
        result = {}
        for species in self.species_list:
            ratio = f_max_values[species] / f_max_dog
            result[species.value] = ratio
        
        return result
    
    def cochlear_scaling_comparison(self) -> Dict[str, float]:
        """
        Vergleiche die Greenwood-Skalierungsparameter (Tabelle 9.1, Eq. 9.9-9.11).
        
        Returns:
            Dictionary mit Skalierungsparametern in mm⁻¹
        """
        result = {}
        for species, system in self.systems.items():
            alpha = system.cochlear_scaling_parameter()
            result[species.value] = alpha
        return result
    
    def cochlear_scaling_ratio(self) -> float:
        """Berechne das Verhältnis der Skalierungsparameter Katze/Hund."""
        cat_system = self.systems[Species.CAT]
        dog_system = self.systems[Species.DOG]
        
        alpha_cat = cat_system.cochlear_scaling_parameter()
        alpha_dog = dog_system.cochlear_scaling_parameter()
        
        return alpha_cat / alpha_dog
    
    def spectral_selectivity_comparison(self) -> Dict[str, float]:
        """
        Vergleiche die Q-Faktoren (Tabelle 9.1, Eq. 9.12).
        
        Returns:
            Dictionary mit Q-Faktoren
        """
        result = {}
        for species, system in self.systems.items():
            q = system.properties.q_factor_middle
            result[species.value] = q
        return result
    
    def spectral_selectivity_ratio(self) -> float:
        """Berechne das Verhältnis der Q-Faktoren Katze/Hund."""
        cat_q = self.systems[Species.CAT].properties.q_factor_middle
        dog_q = self.systems[Species.DOG].properties.q_factor_middle
        return cat_q / dog_q
    
    def cochlear_amplification_comparison(self) -> Dict[str, float]:
        """
        Vergleiche die cochleäre Verstärkung (Tabelle 9.1, Eq. 9.13).
        
        Returns:
            Dictionary mit Verstärkungen in dB
        """
        result = {}
        for species, system in self.systems.items():
            gain_db = system.properties.cochlear_gain_db
            result[species.value] = gain_db
        return result
    
    def cochlear_amplification_ratio(self) -> float:
        """
        Berechne das Verstärkungsverhältnis als linearer Faktor.
        
        Formula: 10^((G_cat - G_dog) / 20)
        """
        gain_cat = self.systems[Species.CAT].properties.cochlear_gain_db
        gain_dog = self.systems[Species.DOG].properties.cochlear_gain_db
        
        return 10 ** ((gain_cat - gain_dog) / 20.0)
    
    def temporal_resolution_comparison(self) -> Dict[str, float]:
        """
        Vergleiche die zeitliche Auflösung (Tabelle 9.1, Eq. 9.14).
        
        Returns:
            Dictionary mit zeitlichen Auflösungen in µs
        """
        result = {}
        for species, system in self.systems.items():
            tau_us = system.properties.refractory_period_us
            result[species.value] = tau_us
        return result
    
    def temporal_resolution_ratio(self) -> float:
        """Berechne das Verhältnis der Refraktärperioden Hund/Katze."""
        tau_cat = self.systems[Species.CAT].properties.refractory_period_us
        tau_dog = self.systems[Species.DOG].properties.refractory_period_us
        return tau_dog / tau_cat
    
    def phase_locking_frequency_comparison(self) -> Dict[str, float]:
        """
        Vergleiche maximale Phasenlocking-Frequenzen (Eq. 9.15).
        
        Returns:
            Dictionary mit Frequenzen in kHz
        """
        result = {}
        for species, system in self.systems.items():
            f_pl = system.phase_locking_frequency() / 1e3
            result[species.value] = f_pl
        return result
    
    def phase_locking_frequency_ratio(self) -> float:
        """Berechne das Verhältnis f_pl(Katze) / f_pl(Hund)."""
        f_pl_cat = self.systems[Species.CAT].phase_locking_frequency()
        f_pl_dog = self.systems[Species.DOG].phase_locking_frequency()
        return f_pl_cat / f_pl_dog
    
    def localization_accuracy_comparison(self) -> Dict[str, Tuple[float, float]]:
        """
        Vergleiche Lokalisierungsgenauigkeiten (Tabelle 9.1, Eq. 9.21).
        
        Returns:
            Dictionary mit (fixed_deg, mobile_deg) für jede Spezies
        """
        result = {}
        for species, system in self.systems.items():
            fixed = system.localization_accuracy_fixed()
            mobile = system.localization_accuracy_mobile()
            result[species.value] = (fixed, mobile)
        return result
    
    def localization_accuracy_ratio_fixed(self) -> float:
        """Berechne das Verhältnis σ_θ(Hund) / σ_θ(Katze) bei fixiertem Kopf."""
        sigma_cat = self.systems[Species.CAT].localization_accuracy_fixed()
        sigma_dog = self.systems[Species.DOG].localization_accuracy_fixed()
        return sigma_dog / sigma_cat
    
    def localization_accuracy_ratio_mobile(self) -> float:
        """Berechne das Verhältnis σ_θ(Hund) / σ_θ(Katze) mit beweglichen Ohren."""
        sigma_cat = self.systems[Species.CAT].localization_accuracy_mobile()
        sigma_dog = self.systems[Species.DOG].localization_accuracy_mobile()
        return sigma_dog / sigma_cat
    
    def superiority_table(
        self, reference_species: Species = Species.DOG
    ) -> Dict[str, Dict[str, float]]:
        """
        Erstelle eine vollständige Überlegenheits-Tabelle (Tabelle 9.1).
        
        Args:
            reference_species: Referenzart für Vergleich
        
        Returns:
            Verschachtelte Dictionary mit allen Überlegenheitsfaktoren
        """
        reference_system = self.systems[reference_species]
        
        table = {}
        
        for species in self.species_list:
            system = self.systems[species]
            factors = system.superiority_index(reference_system)
            table[species.value] = factors
        
        return table
    
    def geometric_mean_superiority(self, include_mobile: bool = True) -> Dict[str, float]:
        """
        Berechne die geometrischen Mittel der Überlegenheit (Eq. 9.22-9.23).
        
        Args:
            include_mobile: Ob mobile Lokalisierung eingeschlossen werden soll
        
        Returns:
            Dictionary mit geometrischen Mitteln
        """
        result = {}
        
        for species in self.species_list:
            system = self.systems[species]
            
            if species == Species.DOG:
                # Referenz hat Verhältnis 1.0
                result[species.value] = 1.0
            else:
                dog_system = self.systems[Species.DOG]
                factors = system.superiority_index(dog_system)
                
                if include_mobile:
                    result[species.value] = factors['geometric_mean_mobile']
                else:
                    result[species.value] = factors['geometric_mean_fixed']
        
        return result
    
    def summary_table(self) -> str:
        """Gib eine formatierte Zusammenfassungs-Tabelle aus."""
        lines = [
            "=" * 100,
            "AUDITORY SYSTEM COMPARISON",
            "=" * 100,
            ""
        ]
        
        # Frequenzbereich
        freq_comp = self.frequency_bandwidth_comparison()
        lines.append("Frequency Range:")
        for species, (f_min, f_max, ratio) in freq_comp.items():
            lines.append(
                f"  {species:30s}: {f_min:8.1f} Hz - {f_max:8.1f} Hz ({ratio:6.2f}x)"
            )
        lines.append("")
        
        # Q-Faktoren
        q_comp = self.spectral_selectivity_comparison()
        lines.append("Quality Factors (Q) at Middle Frequency:")
        for species, q in q_comp.items():
            lines.append(f"  {species:30s}: {q:6.2f}")
        lines.append("")
        
        # Cochleäre Verstärkung
        gain_comp = self.cochlear_amplification_comparison()
        lines.append("Cochlear Amplification:")
        for species, gain_db in gain_comp.items():
            lines.append(f"  {species:30s}: {gain_db:6.1f} dB")
        lines.append("")
        
        # Lokalisierungsgenauigkeit
        loc_comp = self.localization_accuracy_comparison()
        lines.append("Localization Accuracy (degrees):")
        for species, (fixed, mobile) in loc_comp.items():
            lines.append(
                f"  {species:30s}: {fixed:6.2f}° (fixed), {mobile:6.2f}° (mobile)"
            )
        lines.append("")
        
        # Überlegenheit (relativ zu Hund)
        lines.append("Superiority Factors (Cat vs. Dog):")
        cat_system = self.systems[Species.CAT]
        dog_system = self.systems[Species.DOG]
        factors = cat_system.superiority_index(dog_system)
        for key, value in factors.items():
            if key.startswith('geometric'):
                lines.append(f"  {key:35s}: {value:8.2f}")
        
        lines.append("")
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    def comparison_dataframe_format(self) -> str:
        """Gib Vergleichdaten in tabellarischem Format aus."""
        species_names = [s.value for s in self.species_list]
        
        data = {}
        
        # Sammle alle Vergleichdaten
        data['F_max (Hz)'] = {
            s: self.systems[s].properties.f_max_hz
            for s in self.species_list
        }
        
        data['Q_factor'] = {
            s: self.systems[s].properties.q_factor_middle
            for s in self.species_list
        }
        
        data['Gain (dB)'] = {
            s: self.systems[s].properties.cochlear_gain_db
            for s in self.species_list
        }
        
        data['Loc. Fixed (°)'] = {
            s: self.systems[s].localization_accuracy_fixed()
            for s in self.species_list
        }
        
        data['Loc. Mobile (°)'] = {
            s: self.systems[s].localization_accuracy_mobile()
            for s in self.species_list
        }
        
        # Formatiere als Tabelle
        lines = []
        lines.append("Parameter".ljust(25) + " | " + " | ".join(
            [s[:15].ljust(15) for s in species_names]
        ))
        lines.append("-" * (25 + 3 + 15 * len(species_names)))
        
        for param_name, param_data in data.items():
            row = param_name.ljust(25) + " | "
            for species in self.species_list:
                value = param_data[species]
                row += f"{value:15.2f} | "
            lines.append(row)
        
        return "\n".join(lines)
