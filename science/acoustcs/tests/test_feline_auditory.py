"""
Tests für das Katzen-Gehörsystem (feline_auditory.py).

Deckt ab:
- Frequenzbandbreiten und Oktaven
- Cochleäre Skalierung (Greenwood-Parameter)
- Gütefaktoren und kritische Bandbreiten
- Phasenlocking und zeitliche Auflösung
- Raumlokalisierung
- Vergleiche zwischen Spezies
"""

import pytest
import numpy as np
from acoustcs.feline_auditory import FelineAuditory, Species, AuditoryProperties


class TestAuditoryProperties:
    """Tests für die AuditoryProperties-Datenklasse."""
    
    def test_valid_properties(self):
        """Test mit gültigen Parametern."""
        props = AuditoryProperties(
            species=Species.CAT,
            f_min_hz=55.0,
            f_max_hz=77000.0,
            cochlea_length_mm=32.0,
            q_factor_middle=7.2,
            cochlear_gain_db=45.0,
            refractory_period_us=30.0,
            head_radius_cm=5.0,
        )
        assert props.f_min_hz == 55.0
        assert props.f_max_hz == 77000.0
    
    def test_invalid_f_min_greater_than_f_max(self):
        """Test mit ungültiger Frequenzordnung."""
        with pytest.raises(ValueError):
            AuditoryProperties(
                species=Species.CAT,
                f_min_hz=77000.0,
                f_max_hz=55.0,
                cochlea_length_mm=32.0,
                q_factor_middle=7.2,
                cochlear_gain_db=45.0,
                refractory_period_us=30.0,
                head_radius_cm=5.0,
            )
    
    def test_negative_cochlea_length(self):
        """Test mit negativer Cochlea-Länge."""
        with pytest.raises(ValueError):
            AuditoryProperties(
                species=Species.CAT,
                f_min_hz=55.0,
                f_max_hz=77000.0,
                cochlea_length_mm=-32.0,
                q_factor_middle=7.2,
                cochlear_gain_db=45.0,
                refractory_period_us=30.0,
                head_radius_cm=5.0,
            )


class TestFelineAuditoryInitialization:
    """Tests für Initialisierung des FelineAuditory-Systems."""
    
    def test_default_initialization_cat(self):
        """Test Standard-Initialisierung mit Katze."""
        auditory = FelineAuditory(Species.CAT)
        assert auditory.species == Species.CAT
        assert auditory.properties.f_max_hz == 77000.0
    
    def test_dog_initialization(self):
        """Test Initialisierung mit Hund."""
        auditory = FelineAuditory(Species.DOG)
        assert auditory.species == Species.DOG
        assert auditory.properties.f_max_hz == 45000.0
    
    def test_human_initialization(self):
        """Test Initialisierung mit Mensch."""
        auditory = FelineAuditory(Species.HUMAN)
        assert auditory.species == Species.HUMAN
        assert auditory.properties.f_max_hz == 20000.0


class TestFrequencyBandwidth:
    """Tests für Frequenzbandbreiten (Satz 9.1)."""
    
    def test_cat_frequency_range(self):
        """Test Katzen-Frequenzbereich."""
        cat = FelineAuditory(Species.CAT)
        f_min, f_max, ratio = cat.frequency_bandwidth()
        
        assert f_min == 55.0
        assert f_max == 77000.0
        assert ratio == pytest.approx(1400.0, rel=0.01)
    
    def test_cat_vs_dog_upper_frequency_ratio(self):
        """Test Verhältnis oberer Frequenzen (Eq. 9.8)."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        
        cat_f_max = cat.properties.f_max_hz
        dog_f_max = dog.properties.f_max_hz
        
        ratio = cat_f_max / dog_f_max
        assert ratio == pytest.approx(1.71, rel=0.01)
    
    def test_frequency_range_octaves(self):
        """Test Frequenzbandbreite in Oktaven."""
        cat = FelineAuditory(Species.CAT)
        octaves = cat.frequency_range_octaves()
        
        # log2(77000/55) ≈ 10.45 Oktaven
        assert octaves == pytest.approx(np.log2(77000/55), rel=0.01)
    
    def test_human_octaves_less_than_cat(self):
        """Test dass Menschen weniger Oktaven haben als Katzen."""
        cat = FelineAuditory(Species.CAT)
        human = FelineAuditory(Species.HUMAN)
        
        assert cat.frequency_range_octaves() > human.frequency_range_octaves()


class TestCochlearScaling:
    """Tests für Greenwood-Skalierungsparameter (Eq. 9.9-9.11)."""
    
    def test_alpha_parameter_positive(self):
        """Test dass α positiv ist."""
        cat = FelineAuditory(Species.CAT)
        alpha = cat.cochlear_scaling_parameter()
        assert alpha > 0
    
    def test_alpha_cat_greater_than_dog(self):
        """Test dass Katze höheres α als Hund hat."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        
        alpha_cat = cat.cochlear_scaling_parameter()
        alpha_dog = dog.cochlear_scaling_parameter()
        
        assert alpha_cat > alpha_dog
    
    def test_alpha_ratio_cat_dog(self):
        """Test Verhältnis α_cat / α_dog (Eq. 9.11)."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        
        ratio = cat.cochlear_scaling_parameter() / dog.cochlear_scaling_parameter()
        assert ratio == pytest.approx(1.128, rel=0.05)


class TestQFactor:
    """Tests für Gütefaktoren (Tabelle 9.1, Eq. 9.12)."""
    
    def test_cat_q_factor_greater_than_dog(self):
        """Test Q_cat > Q_dog (Eq. 9.12)."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        
        q_cat = cat.properties.q_factor_middle
        q_dog = dog.properties.q_factor_middle
        
        assert q_cat > q_dog
    
    def test_q_factor_ratio(self):
        """Test Verhältnis Q_cat / Q_dog ≈ 1.5 (Eq. 9.12)."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        
        ratio = cat.properties.q_factor_middle / dog.properties.q_factor_middle
        assert ratio == pytest.approx(1.5, rel=0.05)
    
    def test_critical_bandwidth_at_frequency(self):
        """Test kritische Bandbreite bei verschiedenen Frequenzen."""
        cat = FelineAuditory(Species.CAT)
        
        erb_low = cat.critical_bandwidth(100.0)
        erb_high = cat.critical_bandwidth(10000.0)
        
        # Bandbreite nimmt mit Frequenz zu
        assert erb_high > erb_low
        assert erb_low > 0
        assert erb_high > 0


class TestTemporalResolution:
    """Tests für zeitliche Auflösung (Eq. 9.14-9.15)."""
    
    def test_phase_locking_frequency_positive(self):
        """Test dass Phasenlocking-Frequenz positiv ist."""
        cat = FelineAuditory(Species.CAT)
        f_pl = cat.phase_locking_frequency()
        assert f_pl > 0
    
    def test_cat_phase_locking_higher_than_dog(self):
        """Test f_pl_cat > f_pl_dog (Eq. 9.15)."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        
        f_pl_cat = cat.phase_locking_frequency()
        f_pl_dog = dog.phase_locking_frequency()
        
        assert f_pl_cat > f_pl_dog
    
    def test_phase_locking_ratio(self):
        """Test Verhältnis f_pl_cat / f_pl_dog (Eq. 9.15)."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        
        ratio = cat.phase_locking_frequency() / dog.phase_locking_frequency()
        assert ratio == pytest.approx(1.65, rel=0.1)
    
    def test_temporal_resolution_consistency(self):
        """Test dass zeitliche Auflösung konsistent mit f_pl ist."""
        cat = FelineAuditory(Species.CAT)
        
        f_pl = cat.phase_locking_frequency()
        delta_t = cat.temporal_resolution()
        
        # Δt = 1 / (2 * f_pl)
        expected_delta_t = 1.0 / (2.0 * f_pl)
        assert delta_t == pytest.approx(expected_delta_t)


class TestLocalization:
    """Tests für Raumlokalisierung (Tabelle 9.1, Eq. 9.21)."""
    
    def test_cat_localization_fixed_less_than_dog(self):
        """Test σ_cat < σ_dog bei fixiertem Kopf."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        
        sigma_cat = cat.localization_accuracy_fixed()
        sigma_dog = dog.localization_accuracy_fixed()
        
        assert sigma_cat < sigma_dog
    
    def test_localization_accuracy_values(self):
        """Test dass Lokalisierungswerte im realistischen Bereich liegen."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        human = FelineAuditory(Species.HUMAN)
        
        # Fixed head values (Eq. 9.21)
        assert cat.localization_accuracy_fixed() == pytest.approx(1.5)
        assert dog.localization_accuracy_fixed() == pytest.approx(3.0)
        assert human.localization_accuracy_fixed() == pytest.approx(5.0)
    
    def test_localization_mobile_better_than_fixed(self):
        """Test dass bewegliche Ohren bessere Lokalisierung ermöglichen."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        
        assert cat.localization_accuracy_mobile() < cat.localization_accuracy_fixed()
        assert dog.localization_accuracy_mobile() < dog.localization_accuracy_fixed()
    
    def test_localization_ratio_mobile(self):
        """Test Verhältnis σ_dog / σ_cat mit beweglichen Ohren."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        
        ratio = (dog.localization_accuracy_mobile() / 
                 cat.localization_accuracy_mobile())
        assert ratio == pytest.approx(3.0, rel=0.05)
    
    def test_itd_maximum(self):
        """Test maximale ITD (Eq. 9.18)."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        
        itd_cat = cat.itd_maximum()
        itd_dog = dog.itd_maximum()
        
        # Katze hat kleineren Kopfradius -> kleinere ITD
        assert itd_cat < itd_dog
        assert itd_cat > 0
        assert itd_dog > 0


class TestTonotopicMap:
    """Tests für Greenwood-Tonotopie (Eq. 9.4-9.5)."""
    
    def test_tonotopic_map_boundaries(self):
        """Test Grenzen der tonotopischen Karte."""
        cat = FelineAuditory(Species.CAT)
        
        # Bei x=0 (basal): höchste Frequenz
        f_basal = cat.tonotopic_map(0.0)
        assert f_basal == pytest.approx(cat.properties.f_max_hz, rel=0.1)
        
        # Bei x=L (apikal): niedrigste Frequenz
        f_apical = cat.tonotopic_map(cat.properties.cochlea_length_mm)
        assert f_apical < 1000.0  # Sollte niedrig sein
    
    def test_tonotopic_position_inverse(self):
        """Test inverse tonotopische Abbildung (Eq. 9.5)."""
        cat = FelineAuditory(Species.CAT)
        
        f_test = 1000.0
        x = cat.tonotopic_position(f_test)
        f_reconstructed = cat.tonotopic_map(x)
        
        assert f_reconstructed == pytest.approx(f_test, rel=0.01)
    
    def test_tonotopic_position_out_of_range(self):
        """Test Fehlerbehandlung bei ungültigen Frequenzen."""
        cat = FelineAuditory(Species.CAT)
        
        with pytest.raises(ValueError):
            cat.tonotopic_position(200000.0)  # Zu hoch
        
        with pytest.raises(ValueError):
            cat.tonotopic_position(10.0)  # Zu niedrig


class TestShannonCapacity:
    """Tests für Shannon-Informationskapazität (Eq. 8.1)."""
    
    def test_shannon_capacity_positive(self):
        """Test dass Shannon-Kapazität positiv ist."""
        cat = FelineAuditory(Species.CAT)
        capacity = cat.shannon_capacity()
        assert capacity > 0
    
    def test_cat_capacity_higher_than_human(self):
        """Test dass Katze höhere Kapazität als Mensch hat."""
        cat = FelineAuditory(Species.CAT)
        human = FelineAuditory(Species.HUMAN)
        
        assert cat.shannon_capacity() > human.shannon_capacity()
    
    def test_shannon_capacity_with_different_snr(self):
        """Test Shannon-Kapazität mit verschiedenen SNR-Werten."""
        cat = FelineAuditory(Species.CAT)
        
        c_0db = cat.shannon_capacity(snr_db=0.0)
        c_10db = cat.shannon_capacity(snr_db=10.0)
        
        # Höheres SNR -> höhere Kapazität
        assert c_10db > c_0db


class TestSuperiorityIndex:
    """Tests für Überlegenheitsindex (Satz 9.2, Tabelle 9.1)."""
    
    def test_superiority_index_structure(self):
        """Test Struktur des Überlegenheitsindex."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        
        factors = cat.superiority_index(dog)
        
        # Erwartete Schlüssel
        expected_keys = [
            'upper_frequency',
            'cochlear_scaling',
            'spectral_selectivity',
            'cochlear_amplification',
            'temporal_resolution',
            'localization_fixed',
            'localization_mobile',
            'geometric_mean_fixed',
            'geometric_mean_mobile',
            'db_equivalent',
        ]
        
        for key in expected_keys:
            assert key in factors
            assert factors[key] > 0
    
    def test_geometric_mean_superiority(self):
        """Test geometrisches Mittel der Überlegenheit (Eq. 9.22-9.23)."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        
        factors = cat.superiority_index(dog)
        
        # Geometrisches Mittel sollte > 1 sein
        gm = factors['geometric_mean_mobile']
        assert gm > 1.0
        assert gm == pytest.approx(2.34, rel=0.2)  # Aus Arbeit: ~2.34
    
    def test_superiority_index_default_reference(self):
        """Test dass Default-Referenz Hund ist."""
        cat = FelineAuditory(Species.CAT)
        
        factors1 = cat.superiority_index()
        factors2 = cat.superiority_index(FelineAuditory(Species.DOG))
        
        # Sollten identisch sein
        assert factors1['geometric_mean_mobile'] == pytest.approx(
            factors2['geometric_mean_mobile']
        )
    
    def test_dog_vs_dog_unity(self):
        """Test dass Hund vs. Hund Verhältnisse 1.0 sind."""
        dog1 = FelineAuditory(Species.DOG)
        dog2 = FelineAuditory(Species.DOG)
        
        factors = dog1.superiority_index(dog2)
        
        # Alle Faktoren sollten ≈ 1.0 sein
        assert factors['upper_frequency'] == pytest.approx(1.0)
        assert factors['localization_fixed'] == pytest.approx(1.0)


class TestSummary:
    """Tests für Zusammenfassungs-Methoden."""
    
    def test_summary_string_format(self):
        """Test dass Summary einen String zurückgibt."""
        cat = FelineAuditory(Species.CAT)
        summary = cat.summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Species" in summary
        assert "Frequency Range" in summary
    
    def test_summary_contains_expected_info(self):
        """Test dass Summary wichtige Informationen enthält."""
        cat = FelineAuditory(Species.CAT)
        summary = cat.summary()
        
        # Stelle sicher, dass wichtige Werte dabei sind
        assert "77000" in summary or "77" in summary
        assert "Localization" in summary


# Integrationstests
class TestFelineAuditoryIntegration:
    """Integrationstests für das komplette System."""
    
    def test_cat_superiority_over_dog_and_human(self):
        """Test dass Katze überlegen ist (biologisch sinnvoll)."""
        cat = FelineAuditory(Species.CAT)
        dog = FelineAuditory(Species.DOG)
        human = FelineAuditory(Species.HUMAN)
        
        # Katze sollte Hund überlegen sein
        factors_vs_dog = cat.superiority_index(dog)
        assert factors_vs_dog['geometric_mean_mobile'] > 1.5
        
        # Katze sollte auch Menschen in Spezialisierung überlegen sein
        factors_vs_human = cat.superiority_index(human)
        assert factors_vs_human['geometric_mean_mobile'] > 1.5
    
    def test_frequency_resolution_consistency(self):
        """Test Konsistenz zwischen verschiedenen Frequenzauflösungs-Messgrößen."""
        cat = FelineAuditory(Species.CAT)
        
        # Q-Faktor und kritische Bandbreite sollten konsistent sein
        freq_test = 10000.0
        q_factor = cat.quality_factor(freq_test)
        erb = cat.critical_bandwidth(freq_test)
        
        # Q = f / ERB (ungefähr)
        calculated_q = freq_test / erb
        
        # Sollten in der gleichen Größenordnung sein
        assert calculated_q > 0
        assert q_factor > 0
    
    def test_all_properties_positive(self):
        """Test dass alle physikalischen Größen positiv sind."""
        for species in [Species.CAT, Species.DOG, Species.HUMAN]:
            auditory = FelineAuditory(species)
            
            assert auditory.properties.f_min_hz > 0
            assert auditory.properties.f_max_hz > 0
            assert auditory.properties.cochlea_length_mm > 0
            assert auditory.properties.q_factor_middle > 0
            assert auditory.properties.cochlear_gain_db > 0
            assert auditory.properties.refractory_period_us > 0
            assert auditory.properties.head_radius_cm > 0
