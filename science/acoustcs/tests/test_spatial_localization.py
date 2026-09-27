"""
Tests für Spatial Localization (ITD, ILD, Fisher-Information).

Deckt ab:
- Interaurale Zeit-Differenz (ITD)
- Interauraler Pegel-Unterschied (ILD)
- Just Noticeable Difference (JND)
- Fisher-Information und Cramér-Rao-Schranke
- Hörverlust-Effekte
"""

import pytest
import numpy as np
from acoustcs.spatial_localization import SpatialLocalization
from acoustcs.feline_auditory import Species


class TestITDCalculation:
    """Tests für Interaurale Zeit-Differenz (Eq. 8.1)."""
    
    def test_itd_at_center(self):
        """Test ITD bei zentraler Quelle (θ = 0°)."""
        sl = SpatialLocalization(Species.CAT)
        
        itd = sl.itd_woodworth_schlosberg(azimuth_deg=0.0)
        
        # Bei θ=0: sin(0) = 0 -> ITD = 0
        assert itd == pytest.approx(0.0, abs=1e-10)
    
    def test_itd_laterality_maximum(self):
        """Test ITD bei lateraler Quelle (θ = 90°)."""
        sl = SpatialLocalization(Species.CAT)
        
        itd_90 = sl.itd_woodworth_schlosberg(azimuth_deg=90.0)
        itd_max = sl.itd_maximum()
        
        # Bei θ=90: ITD sollte maximal sein
        assert itd_90 == pytest.approx(itd_max, rel=0.01)
    
    def test_itd_symmetry(self):
        """Test dass ITD symmetrisch ist (links/rechts)."""
        sl = SpatialLocalization()
        
        itd_left = sl.itd_woodworth_schlosberg(azimuth_deg=-45.0)
        itd_right = sl.itd_woodworth_schlosberg(azimuth_deg=45.0)
        
        # Sollten entgegengesetzte Vorzeichen haben
        assert itd_left == pytest.approx(-itd_right)
    
    def test_itd_cat_vs_dog(self):
        """Test dass Katze kleinere ITD hat (kleinerer Kopf)."""
        cat = SpatialLocalization(Species.CAT)
        dog = SpatialLocalization(Species.DOG)
        
        itd_cat_max = cat.itd_maximum()
        itd_dog_max = dog.itd_maximum()
        
        # Katze hat kleineren Kopf
        assert itd_cat_max < itd_dog_max
    
    def test_itd_units(self):
        """Test dass ITD in Sekunden ist."""
        sl = SpatialLocalization()
        
        itd = sl.itd_maximum()
        
        # Typisch: einige Hundert Mikrosekunden
        assert 100e-6 < itd < 1000e-6  # 100 bis 1000 µs


class TestILDCalculation:
    """Tests für Interauralen Pegel-Unterschied (Eq. 8.2)."""
    
    def test_ild_at_center(self):
        """Test ILD bei zentraler Quelle."""
        sl = SpatialLocalization(Species.CAT)
        
        ild = sl.ild_frequency_dependent(freq_hz=1000.0, azimuth_deg=0.0)
        
        # Bei θ=0: ILD sollte minimal sein
        assert np.abs(ild) < 1.0
    
    def test_ild_frequency_dependence(self):
        """Test dass ILD frequenzabhängig ist."""
        sl = SpatialLocalization()
        
        ild_low = sl.ild_frequency_dependent(freq_hz=500.0, azimuth_deg=90.0)
        ild_high = sl.ild_frequency_dependent(freq_hz=8000.0, azimuth_deg=90.0)
        
        # Höhere Frequenzen haben mehr ILD
        assert np.abs(ild_high) > np.abs(ild_low)
    
    def test_ild_laterality(self):
        """Test dass ILD mit Azimuth variiert."""
        sl = SpatialLocalization()
        freq = 2000.0
        
        ild_front = sl.ild_frequency_dependent(freq, azimuth_deg=0.0)
        ild_side = sl.ild_frequency_dependent(freq, azimuth_deg=90.0)
        
        # Laterale Quellen haben mehr ILD
        assert np.abs(ild_side) > np.abs(ild_front)


class TestJustNoticeableDifference:
    """Tests für Just Noticeable Difference."""
    
    def test_jnd_itd_positive(self):
        """Test dass JND_ITD positiv ist."""
        sl = SpatialLocalization(Species.CAT)
        
        jnd = sl.just_noticeable_difference_itd()
        
        assert jnd > 0
        # Typisch: einige Mikrosekunden
        assert jnd < 100e-6
    
    def test_jnd_itd_scales_with_refractory_period(self):
        """Test dass JND mit Refraktärperiode skaliert."""
        cat = SpatialLocalization(Species.CAT)
        dog = SpatialLocalization(Species.DOG)
        
        jnd_cat = cat.just_noticeable_difference_itd()
        jnd_dog = dog.just_noticeable_difference_itd()
        
        # Hund hat längere Refraktärperiode -> größeres JND
        assert jnd_dog > jnd_cat


class TestFisherInformation:
    """Tests für Fisher-Information (Eq. 8.3)."""
    
    def test_fisher_information_positive(self):
        """Test dass Fisher-Information positiv ist."""
        sl = SpatialLocalization()
        
        fi = sl.fisher_information_itd()
        
        assert fi > 0
    
    def test_fisher_information_snr_dependence(self):
        """Test dass Fisher-Information mit SNR zunimmt."""
        sl = SpatialLocalization()
        
        fi_0db = sl.fisher_information_itd(snr_db=0.0)
        fi_20db = sl.fisher_information_itd(snr_db=20.0)
        
        # Besseres SNR -> höhere Fisher-Information
        assert fi_20db > fi_0db
    
    def test_fisher_information_signal_dependence(self):
        """Test dass Fisher-Information mit Signalleistung skaliert."""
        sl = SpatialLocalization()
        
        fi_1 = sl.fisher_information_itd(signal_power=1.0)
        fi_2 = sl.fisher_information_itd(signal_power=2.0)
        
        # Höhere Signalleistung -> höhere Fisher-Information
        assert fi_2 > fi_1


class TestCramerRaoBound:
    """Tests für Cramér-Rao-Schranke (Def. 9.1)."""
    
    def test_cramer_rao_bound_positive(self):
        """Test dass CRB positiv ist."""
        sl = SpatialLocalization()
        
        crb = sl.cramer_rao_bound()
        
        assert crb > 0
        assert not np.isinf(crb)
    
    def test_cramer_rao_bound_improves_with_snr(self):
        """Test dass CRB mit SNR besser wird."""
        sl = SpatialLocalization()
        
        crb_0db = sl.cramer_rao_bound(snr_db=0.0)
        crb_20db = sl.cramer_rao_bound(snr_db=20.0)
        
        # Besseres SNR -> bessere Schranke
        assert crb_20db < crb_0db
    
    def test_localization_accuracy_from_crb(self):
        """Test dass Lokalisierungsgenauigkeit von CRB abgeleitet ist."""
        sl = SpatialLocalization()
        
        crb = sl.cramer_rao_bound()
        loc_acc = sl.localization_accuracy()
        
        # Lokalisierungsgenauigkeit >= CRB
        assert loc_acc >= crb


class TestHearingLoss:
    """Tests für Hörverlust-Effekte (Satz 9.1)."""
    
    def test_hearing_loss_jnd_degradation(self):
        """Test dass Hörverlust JND verschlechtert."""
        sl = SpatialLocalization()
        
        jnd_normal, jnd_with_loss = sl.hearing_loss_effect(
            frequency_hz=1000.0, hearing_loss_db=20.0
        )
        
        assert jnd_normal > 0
        assert jnd_with_loss > jnd_normal
    
    def test_hearing_loss_scaling(self):
        """Test dass Hörverlust linear mit Grad skaliert."""
        sl = SpatialLocalization()
        
        jnd_10db = sl.hearing_loss_effect(1000.0, 10.0)[1]
        jnd_20db = sl.hearing_loss_effect(1000.0, 20.0)[1]
        
        # Doppelter Hörverlust -> doppelte Verschlechterung
        ratio = jnd_20db / jnd_10db
        assert ratio == pytest.approx(2.0, rel=0.05)
    
    def test_entropy_with_hearing_loss(self):
        """Test räumliche Entropie mit Hörverlust (Satz 9.1)."""
        sl = SpatialLocalization()
        
        h_normal = sl.informational_entropy_spatial(hearing_loss_db=0.0)
        h_loss = sl.informational_entropy_spatial(hearing_loss_db=20.0)
        
        # Hörverlust erhöht Entropie (mehr Unsicherheit)
        assert h_loss > h_normal
    
    def test_entropy_maximum_with_severe_loss(self):
        """Test dass Entropie mit schwerem Hörverlust gegen Maximum geht."""
        sl = SpatialLocalization()
        n_sources = 100
        
        h_severe = sl.informational_entropy_spatial(
            n_sources=n_sources, hearing_loss_db=100.0
        )
        h_max = np.log2(n_sources)
        
        # Sollte nah bei Maximum sein
        assert h_severe > h_max * 0.9


class TestBilateralHearingLoss:
    """Tests für bilaterale Hörverlust-Asymmetrie."""
    
    def test_symmetric_loss_zero_asymmetry(self):
        """Test dass symmetrischer Hörverlust keine Asymmetrie hat."""
        sl = SpatialLocalization()
        
        asymmetry = sl.bilateral_hearing_loss_asymmetry(20.0, 20.0)
        
        assert asymmetry == pytest.approx(0.0)
    
    def test_unilateral_loss_maximum_asymmetry(self):
        """Test dass einseitiger Hörverlust maximale Asymmetrie hat."""
        sl = SpatialLocalization()
        
        asymmetry = sl.bilateral_hearing_loss_asymmetry(100.0, 0.0)
        
        # Sollte nahe bei 1.0 sein
        assert asymmetry > 0.9


class TestFrontBackConfusion:
    """Tests für Vorne-Hinten-Verwechselung."""
    
    def test_confusion_rate_cat_low(self):
        """Test dass Katzen niedrige Verwechslungsrate haben."""
        sl = SpatialLocalization(Species.CAT)
        
        confusion = sl.front_back_confusion()
        
        # Katzen mit beweglichen Ohren
        assert confusion < 0.1
    
    def test_confusion_rate_human_higher(self):
        """Test dass Menschen höhere Verwechslungsrate haben."""
        cat = SpatialLocalization(Species.CAT)
        human = SpatialLocalization(Species.HUMAN)
        
        assert human.front_back_confusion() > cat.front_back_confusion()
    
    def test_confusion_rates_plausible(self):
        """Test dass Verwechslungsraten plausibel sind."""
        for species in [Species.CAT, Species.DOG, Species.HUMAN]:
            sl = SpatialLocalization(species)
            confusion = sl.front_back_confusion()
            
            assert 0 <= confusion <= 1.0


class TestSpatialLocalizationIntegration:
    """Integrationstests für räumliche Lokalisierung."""
    
    def test_itd_jnd_relationship(self):
        """Test dass ITD_max >> JND_ITD."""
        sl = SpatialLocalization()
        
        itd_max = sl.itd_maximum()
        jnd_itd = sl.just_noticeable_difference_itd()
        
        # ITD_max sollte viel größer sein
        assert itd_max > 100 * jnd_itd
    
    def test_localization_species_ranking(self):
        """Test dass Lokalisierungsgenauigkeit biologisch sinnvoll ist."""
        cat = SpatialLocalization(Species.CAT)
        dog = SpatialLocalization(Species.DOG)
        human = SpatialLocalization(Species.HUMAN)
        
        # Katze > Hund > Mensch (in Spezialisiertheit)
        acc_cat = cat.localization_accuracy()
        acc_dog = dog.localization_accuracy()
        acc_human = human.localization_accuracy()
        
        assert acc_cat < acc_dog < acc_human
    
    def test_all_localization_parameters_positive(self):
        """Test dass alle Lokalisierungs-Parameter positiv sind."""
        sl = SpatialLocalization()
        
        assert sl.itd_maximum() > 0
        assert sl.just_noticeable_difference_itd() > 0
        assert sl.cramer_rao_bound() > 0
        assert sl.localization_accuracy() > 0
        assert 0 <= sl.front_back_confusion() <= 1
    
    def test_summary_output(self):
        """Test Summary-Ausgabe."""
        sl = SpatialLocalization(Species.CAT)
        
        summary = sl.summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "ITD_max" in summary or "ITD" in summary
