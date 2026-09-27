"""
Tests für das Metadistribution-Modul.

Tests für:
- Wahrscheinlichkeitsräume
- Verteilungsräume
- Metaverteilungen
- Verschiebungstrajektorien
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from metadistribution import (
    ProbabilitySpace,
    DistributionSpace,
    MetaDistribution,
    ShiftTrajectory
)


class TestProbabilitySpace:
    """Tests für ProbabilitySpace."""
    
    def test_initialization_default(self):
        """Test: Initialisiere mit Standardverteilung (gleichverteilt)."""
        ps = ProbabilitySpace(3)
        
        assert ps.n_outcomes == 3
        assert len(ps.measure) == 3
        assert np.allclose(ps.measure, [1/3, 1/3, 1/3])
    
    def test_initialization_custom(self):
        """Test: Initialisiere mit benutzerdefinierten Wahrscheinlichkeiten."""
        custom_measure = [0.2, 0.3, 0.5]
        ps = ProbabilitySpace(3, measure=custom_measure)
        
        assert np.allclose(ps.measure, custom_measure)
    
    def test_initialization_normalization(self):
        """Test: Normalisiere nicht normalisierte Wahrscheinlichkeiten."""
        custom_measure = [1, 2, 3]  # Summe = 6
        ps = ProbabilitySpace(3, measure=custom_measure)
        
        assert np.allclose(np.sum(ps.measure), 1.0)
        assert np.allclose(ps.measure, [1/6, 2/6, 3/6])
    
    def test_initialization_invalid_negative(self):
        """Test: Lehne negative Wahrscheinlichkeiten ab."""
        with pytest.raises(ValueError):
            ProbabilitySpace(2, measure=[0.5, -0.5])
    
    def test_entropy_uniform(self):
        """Test: Entropie der Gleichverteilung."""
        ps = ProbabilitySpace(4)  # Gleichverteilung über 4 Outcomes
        
        entropy = ps.entropy()
        
        # H(U) = log_2(4) = 2
        assert np.isclose(entropy, 2.0)
    
    def test_entropy_deterministic(self):
        """Test: Entropie der deterministischen Verteilung."""
        ps = ProbabilitySpace(3, measure=[1, 0, 0])
        
        entropy = ps.entropy()
        
        # H(D) = 0 (keine Unsicherheit)
        assert np.isclose(entropy, 0.0)
    
    def test_entropy_binary(self):
        """Test: Entropie der Binärverteilung."""
        ps = ProbabilitySpace(2, measure=[0.5, 0.5])
        
        entropy = ps.entropy()
        
        # H(B) = 1
        assert np.isclose(entropy, 1.0)
    
    def test_entropy_non_uniform(self):
        """Test: Entropie einer nicht-uniformen Verteilung."""
        ps = ProbabilitySpace(3, measure=[0.5, 0.25, 0.25])
        
        entropy = ps.entropy()
        
        # Sollte zwischen 0 und log_2(3) ≈ 1.585 liegen
        assert 0 < entropy < 1.585
    
    def test_kl_divergence_identical(self):
        """Test: KL-Divergenz identischer Verteilungen."""
        measure = [0.2, 0.3, 0.5]
        ps1 = ProbabilitySpace(3, measure=measure)
        ps2 = ProbabilitySpace(3, measure=measure)
        
        kl = ps1.kl_divergence(ps2)
        
        assert np.isclose(kl, 0.0)
    
    def test_kl_divergence_different(self):
        """Test: KL-Divergenz verschiedener Verteilungen."""
        ps1 = ProbabilitySpace(2, measure=[0.9, 0.1])
        ps2 = ProbabilitySpace(2, measure=[0.1, 0.9])
        
        kl = ps1.kl_divergence(ps2)
        
        assert kl > 0
        assert np.isfinite(kl)
    
    def test_kl_divergence_asymmetric(self):
        """Test: KL-Divergenz ist asymmetrisch."""
        ps1 = ProbabilitySpace(2, measure=[0.9, 0.1])
        ps2 = ProbabilitySpace(2, measure=[0.5, 0.5])
        
        kl_12 = ps1.kl_divergence(ps2)
        kl_21 = ps2.kl_divergence(ps1)
        
        # Sollten unterschiedlich sein
        assert not np.isclose(kl_12, kl_21)
    
    def test_wasserstein_distance_identical(self):
        """Test: Wasserstein-Distanz identischer Verteilungen."""
        measure = [0.2, 0.3, 0.5]
        ps1 = ProbabilitySpace(3, measure=measure)
        ps2 = ProbabilitySpace(3, measure=measure)
        
        wd = ps1.wasserstein_distance(ps2)
        
        assert np.isclose(wd, 0.0)
    
    # def test_wasserstein_distance_different(self):
    #     """Test: Wasserstein-Distanz verschiedener Verteilungen."""
    #     ps1 = ProbabilitySpace(3, measure=[0.5, 0.5, 0])
    #     ps2 = ProbabilitySpace(3, measure=[0, 0.5, 0.5])
        
    #     wd = ps1.wasserstein_distance(ps2)
        
    #     assert wd > 0
    #     assert wd <= 1.0  # Maximum ist 1 für Wahrscheinlichkeitsmasse
    
    def test_wasserstein_distance_symmetric(self):
        """Test: Wasserstein-Distanz ist symmetrisch."""
        ps1 = ProbabilitySpace(2, measure=[0.7, 0.3])
        ps2 = ProbabilitySpace(2, measure=[0.3, 0.7])
        
        wd_12 = ps1.wasserstein_distance(ps2)
        wd_21 = ps2.wasserstein_distance(ps1)
        
        assert np.isclose(wd_12, wd_21)


class TestDistributionSpace:
    """Tests für DistributionSpace."""
    
    def test_initialization(self):
        """Test: Initialisiere Verteilungsraum."""
        ds = DistributionSpace(n_outcomes=3)
        
        assert ds.n_outcomes == 3
        assert len(ds.distributions) == 0
    
    def test_add_distribution(self):
        """Test: Füge Verteilung hinzu."""
        ds = DistributionSpace(n_outcomes=3)
        measure = [0.2, 0.3, 0.5]
        
        ds.add_distribution(measure)
        
        assert len(ds.distributions) == 1
        assert np.allclose(ds.distributions[0].measure, measure)
    
    def test_add_multiple_distributions(self):
        """Test: Füge mehrere Verteilungen hinzu."""
        ds = DistributionSpace(n_outcomes=2)
        
        ds.add_distribution([0.5, 0.5])
        ds.add_distribution([0.7, 0.3])
        ds.add_distribution([0.3, 0.7])
        
        assert len(ds.distributions) == 3
    
    def test_dimension(self):
        """Test: Berechne Dimension des Verteilungsraums."""
        ds = DistributionSpace(n_outcomes=5)
        
        dim = ds.dimension()
        
        # Dimension sollte n - 1 sein
        assert dim == 4
    
    def test_number_of_trajectories(self):
        """Test: Berechne Anzahl Verschiebungstrajektorien."""
        ds = DistributionSpace(n_outcomes=2)
        
        ds.add_distribution([0.5, 0.5])
        ds.add_distribution([0.7, 0.3])
        
        n_traj = ds.number_of_trajectories()
        
        # Für 2 Distributionen: 2! = 2 Permutationen
        assert n_traj == 2


class TestMetaDistribution:
    """Tests für MetaDistribution."""
    
    def test_initialization(self):
        """Test: Initialisiere Metaverteilung."""
        ds = DistributionSpace(n_outcomes=2)
        md = MetaDistribution(ds)
        
        assert len(md.trajectory_probs) == 0
    
    def test_add_trajectory(self):
        """Test: Füge Verschiebungstrajektorie hinzu."""
        ds = DistributionSpace(n_outcomes=2)
        md = MetaDistribution(ds)
        
        trajectory = (0, 1)
        md.add_trajectory(trajectory, 0.5)
        
        assert trajectory in md.trajectory_probs
        assert md.trajectory_probs[trajectory] == 0.5
    
    def test_add_trajectory_invalid_prob(self):
        """Test: Lehne ungültige Wahrscheinlichkeiten ab."""
        ds = DistributionSpace(n_outcomes=2)
        md = MetaDistribution(ds)
        
        with pytest.raises(ValueError):
            md.add_trajectory((0, 1), 1.5)  # > 1
    
    def test_entropy_empty(self):
        """Test: Entropie leerer Metaverteilung."""
        ds = DistributionSpace(n_outcomes=2)
        md = MetaDistribution(ds)
        
        entropy = md.entropy()
        
        assert np.isclose(entropy, 0.0)
    
    def test_entropy_single_trajectory(self):
        """Test: Entropie mit einzelner Trajektorie."""
        ds = DistributionSpace(n_outcomes=2)
        md = MetaDistribution(ds)
        
        md.add_trajectory((0,), 1.0)
        
        entropy = md.entropy()
        
        # Deterministische Verteilung → Entropie = 0
        assert np.isclose(entropy, 0.0)
    
    def test_entropy_uniform(self):
        """Test: Entropie mit uniformer Verteilung."""
        ds = DistributionSpace(n_outcomes=2)
        md = MetaDistribution(ds)
        
        md.add_trajectory((0,), 0.5)
        md.add_trajectory((1,), 0.5)
        
        entropy = md.entropy()
        
        # Gleichverteilung über 2 Trajektorien → H = 1 bit
        assert np.isclose(entropy, 1.0)
    
    def test_average_distribution_entropy(self):
        """Test: Durchschnittliche Entropie der Distributionen."""
        ds = DistributionSpace(n_outcomes=2)
        ds.add_distribution([0.5, 0.5])  # H = 1
        ds.add_distribution([0.9, 0.1])  # H < 1
        
        md = MetaDistribution(ds)
        md.add_trajectory((0,), 0.5)
        md.add_trajectory((1,), 0.5)
        
        avg_entropy = md.average_distribution_entropy()
        
        assert 0 < avg_entropy < 1
    
    def test_shift_degree_empty(self):
        """Test: Verschiebungsgrad für leeren Raum."""
        ds = DistributionSpace(n_outcomes=2)
        md = MetaDistribution(ds)
        
        degree = md.shift_degree()
        
        assert np.isclose(degree, 0.0)
    
    def test_shift_degree_identical(self):
        """Test: Verschiebungsgrad für identische Distributionen."""
        ds = DistributionSpace(n_outcomes=2)
        measure = [0.5, 0.5]
        ds.add_distribution(measure)
        ds.add_distribution(measure)
        
        md = MetaDistribution(ds)
        
        degree = md.shift_degree()
        
        # Identische Verteilungen → Distanz = 0
        assert np.isclose(degree, 0.0)
    
    # def test_shift_degree_different(self):
    #     """Test: Verschiebungsgrad für unterschiedliche Distributionen."""
    #     ds = DistributionSpace(n_outcomes=2)
    #     ds.add_distribution([0.1, 0.9])
    #     ds.add_distribution([0.9, 0.1])
        
    #     md = MetaDistribution(ds)
        
    #     degree = md.shift_degree()
        
    #     # Sehr unterschiedliche Verteilungen → hoher Grad
    #     assert degree > 0.5


class TestShiftTrajectory:
    """Tests für ShiftTrajectory."""
    
    def test_initialization(self):
        """Test: Initialisiere Verschiebungstrajektorie."""
        ps1 = ProbabilitySpace(2, measure=[0.5, 0.5])
        ps2 = ProbabilitySpace(2, measure=[0.7, 0.3])
        
        trajectory = ShiftTrajectory([ps1, ps2])
        
        assert len(trajectory.time_steps) == 2
    
    def test_convergence_rate_empty(self):
        """Test: Konvergenzrate für leere Trajektorie."""
        trajectory = ShiftTrajectory([])
        target = ProbabilitySpace(2)
        
        rate = trajectory.convergence_rate(target)
        
        assert rate == 0.0
    
    # def test_convergence_rate_single(self):
    #     """Test: Konvergenzrate mit einzelnem Zeitpunkt."""
    #     ps = ProbabilitySpace(2, measure=[0.5, 0.5])
    #     trajectory = ShiftTrajectory([ps])
    #     target = ProbabilitySpace(2, measure=[0.5, 0.5])
        
    #     rate = trajectory.convergence_rate(target)
        
    #     # Bereits am Ziel → Rate sollte hoch sein
    #     assert rate > 0
    
    # def test_convergence_rate_toward_target(self):
    #     """Test: Konvergenzrate bei Annäherung an Ziel."""
    #     steps = [
    #         ProbabilitySpace(2, measure=[0.5, 0.5]),
    #         ProbabilitySpace(2, measure=[0.6, 0.4]),
    #         ProbabilitySpace(2, measure=[0.8, 0.2]),
    #         ProbabilitySpace(2, measure=[0.9, 0.1])
    #     ]
    #     trajectory = ShiftTrajectory(steps)
    #     target = ProbabilitySpace(2, measure=[1.0, 0.0])
        
    #     rate = trajectory.convergence_rate(target)
        
    #     # Sollte konvergieren
    #     assert rate > 0
    
    def test_phase_transition_points(self):
        """Test: Erkenne Übergangspunkte in Trajektorie."""
        steps = [
            ProbabilitySpace(2, measure=[0.5, 0.5]),
            ProbabilitySpace(2, measure=[0.5, 0.5]),
            ProbabilitySpace(2, measure=[0.8, 0.2]),  # Übergang
            ProbabilitySpace(2, measure=[0.9, 0.1]),
            ProbabilitySpace(2, measure=[0.95, 0.05])
        ]
        trajectory = ShiftTrajectory(steps)
        
        transitions = trajectory.phase_transition_points(threshold=0.2)
        
        # Sollte mindestens einen Übergang finden
        assert len(transitions) > 0


class TestIntegration:
    """Integrationstests für Metaverteilungs-Theorie."""
    
    def test_probability_space_to_metadistribution_pipeline(self):
        """Test: Pipeline von ProbabilitySpace zu MetaDistribution."""
        # Erstelle Verteilungsraum mit mehreren Distributionen
        ds = DistributionSpace(n_outcomes=3)
        ds.add_distribution([0.5, 0.3, 0.2])
        ds.add_distribution([0.3, 0.5, 0.2])
        ds.add_distribution([0.2, 0.3, 0.5])
        
        # Erstelle Metaverteilung
        md = MetaDistribution(ds)
        md.add_trajectory((0, 1, 2), 0.4)
        md.add_trajectory((0, 2, 1), 0.6)
        
        # Alle sollten konsistent sein
        assert md.entropy() >= 0
        assert md.average_distribution_entropy() >= 0
        assert md.shift_degree() >= 0
    
    def test_shift_trajectory_with_entropy_decrease(self):
        """Test: Verschiebungstrajektorie mit fallender Entropie."""
        # Erstelle Trajektorie mit fallender Entropie
        steps = [
            ProbabilitySpace(3, measure=[0.33, 0.33, 0.34]),  # Hohe Entropie
            ProbabilitySpace(3, measure=[0.5, 0.3, 0.2]),
            ProbabilitySpace(3, measure=[0.7, 0.2, 0.1]),
            ProbabilitySpace(3, measure=[0.9, 0.05, 0.05])   # Niedrige Entropie
        ]
        trajectory = ShiftTrajectory(steps)
        
        # Entropie sollte fallen
        entropies = [ps.entropy() for ps in steps]
        assert entropies[0] > entropies[-1]
    
    # def test_phase_transition_with_convergence(self):
    #     """Test: Phase-Übergänge mit Konvergenz."""
    #     # Erstelle Trajektorie mit drei Phasen
    #     phase1 = [ProbabilitySpace(2, measure=[0.5, 0.5]) for _ in range(3)]
    #     phase2 = [ProbabilitySpace(2, measure=[0.5 + 0.1*i, 0.5 - 0.1*i]) for i in range(3)]
    #     phase3 = [ProbabilitySpace(2, measure=[0.95 - 0.01*i, 0.05 + 0.01*i]) for i in range(3)]
        
    #     all_steps = phase1 + phase2 + phase3
    #     trajectory = ShiftTrajectory(all_steps)
        
    #     target = ProbabilitySpace(2, measure=[1.0, 0.0])
    #     rate = trajectory.convergence_rate(target)
        
    #     assert rate > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
