"""
Pytest Test-Suite für die Bacteria-DNA Wechselwirkungsmodellierung

Tests für alle Komponenten:
- Genome und Bakterien
- Modifikationsmechanismen
- Genomische Einflussmetrik
- Populationsdynamik
- Mikrobiom-Netzwerke
"""

import pytest
import numpy as np
from bacteria_dna_model import (
    Genome, Bacterium, Host, ModificationMechanisms, GenomicInfluenceMetric,
    PopulationDynamics, MicrobiomeNetwork, ModificationMechanism
)


class TestGenome:
    """Tests für die Genome-Klasse"""
    
    def test_genome_creation(self):
        """Test: Genome-Erstellung"""
        genome = Genome("ATGCATGC")
        assert len(genome.sequence) == 8
        assert genome.size == 8
        assert 0.0 <= genome.gc_content <= 1.0
    
    def test_gc_content_calculation(self):
        """Test: GC-Content-Berechnung"""
        genome_high_gc = Genome("GCGCGCGC")
        assert genome_high_gc.gc_content == 1.0
        
        genome_low_gc = Genome("ATATATAT")
        assert genome_low_gc.gc_content == 0.0
        
        genome_mixed = Genome("ATGCATGC")
        assert 0.0 < genome_mixed.gc_content < 1.0
    
    def test_genome_to_dict(self):
        """Test: Genome zu Dictionary Konvertierung"""
        genome = Genome("ATGCATGC")
        d = genome.to_dict()
        
        assert 'size' in d
        assert 'gc_content' in d
        assert 'sequence_preview' in d
        assert d['size'] == 8
    
    def test_hgt_regions(self):
        """Test: HGT-Regionen Tracking"""
        genome = Genome("ATGCATGC")
        genome.hgt_regions.append((0, 4))
        
        assert len(genome.hgt_regions) == 1
        assert genome.hgt_regions[0] == (0, 4)
    
    def test_epigenetic_marks(self):
        """Test: Epigenetische Marker"""
        genome = Genome("ATGCATGC")
        genome.epigenetic_marks[2] = 0.5
        
        assert len(genome.epigenetic_marks) == 1
        assert genome.epigenetic_marks[2] == 0.5


class TestBacterium:
    """Tests für die Bacterium-Klasse"""
    
    def test_bacterium_creation(self):
        """Test: Bakterium-Erstellung"""
        genome = Genome("ATGCATGC")
        bacterium = Bacterium("E_coli", genome)
        
        assert bacterium.name == "E_coli"
        assert bacterium.genome == genome
        assert len(bacterium.metabolic_functions) == 0
    
    def test_metabolic_functions(self):
        """Test: Metabolische Funktionen"""
        genome = Genome("ATGCATGC")
        bacterium = Bacterium("E_coli", genome)
        bacterium.metabolic_functions.add("lactose_metabolism")
        bacterium.metabolic_functions.add("glucose_fermentation")
        
        assert bacterium.has_function("lactose_metabolism")
        assert not bacterium.has_function("arabinose_metabolism")
    
    def test_functional_space(self):
        """Test: Funktionaler Raum"""
        genome = Genome("ATGCATGC")
        bacterium = Bacterium("E_coli", genome)
        bacterium.metabolic_functions.add("func1")
        bacterium.metabolic_functions.add("func2")
        
        f_space = bacterium.functional_space()
        assert "func1" in f_space
        assert "func2" in f_space
        assert len(f_space) == 2


class TestHost:
    """Tests für die Host-Klasse"""
    
    def test_host_creation(self):
        """Test: Wirt-Erstellung"""
        genome = Genome("ATGCATGC" * 1000)
        host = Host("human", genome)
        
        assert host.name == "human"
        assert host.fitness == 1.0
        assert host.immune_response == 0.0
    
    def test_host_stress(self):
        """Test: Wirt-Stresslevel"""
        genome = Genome("ATGCATGC" * 1000)
        host = Host("human", genome, stress_level=0.5)
        
        assert host.stress_level == 0.5


class TestModificationMechanisms:
    """Tests für Modifikationsmechanismen"""
    
    def test_hgt_integration(self):
        """Test: Horizontaler Gentransfer"""
        donor = Genome("ATGCATGCATGCATGC")
        recipient = Genome("TTTTTTTTTTTTTTTT")
        
        modified = ModificationMechanisms.horizontal_gene_transfer(
            donor, recipient, transfer_size=4, efficiency=1.0
        )
        
        # Größe sollte sich geändert haben
        assert len(modified.sequence) > len(recipient.sequence)
        # HGT-Region sollte markiert sein
        assert len(modified.hgt_regions) > 0
    
    def test_hgt_no_transfer_with_low_efficiency(self):
        """Test: HGT mit niedriger Effizienz"""
        donor = Genome("ATGCATGCATGCATGC")
        recipient = Genome("TTTTTTTTTTTTTTTT")
        
        np.random.seed(42)
        modified = ModificationMechanisms.horizontal_gene_transfer(
            donor, recipient, transfer_size=4, efficiency=0.0
        )
        
        # Mit efficiency=0.0 sollte nichts passieren
        assert modified.sequence == recipient.sequence
    
    def test_plasmid_insertion(self):
        """Test: Plasmid-Integration"""
        host_genome = Genome("ATGCATGC" * 100)
        original_size = len(host_genome.sequence)
        
        modified = ModificationMechanisms.plasmid_insertion(
            host_genome, plasmid_size=1000, plasmid_genes=10
        )
        
        # Größe sollte sich erhöht haben
        assert len(modified.sequence) > original_size
    
    def test_mutagenesis_rate(self):
        """Test: Mutagenese-Rate"""
        genome = Genome("ATGCATGC" * 1000)
        
        # Mit hoher Mutationsrate sollten viele Mutationen entstehen
        mutated = ModificationMechanisms.mutagenesis(
            genome, mutation_rate=0.01, metabolite_concentration=0.5
        )
        
        # Vergleiche Sequenzen
        differences = sum(
            1 for a, b in zip(genome.sequence, mutated.sequence)
            if a != b
        )
        
        assert differences > 0
    
    def test_mutagenesis_metabolite_effect(self):
        """Test: Metaboliten erhöhen Mutationsrate"""
        genome = Genome("ATGCATGC" * 1000)
        
        # Ohne Metaboliten
        mutated_low = ModificationMechanisms.mutagenesis(
            genome, mutation_rate=0.001, metabolite_concentration=0.0
        )
        
        # Mit Metaboliten
        mutated_high = ModificationMechanisms.mutagenesis(
            genome, mutation_rate=0.001, metabolite_concentration=0.5
        )
        
        diff_low = sum(
            1 for a, b in zip(genome.sequence, mutated_low.sequence)
            if a != b
        )
        diff_high = sum(
            1 for a, b in zip(genome.sequence, mutated_high.sequence)
            if a != b
        )
        
        # Mit Metaboliten sollten mehr Mutationen sein
        assert diff_high > diff_low
    
    def test_epigenetic_modification(self):
        """Test: Epigenetische Modifikation"""
        genome = Genome("ATGCATGC" * 100)
        
        modified = ModificationMechanisms.epigenetic_modification(
            genome, affected_genes_fraction=0.1, metabolite_effect=0.5
        )
        
        # Epigenetische Marker sollten gesetzt sein
        assert len(modified.epigenetic_marks) > 0
        
        # Werte sollten zwischen 0 und 0.5 sein
        for pos, value in modified.epigenetic_marks.items():
            assert 0.0 <= value <= 0.5
    
    def test_recombination(self):
        """Test: Rekombination"""
        genome = Genome("ATGCATGC" * 1000)
        original_size = len(genome.sequence)
        
        recombined = ModificationMechanisms.recombination(
            genome, recombination_sites=5
        )
        
        # Größe kann sich ändern (durch Deletionen)
        assert len(recombined.sequence) <= original_size
    
    def test_stress_response_mutagenesis(self):
        """Test: Stress-induzierte Mutagenese"""
        genome = Genome("ATGCATGC" * 1000)
        
        # Ohne Stress
        mutated_low = ModificationMechanisms.stress_response_mutagenesis(
            genome, stress_level=0.0, adaptation_rate=0.01
        )
        
        # Mit Stress
        mutated_high = ModificationMechanisms.stress_response_mutagenesis(
            genome, stress_level=1.0, adaptation_rate=0.01
        )
        
        diff_low = sum(
            1 for a, b in zip(genome.sequence, mutated_low.sequence)
            if a != b
        )
        diff_high = sum(
            1 for a, b in zip(genome.sequence, mutated_high.sequence)
            if a != b
        )
        
        # Mit Stress sollten mehr Mutationen sein
        assert diff_high > diff_low


class TestGenomicInfluenceMetric:
    """Tests für die Genomische Einflussmetrik"""
    
    def test_hgt_component(self):
        """Test: HGT-Komponente"""
        genome = Genome("ATGCATGC" * 100)
        genome.hgt_regions.append((0, 100))
        bacteria = []
        
        metric = GenomicInfluenceMetric.calculate_hgt_component(genome, bacteria)
        
        assert 0.0 <= metric <= 1.0
    
    def test_plasmid_component(self):
        """Test: Plasmid-Komponente"""
        genome = Genome("ATGCATGC" * 100)
        
        metric = GenomicInfluenceMetric.calculate_plasmid_component(genome)
        
        assert 0.0 <= metric <= 1.0
    
    def test_mutagenesis_component(self):
        """Test: Mutagenese-Komponente"""
        genome1 = Genome("ATGCATGC" * 100)
        genome2 = Genome("TTTTTTTT" * 100)
        
        metric = GenomicInfluenceMetric.calculate_mutagenesis_component(
            genome2, genome1
        )
        
        assert 0.0 <= metric <= 1.0
    
    def test_epigenetic_component(self):
        """Test: Epigenetische Komponente"""
        genome = Genome("ATGCATGC" * 100)
        genome.epigenetic_marks = {i: 0.5 for i in range(50)}
        
        metric = GenomicInfluenceMetric.calculate_epigenetic_component(genome)
        
        assert 0.0 <= metric <= 1.0
    
    def test_total_influence_metric(self):
        """Test: Gesamte Einflussmetrik"""
        host_genome = Genome("ATGCATGC" * 100)
        host = Host("human", host_genome)
        
        original_genome = Genome("ATGCATGC" * 100)
        bacteria = []
        
        result = GenomicInfluenceMetric.calculate_total_influence(
            host, original_genome, bacteria
        )
        
        assert 'total' in result
        assert 'components' in result
        assert 0.0 <= result['total'] <= 1.0
        assert len(result['components']) == 6
    
    def test_influence_metric_weights(self):
        """Test: Benutzerdefinierte Gewichte"""
        host_genome = Genome("ATGCATGC" * 100)
        host = Host("human", host_genome)
        original_genome = Genome("ATGCATGC" * 100)
        
        custom_weights = {
            'hgt': 0.5,
            'plasmid': 0.1,
            'mutagenese': 0.1,
            'epigen': 0.1,
            'rekomb': 0.1,
            'stress': 0.1
        }
        
        result = GenomicInfluenceMetric.calculate_total_influence(
            host, original_genome, [], weights=custom_weights
        )
        
        assert result['weights'] == custom_weights


class TestPopulationDynamics:
    """Tests für Populationsdynamik"""
    
    def test_growth_rate(self):
        """Test: Wachstumsrate"""
        rate = PopulationDynamics.growth_rate(
            bacteria_population=1e6,
            host_genome_size=1e7,
            metabolite_level=0.5,
            resources=1.0
        )
        
        assert rate >= 0.0
    
    def test_growth_rate_logistic(self):
        """Test: Logistisches Wachstum"""
        # Mit niedriger Population sollte das Wachstum positiv sein
        rate_low = PopulationDynamics.growth_rate(
            bacteria_population=1e3,
            host_genome_size=1e7,
            metabolite_level=0.5,
            resources=1.0
        )
        
        # Mit hoher Population (nahe Carrying Capacity) sollte Wachstum niedrig sein
        rate_high = PopulationDynamics.growth_rate(
            bacteria_population=1e9,
            host_genome_size=1e7,
            metabolite_level=0.5,
            resources=1.0
        )
        
        assert rate_low > rate_high
    
    def test_death_rate(self):
        """Test: Sterberate"""
        rate = PopulationDynamics.death_rate(
            immune_response=0.5,
            host_fitness=0.8,
            bacteria_population=1e6
        )
        
        assert rate > 0.0
    
    def test_fitness_change(self):
        """Test: Fitness-Änderung"""
        # Mit neuen Funktionen sollte Fitness erhöht werden
        change_positive = PopulationDynamics.fitness_change(
            new_functions={'func1', 'func2', 'func3'},
            metabolite_costs=0.0,
            stress_level=0.0
        )
        
        # Mit hohen Kosten sollte Fitness sinken
        change_negative = PopulationDynamics.fitness_change(
            new_functions=set(),
            metabolite_costs=1.0,
            stress_level=1.0
        )
        
        assert change_positive > change_negative
    
    def test_simulate_dynamics(self):
        """Test: Simulation der Dynamik"""
        time_points = np.linspace(0, 100, 50)
        immune_trajectory = np.linspace(0, 1, 50)
        
        time, bacteria, fitness = PopulationDynamics.simulate_dynamics(
            initial_bacteria=1e6,
            initial_host_fitness=1.0,
            resources=1.0,
            metabolite_level=0.5,
            immune_response_trajectory=immune_trajectory,
            time_points=time_points
        )
        
        assert len(time) == len(time_points)
        assert len(bacteria) == len(time_points)
        assert len(fitness) == len(time_points)
        
        # Fitness sollte sinken (wegen erhöhter Immunantwort)
        assert fitness[-1] < fitness[0]


class TestMicrobiomeNetwork:
    """Tests für Mikrobiom-Netzwerk"""
    
    def test_network_creation(self):
        """Test: Netzwerk-Erstellung"""
        network = MicrobiomeNetwork()
        
        assert len(network.graph.nodes()) == 0
        assert len(network.bacteria_species) == 0
    
    def test_add_species(self):
        """Test: Hinzufügen von Arten"""
        network = MicrobiomeNetwork()
        
        genome = Genome("ATGCATGC")
        bacterium = Bacterium("E_coli", genome)
        
        network.add_species(bacterium)
        
        assert len(network.graph.nodes()) == 1
        assert "E_coli" in network.bacteria_species
    
    def test_add_interaction(self):
        """Test: Hinzufügen von Interaktionen"""
        network = MicrobiomeNetwork()
        
        genome1 = Genome("ATGCATGC")
        genome2 = Genome("TTTTTTTT")
        
        b1 = Bacterium("B1", genome1)
        b2 = Bacterium("B2", genome2)
        
        network.add_species(b1)
        network.add_species(b2)
        network.add_interaction("B1", "B2", "substrate", 0.7)
        
        assert network.graph.has_edge("B1", "B2")
        assert network.graph["B1"]["B2"]["strength"] == 0.7
    
    def test_find_functional_clusters(self):
        """Test: Funktionale Cluster-Erkennung"""
        network = MicrobiomeNetwork()
        
        # Erstelle kleine Netzwerk
        for i in range(5):
            genome = Genome("ATGC" * 25)
            bacterium = Bacterium(f"B{i}", genome)
            network.add_species(bacterium)
        
        # Füge Interaktionen hinzu
        network.add_interaction("B0", "B1", "substrate", 0.8)
        network.add_interaction("B1", "B2", "substrate", 0.7)
        
        clusters = network.find_functional_clusters(min_cluster_size=1)
        
        assert len(clusters) > 0
    
    def test_calculate_complementarity(self):
        """Test: Komplementarität-Berechnung"""
        network = MicrobiomeNetwork()
        
        genome1 = Genome("ATGCATGC")
        genome2 = Genome("TTTTTTTT")
        
        b1 = Bacterium("B1", genome1)
        b2 = Bacterium("B2", genome2)
        
        b1.metabolic_functions.add("func1")
        b1.metabolic_functions.add("func2")
        
        b2.metabolic_functions.add("func3")
        b2.metabolic_functions.add("func4")
        
        network.add_species(b1)
        network.add_species(b2)
        
        comp = network.calculate_complementarity("B1", "B2")
        
        assert 0.0 <= comp <= 1.0
        # Keine überlappenden Funktionen sollten hohe Komplementarität bedeuten
        assert comp > 0.5
    
    def test_information_cascade(self):
        """Test: Informationskaskade"""
        network = MicrobiomeNetwork()
        
        for i in range(5):
            genome = Genome("ATGC" * 25)
            bacterium = Bacterium(f"B{i}", genome)
            network.add_species(bacterium)
        
        # Erstelle lineares Netzwerk
        for i in range(4):
            network.add_interaction(f"B{i}", f"B{i+1}", "substrate", 0.8)
        
        cascade = network.propagate_information_cascade(
            "B0", "metabolite", steps=3
        )
        
        assert len(cascade) == 5
        assert cascade["B0"] > 0.0
        # Konzentration sollte weiter weg niedriger sein
        assert cascade["B4"] < cascade["B0"]
    
    def test_network_to_dict(self):
        """Test: Netzwerk zu Dictionary"""
        network = MicrobiomeNetwork()
        
        genome = Genome("ATGCATGC")
        bacterium = Bacterium("E_coli", genome)
        
        network.add_species(bacterium)
        
        d = network.to_dict()
        
        assert 'nodes' in d
        assert 'edges' in d
        assert d['num_nodes'] == 1
        assert d['num_edges'] == 0


class TestIntegration:
    """Integrationstests für vollständiges System"""
    
    def test_full_workflow(self):
        """Test: Kompletter Workflow"""
        # Erstelle Host und Bakterium
        host_genome = Genome("ATGCATGC" * 1000)
        host = Host("human", host_genome)
        
        bacteria_genome = Genome("TTTTTTTT" * 1000)
        bacteria_genome.metabolic_functions = {
            "butyrate_production",
            "lactose_fermentation"
        }
        bacterium = Bacterium("Faecalibacterium", bacteria_genome)
        
        # Appliziere Modifikationen
        modified_genome = ModificationMechanisms.horizontal_gene_transfer(
            bacteria_genome, host_genome, efficiency=0.5
        )
        
        modified_genome = ModificationMechanisms.epigenetic_modification(
            modified_genome, affected_genes_fraction=0.05
        )
        
        # Aktualisiere Host
        host.genome = modified_genome
        
        # Berechne Einflussmetrik
        result = GenomicInfluenceMetric.calculate_total_influence(
            host, host_genome, [bacterium]
        )
        
        assert result['total'] >= 0.0
    
    def test_microbiome_ecosystem(self):
        """Test: Mikrobiom-Ökosystem"""
        # Erstelle Mikrobiom mit mehreren Arten
        network = MicrobiomeNetwork()
        
        species_data = {
            "Faecalibacterium": {"butyrate_production", "fiber_degradation"},
            "Roseburia": {"butyrate_production", "acetate_utilization"},
            "Akkermansia": {"mucin_degradation", "metabolite_production"},
        }
        
        for name, functions in species_data.items():
            genome = Genome("ATGCATGC" * 500)
            bacterium = Bacterium(name, genome)
            bacterium.metabolic_functions = functions
            network.add_species(bacterium)
        
        # Füge Interaktionen hinzu
        network.add_interaction(
            "Faecalibacterium", "Roseburia", "substrate", 0.8
        )
        network.add_interaction(
            "Akkermansia", "Faecalibacterium", "synergy", 0.6
        )
        
        # Finde Cluster
        clusters = network.find_functional_clusters(min_cluster_size=2)
        assert len(clusters) > 0
        
        # Teste Komplementarität
        comp = network.calculate_complementarity(
            "Faecalibacterium", "Roseburia"
        )
        assert 0.0 <= comp <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
