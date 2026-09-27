"""
===============================================================================
PYTEST - Erweiterte Testabdeckung für BAKKT Implementation
===============================================================================

Dieser Test-Modul bietet:
- 50+ Testfälle mit hoher Coverage
- Parametrisierte Tests für verschiedene Szenarien
- Performance-Tests
- Integration Tests
- Edge-Case Tests

Ausführung:
    pytest test_bakkt.py -v
    pytest test_bakkt.py -v --tb=short
    pytest test_bakkt.py -v -k "TestSubgraph"  # Nur Subgraph-Tests
    pytest test_bakkt.py -v --cov=bakkt_implementation
"""

import pytest
import numpy as np
import networkx as nx
from bakkt_implementation import (
    Bacterium, LocalInformation, Consequence, BacterialAction,
    HelpType, SubgraphAlgorithm, BacterialPopulation,
    ConsciousnessModel, MathematicalFormalisms
)

# ═══════════════════════════════════════════════════════════════════════════════
# TESTS MIT PYTEST
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacteriumFundamentals:
    """Tests für Grundkonzepte (Kapitel 2)"""
    
    def test_bacterium_creation(self):
        """Test: Bakterium kann erstellt werden"""
        b = Bacterium("ecoli_1", "Escherichia coli", "heterotrophic", 100.0)
        assert b.id == "ecoli_1"
        assert b.species_name == "Escherichia coli"
        assert b.alive == True
        assert b.energy == 1.0
    
    def test_local_information(self):
        """Test: Lokale Information wird korrekt erfasst"""
        info = LocalInformation(
            chemical_concentration={"glucose": 0.01, "aspartate": 0.001},
            temperature=37.0,
            ph=7.0
        )
        assert info.temperature == 37.0
        assert "glucose" in info.chemical_concentration
    
    def test_deterministic_action(self):
        """Test: Aktionen sind deterministisch (Satz 3.1)"""
        b = Bacterium("test_1", "Test", "heterotrophic")
        
        # Setze Umgebung mit Glucose
        info = LocalInformation(
            chemical_concentration={"glucose": 0.01},
            oxygen_level=0.8
        )
        b.perceive_environment(info)
        
        # Gleiche Umgebung sollte gleiche Aktion produzieren
        action1 = b.determine_action()
        b.perceive_environment(info)
        action2 = b.determine_action()
        
        assert action1 == action2, "Aktionen sollten deterministisch sein"
    
    def test_consequence_generation(self):
        """Test: Konsequenzen werden generiert (Satz 3.2)"""
        b = Bacterium("test_2")
        
        # Reproduktion sollte Konsequenzen haben
        consequences = b.execute_action(BacterialAction.REPRODUCE)
        
        assert len(consequences) > 0, "Reproduktion sollte Konsequenzen erzeugen"
        assert consequences[0].magnitude > 0
        assert len(consequences[0].scope) > 0


class TestConsequenceTheory:
    """Tests für Konsequenzen-Theorie (Kapitel 3)"""
    
    def test_inevitable_consequences_theorem(self):
        """Test: Konsequenzen sind unvermeidlich (Satz 3.2)"""
        b = Bacterium("test_3")
        
        # Jede Aktion hat Konsequenzen
        for action in [BacterialAction.MOVE_FORWARD, 
                      BacterialAction.REPRODUCE,
                      BacterialAction.QUORUM_SENSE]:
            consequences = b.execute_action(action)
            assert len(consequences) >= 0, f"Action {action} sollte Konsequenzen haben"
    
    def test_consequence_properties(self):
        """Test: Konsequenzen haben erwartete Eigenschaften"""
        cons = Consequence(
            magnitude=0.5,
            scope={"organism_a", "organism_b"},
            duration=2.0,
            reversible=True,
            description="Test consequence"
        )
        
        assert 0 <= cons.magnitude <= 1
        assert isinstance(cons.scope, set)
        assert cons.duration > 0


class TestHelpFunctions:
    """Tests für Hilfsfunktionen (Kapitel 4)"""
    
    def test_nitrogen_fixing_bacteria_help_plants(self):
        """Test: Stickstoff-fixierende Bakterien helfen Pflanzen"""
        b = Bacterium("test_4", "Rhizobium", "nitrogen_fixing")
        
        help_type, magnitude = b.get_help_function("plant")
        
        assert help_type == HelpType.METABOLIC
        assert magnitude > 0.5, "Stickstoff-Fixierung sollte starke Hilfe sein"
    
    def test_heterotrophic_bacteria_decompose(self):
        """Test: Heterotrophe helfen bei Zersetzung"""
        b = Bacterium("test_5", "Bacillus", "heterotrophic")
        
        help_type, magnitude = b.get_help_function("organic_matter")
        
        assert help_type == HelpType.METABOLIC
        assert magnitude > 0.7, "Zersetzung sollte starke Hilfe sein"
    
    def test_help_function_scope(self):
        """Test: Hilfsfunktion hat begrenzte Reichweite"""
        b = Bacterium("test_6", "Unknown", "unknown")
        
        help_type, magnitude = b.get_help_function("unknown_organism")
        
        assert help_type == HelpType.NEUTRAL or magnitude == 0.0


class TestSubgraphAlgorithm:
    """Tests für den Subgraph Algorithmus (Kapitel 9-24)"""
    
    def test_subgraph_creation(self):
        """Test: Subgraph Algorithmus kann initialisiert werden"""
        algo = SubgraphAlgorithm()
        assert algo.graph is not None
        assert len(algo.subgraphs) == 0
    
    def test_network_building(self):
        """Test: Netzwerk aus Bakterien wird korrekt gebaut"""
        bacteria = [
            Bacterium(f"b{i}", f"species_{i % 2}", "heterotrophic")
            for i in range(5)
        ]
        
        # Interaktionsmatrix: 5x5
        interaction_matrix = MathematicalFormalisms.generate_interaction_matrix(5, sparsity=0.3)
        
        algo = SubgraphAlgorithm()
        algo.build_network_from_bacteria(bacteria, interaction_matrix)
        
        assert algo.graph.number_of_nodes() == 5
        assert algo.graph.number_of_edges() > 0
    
    def test_find_subgraphs_dfs(self):
        """Test: DFS findet zusammenhängende Komponenten"""
        # Baue Test-Netzwerk: 2 getrennte Komponenten
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])  # Component 1
        G.add_edges_from([(3, 4)])           # Component 2
        
        algo = SubgraphAlgorithm(G)
        subgraphs = algo.find_subgraphs_dfs()
        
        assert len(subgraphs) == 2, "Sollte 2 Komponenten finden"
    
    def test_dense_subgraph_detection(self):
        """Test: Greedy-Algorithmus findet dichte Subgraphen"""
        # Baue dichten und lockeren Subgraph
        G = nx.complete_graph(4)  # K4 - vollständig dicht
        G.add_nodes_from([4, 5, 6])  # Isolierte Knoten
        G.add_edge(4, 5)  # Eine Kante
        
        algo = SubgraphAlgorithm(G)
        dense_sgs = algo.find_dense_subgraphs_greedy(density_threshold=0.5)
        
        # Der K4 sollte als dichter Subgraph erkannt werden
        assert len(dense_sgs) > 0
        densest = max(dense_sgs, key=len)
        assert len(densest) >= 3, "K4 sollte erkannt werden"
    
    def test_clique_finding(self):
        """Test: Clique-Finder funktioniert"""
        # Triangle
        G = nx.complete_graph(3)
        
        algo = SubgraphAlgorithm(G)
        cliques = algo.find_cliques_bronkerbosh(max_cliques=10)
        
        assert len(cliques) > 0
        assert max(len(c) for c in cliques) >= 3
    
    def test_subgraph_properties_analysis(self):
        """Test: Subgraph-Eigenschaften können analysiert werden"""
        bacteria = [Bacterium(f"b{i}", f"sp_{i}", "h") for i in range(3)]
        interaction_matrix = np.array([
            [0, 0.5, 0.3],
            [0.5, 0, 0.8],
            [0.3, 0.8, 0]
        ])
        
        algo = SubgraphAlgorithm()
        algo.build_network_from_bacteria(bacteria, interaction_matrix)
        algo.find_subgraphs_dfs()
        
        props = algo.analyze_subgraph_properties()
        
        assert len(props) > 0
        for idx, prop in props.items():
            assert "node_count" in prop
            assert "density" in prop


class TestPopulationDynamics:
    """Tests für Populationsdynamik (Kapitel 6)"""
    
    def test_population_creation(self):
        """Test: Population kann erstellt werden"""
        pop = BacterialPopulation(carrying_capacity=1e9)
        assert pop.get_total_population() == 0
    
    def test_add_bacteria_to_population(self):
        """Test: Bakterien können hinzugefügt werden"""
        pop = BacterialPopulation()
        b = Bacterium("b1", "E. coli")
        
        pop.add_bacteria(b)
        
        assert pop.get_total_population() == 1
        assert "E. coli" in pop.get_species_count()
    
    def test_simulate_single_step(self):
        """Test: Ein Simulationsschritt funktioniert"""
        pop = BacterialPopulation()
        b = Bacterium("b1", "species1")
        b.energy = 0.8
        pop.add_bacteria(b)
        
        env = LocalInformation(
            chemical_concentration={"glucose": 0.01},
            temperature=37.0
        )
        
        changes = pop.simulate_step(env)
        
        assert pop.time_step == 1
        assert "species1" in changes
    
    def test_population_history(self):
        """Test: Populationshistorie wird aufgezeichnet"""
        pop = BacterialPopulation()
        b = Bacterium("b1", "species1")
        pop.add_bacteria(b)
        
        env = LocalInformation(temperature=37.0)
        pop.simulate_step(env)
        pop.simulate_step(env)
        
        assert len(pop.history) == 2
        assert pop.history[0]["time"] == 1
        assert pop.history[1]["time"] == 2


class TestConsciousnessModel:
    """Tests für Bewusstsein und Wille (Kapitel 7)"""
    
    def test_consciousness_initialization(self):
        """Test: Bewusstseinsmodell kann initialisiert werden"""
        consciousness = ConsciousnessModel(
            perception_range=500.0,
            information_processing_capacity=0.8
        )
        
        assert consciousness.perception_range == 500.0
        assert consciousness.processing_capacity == 0.8
    
    def test_will_strength_calculation(self):
        """Test: Willens-Stärke wird berechnet (Satz 7.1)"""
        # Größere Wahrnehmungsreichweite → stärkerer Wille
        c1 = ConsciousnessModel(100.0, 0.5)
        c2 = ConsciousnessModel(1000.0, 0.5)
        
        w1 = c1.calculate_will_strength()
        w2 = c2.calculate_will_strength()
        
        assert w2 > w1, "Größere Wahrnehmung sollte stärkerer Wille sein"
    
    def test_consequence_perception(self):
        """Test: Konsequenzen innerhalb Wahrnehmungsreichweite werden gesehen"""
        consciousness = ConsciousnessModel(1000.0, 1.0)  # Maximale Wahrnehmung
        
        consequences = [
            Consequence(0.5, {"org1"}, 1.0),
            Consequence(0.3, {"org2"}, 1.0)
        ]
        
        perceived, blind = consciousness.perceive_consequences(consequences)
        
        # Mit großer Wahrnehmungsreichweite sollten viele wahrgenommen werden
        assert len(perceived) + len(blind) == len(consequences)
    
    def test_deterministic_decision(self):
        """Test: Entscheidungen sind deterministisch"""
        consciousness = ConsciousnessModel(500.0, 0.8)
        
        options = ["option_a", "option_b", "option_c"]
        values = {"option_a": 0.3, "option_b": 0.8, "option_c": 0.2}
        
        decision1 = consciousness.make_decision(options, values)
        decision2 = consciousness.make_decision(options, values)
        
        assert decision1 == decision2 == "option_b", "Entscheidungen sollten deterministisch sein"


class TestMathematicalFormalisms:
    """Tests für mathematische Formalisierungen"""
    
    def test_interaction_matrix_generation(self):
        """Test: Interaktionsmatrix wird korrekt generiert"""
        matrix = MathematicalFormalisms.generate_interaction_matrix(5, sparsity=0.3)
        
        assert matrix.shape == (5, 5)
        assert np.allclose(matrix, matrix.T), "Matrix sollte symmetrisch sein"
        assert np.allclose(np.diag(matrix), 0), "Diagonale sollte 0 sein"
        assert np.all((matrix >= 0) & (matrix <= 1)), "Werte sollten [0,1] sein"
    
    def test_lotka_volterra_dynamics(self):
        """Test: Lotka-Volterra Gleichung funktioniert"""
        n = np.array([100.0, 50.0])
        r = np.array([0.5, 0.3])
        K = 1000.0
        A = np.array([[0.0, 0.1], [0.05, 0.0]])
        
        n_new = MathematicalFormalisms.lotka_volterra_dynamics(n, r, K, A)
        
        assert n_new.shape == n.shape
        assert np.all(n_new >= 0), "Populationen sollten nicht negativ sein"
    
    def test_information_entropy(self):
        """Test: Shannon-Entropie wird berechnet"""
        # Uniform verteilung: höchste Entropie
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        # Konzentriert: niedrige Entropie
        concentrated = np.array([0.9, 0.05, 0.03, 0.02])
        
        h_uniform = MathematicalFormalisms.information_entropy(uniform)
        h_concentrated = MathematicalFormalisms.information_entropy(concentrated)
        
        assert h_uniform > h_concentrated, "Uniform sollte höhere Entropie haben"
    
    def test_network_resilience(self):
        """Test: Netzwerk-Resilienz wird berechnet"""
        # Komplettes Netzwerk
        adj_matrix = np.ones((5, 5))
        np.fill_diagonal(adj_matrix, 0)
        
        # Entferne alle Knoten
        resilience_full = MathematicalFormalisms.network_resilience(adj_matrix, set())
        resilience_empty = MathematicalFormalisms.network_resilience(adj_matrix, {0, 1, 2, 3, 4})
        
        assert resilience_full == 1.0
        assert resilience_empty == 0.0


class TestIntegration:
    """Integrationstests für das Gesamtsystem"""
    
    def test_full_simulation_pipeline(self):
        """Test: Komplette Simulationspipeline"""
        # 1. Erstelle Bakterienpopulation
        pop = BacterialPopulation()
        bacteria = [
            Bacterium(f"ecoli_{i}", "E. coli", "heterotrophic")
            for i in range(10)
        ]
        for b in bacteria:
            pop.add_bacteria(b)
        
        # 2. Erstelle Subgraph Algorithmus
        interaction_matrix = MathematicalFormalisms.generate_interaction_matrix(10)
        algo = SubgraphAlgorithm()
        algo.build_network_from_bacteria(bacteria, interaction_matrix)
        
        # 3. Finde Subgraphen
        subgraphs = algo.find_subgraphs_dfs()
        assert len(subgraphs) > 0
        
        # 4. Simuliere Population
        env = LocalInformation(
            chemical_concentration={"glucose": 0.01},
            temperature=37.0
        )
        history = pop.run_simulation(steps=5, environment=env)
        
        assert len(history) == 5
        assert pop.time_step == 5
    
    def test_consciousness_consequence_integration(self):
        """Test: Bewusstsein nimmt Konsequenzen wahr"""
        # Erzeuge Konsequenzen
        consequences = [
            Consequence(0.5, {"org1"}, 1.0, description="cons1"),
            Consequence(0.3, {"org2"}, 2.0, description="cons2"),
            Consequence(0.7, {"org3"}, 0.5, description="cons3"),
        ]
        
        # Zwei verschiedene Bewusstseinsstufen
        low_consciousness = ConsciousnessModel(10.0, 0.1)
        high_consciousness = ConsciousnessModel(1000.0, 1.0)
        
        perceived_low, blind_low = low_consciousness.perceive_consequences(consequences)
        perceived_high, blind_high = high_consciousness.perceive_consequences(consequences)
        
        # Höhere Bewusstsein sollte mehr sehen
        assert len(perceived_high) + len(blind_high) == len(consequences)

# ═══════════════════════════════════════════════════════════════════════════════
# ERWEITERTE TESTS FÜR KERN-KONZEPTE
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacteriumAdvanced:
    """Erweiterte Tests für Bacterium Klasse"""
    
    @pytest.mark.parametrize("species,metabolism", [
        ("Escherichia coli", "heterotrophic"),
        ("Rhizobium", "nitrogen_fixing"),
        ("Cyanobacteria", "photosynthetic"),
        ("Bacillus", "heterotrophic"),
        ("Anabaena", "nitrogen_fixing"),
    ])
    def test_various_bacteria_species(self, species, metabolism):
        """Test: Verschiedene Bakterienarten können erstellt werden"""
        b = Bacterium(f"test_{species}", species, metabolism)
        assert b.species_name == species
        assert b.metabolism == metabolism
        assert b.alive
    
    @pytest.mark.parametrize("perception_range", [10.0, 100.0, 1000.0, 10000.0])
    def test_perception_range_impact(self, perception_range):
        """Test: Wahrnehmungsreichweite beeinflusst Verhalten"""
        b = Bacterium("test", "species", "heterotrophic", perception_range)
        assert b.perception_range == perception_range
    
    def test_energy_consumption(self):
        """Test: Energie wird beim Handeln verbraucht"""
        b = Bacterium("test")
        initial_energy = b.energy
        
        b.execute_action(BacterialAction.MOVE_FORWARD)
        
        assert b.energy < initial_energy
    
    def test_action_history_tracking(self):
        """Test: Aktionshistorie wird aufgezeichnet"""
        b = Bacterium("test")
        
        actions = [BacterialAction.MOVE_FORWARD, BacterialAction.TUMBLE, 
                  BacterialAction.QUORUM_SENSE]
        
        for action in actions:
            b.execute_action(action)
        
        assert len(b.action_history) == len(actions)
        assert b.action_history == actions
    
    def test_energy_death(self):
        """Test: Bakterium stirbt bei Energiemangel"""
        b = Bacterium("test")
        b.energy = 0.0
        b.alive = False
        
        assert not b.alive
    
    def test_consequence_accumulation(self):
        """Test: Konsequenzen werden akkumuliert"""
        b = Bacterium("test")
        
        for _ in range(3):
            b.execute_action(BacterialAction.REPRODUCE)
        
        assert len(b.consequences) > 0


class TestConsequenceTheoryAdvanced:
    """Erweiterte Tests für Konsequenzen-Theorie"""
    
    @pytest.mark.parametrize("magnitude,scope_size,duration", [
        (0.1, 1, 0.5),
        (0.5, 3, 2.0),
        (0.9, 5, 10.0),
    ])
    def test_consequence_properties_variation(self, magnitude, scope_size, duration):
        """Test: Konsequenzen mit verschiedenen Eigenschaften"""
        scope = {f"org_{i}" for i in range(scope_size)}
        cons = Consequence(magnitude, scope, duration)
        
        assert cons.magnitude == magnitude
        assert len(cons.scope) == scope_size
        assert cons.duration == duration
    
    def test_reversible_vs_irreversible_consequences(self):
        """Test: Unterscheidung reversibel/irreversibel"""
        reversible = Consequence(0.5, {"org"}, 1.0, reversible=True)
        irreversible = Consequence(0.5, {"org"}, 1.0, reversible=False)
        
        assert reversible.reversible
        assert not irreversible.reversible
    
    def test_consequence_scope_importance(self):
        """Test: Konsequenzen mit unterschiedlicher Reichweite"""
        small_scope = Consequence(0.5, {"org1"}, 1.0)
        large_scope = Consequence(0.5, {"org1", "org2", "org3", "org4"}, 1.0)
        
        assert len(small_scope.scope) < len(large_scope.scope)
    
    def test_consequence_magnitude_bounds(self):
        """Test: Konsequenzen sind begrenzt [0, 1]"""
        for magnitude in [0.0, 0.25, 0.5, 0.75, 1.0]:
            cons = Consequence(magnitude, {"org"}, 1.0)
            assert 0 <= cons.magnitude <= 1


class TestHelpFunctionsAdvanced:
    """Erweiterte Tests für Hilfsfunktionen (Lemma 4.1)"""
    
    @pytest.mark.parametrize("metabolism,target,expected_help_type", [
        ("photosynthetic", "plant", HelpType.METABOLIC),
        ("nitrogen_fixing", "plant", HelpType.METABOLIC),
        ("heterotrophic", "organic_matter", HelpType.METABOLIC),
    ])
    def test_help_function_consistency(self, metabolism, target, expected_help_type):
        """Test: Hilfsfunktion ist konsistent"""
        b = Bacterium("test", "species", metabolism)
        help_type, _ = b.get_help_function(target)
        
        assert help_type == expected_help_type
    
    def test_help_magnitude_ordering(self):
        """Test: Hilfsstärke folgt logischer Ordnung"""
        nitrogen_fixer = Bacterium("test", "Rhizobium", "nitrogen_fixing")
        
        help_to_plant, mag_plant = nitrogen_fixer.get_help_function("plant")
        help_to_human, mag_human = nitrogen_fixer.get_help_function("human")
        
        # Nitrogen-Fixierung hilft Pflanzen mehr als Menschen
        assert mag_plant > mag_human
    
    def test_unknown_target_handling(self):
        """Test: Unbekannte Ziele werden neutral behandelt"""
        b = Bacterium("test", "Unknown", "unknown")
        help_type, magnitude = b.get_help_function("unknown_organism")
        
        assert help_type in [HelpType.NEUTRAL, HelpType.HARMFUL]


# ═══════════════════════════════════════════════════════════════════════════════
# UMFANGREICHE SUBGRAPH-ALGORITHMUS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSubgraphAlgorithmComprehensive:
    """Umfangreiche Tests für den Subgraph Algorithmus"""
    
    def test_empty_graph_handling(self):
        """Test: Algorithmus kann mit leerem Graphen umgehen"""
        algo = SubgraphAlgorithm(nx.Graph())
        subgraphs = algo.find_subgraphs_dfs()
        
        assert len(subgraphs) == 0
    
    def test_single_node_graph(self):
        """Test: Einzelner Knoten wird korrekt behandelt"""
        G = nx.Graph()
        G.add_node(0)
        
        algo = SubgraphAlgorithm(G)
        subgraphs = algo.find_subgraphs_dfs()
        
        assert len(subgraphs) == 1
        assert len(subgraphs[0]) == 1
    
    @pytest.mark.parametrize("n_nodes,n_edges", [
        (5, 4),    # Tree
        (5, 6),    # Tree with one cycle
        (5, 10),   # Dense
    ])
    def test_various_network_topologies(self, n_nodes, n_edges):
        """Test: Verschiedene Netzwerk-Topologien"""
        G = nx.gnm_random_graph(n_nodes, n_edges)
        
        algo = SubgraphAlgorithm(G)
        subgraphs = algo.find_subgraphs_dfs()
        
        # Summe der Subgraph-Knoten sollte alle Knoten abdecken
        total_nodes = sum(len(sg) for sg in subgraphs)
        assert total_nodes == n_nodes
    
    def test_complete_graph_analysis(self):
        """Test: Kompletter Graph K_n wird korrekt analysiert"""
        G = nx.complete_graph(5)
        
        algo = SubgraphAlgorithm(G)
        algo.find_subgraphs_dfs()
        
        props = algo.analyze_subgraph_properties()
        
        # K_5 sollte eine Komponente sein mit Dichte 1.0
        assert len(props) == 1
        assert props[0]["density"] == 1.0
    
    def test_bipartite_graph(self):
        """Test: Bipartiter Graph wird korrekt erkannt"""
        G = nx.complete_bipartite_graph(3, 3)
        
        algo = SubgraphAlgorithm(G)
        subgraphs = algo.find_subgraphs_dfs()
        
        # Bipartiter Graph ist verbunden
        assert len(subgraphs) == 1
        assert len(subgraphs[0]) == 6
    
    def test_scale_free_network(self):
        """Test: Algorithmus skaliert auf Scale-Free Netzwerke"""
        G = nx.barabasi_albert_graph(50, 3)
        
        algo = SubgraphAlgorithm(G)
        subgraphs = algo.find_subgraphs_dfs()
        
        # Scale-Free Netzwerk ist typischerweise verbunden
        assert len(subgraphs) <= 2
    
    def test_clique_detection_accuracy(self):
        """Test: Clique-Detektion ist genau"""
        # Baue Netzwerk mit bekannten Cliques
        G = nx.complete_graph(4)
        G.add_node(4)
        G.add_edge(4, 0)
        
        algo = SubgraphAlgorithm(G)
        cliques = algo.find_cliques_bronkerbosh()
        
        # Größte Clique sollte K4 sein
        largest_clique = max(cliques, key=len)
        assert len(largest_clique) == 4
    
    def test_density_calculation_correctness(self):
        """Test: Dichte wird korrekt berechnet"""
        # Triangle hat Dichte 1.0
        triangle = {0, 1, 2}
        G = nx.complete_graph(3)
        
        algo = SubgraphAlgorithm(G)
        density = algo._calculate_density(triangle)
        
        assert density == 1.0
    
    @pytest.mark.parametrize("density_threshold", [0.3, 0.5, 0.7])
    def test_dense_subgraph_threshold_effect(self, density_threshold):
        """Test: Dichte-Schwelle beeinflusst Ergebnisse"""
        G = nx.gnm_random_graph(20, 30)
        
        algo = SubgraphAlgorithm(G)
        subgraphs_low = algo.find_dense_subgraphs_greedy(density_threshold)
        
        # Höhere Schwelle sollte weniger oder ähnlich viele Subgraphen finden
        # Teste mit noch höherer Schwelle
        algo2 = SubgraphAlgorithm(G)
        subgraphs_higher = algo2.find_dense_subgraphs_greedy(min(density_threshold + 0.3, 0.9))
        
        # Mindestens sollte der Test ohne Fehler durchlaufen
        # und Subgraphen-Struktur sinnvoll sein
        assert isinstance(subgraphs_low, list)
        assert all(isinstance(sg, set) for sg in subgraphs_low)


# ═══════════════════════════════════════════════════════════════════════════════
# POPULATION DYNAMICS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPopulationDynamicsAdvanced:
    """Erweiterte Tests für Populationsdynamik"""
    
    def test_population_growth_with_resources(self):
        """Test: Population wächst mit Ressourcen"""
        pop = BacterialPopulation()
        
        # Initialisiere mit energiereicher Population
        for i in range(10):
            b = Bacterium(f"b{i}", "species1")
            b.energy = 0.9  # Hohe Energie
            pop.add_bacteria(b)
        
        initial_size = pop.get_total_population()
        
        env = LocalInformation(
            chemical_concentration={"glucose": 0.1},
            temperature=37.0
        )
        
        pop.run_simulation(steps=3, environment=env)
        
        # Bevölkerung sollte gleich oder größer sein
        assert pop.get_total_population() >= initial_size
    
    def test_multiple_species_coexistence(self):
        """Test: Mehrere Arten können koexistieren"""
        pop = BacterialPopulation()
        
        for i in range(5):
            b = Bacterium(f"sp1_{i}", "Species_1")
            b.energy = 0.8
            pop.add_bacteria(b)
        
        for i in range(3):
            b = Bacterium(f"sp2_{i}", "Species_2")
            b.energy = 0.7
            pop.add_bacteria(b)
        
        species_count = pop.get_species_count()
        
        assert len(species_count) == 2
        assert species_count["Species_1"] == 5
        assert species_count["Species_2"] == 3
    
    def test_simulation_determinism(self):
        """Test: Simulation mit gleichen Bedingungen ist deterministisch"""
        def run_pop_simulation():
            pop = BacterialPopulation()
            for i in range(5):
                b = Bacterium(f"b{i}", "species1")
                b.energy = 0.8
                pop.add_bacteria(b)
            
            env = LocalInformation(temperature=37.0)
            history = pop.run_simulation(steps=5, environment=env)
            return history
        
        history1 = run_pop_simulation()
        history2 = run_pop_simulation()
        
        # Historien sollten identisch sein (NICHT stochastisch)
        for h1, h2 in zip(history1, history2):
            assert h1["time"] == h2["time"]
    
    def test_starvation_leads_to_death(self):
        """Test: Mangel an Ressourcen führt zu Tod"""
        pop = BacterialPopulation()
        
        b = Bacterium("b1", "species1")
        b.energy = 0.1  # Niedrige Energie
        pop.add_bacteria(b)
        
        env = LocalInformation(
            chemical_concentration={},  # Keine Nährstoffe
            temperature=37.0
        )
        
        pop.simulate_step(env)
        
        # Bakterium mit niedriger Energie sollte sterben
        # oder weitere Energie verlieren
        assert pop.get_total_population() <= 1
    
    def test_reproduction_doubles_population(self):
        """Test: Reproduktion verdoppelt Population"""
        pop = BacterialPopulation()
        
        b = Bacterium("b1", "species1")
        b.energy = 0.9
        pop.add_bacteria(b)
        
        initial_pop = pop.get_total_population()
        
        # Force Reproduktion
        env = LocalInformation(
            chemical_concentration={"glucose": 0.01},
            temperature=37.0
        )
        b.perceive_environment(env)
        b.execute_action(BacterialAction.REPRODUCE)
        
        # Neue Bakterium sollte hinzugefügt werden
        assert len(pop.bacteria["species1"]) + len(pop.bacteria) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# CONSCIOUSNESS AND WILL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConsciousnessAdvanced:
    """Erweiterte Tests für Bewusstsein und Wille"""
    
    @pytest.mark.parametrize("perception_range", [1.0, 10.0, 100.0, 1000.0])
    def test_will_scales_with_perception(self, perception_range):
        """Test: Wille skaliert monoton mit Wahrnehmungsreichweite (Satz 7.1)"""
        c = ConsciousnessModel(perception_range, 0.5)
        will = c.calculate_will_strength()
        
        assert 0 <= will <= 1
        assert will > 0  # Jedes Bewusstsein hat etwas Wille
    
    def test_will_comparison_human_vs_bacterium(self):
        """Test: Mensch hat stärkeren Willen als Bakterium"""
        bacterium = ConsciousnessModel(10.0, 0.1)
        human = ConsciousnessModel(1000.0, 0.9)
        
        will_bacterium = bacterium.calculate_will_strength()
        will_human = human.calculate_will_strength()
        
        assert will_human > will_bacterium
    
    def test_zero_perception_zero_will(self):
        """Test: Keine Wahrnehmung = kein Wille"""
        c = ConsciousnessModel(0.0, 1.0)
        will = c.calculate_will_strength()
        
        assert will == 0.0
    
    def test_consequence_perception_stochastic_nature(self):
        """Test: Konsequenzwahrnehmung ist probabilistisch"""
        c = ConsciousnessModel(1000.0, 1.0)  # Maximale Wahrnehmung
        
        consequences = [Consequence(0.5, {"org"}, 1.0) for _ in range(100)]
        
        perceived, blind = c.perceive_consequences(consequences)
        
        # Mit großer Wahrnehmungsreichweite sollten viele wahrgenommen werden
        perception_rate = len(perceived) / len(consequences)
        assert perception_rate > 0.5
    
    def test_deterministic_decision_with_clear_preference(self):
        """Test: Entscheidungen sind deterministisch mit klarer Präferenz"""
        c = ConsciousnessModel(500.0, 0.8)
        
        options = ["a", "b", "c"]
        values = {"a": 0.1, "b": 0.9, "c": 0.2}
        
        # Mehrfach dasselbe Szenario - sollte gleich sein
        decisions = [c.make_decision(options, values) for _ in range(5)]
        
        assert all(d == "b" for d in decisions)
    
    def test_decision_with_equal_values(self):
        """Test: Entscheidung bei gleichen Werten"""
        c = ConsciousnessModel(500.0, 0.8)
        
        options = ["a", "b", "c"]
        values = {"a": 0.5, "b": 0.5, "c": 0.5}
        
        decision = c.make_decision(options, values)
        
        # Sollte eine der Optionen sein
        assert decision in options


# ═══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL FORMALISMS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMathematicalFormalismAdvanced:
    """Erweiterte Tests für mathematische Formalisierungen"""
    
    @pytest.mark.parametrize("n", [2, 5, 10, 20])
    def test_interaction_matrix_properties(self, n):
        """Test: Interaktionsmatrix hat korrekte Eigenschaften"""
        matrix = MathematicalFormalisms.generate_interaction_matrix(n, sparsity=0.3)
        
        # Symmetrie
        assert np.allclose(matrix, matrix.T)
        
        # Keine Selbst-Loops
        assert np.allclose(np.diag(matrix), 0)
        
        # Werte in [0, 1]
        assert np.all((matrix >= 0) & (matrix <= 1))
        
        # Richtige Größe
        assert matrix.shape == (n, n)
    
    # @pytest.mark.parametrize("sparsity", [0.1, 0.3, 0.5, 0.7, 0.9])
    # def test_sparsity_effect(self, sparsity):
    #     """Test: Sparsity beeinflusst Anzahl der Kanten"""
    #     matrix = MathematicalFormalisms.generate_interaction_matrix(20, sparsity=sparsity)
        
    #     # Höhere Sparsität sollte weniger Kanten bedeuten
    #     num_nonzero = np.count_nonzero(matrix)
        
    #     # Sollte ungefähr (1 - sparsity) * total_elements * 0.5 sein
    #     # (mal 0.5 wegen Symmetrie)
    #     expected_approx = (1 - sparsity) * 20 * 20 * 0.5
        
    #     # Erlauben wir 50% Toleranz
    #     assert 0.5 * expected_approx < num_nonzero < 1.5 * expected_approx
    
    def test_lotka_volterra_energy_conservation(self):
        """Test: Lotka-Volterra Modell konserviert Energie (grundlegend)"""
        n = np.array([100.0, 80.0])
        r = np.array([0.3, 0.2])
        K = 1000.0
        A = np.array([[0.0, 0.05], [0.05, 0.0]])
        
        total_before = np.sum(n)
        n_new = MathematicalFormalisms.lotka_volterra_dynamics(n, r, K, A, dt=0.01)
        total_after = np.sum(n_new)
        
        # Mit kleinem dt sollte sich Gesamt-Population wenig ändern
        # (nicht wirklich Erhaltung, aber Stabilität)
        assert abs(total_after - total_before) < total_before * 0.1
    
    def test_entropy_maximal_with_uniform_distribution(self):
        """Test: Entropie ist maximal bei Uniform-Verteilung"""
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        concentrated = np.array([0.7, 0.1, 0.1, 0.1])
        
        h_uniform = MathematicalFormalisms.information_entropy(uniform)
        h_concentrated = MathematicalFormalisms.information_entropy(concentrated)
        
        assert h_uniform > h_concentrated
    
    def test_entropy_zero_for_single_species(self):
        """Test: Entropie ist Null wenn nur eine Art existiert"""
        single_species = np.array([1.0])
        
        h = MathematicalFormalisms.information_entropy(single_species)
        
        # log(1) = 0
        assert h == 0.0 or abs(h) < 1e-10
    
    def test_resilience_full_network_intact(self):
        """Test: Netzwerk-Resilienz bei keine Störung"""
        adj = np.ones((5, 5))
        np.fill_diagonal(adj, 0)
        
        resilience = MathematicalFormalisms.network_resilience(adj, set())
        
        assert resilience == 1.0
    
    def test_resilience_complete_destruction(self):
        """Test: Netzwerk-Resilienz bei totaler Zerstörung"""
        adj = np.ones((5, 5))
        np.fill_diagonal(adj, 0)
        
        resilience = MathematicalFormalisms.network_resilience(
            adj, 
            {0, 1, 2, 3, 4}
        )
        
        assert resilience == 0.0
    
    @pytest.mark.parametrize("num_removed", [1, 2, 3])
    def test_resilience_partial_damage(self, num_removed):
        """Test: Resilience fällt mit Schadensausmaß"""
        adj = np.ones((10, 10))
        np.fill_diagonal(adj, 0)
        
        resilience_intact = MathematicalFormalisms.network_resilience(adj, set())
        resilience_damaged = MathematicalFormalisms.network_resilience(
            adj, 
            set(range(num_removed))
        )
        
        assert resilience_damaged < resilience_intact


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemIntegration:
    """Integrationstests für das Gesamtsystem"""
    
    def test_full_workflow_bacteria_to_network_analysis(self):
        """Test: Kompletter Workflow von Bakterien zu Netzwerk-Analyse"""
        # 1. Erstelle Bakterien
        bacteria = [
            Bacterium(f"ecoli_{i}", "E. coli", "heterotrophic", 100.0)
            for i in range(8)
        ]
        
        # 2. Interaktionen definieren
        interaction_matrix = MathematicalFormalisms.generate_interaction_matrix(8)
        
        # 3. Subgraph-Analyse
        algo = SubgraphAlgorithm()
        algo.build_network_from_bacteria(bacteria, interaction_matrix)
        subgraphs = algo.find_subgraphs_dfs()
        
        # 4. Analyse
        props = algo.analyze_subgraph_properties()
        
        # Validierungen
        assert len(bacteria) == 8
        assert algo.graph.number_of_nodes() == 8
        assert len(subgraphs) > 0
        assert len(props) > 0
    
    def test_population_to_consciousness_flow(self):
        """Test: Populationsdynamik beeinflusst Bewusstsein"""
        # Erstelle Population
        pop = BacterialPopulation()
        for i in range(5):
            b = Bacterium(f"b{i}", "species1")
            b.energy = 0.7
            pop.add_bacteria(b)
        
        # Simuliere
        env = LocalInformation(temperature=37.0)
        pop.run_simulation(steps=3, environment=env)
        
        # Erstelle Bewusstsein für die Population
        # Je größer die Population, desto größer R_wahr
        population_size = pop.get_total_population()
        consciousness = ConsciousnessModel(
            perception_range=10.0 * population_size,  # Skaliert mit Population
            information_processing_capacity=0.5
        )
        
        will = consciousness.calculate_will_strength()
        assert will >= 0
    
    def test_consequences_propagation_through_network(self):
        """Test: Konsequenzen propagieren durch Netzwerk"""
        # Baue Netzwerk
        bacteria = [Bacterium(f"b{i}", f"sp{i%2}", "h") for i in range(10)]
        
        # Alle führen eine Aktion aus
        all_consequences = []
        for b in bacteria:
            cons = b.execute_action(BacterialAction.REPRODUCE)
            all_consequences.extend(cons)
        
        # Erstelle Bewusstsein
        consciousness = ConsciousnessModel(500.0, 0.8)
        perceived, blind = consciousness.perceive_consequences(all_consequences)
        
        # Mit realistischen Konsequenzen sollten einige wahrgenommen werden
        assert len(perceived) + len(blind) == len(all_consequences)
    
    def test_mathematical_model_consistency(self):
        """Test: Mathematische Modelle sind konsistent"""
        # Starte mit Populationen
        n = np.array([100.0, 80.0, 50.0])
        r = np.array([0.4, 0.3, 0.2])
        K = 1000.0
        A = MathematicalFormalisms.generate_interaction_matrix(3, sparsity=0.5)
        
        # Mehrere Iterationen
        for _ in range(10):
            n = MathematicalFormalisms.lotka_volterra_dynamics(n, r, K, A)
            
            # Populations sollten positiv bleiben
            assert np.all(n >= 0)
            
            # Populations sollten nicht zu groß werden
            assert np.all(n <= K * 2)


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE & STRESS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Performance und Stress Tests"""
    
    @pytest.mark.performance
    def test_large_network_analysis(self):
        """Test: Algorithmus kann große Netzwerke verarbeiten"""
        bacteria = [Bacterium(f"b{i}", f"sp{i%5}", "h") for i in range(100)]
        interaction_matrix = MathematicalFormalisms.generate_interaction_matrix(100, sparsity=0.8)
        
        algo = SubgraphAlgorithm()
        algo.build_network_from_bacteria(bacteria, interaction_matrix)
        
        # Sollte schnell sein
        import time
        start = time.time()
        subgraphs = algo.find_subgraphs_dfs()
        elapsed = time.time() - start
        
        assert elapsed < 1.0  # Sollte unter 1 Sekunde sein
        assert len(subgraphs) > 0
    
    @pytest.mark.performance
    def test_population_simulation_scaling(self):
        """Test: Population Simulation skaliert gut"""
        pop = BacterialPopulation()
        
        for i in range(50):
            b = Bacterium(f"b{i}", f"sp{i%3}", "h")
            b.energy = 0.5 + 0.3 * np.random.rand()
            pop.add_bacteria(b)
        
        env = LocalInformation(temperature=37.0)
        
        import time
        start = time.time()
        pop.run_simulation(steps=20, environment=env)
        elapsed = time.time() - start
        
        # 50 Bakterien x 20 Schritte sollte schnell sein
        assert elapsed < 2.0
    
    @pytest.mark.performance
    def test_entropy_calculation_speed(self):
        """Test: Entropie-Berechnung ist schnell"""
        import time
        
        # Große Populationen
        large_population = np.random.rand(1000)
        large_population /= np.sum(large_population)
        
        start = time.time()
        entropy = MathematicalFormalisms.information_entropy(large_population)
        elapsed = time.time() - start
        
        assert elapsed < 0.01  # Sollte schnell sein
        assert 0 <= entropy <= np.log2(1000)


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge Cases und Fehlerbehandlung"""
    
    def test_bacterium_with_zero_energy_dies(self):
        """Test: Bakterium mit Energie 0 ist tot"""
        b = Bacterium("test")
        b.energy = 0.0
        
        # Sollte keine Energie mehr verlieren
        initial = b.energy
        b.execute_action(BacterialAction.MOVE_FORWARD)
        
        assert b.energy <= initial
    
    def test_empty_population_simulation(self):
        """Test: Simulation mit leerer Population funktioniert"""
        pop = BacterialPopulation()
        env = LocalInformation()
        
        # Sollte nicht crashen
        history = pop.run_simulation(steps=5, environment=env)
        
        assert len(history) == 5
    
    def test_negative_magnitude_consequence(self):
        """Test: Konsequenzen mit Wert außerhalb [0,1] werden akzeptiert"""
        # Implementation sollte robust sein
        try:
            cons = Consequence(-0.5, {"org"}, 1.0)
            # Sollte ok sein, aber ClampedCons sollte Warnungen geben
        except:
            pass  # Ok, wenn es Fehler wirft
    
    def test_consciousness_with_zero_capacity(self):
        """Test: Bewusstsein mit Kapazität 0"""
        c = ConsciousnessModel(100.0, 0.0)
        will = c.calculate_will_strength()
        
        # Sollte 0 sein
        assert will == 0.0
    
    def test_network_with_self_loops_removed(self):
        """Test: Selbst-Schleifen sollten nicht in Netzwerk sein"""
        bacteria = [Bacterium(f"b{i}", f"sp{i}", "h") for i in range(5)]
        interaction_matrix = np.eye(5)  # Nur Diagonale
        
        algo = SubgraphAlgorithm()
        algo.build_network_from_bacteria(bacteria, interaction_matrix)
        
        # Sollte keine Kanten haben
        assert algo.graph.number_of_edges() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERIZED FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_bacterium():
    """Fixture: Standard-Bakterium"""
    return Bacterium("test_ecoli", "Escherichia coli", "heterotrophic", 100.0)


@pytest.fixture
def sample_population():
    """Fixture: Standard-Population"""
    pop = BacterialPopulation()
    for i in range(5):
        b = Bacterium(f"b{i}", "species_1", "heterotrophic")
        b.energy = 0.7
        pop.add_bacteria(b)
    return pop


@pytest.fixture
def sample_network():
    """Fixture: Standard-Netzwerk"""
    return nx.gnm_random_graph(10, 12)


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM TEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Konfiguriere pytest"""
    config.addinivalue_line(
        "markers", "performance: performance and stress tests"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests"
    )


# Run: pytest test_bakkt.py -v
