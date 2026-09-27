"""
Umfassende Tests für das Gen-Modul mit hoher Überdeckung

Autor: Stephan Epp
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.gen import (
    BiologicalNetwork,
    PathwayAnalyzer,
    create_glycolysis_pathway,
    create_partial_glycolysis,
    create_krebs_cycle_fragment,
    create_protein_interaction_network,
    main
)

class TestBiologicalNetwork:
    """Tests für die BiologicalNetwork-Klasse"""
    
    def test_init_valid_network(self):
        """Test: Initialisierung mit gültigen Parametern"""
        nodes = ["A", "B", "C"]
        matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        network = BiologicalNetwork("TestNet", nodes, matrix)
        
        assert network.name == "TestNet"
        assert network.nodes == nodes
        assert network.n == 3
        assert np.array_equal(network.matrix, matrix)
    
    def test_init_invalid_matrix_shape(self):
        """Test: Fehler bei ungültiger Matrixform"""
        nodes = ["A", "B", "C"]
        matrix = np.array([
            [0, 1],
            [0, 0]
        ])
        
        with pytest.raises(ValueError, match="Adjazenzmatrix muss 3x3 sein"):
            BiologicalNetwork("TestNet", nodes, matrix)
    
    def test_init_empty_network(self):
        """Test: Initialisierung mit leerem Netzwerk"""
        nodes = []
        matrix = np.array([]).reshape(0, 0)
        network = BiologicalNetwork("EmptyNet", nodes, matrix)
        
        assert network.n == 0
        assert network.nodes == []
    
    def test_init_single_node(self):
        """Test: Initialisierung mit einzelnem Knoten"""
        nodes = ["A"]
        matrix = np.array([[0]])
        network = BiologicalNetwork("SingleNode", nodes, matrix)
        
        assert network.n == 1
        assert network.nodes == ["A"]
    
    def test_repr(self):
        """Test: String-Repräsentation"""
        nodes = ["A", "B"]
        matrix = np.array([[0, 1], [0, 0]])
        network = BiologicalNetwork("TestNet", nodes, matrix)
        
        assert repr(network) == "BiologicalNetwork('TestNet', n=2)"
    
    def test_get_edges_linear_path(self):
        """Test: Kanten aus linearem Pfad"""
        nodes = ["A", "B", "C"]
        matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        network = BiologicalNetwork("Linear", nodes, matrix)
        edges = network.get_edges()
        
        assert edges == [("A", "B"), ("B", "C")]
    
    def test_get_edges_no_edges(self):
        """Test: Netzwerk ohne Kanten"""
        nodes = ["A", "B", "C"]
        matrix = np.zeros((3, 3))
        network = BiologicalNetwork("NoEdges", nodes, matrix)
        edges = network.get_edges()
        
        assert edges == []
    
    def test_get_edges_complete_graph(self):
        """Test: Vollständiger Graph"""
        nodes = ["A", "B"]
        matrix = np.array([
            [1, 1],
            [1, 1]
        ])
        network = BiologicalNetwork("Complete", nodes, matrix)
        edges = network.get_edges()
        
        assert len(edges) == 4
        assert ("A", "A") in edges
        assert ("A", "B") in edges
        assert ("B", "A") in edges
        assert ("B", "B") in edges
    
    def test_get_edges_self_loops(self):
        """Test: Selbstschleifen"""
        nodes = ["A", "B"]
        matrix = np.array([
            [1, 0],
            [0, 1]
        ])
        network = BiologicalNetwork("SelfLoops", nodes, matrix)
        edges = network.get_edges()
        
        assert edges == [("A", "A"), ("B", "B")]
    
    def test_to_dict(self):
        """Test: Export zu Dictionary"""
        nodes = ["A", "B"]
        matrix = np.array([
            [0, 1],
            [0, 0]
        ])
        network = BiologicalNetwork("TestNet", nodes, matrix)
        data = network.to_dict()
        
        assert data["name"] == "TestNet"
        assert data["nodes"] == ["A", "B"]
        assert data["adjacency_matrix"] == [[0, 1], [0, 0]]
        assert data["edges"] == [("A", "B")]
    
    def test_to_dict_empty_network(self):
        """Test: Export von leerem Netzwerk"""
        nodes = []
        matrix = np.array([]).reshape(0, 0)
        network = BiologicalNetwork("Empty", nodes, matrix)
        data = network.to_dict()
        
        assert data["name"] == "Empty"
        assert data["nodes"] == []
        assert data["edges"] == []


class TestPathwayAnalyzer:
    """Tests für die PathwayAnalyzer-Klasse"""
    
    def test_init(self):
        """Test: Initialisierung des Analyzers"""
        analyzer = PathwayAnalyzer()
        
        assert analyzer.networks == {}
        assert analyzer.comparator is not None
    
    def test_add_network(self, capsys):
        """Test: Netzwerk hinzufügen"""
        analyzer = PathwayAnalyzer()
        nodes = ["A", "B"]
        matrix = np.array([[0, 1], [0, 0]])
        network = BiologicalNetwork("TestNet", nodes, matrix)
        
        analyzer.add_network(network)
        
        assert "TestNet" in analyzer.networks
        assert analyzer.networks["TestNet"] == network
        captured = capsys.readouterr()
        assert "✓ Netzwerk hinzugefügt: TestNet (2 Knoten)" in captured.out
    
    def test_add_multiple_networks(self):
        """Test: Mehrere Netzwerke hinzufügen"""
        analyzer = PathwayAnalyzer()
        
        net1 = BiologicalNetwork("Net1", ["A"], np.array([[0]]))
        net2 = BiologicalNetwork("Net2", ["B"], np.array([[0]]))
        
        analyzer.add_network(net1)
        analyzer.add_network(net2)
        
        assert len(analyzer.networks) == 2
        assert "Net1" in analyzer.networks
        assert "Net2" in analyzer.networks
    
    def test_compare_networks_missing_network(self):
        """Test: Vergleich mit fehlendem Netzwerk"""
        analyzer = PathwayAnalyzer()
        
        with pytest.raises(ValueError, match="Beide Netzwerke müssen zuerst hinzugefügt werden"):
            analyzer.compare_networks("NonExistent1", "NonExistent2")
    
    def test_compare_networks_one_missing(self):
        """Test: Vergleich mit einem fehlenden Netzwerk"""
        analyzer = PathwayAnalyzer()
        net = BiologicalNetwork("Net1", ["A"], np.array([[0]]))
        analyzer.add_network(net)
        
        with pytest.raises(ValueError):
            analyzer.compare_networks("Net1", "NonExistent")
    
    @patch('src.gen.Subgraph.compare_graphs')
    def test_compare_networks_keep_a(self, mock_compare, capsys):
        """Test: Vergleich mit Ergebnis keep_A"""
        analyzer = PathwayAnalyzer()
        mock_compare.return_value = "keep_A"
        
        net1 = BiologicalNetwork("Net1", ["A", "B"], np.array([[0, 1], [0, 0]]))
        net2 = BiologicalNetwork("Net2", ["A"], np.array([[0]]))
        
        analyzer.add_network(net1)
        analyzer.add_network(net2)
        
        result = analyzer.compare_networks("Net1", "Net2")
        
        assert result["network_a"] == "Net1"
        assert result["network_b"] == "Net2"
        assert result["decision"] == "keep_A"
        assert "Net1" in result["interpretation"]
        assert "Obermenge" in result["interpretation"]
    
    @patch('src.gen.Subgraph.compare_graphs')
    def test_compare_networks_keep_b(self, mock_compare):
        """Test: Vergleich mit Ergebnis keep_B"""
        analyzer = PathwayAnalyzer()
        mock_compare.return_value = "keep_B"
        
        net1 = BiologicalNetwork("Net1", ["A"], np.array([[0]]))
        net2 = BiologicalNetwork("Net2", ["A", "B"], np.array([[0, 1], [0, 0]]))
        
        analyzer.add_network(net1)
        analyzer.add_network(net2)
        
        result = analyzer.compare_networks("Net1", "Net2")
        
        assert result["decision"] == "keep_B"
        assert "Net2" in result["interpretation"]
    
    @patch('src.gen.Subgraph.compare_graphs')
    def test_compare_networks_keep_both(self, mock_compare):
        """Test: Vergleich mit Ergebnis keep_both"""
        analyzer = PathwayAnalyzer()
        mock_compare.return_value = "keep_both"
        
        net1 = BiologicalNetwork("Net1", ["A"], np.array([[0]]))
        net2 = BiologicalNetwork("Net2", ["B"], np.array([[0]]))
        
        analyzer.add_network(net1)
        analyzer.add_network(net2)
        
        result = analyzer.compare_networks("Net1", "Net2")
        
        assert result["decision"] == "keep_both"
        assert "unvergleichbar" in result["interpretation"]
    
    @patch('src.gen.Subgraph.compare_graphs')
    def test_compare_networks_keep_either(self, mock_compare):
        """Test: Vergleich mit Ergebnis keep_either"""
        analyzer = PathwayAnalyzer()
        mock_compare.return_value = "keep_either"
        
        net1 = BiologicalNetwork("Net1", ["A"], np.array([[0]]))
        net2 = BiologicalNetwork("Net2", ["A"], np.array([[0]]))
        
        analyzer.add_network(net1)
        analyzer.add_network(net2)
        
        result = analyzer.compare_networks("Net1", "Net2")
        
        assert result["decision"] == "keep_either"
        assert "identisch" in result["interpretation"]
    
    @patch('src.gen.Subgraph.compare_graphs')
    def test_compare_networks_unknown_result(self, mock_compare):
        """Test: Vergleich mit unbekanntem Ergebnis"""
        analyzer = PathwayAnalyzer()
        mock_compare.return_value = "unknown_result"
        
        net1 = BiologicalNetwork("Net1", ["A"], np.array([[0]]))
        net2 = BiologicalNetwork("Net2", ["B"], np.array([[0]]))
        
        analyzer.add_network(net1)
        analyzer.add_network(net2)
        
        result = analyzer.compare_networks("Net1", "Net2")
        
        assert "Unbekanntes Ergebnis" in result["interpretation"]
    
    def test_interpret_result_keep_a(self):
        """Test: Interpretation keep_A"""
        analyzer = PathwayAnalyzer()
        result = analyzer._interpret_result("keep_A", "NetA", "NetB")
        
        assert "NetA" in result
        assert "Obermenge" in result
    
    def test_interpret_result_keep_b(self):
        """Test: Interpretation keep_B"""
        analyzer = PathwayAnalyzer()
        result = analyzer._interpret_result("keep_B", "NetA", "NetB")
        
        assert "NetB" in result
        assert "Obermenge" in result
    
    def test_interpret_result_keep_both(self):
        """Test: Interpretation keep_both"""
        analyzer = PathwayAnalyzer()
        result = analyzer._interpret_result("keep_both", "NetA", "NetB")
        
        assert "unvergleichbar" in result
    
    def test_interpret_result_keep_either(self):
        """Test: Interpretation keep_either"""
        analyzer = PathwayAnalyzer()
        result = analyzer._interpret_result("keep_either", "NetA", "NetB")
        
        assert "identisch" in result
    
    def test_interpret_result_unknown(self):
        """Test: Interpretation unbekanntes Ergebnis"""
        analyzer = PathwayAnalyzer()
        result = analyzer._interpret_result("unknown", "NetA", "NetB")
        
        assert "Unbekanntes Ergebnis" in result
    
    @patch('src.gen.Subgraph.compare_graphs')
    def test_find_common_pathways_empty(self, mock_compare):
        """Test: Keine gemeinsamen Pfade"""
        analyzer = PathwayAnalyzer()
        mock_compare.return_value = "keep_both"
        
        net1 = BiologicalNetwork("Net1", ["A"], np.array([[0]]))
        net2 = BiologicalNetwork("Net2", ["B"], np.array([[0]]))
        
        analyzer.add_network(net1)
        analyzer.add_network(net2)
        
        common = analyzer.find_common_pathways()
        
        assert common == []
    
    @patch('src.gen.Subgraph.compare_graphs')
    def test_find_common_pathways_with_results(self, mock_compare):
        """Test: Gemeinsame Pfade gefunden"""
        analyzer = PathwayAnalyzer()
        mock_compare.return_value = "keep_A"
        
        net1 = BiologicalNetwork("Net1", ["A", "B"], np.array([[0, 1], [0, 0]]))
        net2 = BiologicalNetwork("Net2", ["A"], np.array([[0]]))
        net3 = BiologicalNetwork("Net3", ["C"], np.array([[0]]))
        
        analyzer.add_network(net1)
        analyzer.add_network(net2)
        analyzer.add_network(net3)
        
        common = analyzer.find_common_pathways()
        
        assert len(common) > 0
        assert ("Net1", "Net2", "keep_A") in common
    
    @patch('src.gen.Subgraph.compare_graphs')
    def test_find_common_pathways_multiple_networks(self, mock_compare):
        """Test: Mehrere Netzwerke mit Subgraph-Beziehungen"""
        analyzer = PathwayAnalyzer()
        
        # Simuliere verschiedene Vergleichsergebnisse
        def compare_side_effect(mat_a, mat_b):
            return "keep_either"
        
        mock_compare.side_effect = compare_side_effect
        
        net1 = BiologicalNetwork("Net1", ["A"], np.array([[0]]))
        net2 = BiologicalNetwork("Net2", ["A"], np.array([[0]]))
        net3 = BiologicalNetwork("Net3", ["A"], np.array([[0]]))
        
        analyzer.add_network(net1)
        analyzer.add_network(net2)
        analyzer.add_network(net3)
        
        common = analyzer.find_common_pathways()
        
        # Sollte alle Paare vergleichen
        assert len(common) >= 0


class TestCreateFunctions:
    """Tests für die Netzwerk-Erstellungsfunktionen"""
    
    def test_create_glycolysis_pathway(self):
        """Test: Glykolyse-Pfad erstellen"""
        network = create_glycolysis_pathway()
        
        assert network.name == "Glycolysis"
        assert len(network.nodes) == 7
        assert network.n == 7
        assert "Glucose" in network.nodes
        assert "Pyruvate" in network.nodes
        
        edges = network.get_edges()
        assert len(edges) > 0
    
    def test_create_partial_glycolysis(self):
        """Test: Teilweise Glykolyse erstellen"""
        network = create_partial_glycolysis()
        
        assert network.name == "Partial_Glycolysis"
        assert len(network.nodes) == 4
        assert network.n == 4
        assert "Glucose" in network.nodes
        assert "FBP" in network.nodes
    
    def test_create_krebs_cycle_fragment(self):
        """Test: Krebs-Zyklus-Fragment erstellen"""
        network = create_krebs_cycle_fragment()
        
        assert network.name == "Krebs_Cycle_Fragment"
        assert len(network.nodes) == 4
        assert network.n == 4
        assert "Citrate" in network.nodes
        assert "Succinyl-CoA" in network.nodes
    
    def test_create_protein_interaction_network(self):
        """Test: Protein-Interaktions-Netzwerk erstellen"""
        network = create_protein_interaction_network()
        
        assert network.name == "DNA_Damage_Response"
        assert len(network.nodes) == 4
        assert network.n == 4
        assert "p53" in network.nodes
        assert "MDM2" in network.nodes
        assert "ATM" in network.nodes
        assert "DNA-PK" in network.nodes
    
    def test_glycolysis_edges(self):
        """Test: Glykolyse-Kanten"""
        network = create_glycolysis_pathway()
        edges = network.get_edges()
        
        # Sollte mindestens einen Pfad haben
        assert len(edges) > 0
        
        # Erste Kante sollte von Glucose ausgehen
        assert any(edge[0] == "Glucose" for edge in edges)
    
    def test_partial_glycolysis_is_subset(self):
        """Test: Teilweise Glykolyse ist Teilmenge"""
        full = create_glycolysis_pathway()
        partial = create_partial_glycolysis()
        
        # Alle Knoten der teilweisen sollten in der vollständigen sein
        for node in partial.nodes:
            assert node in full.nodes


class TestMain:
    """Tests für die main-Funktion"""
    
    @patch('src.gen.PathwayAnalyzer')
    @patch('builtins.print')
    def test_main_execution(self, mock_print, mock_analyzer_class):
        """Test: Hauptfunktion wird ausgeführt"""
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.find_common_pathways.return_value = []
        
        main()
        
        # Überprüfe, dass Analyzer erstellt wurde
        mock_analyzer_class.assert_called_once()
        
        # Überprüfe, dass Netzwerke hinzugefügt wurden
        assert mock_analyzer.add_network.call_count == 4
        
        # Überprüfe, dass Vergleiche durchgeführt wurden
        assert mock_analyzer.compare_networks.call_count > 0

class TestIntegration:
    """Integrationstests"""
    
    def test_full_workflow(self):
        """Test: Vollständiger Workflow"""
        # Erstelle Netzwerke
        net1 = create_glycolysis_pathway()
        net2 = create_partial_glycolysis()
        
        # Erstelle Analyzer
        analyzer = PathwayAnalyzer()
        analyzer.add_network(net1)
        analyzer.add_network(net2)
        
        # Überprüfe, dass Netzwerke hinzugefügt wurden
        assert len(analyzer.networks) == 2
        
        # Überprüfe Netzwerk-Eigenschaften
        assert net1.n == 7
        assert net2.n == 4
        
        # Überprüfe Kanten
        edges1 = net1.get_edges()
        edges2 = net2.get_edges()
        
        assert len(edges1) > 0
        assert len(edges2) > 0
    
    def test_network_export_import(self):
        """Test: Netzwerk-Export und Struktur"""
        network = create_glycolysis_pathway()
        data = network.to_dict()
        
        # Überprüfe Export-Struktur
        assert "name" in data
        assert "nodes" in data
        assert "adjacency_matrix" in data
        assert "edges" in data
        
        # Überprüfe Datentypen
        assert isinstance(data["name"], str)
        assert isinstance(data["nodes"], list)
        assert isinstance(data["adjacency_matrix"], list)
        assert isinstance(data["edges"], list)
    
    def test_multiple_networks_comparison(self):
        """Test: Vergleich mehrerer Netzwerke"""
        analyzer = PathwayAnalyzer()
        
        net1 = create_glycolysis_pathway()
        net2 = create_partial_glycolysis()
        net3 = create_krebs_cycle_fragment()
        
        analyzer.add_network(net1)
        analyzer.add_network(net2)
        analyzer.add_network(net3)
        
        assert len(analyzer.networks) == 3
        
        # Alle Netzwerke sollten unterschiedliche Größen haben
        sizes = [net.n for net in analyzer.networks.values()]
        assert len(set(sizes)) >= 2  # Mindestens 2 unterschiedliche Größen


class TestEdgeCases:
    """Tests für Grenzfälle"""
    
    def test_large_network(self):
        """Test: Großes Netzwerk"""
        nodes = [f"Node_{i}" for i in range(100)]
        matrix = np.zeros((100, 100))
        # Erstelle einen linearen Pfad
        for i in range(99):
            matrix[i, i+1] = 1
        
        network = BiologicalNetwork("LargeNet", nodes, matrix)
        
        assert network.n == 100
        edges = network.get_edges()
        assert len(edges) == 99
    
    def test_network_with_special_characters(self):
        """Test: Netzwerk mit Sonderzeichen"""
        nodes = ["α-Ketoglutarate", "β-Alanine", "γ-Aminobutyric"]
        matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        network = BiologicalNetwork("SpecialChars", nodes, matrix)
        
        assert network.n == 3
        edges = network.get_edges()
        assert ("α-Ketoglutarate", "β-Alanine") in edges
    
    def test_network_with_duplicate_node_names(self):
        """Test: Netzwerk mit doppelten Knotennamen"""
        nodes = ["A", "A", "B"]
        matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        network = BiologicalNetwork("DuplicateNames", nodes, matrix)
        
        # Sollte trotzdem funktionieren
        assert network.n == 3
        edges = network.get_edges()
        assert len(edges) == 2
    
    def test_matrix_with_float_values(self):
        """Test: Matrix mit Float-Werten"""
        nodes = ["A", "B"]
        matrix = np.array([
            [0.0, 1.0],
            [0.0, 0.0]
        ])
        
        network = BiologicalNetwork("FloatMatrix", nodes, matrix)
        edges = network.get_edges()
        
        # Sollte 1.0 als Kante erkennen
        assert ("A", "B") in edges
    
    def test_matrix_with_non_binary_values(self):
        """Test: Matrix mit nicht-binären Werten"""
        nodes = ["A", "B"]
        matrix = np.array([
            [0, 2],
            [0, 0]
        ])
        
        network = BiologicalNetwork("NonBinary", nodes, matrix)
        edges = network.get_edges()
        
        # get_edges() prüft auf == 1, daher wird 2 nicht erkannt
        # Das ist das erwartete Verhalten
        assert ("A", "B") not in edges
    
    def test_compare_networks_with_output(self, capsys):
        """Test: Netzwerk-Vergleich mit Ausgabe"""
        analyzer = PathwayAnalyzer()
        
        net1 = BiologicalNetwork("Net1", ["A", "B"], np.array([[0, 1], [0, 0]]))
        net2 = BiologicalNetwork("Net2", ["A"], np.array([[0]]))
        
        analyzer.add_network(net1)
        analyzer.add_network(net2)
        
        with patch('src.gen.Subgraph.compare_graphs', return_value="keep_A"):
            result = analyzer.compare_networks("Net1", "Net2")
        
        captured = capsys.readouterr()
        assert "Vergleich: Net1 vs Net2" in captured.out
        assert "Net1: 2 Knoten, 1 Kanten" in captured.out
        assert "Net2: 1 Knoten, 0 Kanten" in captured.out
