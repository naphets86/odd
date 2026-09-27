"""
Demo Biologische Netzwerkanalyse 

Diese Demo zeigt die Fähigkeiten des Gen-Frameworks zur Analyse biologischer Netzwerke mit dem Subgraph Algorithmus.

Autor: Stephan Epp
"""

import numpy as np
from gen import BiologicalNetwork, PathwayAnalyzer

def print_section(title: str):
    """Druckt formatierte Abschnittsüberschrift"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_subsection(title: str):
    """Druckt formatierte Unterabschnittsüberschrift"""
    print(f"\n--- {title} ---")


def create_example_networks():
    """Erstellt Beispiel-Biologische Netzwerke für die Demonstration"""
    
    # 1. Glykolyse-Pfad (vereinfacht)
    glycolysis = BiologicalNetwork(
        name="Glycolysis",
        nodes=["Glucose", "G6P", "F6P", "FBP", "DHAP", "G3P", "Pyruvate"],
        adjacency_matrix=np.array([
            [0, 1, 0, 0, 0, 0, 0],  # Glucose -> G6P
            [0, 0, 1, 0, 0, 0, 0],  # G6P -> F6P
            [0, 0, 0, 1, 0, 0, 0],  # F6P -> FBP
            [0, 0, 0, 0, 1, 1, 0],  # FBP -> DHAP/G3P
            [0, 0, 0, 0, 0, 1, 0],  # DHAP -> G3P
            [0, 0, 0, 0, 0, 0, 1],  # G3P -> Pyruvate
            [0, 0, 0, 0, 0, 0, 0]   # Pyruvate (Ende)
        ])
    )
    
    # 2. Teilweise Glykolyse (Teilmenge)
    partial_glycolysis = BiologicalNetwork(
        name="Partial_Glycolysis",
        nodes=["Glucose", "G6P", "F6P", "FBP"],
        adjacency_matrix=np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
    )
    
    # 3. Krebs-Zyklus-Fragment
    krebs_fragment = BiologicalNetwork(
        name="Krebs_Cycle_Fragment",
        nodes=["Citrate", "Isocitrate", "α-Ketoglutarate", "Succinyl-CoA"],
        adjacency_matrix=np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
    )
    
    # 4. DNA-Schadensantwort-Netzwerk (Protein-Interaktionen)
    dna_damage_nodes = ["p53", "MDM2", "ATM", "DNA-PK", "CHK2"]
    dna_damage_edges = [
        ("p53", "MDM2"),    # p53 aktiviert MDM2
        ("MDM2", "p53"),    # MDM2 hemmt p53 (Rückkopplung)
        ("ATM", "p53"),     # ATM phosphoryliert p53
        ("ATM", "CHK2"),    # ATM aktiviert CHK2
        ("DNA-PK", "ATM"),  # DNA-PK aktiviert ATM
        ("CHK2", "p53")     # CHK2 phosphoryliert p53
    ]
    # Erstelle Adjazenzmatrix aus Kanten
    dna_damage_matrix = np.zeros((len(dna_damage_nodes), len(dna_damage_nodes)))
    for src, tgt in dna_damage_edges:
        src_idx = dna_damage_nodes.index(src)
        tgt_idx = dna_damage_nodes.index(tgt)
        dna_damage_matrix[src_idx, tgt_idx] = 1
    
    dna_damage = BiologicalNetwork(
        name="DNA_Damage_Response",
        nodes=dna_damage_nodes,
        adjacency_matrix=dna_damage_matrix
    )
    
    # 5. Vereinfachter Pentose-Phosphat-Pfad
    pentose_phosphate = BiologicalNetwork(
        name="Pentose_Phosphate",
        nodes=["G6P", "6PG", "Ru5P", "R5P"],
        adjacency_matrix=np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
    )
    
    # 6. Alternativer Glukose-Pfad
    alternative_glucose = BiologicalNetwork(
        name="Alternative_Glucose",
        nodes=["Glucose", "G6P", "6PG", "Ru5P"],
        adjacency_matrix=np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
    )
    
    return {
        "glycolysis": glycolysis,
        "partial_glycolysis": partial_glycolysis,
        "krebs_fragment": krebs_fragment,
        "dna_damage": dna_damage,
        "pentose_phosphate": pentose_phosphate,
        "alternative_glucose": alternative_glucose
    }


def demo_network_properties(networks):
    """Demonstriert die Analyse von Netzwerk-Eigenschaften"""
    print_section("1. NETZWERK-EIGENSCHAFTEN")
    
    for name, network in networks.items():
        print(f"{network.name}")
        print(f"   Knoten: {network.n}")
        print(f"   Kanten: {len(network.get_edges())}")
        
        # Zeige erste paar Kanten
        edges = network.get_edges()
        if edges:
            print(f"   Beispiel-Kanten: {edges[:3]}")
            if len(edges) > 3:
                print(f"                   ... und {len(edges) - 3} weitere")
        print()


def demo_pathway_comparison(analyzer):
    """Demonstriert den Pfad-Vergleich"""
    print_section("2. PFAD-VERGLEICH")
    
    # Vergleiche vollständige vs. teilweise Glykolyse
    print_subsection("Vergleich: Vollständige vs. Teilweise Glykolyse")
    result = analyzer.compare_networks("Glycolysis", "Partial_Glycolysis")
    print(f"Entscheidung: {result['decision']}")
    print(f"Interpretation: {result['interpretation']}")
    
    # Vergleiche verschiedene metabolische Pfade
    print_subsection("Vergleich: Glykolyse vs. Krebs-Zyklus")
    result = analyzer.compare_networks("Glycolysis", "Krebs_Cycle_Fragment")
    print(f"Entscheidung: {result['decision']}")
    print(f"Interpretation: {result['interpretation']}")
    
    # Vergleiche metabolisches vs. Protein-Netzwerk
    print_subsection("Vergleich: Metabolisches vs. Protein-Interaktions-Netzwerk")
    result = analyzer.compare_networks("Glycolysis", "DNA_Damage_Response")
    print(f"Entscheidung: {result['decision']}")
    print(f"Interpretation: {result['interpretation']}")
    
    # Vergleiche verwandte Pfade
    print_subsection("Vergleich: Pentose-Phosphat vs. Alternativer Glukose-Pfad")
    result = analyzer.compare_networks("Pentose_Phosphate", "Alternative_Glucose")
    print(f"Entscheidung: {result['decision']}")
    print(f"Interpretation: {result['interpretation']}")


def demo_common_pathways(analyzer):
    """Demonstriert das Finden von gemeinsamen Pfaden"""
    print_section("3. SUBGRAPH-BEZIEHUNGEN")
    
    common = analyzer.find_common_pathways()
    
    if common:
        print("✓ Subgraph-Beziehungen gefunden:\n")
        for i, (net_a, net_b, decision) in enumerate(common, 1):
            print(f"{i}. {net_a} ↔ {net_b}")
            print(f"   Entscheidung: {decision}")
            
            # Hole weitere Details
            result = analyzer.compare_networks(net_a, net_b)
            print(f"   {result['interpretation']}\n")
    else:
        print("✗ Keine Subgraph-Beziehungen zwischen den Netzwerken gefunden.")


def demo_protein_network_analysis(dna_damage_network):
    """Demonstriert die Analyse von Protein-Interaktions-Netzwerken"""
    print_section("4. PROTEIN-INTERAKTIONS-NETZWERK-ANALYSE")
    
    print(f"   Analysiere {dna_damage_network.name}")
    print(f"   Gesamtzahl Proteine: {dna_damage_network.n}")
    print(f"   Gesamtzahl Interaktionen: {len(dna_damage_network.get_edges())}\n")
    
    # Zeige Interaktions-Netzwerk
    print("  Interaktions-Netzwerk:")
    edges = dna_damage_network.get_edges()
    for src, tgt in edges:
        print(f"  {src} → {tgt}")


def demo_statistics(analyzer):
    """Demonstriert Statistiken"""
    print_section("5. GLOBALE STATISTIKEN")
    
    print("  Übersicht:")
    print(f"   Gesamtzahl analysierter Netzwerke: {len(analyzer.networks)}")
    
    total_nodes = sum(net.n for net in analyzer.networks.values())
    total_edges = sum(len(net.get_edges()) for net in analyzer.networks.values())
    
    print(f"   Gesamtzahl Knoten: {total_nodes}")
    print(f"   Gesamtzahl Kanten: {total_edges}")
    
    if len(analyzer.networks) > 0:
        avg_nodes = total_nodes / len(analyzer.networks)
        avg_edges = total_edges / len(analyzer.networks)
        print(f"   Durchschnittliche Knoten pro Netzwerk: {avg_nodes:.1f}")
        print(f"   Durchschnittliche Kanten pro Netzwerk: {avg_edges:.1f}")
    
    print("\n  Alle Netzwerke:")
    for i, (name, net) in enumerate(analyzer.networks.items(), 1):
        print(f"   {i}. {name:25} ({net.n} Knoten, {len(net.get_edges())} Kanten)")


def demo_export(network):
    """Demonstriert den Netzwerk-Export"""
    print_section("6. NETZWERK-EXPORT")
    
    print(f"  Exportiere {network.name} in Wörterbuch-Format:\n")
    
    data = network.to_dict()
    
    print(f"Name: {data['name']}")
    print(f"Knoten: {', '.join(data['nodes'])}")
    print(f"\nKanten ({len(data['edges'])}):")
    for src, tgt in data['edges']:
        print(f"  {src} → {tgt}")
    
    print(f"\nAdjazenzmatrix-Form: {len(data['adjacency_matrix'])}x{len(data['adjacency_matrix'][0])}")


def main():
    """Führt die vollständige Demonstration aus"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  GEN - Biologische Netzwerkanalyse mit Subgraph Algorithmus".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Demo: Metabolische Pfade & Protein-Interaktions-Netzwerke".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Erstelle Beispiel-Netzwerke
    networks = create_example_networks()
    
    # Initialisiere Analyzer
    analyzer = PathwayAnalyzer()
    
    # Füge alle Netzwerke hinzu
    print("\n🔧 Initialisiere Analyzer und lade Netzwerke...")
    for network in networks.values():
        analyzer.add_network(network)
    print(f"✓ {len(networks)} biologische Netzwerke geladen\n")
    
    # Führe Demonstrationen aus
    demo_network_properties(networks)
    demo_pathway_comparison(analyzer)
    demo_common_pathways(analyzer)
    demo_protein_network_analysis(networks["dna_damage"])
    demo_statistics(analyzer)
    demo_export(networks["partial_glycolysis"])
    
    # Abschließender Abschnitt
    print_section("DEMONSTRATION ABGESCHLOSSEN")
    print("  Das Gen-Framework hat erfolgreich analysiert:")
    print("   • Metabolische Pfade (Glykolyse, Krebs-Zyklus, Pentose-Phosphat)")
    print("   • Protein-Interaktions-Netzwerke (DNA-Schadensantwort)")
    print("   • Subgraph-Beziehungen zwischen Pfaden")
    print("   • Netzwerk-Statistiken und Eigenschaften")
    
    print("\n  Weitere Informationen:")
    print("   • GitHub: https://github.com/hjstephan/gen")
    print("   • Subgraph Algorithmus: https://github.com/hjstephan/subgraph")
    print("   • Dokumentation: Siehe README.md")
    
    print("\n  Versuche, diese Demo zu modifizieren:")
    print("   • Füge deine eigenen biologischen Netzwerke hinzu")
    print("   • Importiere Daten aus KEGG oder anderen Datenbanken")
    print("   • Analysiere krankheitsbezogene Pfadveränderungen")
    print("   • Vergleiche Netzwerke über verschiedene Organismen hinweg\n")


if __name__ == "__main__":
    main()
