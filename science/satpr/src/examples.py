#!/usr/bin/env python3
"""
Beispiel-Skript demonstriert die Verwendung aller SATPR-Module.

Zeigt:
1. SAT-Solver Grundlagen
2. Wahrscheinlichkeitsverschiebungs-Analyse
3. Logarithmisches Belegungsverfahren
4. Simulationen
5. Statistische Analysen
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from metadistribution import ProbabilitySpace, DistributionSpace, MetaDistribution
from sat_solver import Clause, SATFormula, CombinationPyramid, SubgraphSATSolver
from probability_shift import ProbabilityShiftAnalyzer, ThreeWayPartition
from lbv_solver import LogarithmicAssignmentProcedure, CombinedLBVSubgraphSolver
from simulation import MonteCarloSimulator, LBVSimulator
from statistics import EntropyAnalysis, PhaseTransitionAnalysis


def example_1_basic_sat():
    """Beispiel 1: Grundlegende SAT-Formel lösen."""
    print("\n" + "="*60)
    print("BEISPIEL 1: Grundlegende SAT-Formel Lösung")
    print("="*60)
    
    # Erstelle einfache Formel: (x1 OR x2) AND (NOT x1 OR x3)
    clauses = [
        Clause([1, 2]),      # x1 OR x2
        Clause([-1, 3])      # NOT x1 OR x3
    ]
    formula = SATFormula(clauses, n_vars=3)
    
    print(f"\nFormel: ({1} OR {2}) AND (NOT {1} OR {3})")
    print(f"Anzahl Variablen: {formula.n_vars}")
    print(f"Anzahl Klauseln: {len(formula.clauses)}")
    
    # Löse mit Subgraph-Solver
    solver = SubgraphSATSolver(formula)
    satisfiable, assignment, ops = solver.solve()
    
    print(f"\nErgebnis:")
    print(f"  Erfüllbar: {satisfiable}")
    if satisfiable:
        print(f"  Belegung: {assignment}")
    print(f"  Operationen: {ops}")


def example_2_random_sat():
    """Beispiel 2: Zufällige 3-SAT Formel generieren und lösen."""
    print("\n" + "="*60)
    print("BEISPIEL 2: Zufällige 3-SAT Formel")
    print("="*60)
    
    n_vars = 6
    n_clauses = 15
    
    # Generiere zufällige Formel
    formula = SATFormula.random_formula(n_vars, n_clauses, k_literals=3)
    
    print(f"\nGenerierte zufällige 3-SAT Formel:")
    print(f"  Variablen: {n_vars}")
    print(f"  Klauseln: {n_clauses}")
    print(f"  Klauseln pro Variable: {n_clauses / n_vars:.1f}")
    
    # Löse mit Solver
    solver = SubgraphSATSolver(formula)
    satisfiable, assignment, ops = solver.solve()
    
    print(f"\nLösungs-Statistiken:")
    print(f"  Erfüllbar: {satisfiable}")
    print(f"  Benötigte Operationen: {ops}")
    print(f"  Geschätzte Laufzeit O(n³): {n_vars**3}")


def example_3_combination_pyramid():
    """Beispiel 3: Kombinationspyramide Analyse."""
    print("\n" + "="*60)
    print("BEISPIEL 3: Kombinationspyramide")
    print("="*60)
    
    n = 5
    pyramid = CombinationPyramid(n)
    
    print(f"\nKombinationspyramide für {n} Variablen:")
    print(f"  Gesamtanzahl Knoten (2^n): {pyramid.total_nodes()}")
    print(f"  Blattknoten (vollständige Belegungen): {pyramid.leaf_nodes()}")
    
    print(f"\n  Knoten pro Ebene:")
    for k in range(n + 1):
        level_size = pyramid.level_size(k)
        print(f"    Ebene {k}: C({n},{k}) = {level_size}")
    
    print(f"\n  Dreiteilung (p_left, p_right, p_undefined):")
    split = pyramid.three_way_split()
    print(f"    {split[0]:.3f}, {split[1]:.3f}, {split[2]:.3f}")
    
    print(f"\n  Erwartete Pfadtiefe (logarithmisch):")
    depth = pyramid.expected_depth()
    print(f"    {depth:.2f} Schritte (vs n={n})")


def example_4_probability_shift():
    """Beispiel 4: Wahrscheinlichkeitsverschiebungs-Analyse."""
    print("\n" + "="*60)
    print("BEISPIEL 4: Wahrscheinlichkeitsverschiebung")
    print("="*60)
    
    n = 5
    analyzer = ProbabilityShiftAnalyzer(n)
    
    print(f"\nWahrscheinlichkeitsverschiebung für {n} Variablen:")
    
    # Bestimme Phasengrenzen
    phase1_end, phase2_end = analyzer.phase_transition_points()
    
    print(f"  Phase 1 (Logarithmisch): k = 1 bis {phase1_end}")
    print(f"  Phase 2 (Übergang): k = {phase1_end+1} bis {phase2_end}")
    print(f"  Phase 3 (Asymptotik): k = {phase2_end+1} bis {2**n}")
    
    print(f"\n  Erwartete Laufzeiten:")
    print(f"    {'k':<5} {'Phase':<7} {'E[T_k]':<10} {'O-Notation'}")
    print(f"    " + "-" * 35)
    
    for k in [1, phase1_end//2, phase1_end, (phase1_end+phase2_end)//2, 
              phase2_end, phase2_end + 5, 2**n - 5]:
        if k <= analyzer.max_formulas:
            phase = analyzer.get_phase(k)
            runtime = analyzer.expected_runtime(k)
            
            if phase == 1:
                notation = "O(log n)"
            elif phase == 2:
                notation = "O(n^2)"
            else:
                notation = "O(n³)"
            
            print(f"    {k:<5} {phase:<7} {runtime:<10.2f} {notation}")


def example_5_lbv_algorithm():
    """Beispiel 5: Logarithmisches Belegungsverfahren."""
    print("\n" + "="*60)
    print("BEISPIEL 5: Logarithmisches Belegungsverfahren (LBV)")
    print("="*60)
    
    m = 8
    lbv = LogarithmicAssignmentProcedure(m)
    
    print(f"\nLBV für {m} Variablen:")
    print(f"  Maximale Iterationen K = ⌈log₂({m})⌉ + 2 = {lbv.iteration_count()}")
    
    print(f"\n  Flip-Sequenz:")
    print(f"    {'Iteration':<10} {'Flip-Größe':<15} {'Belegung-Typ'}")
    print(f"    " + "-" * 40)
    
    for k in range(min(6, lbv.iteration_count())):
        flip = lbv.flip_size(k)
        assignment = lbv.generate_assignment(k)
        
        if k == 0:
            desc = "Alle True"
        elif k == 1:
            desc = "Alle False"
        else:
            desc = f"First {flip} flipped"
        
        print(f"    {k:<10} {flip:<15} {desc}")
    
    # Erfolgswahrscheinlichkeit
    prob = lbv.expected_success_probability(1/3)
    print(f"\n  Erfolgswahrscheinlichkeit (p=1/3): {prob:.4f}")
    print(f"  Kombiniert mit Subgraph: O(m³ log m) = O({m**3} * {int(np.log2(m))+2})")


def example_6_simulation():
    """Beispiel 6: Monte-Carlo Simulation."""
    print("\n" + "="*60)
    print("BEISPIEL 6: Monte-Carlo Simulation")
    print("="*60)
    
    n_vars = 4
    simulator = MonteCarloSimulator(n_variables=n_vars, seed=42)
    
    print(f"\nFühre Simulation durch für {n_vars} Variablen...")
    print(f"  Pyramidengröße: 2^{n_vars} = {2**n_vars}")
    
    # Simuliere Phase 1
    print(f"\n  Simuliere Phase 1 (Logarithmisch)...")
    results_p1 = simulator.run_phase_simulation(phase=1, n_samples=3)
    
    print(f"    Durchschnittliche Laufzeit: {np.mean([r.actual_runtime for r in results_p1]):.2f}")
    print(f"    Erwartete Laufzeit: {np.mean([r.expected_runtime for r in results_p1]):.2f}")
    
    # Simuliere Phase 3
    print(f"\n  Simuliere Phase 3 (Worst-Case)...")
    results_p3 = simulator.run_phase_simulation(phase=3, n_samples=3)
    
    print(f"    Durchschnittliche Laufzeit: {np.mean([r.actual_runtime for r in results_p3]):.2f}")
    print(f"    Erwartete Laufzeit: {np.mean([r.expected_runtime for r in results_p3]):.2f}")
    print(f"    Worst-Case O(n³): {n_vars**3}")


def example_7_lbv_simulation():
    """Beispiel 7: LBV Simulation."""
    print("\n" + "="*60)
    print("BEISPIEL 7: LBV Simulation")
    print("="*60)
    
    n_vars = 5
    lbv_sim = LBVSimulator(n_variables=n_vars, seed=42)
    
    print(f"\nSimuliere LBV für {n_vars} Variablen...")
    
    result = lbv_sim.simulate_lbv_success_rate(n_formulas=10)
    
    print(f"\n  Erfolgsrate: {result['success_rate']:.2%}")
    print(f"  Durchschnittliche Iterationen: {result['mean_iterations']:.1f}")
    print(f"  Theoretische Iterationen: {result['theoretical_iterations']}")
    print(f"  Theoretische Erfolgswahrscheinlichkeit: {result['expected_success_prob']:.4f}")


def example_8_metadistribution():
    """Beispiel 8: Metaverteilungs-Theorie."""
    print("\n" + "="*60)
    print("BEISPIEL 8: Metaverteilungs-Theorie")
    print("="*60)
    
    # Erstelle Wahrscheinlichkeitsräume
    ps1 = ProbabilitySpace(3, np.array([0.5, 0.3, 0.2]))
    ps2 = ProbabilitySpace(3, np.array([0.2, 0.3, 0.5]))
    
    print(f"\nZwei Wahrscheinlichkeitsverteilungen:")
    print(f"  P1 = {ps1.measure}")
    print(f"  P2 = {ps2.measure}")
    
    # Berechne Entropien
    h1 = ps1.entropy()
    h2 = ps2.entropy()
    
    print(f"\n  Shannon-Entropie:")
    print(f"    H(P1) = {h1:.4f} bits")
    print(f"    H(P2) = {h2:.4f} bits")
    
    # Berechne Divergenzen
    kl = ps1.kl_divergence(ps2)
    ws = ps1.wasserstein_distance(ps2)
    
    print(f"\n  Divergenzen:")
    print(f"    KL-Divergenz D_KL(P1||P2) = {kl:.4f}")
    print(f"    Wasserstein-Distanz = {ws:.4f}")
    
    # Erstelle Metaverteilung
    space = DistributionSpace(3)
    space.add_distribution(ps1.measure)
    space.add_distribution(ps2.measure)
    
    meta = MetaDistribution(space)
    meta.add_trajectory((0, 1), 1.0)
    
    print(f"\n  Metaverteilung:")
    print(f"    Verteilungsraum-Dimension: {space.dimension()}")
    print(f"    Entropie der Metaverteilung: {meta.entropy():.4f}")
    print(f"    Grad der Verschiebung: {meta.shift_degree():.4f}")


def example_9_statistics():
    """Beispiel 9: Statistische Analysen."""
    print("\n" + "="*60)
    print("BEISPIEL 9: Statistische Analysen")
    print("="*60)
    
    # Generiere Daten mit 3-Phasen-Struktur
    data = (
        list(np.random.normal(2, 0.5, 5)) +          # Phase 1: klein
        list(np.random.normal(10, 2, 5)) +           # Phase 2: mittel
        list(np.random.normal(50, 3, 5))             # Phase 3: groß
    )
    
    print(f"\nAnalysiere Laufzeit-Daten mit 3 Phasen:")
    
    # Erkenne Übergänge
    transitions = PhaseTransitionAnalysis.detect_transition_points(data, window_size=2)
    print(f"\n  Erkannte Übergangspunkte: {transitions}")
    
    # Berechne Entropien
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    entropy = EntropyAnalysis.shannon_entropy(probs)
    
    print(f"\n  Shannon-Entropie einer Gleichverteilung: {entropy:.4f} bits")
    
    renyi2 = EntropyAnalysis.renyi_entropy(probs, alpha=2.0)
    print(f"  Rényi-Entropie (α=2): {renyi2:.4f} bits")
    
    # Zeige Phasen-Merkmale
    print(f"\n  Phasen-Merkmale:")
    print(f"    Phase 1 (früh):")
    print(f"      Mittelwert: {np.mean(data[:5]):.2f}")
    print(f"      Std-Abw.: {np.std(data[:5]):.2f}")
    
    print(f"    Phase 3 (spät):")
    print(f"      Mittelwert: {np.mean(data[-5:]):.2f}")
    print(f"      Std-Abw.: {np.std(data[-5:]):.2f}")


def main():
    """Führe alle Beispiele aus."""
    print("\n" + "="*60)
    print("SATPR - BEISPIEL DEMONSTRATIONEN")
    print("Wahrscheinlichkeitsverteilung und Laufzeitanalyse")
    print("des Subgraph-SAT-Solvers")
    print("="*60)
    
    try:
        example_1_basic_sat()
        example_2_random_sat()
        example_3_combination_pyramid()
        example_4_probability_shift()
        example_5_lbv_algorithm()
        example_6_simulation()
        example_7_lbv_simulation()
        example_8_metadistribution()
        example_9_statistics()
        
        print("\n" + "="*60)
        print("ALLE BEISPIELE ERFOLGREICH AUSGEFÜHRT!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nFEHLER: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
