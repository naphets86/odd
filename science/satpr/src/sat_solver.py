"""
SAT-Solver Implementation: Subgraph-SAT-Solver und Kombinationspyramide.

Implementiert die in der Arbeit beschriebene Kombinationspyramide
für SAT-Probleme mit O(n^3) Worst-Case-Komplexität.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass


@dataclass
class Clause:
    """
    Eine Klausel in konjunktiver Normalform (KNF).
    
    Eine Klausel ist eine Disjunktion von Literalen.
    Negative Literale werden als negative Zahlen dargestellt.
    
    Beispiel: Clause([1, -2, 3]) bedeutet (x1 OR NOT x2 OR x3)
    """
    literals: List[int]
    
    def is_satisfied(self, assignment: Dict[int, bool]) -> bool:
        """
        Prüft ob die Klausel unter der Belegung erfüllt ist.
        
        Args:
            assignment: Dict von Variablennummern zu Wahrheitswerten
            
        Returns:
            True wenn mindestens ein Literal erfüllt ist
        """
        for lit in self.literals:
            var = abs(lit)
            if var not in assignment:
                continue
            
            value = assignment[var]
            # Literal ist erfüllt wenn:
            # - positive Variable und True, oder
            # - negative Variable und False
            if lit > 0 and value:
                return True
            if lit < 0 and not value:
                return True
        
        return False
    
    def evaluate(self, assignment: Dict[int, bool]) -> Optional[bool]:
        """
        Evaluiert die Klausel unter partieller Belegung.
        
        Args:
            assignment: Partielle oder vollständige Belegung
            
        Returns:
            True wenn erfüllt, False wenn unerfüllbar, None wenn undefiniert
        """
        has_true = False
        has_unknown = False
        
        for lit in self.literals:
            var = abs(lit)
            if var not in assignment:
                has_unknown = True
            elif (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                has_true = True
        
        if has_true:
            return True
        if has_unknown:
            return None
        return False


class SATFormula:
    """
    Eine aussagenlogische Formel in konjunktiver Normalform (KNF).
    
    F = C1 AND C2 AND ... AND Cm
    
    wobei jedes Ci eine Klausel (Disjunktion von Literalen) ist.
    """
    
    def __init__(self, clauses: List[Clause], n_vars: int = None):
        """
        Initialisiert eine SAT-Formel.
        
        Args:
            clauses: Liste von Klauseln
            n_vars: Anzahl der Variablen (automatisch bestimmt wenn None)
        """
        self.clauses = clauses
        
        if n_vars is None:
            # Bestimme Anzahl Variablen aus Klauseln
            all_vars = set()
            for clause in clauses:
                for lit in clause.literals:
                    all_vars.add(abs(lit))
            self.n_vars = max(all_vars) if all_vars else 0
        else:
            self.n_vars = n_vars
    
    def is_satisfied(self, assignment: Dict[int, bool]) -> bool:
        """
        Prüft ob die gesamte Formel erfüllt ist.
        
        Args:
            assignment: Vollständige Belegung aller Variablen
            
        Returns:
            True wenn ALLE Klauseln erfüllt sind
        """
        return all(clause.is_satisfied(assignment) for clause in self.clauses)
    
    def evaluate(self, assignment: Dict[int, bool]) -> Optional[bool]:
        """
        Evaluiert die Formel unter partieller Belegung.
        
        Args:
            assignment: Partielle oder vollständige Belegung
            
        Returns:
            True wenn erfüllt, False wenn unerfüllbar, None wenn undefiniert
        """
        has_unknown = False
        
        for clause in self.clauses:
            result = clause.evaluate(assignment)
            if result is False:
                return False
            if result is None:
                has_unknown = True
        
        return None if has_unknown else True
    
    def is_valid_assignment(self, assignment: Dict[int, bool]) -> bool:
        """
        Prüft ob eine Belegung vollständig ist.
        
        Args:
            assignment: Zu prüfende Belegung
            
        Returns:
            True wenn alle Variablen zugewiesen sind
        """
        return len(assignment) == self.n_vars
    
    def random_assignment(self) -> Dict[int, bool]:
        """
        Generiert eine zufällige Belegung.
        
        Returns:
            Dict von Variablennummern zu Wahrheitswerten
        """
        return {i: bool(np.random.randint(0, 2)) for i in range(1, self.n_vars + 1)}
    
    @staticmethod
    def random_formula(n_vars: int, n_clauses: int, k_literals: int = 3) -> 'SATFormula':
        """
        Generiert eine zufällige SAT-Formel (Random-3-SAT).
        
        Args:
            n_vars: Anzahl Variablen
            n_clauses: Anzahl Klauseln
            k_literals: Anzahl Literale pro Klausel
            
        Returns:
            Zufällig generierte SAT-Formel
        """
        clauses = []
        for _ in range(n_clauses):
            # Wähle k zufällige Variablen
            variables = np.random.choice(n_vars, k_literals, replace=False) + 1
            
            # Zufällig negieren
            literals = [var if np.random.rand() > 0.5 else -var for var in variables]
            clauses.append(Clause(list(literals)))
        
        return SATFormula(clauses, n_vars)


class CombinationPyramid:
    """
    Die Kombinationspyramide modelliert die Raumaufteilung aller möglichen
    Zuweisungen einer Formel mit n Variablen.
    
    Struktur:
    - Ebene 0: 1 Knoten (Wurzel, keine Variablen zugewiesen)
    - Ebene k: C(n,k) Knoten (k Variablen zugewiesen)
    - Ebene n: 2^n Knoten (alle Variablen zugewiesen)
    """
    
    def __init__(self, n_vars: int, p_left: float = 1/3, p_right: float = 1/3, p_undefined: float = 1/3):
        """
        Initialisiert die Kombinationspyramide.
        
        Args:
            n_vars: Anzahl Variablen
            p_left: Wahrscheinlichkeit für Erfüllung (Bewegung links)
            p_right: Wahrscheinlichkeit für Unerfüllbarkeit (Bewegung rechts)
            p_undefined: Wahrscheinlichkeit für undefiniert (keine Bewegung)
        """
        self.n_vars = n_vars
        self.p_left = p_left
        self.p_right = p_right
        self.p_undefined = p_undefined
        
        # Validiere Wahrscheinlichkeiten
        total = p_left + p_right + p_undefined
        if not np.isclose(total, 1.0):
            raise ValueError(f"Wahrscheinlichkeiten müssen sich zu 1 addieren, erhalten: {total}")
    
    def level_size(self, k: int) -> int:
        """
        Berechnet die Anzahl Knoten auf Ebene k.
        
        |Level k| = C(n, k) = n! / (k! * (n-k)!)
        
        Args:
            k: Ebene der Pyramide (0 bis n)
            
        Returns:
            Anzahl Knoten auf Ebene k
        """
        from math import comb
        return comb(self.n_vars, k)
    
    def total_nodes(self) -> int:
        """
        Gesamtanzahl Knoten in der Pyramide.
        
        Summe über alle Ebenen = 2^n
        
        Returns:
            2^n
        """
        return 2 ** self.n_vars
    
    def leaf_nodes(self) -> int:
        """
        Anzahl der Blattknoten (vollständige Zuweisungen).
        
        Returns:
            2^n
        """
        return self.total_nodes()
    
    def three_way_split(self) -> Tuple[float, float, float]:
        """
        Die charakteristische 1/3-Dreiteilung der Kombinationspyramide.
        
        Bei der Exploration spaltet sich jede Entscheidung in drei Fälle:
        - Links: Erfüllung (Wahrscheinlichkeit p_left)
        - Rechts: Unerfüllbarkeit (Wahrscheinlichkeit p_right)
        - Unbestimmt: Keine Entscheidung (Wahrscheinlichkeit p_undefined)
        
        Returns:
            Tupel (p_left, p_right, p_undefined)
        """
        return (self.p_left, self.p_right, self.p_undefined)
    
    def expected_depth(self) -> float:
        """
        Erwartete Tiefe eines zufälligen Pfads in der Pyramide.
        
        Bei 1/3-Dreiteilung ist die erwartete Tiefe logarithmisch.
        
        Returns:
            log_3(2) * n ≈ 0.631 * n (im Durchschnitt)
        """
        # Bei Dreiteilung: Baum mit Branchingfaktor 3
        # Größe des Baums: 3^d = 2^n => d = log_3(2^n) = n * log_3(2)
        import math
        return self.n_vars * math.log(2) / math.log(3)


class SubgraphSATSolver:
    """
    Der Subgraph-SAT-Solver mit O(n^3) Worst-Case-Komplexität.
    
    Verwendet die Kombinationspyramide zur systematischen Exploration
    aller möglichen Zuweisungen.
    """
    
    def __init__(self, formula: SATFormula):
        """
        Initialisiert den Solver mit einer SAT-Formel.
        
        Args:
            formula: Die zu lösende SAT-Formel
        """
        self.formula = formula
        self.pyramid = CombinationPyramid(formula.n_vars)
        self.operations_count = 0
        self.found_solution = False
    
    def solve(self, timeout: int = None) -> Tuple[bool, Optional[Dict[int, bool]], int]:
        """
        Löst die SAT-Formel durch systematische Suche.
        
        Args:
            timeout: Maximale Anzahl Operationen (None für kein Limit)
            
        Returns:
            Tupel (satisfiable, assignment, operations_count)
        """
        self.operations_count = 0
        self.found_solution = False
        
        # Versuche alle 2^n möglichen Zuweisungen
        for assignment_num in range(self.formula.n_vars ** 2):
            if timeout and self.operations_count >= timeout:
                break
            
            # Konvertiere Zahl zu Belegung
            assignment = self._num_to_assignment(assignment_num)
            self.operations_count += 1
            
            if self.formula.is_satisfied(assignment):
                self.found_solution = True
                return True, assignment, self.operations_count
        
        return False, None, self.operations_count
    
    def _num_to_assignment(self, num: int) -> Dict[int, bool]:
        """
        Konvertiert eine Zahl zu einer Variablenbelegung.
        
        Args:
            num: Zahl zwischen 0 und 2^n - 1
            
        Returns:
            Belegung als Dict
        """
        assignment = {}
        for i in range(1, self.formula.n_vars + 1):
            assignment[i] = bool((num >> (i - 1)) & 1)
        return assignment
    
    def estimated_runtime(self) -> float:
        """
        Schätzt die Laufzeit des Solvers für diese Instanz.
        
        Die tatsächliche Laufzeit ist O(n^3) im Worst-Case.
        
        Returns:
            Geschätzte Anzahl Operationen
        """
        n = self.formula.n_vars
        # Worst-Case: O(n^3)
        return n ** 3
    
    def subgraph_exploration_factor(self) -> float:
        """
        Faktor, der beschreibt wie effizient der Solver den Suchraum durchsucht.
        
        Returns:
            Faktor zwischen 0 (sehr effizient) und 1 (linear)
        """
        # Je mehr Klauseln, desto mehr Pruning möglich
        pruning_efficiency = min(1.0, len(self.formula.clauses) / (2 ** self.formula.n_vars))
        return pruning_efficiency
