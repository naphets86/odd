"""
markov_analyzer.py
==================
Markov-Ketten-Analysemodul fuer UML-Profil-basierte Echtzeitsysteme.

Dieses Modul implementiert diskrete und kontinuierliche Markov-Ketten
zur stochastischen Analyse von Zustandsuebergaengen in Echtzeitsystemen.
Es unterstuetzt Stabilitaetsanalysen (Eigenwertmethode), stationaere
Verteilungen, MFPT (Mean First Passage Time), Absorptionsanalysen
und Sensitivitaetsanalysen fuer UML-Profil-annotierte Systeme.

Autor: Stephan Epp
"""

from __future__ import annotations

import math
import copy
from typing import Optional


# ---------------------------------------------------------------------------
# Hilfsfunktionen fuer lineare Algebra (ohne externe Abhaengigkeiten)
# ---------------------------------------------------------------------------

def _mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Matrix-Multiplikation A @ B."""
    n = len(A)
    m = len(B[0])
    k = len(B)
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for l in range(k):
                C[i][j] += A[i][l] * B[l][j]
    return C


def _mat_vec(A: list[list[float]], v: list[float]) -> list[float]:
    """Matrix-Vektor-Multiplikation A @ v."""
    n = len(A)
    result = [0.0] * n
    for i in range(n):
        for j in range(len(v)):
            result[i] += A[i][j] * v[j]
    return result


def _vec_scale(v: list[float], s: float) -> list[float]:
    return [x * s for x in v]


def _vec_norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _identity(n: int) -> list[list[float]]:
    I = [[0.0] * n for _ in range(n)]
    for i in range(n):
        I[i][i] = 1.0
    return I


def _gauss_solve(A: list[list[float]], b: list[float]) -> list[float]:
    """
    Loest Ax = b mit Gauss-Elimination (in-place, mit Pivotisierung).
    Wirft ValueError, wenn das System singulaer ist.
    """
    n = len(A)
    # Kopie erstellen
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        # Pivotsuche
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot_row] = M[pivot_row], M[col]

        pivot = M[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Singulaere Matrix – kein eindeutiger Loesung.")

        for row in range(n):
            if row == col:
                continue
            factor = M[row][col] / pivot
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    return [M[i][n] / M[i][i] for i in range(n)]


def _power_iteration(
    A: list[list[float]], max_iter: int = 1000, tol: float = 1e-10
) -> tuple[float, list[float]]:
    """
    Power-Iteration fuer den dominanten Eigenwert/-vektor.
    Gibt (Eigenwert, normierter Eigenvektor) zurueck.
    """
    n = len(A)
    v = [1.0 / math.sqrt(n)] * n
    eigenvalue = 0.0
    for _ in range(max_iter):
        w = _mat_vec(A, v)
        norm = _vec_norm(w)
        if norm < 1e-15:
            break
        v_new = _vec_scale(w, 1.0 / norm)
        eigenvalue_new = sum(w[i] * v[i] for i in range(n))
        if _vec_norm(_vec_sub(v_new, v)) < tol:
            eigenvalue = eigenvalue_new
            v = v_new
            break
        v = v_new
        eigenvalue = eigenvalue_new
    return eigenvalue, v


# ---------------------------------------------------------------------------
# Klassen
# ---------------------------------------------------------------------------

class MarkovState:
    """Repraesentiert einen Zustand in einer Markov-Kette."""

    def __init__(self, name: str, state_id: int, is_absorbing: bool = False,
                 timing_annotation: Optional[dict] = None):
        """
        Parameters
        ----------
        name : str
            Bezeichner des Zustands (z.B. 'Closed', 'Opening').
        state_id : int
            Numerischer Index.
        is_absorbing : bool
            True, wenn der Zustand absorbierend ist (keine Ausgaenge).
        timing_annotation : dict, optional
            UML-Timing-Annotationen (minDuration, maxDuration, typicalDuration).
        """
        self.name = name
        self.state_id = state_id
        self.is_absorbing = is_absorbing
        self.timing_annotation = timing_annotation or {}

    def __repr__(self) -> str:
        return f"MarkovState(id={self.state_id}, name='{self.name}')"

    def get_timing_info(self) -> dict:
        """Gibt Timing-Informationen des Zustands zurueck."""
        return {
            "state": self.name,
            "minDuration": self.timing_annotation.get("minDuration", None),
            "maxDuration": self.timing_annotation.get("maxDuration", None),
            "typicalDuration": self.timing_annotation.get("typicalDuration", None),
        }


class DiscreteMarkovChain:
    """
    Diskrete zeitlich-homogene Markov-Kette.

    Unterstuetzt:
    - Stationaere Verteilung (Gleichungssystem-Methode)
    - Stabilitaetsanalyse via Eigenwerte
    - n-Schritt-Uebergangswahrscheinlichkeiten
    - Mean First Passage Time (MFPT)
    - Absorptionswahrscheinlichkeiten
    - Sensitivitaetsanalyse
    """

    def __init__(self, states: list[MarkovState],
                 transition_matrix: list[list[float]]):
        """
        Parameters
        ----------
        states : list[MarkovState]
            Liste der Zustaende.
        transition_matrix : list[list[float]]
            Stochastische Uebergangsmatrix P mit P[i][j] = P(X_{n+1}=j | X_n=i).
        """
        n = len(states)
        if len(transition_matrix) != n:
            raise ValueError("Anzahl der Zustaende muss Matrixgroesse entsprechen.")
        for i, row in enumerate(transition_matrix):
            if len(row) != n:
                raise ValueError(f"Zeile {i} hat falsche Laenge.")
            row_sum = sum(row)
            if abs(row_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"Zeile {i} summiert zu {row_sum:.6f}, nicht 1.0."
                )

        self.states = states
        self.n = n
        self.P = [row[:] for row in transition_matrix]
        self._stationary: Optional[list[float]] = None

    # ------------------------------------------------------------------
    # Kernmethoden
    # ------------------------------------------------------------------

    def stationary_distribution(self) -> list[float]:
        """
        Berechnet die stationaere Verteilung pi mit pi @ P = pi, sum(pi) = 1.

        Nutzt die lineare Gleichungssystem-Methode:
        (P^T - I) erweitert um Normierungsbedingung.

        Returns
        -------
        list[float]
            Stationaere Wahrscheinlichkeitsverteilung.
        """
        if self._stationary is not None:
            return self._stationary[:]

        n = self.n
        # Aufstellen: (P^T - I) x = 0, letzte Zeile: sum = 1
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                A[i][j] = self.P[j][i] - (1.0 if i == j else 0.0)
        # Letzte Zeile durch Normierungsbedingung ersetzen
        A[n - 1] = [1.0] * n
        b = [0.0] * (n - 1) + [1.0]

        self._stationary = _gauss_solve(A, b)
        # Negativitaet korrigieren (numerische Fehler)
        self._stationary = [max(0.0, x) for x in self._stationary]
        total = sum(self._stationary)
        self._stationary = [x / total for x in self._stationary]
        return self._stationary[:]

    def n_step_transition(self, n_steps: int) -> list[list[float]]:
        """
        Berechnet P^n (n-Schritt-Uebergangsmatrix) durch wiederholte Multiplikation.

        Parameters
        ----------
        n_steps : int
            Anzahl der Schritte (>= 0).
        """
        if n_steps < 0:
            raise ValueError("n_steps muss >= 0 sein.")
        result = _identity(self.n)
        base = [row[:] for row in self.P]
        exp = n_steps
        while exp > 0:
            if exp % 2 == 1:
                result = _mat_mul(result, base)
            base = _mat_mul(base, base)
            exp //= 2
        return result

    def analyze_stability(self) -> dict:
        """
        Analysiert die Stabilitaet der Markov-Kette ueber den zweiten Eigenwert.

        Fuer eine ergodische Kette gilt:
        - Dominanter Eigenwert = 1.0
        - Zweiter Eigenwert |lambda_2| < 1 => stabil / Konvergenz garantiert
        - Zerfallsrate alpha = -ln(|lambda_2|)

        Returns
        -------
        dict mit Schluesseln:
            dominant_eigenvalue, is_ergodic, second_eigenvalue,
            decay_rate, mixing_time_estimate, convergence_guaranteed
        """
        # Dominanten Eigenwert per Power-Iteration schaetzen
        dominant_ev, _ = _power_iteration(self.P)

        is_ergodic = abs(dominant_ev - 1.0) < 1e-6

        # Zweiten Eigenwert: Deflation
        _, v1 = _power_iteration(self.P)
        # Deflation: A2 = A - lambda1 * v1 * v1^T
        n = self.n
        A2 = [[self.P[i][j] - dominant_ev * v1[i] * v1[j]
               for j in range(n)] for i in range(n)]
        second_ev, _ = _power_iteration(A2, max_iter=500)

        abs_second = abs(second_ev)
        if abs_second >= 1.0 - 1e-9:
            decay_rate = 0.0
            mixing_time = float("inf")
        else:
            decay_rate = -math.log(abs_second) if abs_second > 1e-15 else float("inf")
            mixing_time = 1.0 / decay_rate if decay_rate > 0 else float("inf")

        return {
            "dominant_eigenvalue": dominant_ev,
            "is_ergodic": is_ergodic,
            "second_eigenvalue": second_ev,
            "abs_second_eigenvalue": abs_second,
            "decay_rate": decay_rate,
            "mixing_time_estimate": mixing_time,
            "convergence_guaranteed": is_ergodic and abs_second < 1.0,
        }

    def mean_first_passage_time(self, target_state: int) -> list[float]:
        """
        Berechnet die mittlere Erstpassagezeit (MFPT) von jedem Zustand
        zum Zielzustand.

        MFPT[i] = erwartete Anzahl Schritte von Zustand i zu target_state.
        MFPT[target] = 0 (per Konvention).

        Parameters
        ----------
        target_state : int
            Index des Zielzustands.
        """
        n = self.n
        if not (0 <= target_state < n):
            raise ValueError(f"target_state {target_state} ausserhalb [0, {n-1}].")

        # Aufstellen: m_i = 1 + sum_j P[i][j] * m_j  (fuer i != target)
        # => m_i - sum_{j!=target} P[i][j]*m_j = 1 + P[i][target]*0
        # => (I - P_reduced) m = 1
        indices = [i for i in range(n) if i != target_state]
        k = len(indices)
        idx_map = {v: i for i, v in enumerate(indices)}

        A = [[0.0] * k for _ in range(k)]
        b = [1.0] * k
        for row_i, i in enumerate(indices):
            A[row_i][row_i] = 1.0
            for j in indices:
                A[row_i][idx_map[j]] -= self.P[i][j]

        solution = _gauss_solve(A, b)
        mfpt = [0.0] * n
        for row_i, i in enumerate(indices):
            mfpt[i] = solution[row_i]
        return mfpt

    def absorption_probabilities(self) -> dict:
        """
        Berechnet Absorptionswahrscheinlichkeiten fuer Ketten mit
        absorbierenden Zustaenden.

        Returns
        -------
        dict mit 'absorbing_states', 'transient_states',
        'absorption_probs' (n x num_absorbing Matrix).
        """
        absorbing = [s.state_id for s in self.states if s.is_absorbing]
        transient = [s.state_id for s in self.states if not s.is_absorbing]

        if not absorbing:
            return {
                "absorbing_states": [],
                "transient_states": list(range(self.n)),
                "absorption_probs": [],
                "message": "Keine absorbierenden Zustaende.",
            }

        # Fundamentalmatrix N = (I - Q)^{-1}
        # Q = transient->transient Teilmatrix, R = transient->absorbing
        t = len(transient)
        a = len(absorbing)
        t_map = {v: i for i, v in enumerate(transient)}

        Q = [[self.P[transient[i]][transient[j]] for j in range(t)] for i in range(t)]
        R = [[self.P[transient[i]][absorbing[j]] for j in range(a)] for i in range(t)]

        # (I - Q)
        IminusQ = [[(1.0 if i == j else 0.0) - Q[i][j] for j in range(t)] for i in range(t)]

        # B = N @ R, aber wir loesen (I-Q) B_col = R_col fuer jede Spalte
        B = [[0.0] * a for _ in range(t)]
        for col in range(a):
            r_col = [R[i][col] for i in range(t)]
            sol = _gauss_solve(IminusQ, r_col)
            for i in range(t):
                B[i][col] = sol[i]

        return {
            "absorbing_states": absorbing,
            "transient_states": transient,
            "absorption_probs": B,
        }

    def sensitivity_analysis(self, delta: float = 0.01) -> list[list[float]]:
        """
        Sensitivitaetsanalyse: Aenderung der stationaeren Verteilung
        bei kleiner Stoerung jedes Eintrags P[i][j].

        Returns
        -------
        list[list[float]]
            Sensitivitaetsmatrix S[i][j] = ||d_pi / d_P_ij|| (numerisch).
        """
        pi_base = self.stationary_distribution()
        n = self.n
        S = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if self.P[i][j] == 0.0:
                    continue
                # Stoerung
                P_perturbed = [row[:] for row in self.P]
                P_perturbed[i][j] += delta
                # Renormieren der Zeile i
                row_sum = sum(P_perturbed[i])
                P_perturbed[i] = [x / row_sum for x in P_perturbed[i]]

                try:
                    dmc = DiscreteMarkovChain(self.states, P_perturbed)
                    pi_pert = dmc.stationary_distribution()
                    diff = _vec_norm(_vec_sub(pi_pert, pi_base))
                    S[i][j] = diff / delta
                except ValueError:
                    S[i][j] = float("nan")

        return S

    def state_distribution_after_steps(
        self, initial_dist: list[float], n_steps: int
    ) -> list[float]:
        """
        Berechnet die Verteilung nach n_steps Schritten ausgehend von initial_dist.

        Parameters
        ----------
        initial_dist : list[float]
            Anfangsverteilung (muss zu 1 summieren).
        n_steps : int
            Anzahl Schritte.
        """
        if abs(sum(initial_dist) - 1.0) > 1e-6:
            raise ValueError("Anfangsverteilung muss zu 1 summieren.")
        Pn = self.n_step_transition(n_steps)
        # dist @ Pn
        result = [0.0] * self.n
        for j in range(self.n):
            for i in range(self.n):
                result[j] += initial_dist[i] * Pn[i][j]
        return result

    def summary(self) -> dict:
        """Gibt eine Zusammenfassung der Kette zurueck."""
        pi = self.stationary_distribution()
        stab = self.analyze_stability()
        state_names = [s.name for s in self.states]
        return {
            "num_states": self.n,
            "state_names": state_names,
            "stationary_distribution": {
                state_names[i]: round(pi[i], 6) for i in range(self.n)
            },
            "stability": stab,
        }


class ContinuousMarkovChain:
    """
    Kontinuierliche Markov-Kette (CTMC) mit Generator-Matrix Q.

    Die Eintraege erfuellen:
    - Q[i][j] >= 0 fuer i != j  (Uebergangsraten)
    - Q[i][i] = -sum_{j!=i} Q[i][j]  (Zeilenabschnitt-Bedingung)

    Unterstuetzt:
    - Stationaere Verteilung
    - Transiente Verteilung (Uniformisierung)
    - Konversation zu eingebetteter diskreter Kette
    """

    def __init__(self, states: list[MarkovState],
                 generator_matrix: list[list[float]]):
        """
        Parameters
        ----------
        generator_matrix : list[list[float]]
            Q-Matrix mit Zeilensumme 0 und positiven Off-Diagonal-Eintraegen.
        """
        n = len(states)
        if len(generator_matrix) != n:
            raise ValueError("Matrixgroesse muss Anzahl der Zustaende entsprechen.")
        for i, row in enumerate(generator_matrix):
            if len(row) != n:
                raise ValueError(f"Zeile {i} hat falsche Laenge.")
            row_sum = sum(row)
            if abs(row_sum) > 1e-6:
                raise ValueError(
                    f"Zeile {i} summiert zu {row_sum:.6f}, nicht 0."
                )

        self.states = states
        self.n = n
        self.Q = [row[:] for row in generator_matrix]

    def stationary_distribution(self) -> list[float]:
        """
        Berechnet stationaere Verteilung: pi Q = 0, sum(pi) = 1.
        """
        n = self.n
        # (Q^T) x = 0, letzte Zeile = Normierung
        A = [[self.Q[j][i] for j in range(n)] for i in range(n)]
        A[n - 1] = [1.0] * n
        b = [0.0] * (n - 1) + [1.0]
        pi = _gauss_solve(A, b)
        pi = [max(0.0, x) for x in pi]
        total = sum(pi)
        return [x / total for x in pi]

    def to_embedded_chain(self) -> DiscreteMarkovChain:
        """
        Konvertiert CTMC in eingebettete diskrete Markov-Kette.
        P[i][j] = Q[i][j] / |Q[i][i]| fuer i != j, P[i][i] = 0.
        """
        n = self.n
        P = [[0.0] * n for _ in range(n)]
        for i in range(n):
            rate_out = -self.Q[i][i]
            if rate_out > 1e-15:
                for j in range(n):
                    if i != j:
                        P[i][j] = self.Q[i][j] / rate_out
            else:
                P[i][i] = 1.0  # absorbierender Zustand
        return DiscreteMarkovChain(self.states, P)

    def transient_distribution(
        self, initial_dist: list[float], t: float, num_terms: int = 50
    ) -> list[float]:
        """
        Berechnet transiente Verteilung pi(t) = pi(0) * exp(Qt) via Uniformisierung.

        Parameters
        ----------
        initial_dist : list[float]
            Startverteilung.
        t : float
            Zeitpunkt.
        num_terms : int
            Anzahl Tayloer-Terme (Genauigkeit).
        """
        if abs(sum(initial_dist) - 1.0) > 1e-6:
            raise ValueError("Anfangsverteilung muss zu 1 summieren.")

        n = self.n
        # Uniformisierungsrate: q = max |Q[i][i]|
        q = max(-self.Q[i][i] for i in range(n))
        if q < 1e-15:
            return initial_dist[:]

        # Uniformisierte Matrix P_u = I + Q/q
        Pu = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                Pu[i][j] = (1.0 if i == j else 0.0) + self.Q[i][j] / q

        # pi(t) = sum_{k=0}^{inf} e^{-qt} * (qt)^k/k! * pi(0) P_u^k
        qt = q * t
        prob = math.exp(-qt)
        Pk = _identity(n)  # P_u^0
        result = [0.0] * n
        poisson_weight = prob  # e^{-qt} * (qt)^0 / 0!

        for k in range(num_terms):
            # pi(0) @ Pk
            contrib = [sum(initial_dist[i] * Pk[i][j] for i in range(n)) for j in range(n)]
            for j in range(n):
                result[j] += poisson_weight * contrib[j]
            # Naechste Iteration
            if k < num_terms - 1:
                Pk = _mat_mul(Pk, Pu)
                poisson_weight *= qt / (k + 1)
                if poisson_weight < 1e-15:
                    break

        total = sum(result)
        if total > 1e-15:
            result = [x / total for x in result]
        else:
            # Numerisch degeneriert: Stationaere Verteilung zurueckgeben
            result = self.stationary_distribution()
        return result

    def summary(self) -> dict:
        """Gibt Zusammenfassung der CTMC zurueck."""
        pi = self.stationary_distribution()
        embedded = self.to_embedded_chain()
        stab = embedded.analyze_stability()
        state_names = [s.name for s in self.states]
        return {
            "num_states": self.n,
            "state_names": state_names,
            "stationary_distribution": {
                state_names[i]: round(pi[i], 6) for i in range(self.n)
            },
            "embedded_chain_stability": stab,
        }


class MarkovAnalyzer:
    """
    Hochrangige Analyseklasse, die DiscreteMarkovChain oder
    ContinuousMarkovChain entgegennimmt und umfassende Reports erzeugt.
    """

    def __init__(self, chain):
        """
        Parameters
        ----------
        chain : DiscreteMarkovChain | ContinuousMarkovChain
        """
        if not isinstance(chain, (DiscreteMarkovChain, ContinuousMarkovChain)):
            raise TypeError("chain muss DiscreteMarkovChain oder ContinuousMarkovChain sein.")
        self.chain = chain
        self._is_discrete = isinstance(chain, DiscreteMarkovChain)

    def full_report(self) -> dict:
        """Erstellt vollstaendigen Analysebericht."""
        report = {
            "chain_type": "discrete" if self._is_discrete else "continuous",
            "summary": self.chain.summary(),
        }
        if self._is_discrete:
            stab = self.chain.analyze_stability()
            report["stability_analysis"] = stab
            report["decay_rate"] = stab["decay_rate"]
            report["mixing_time"] = stab["mixing_time_estimate"]
        else:
            embedded = self.chain.to_embedded_chain()
            stab = embedded.analyze_stability()
            report["stability_analysis"] = stab
            report["decay_rate"] = stab["decay_rate"]

        return report

    def timing_compliance_check(self) -> list[dict]:
        """
        Prueft, ob die Verweildauern (1/rate fuer CTMC) mit den
        UML-Timing-Annotationen der Zustaende konform sind.
        """
        results = []
        if self._is_discrete:
            chain = self.chain
        else:
            chain = self.chain

        for state in chain.states:
            ta = state.timing_annotation
            if not ta:
                continue
            result = {"state": state.name, "compliant": True, "issues": []}
            min_d = ta.get("minDuration")
            max_d = ta.get("maxDuration")
            typical = ta.get("typicalDuration")

            if not self._is_discrete:
                idx = state.state_id
                rate_out = -self.chain.Q[idx][idx]
                if rate_out > 1e-15:
                    mean_sojourn = 1.0 / rate_out
                    result["mean_sojourn_time"] = mean_sojourn
                    if min_d is not None and mean_sojourn < min_d:
                        result["compliant"] = False
                        result["issues"].append(
                            f"Mittlere Verweildauer {mean_sojourn:.3f} < minDuration {min_d}"
                        )
                    if max_d is not None and mean_sojourn > max_d:
                        result["compliant"] = False
                        result["issues"].append(
                            f"Mittlere Verweildauer {mean_sojourn:.3f} > maxDuration {max_d}"
                        )
            results.append(result)
        return results
