"""
queue_analyzer.py
=================
Warteschlangen-Analysemodul fuer UML-Profil-basierte Echtzeitsysteme.

Implementiert M/M/1, M/M/c, M/D/1 und M/G/1 Warteschlangenmodelle
sowie Little's Law, Stabilitaetspruefung (rho < 1), Sojourn-Time-
Verteilung, Percentile-Analyse und Batch-Analyse fuer Systeme mit
mehreren Diensten.

Autor: Stephan Epp
"""

from __future__ import annotations

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _factorial(n: int) -> int:
    """Berechnet n! iterativ."""
    if n < 0:
        raise ValueError("Fakultaet fuer negative Zahlen nicht definiert.")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _poisson_cdf(k: int, lam: float) -> float:
    """Kumulierte Poisson-Verteilung P(X <= k) fuer Parameter lam."""
    total = 0.0
    prob = math.exp(-lam)
    total = prob
    for i in range(1, k + 1):
        prob *= lam / i
        total += prob
    return total


# ---------------------------------------------------------------------------
# Basisklasse
# ---------------------------------------------------------------------------

class QueueMetrics:
    """Buendelt Kenngroessen eines Warteschlangensystems."""

    def __init__(
        self,
        model_name: str,
        arrival_rate: float,
        service_rate: float,
        num_servers: int,
        utilization: float,
        L: float,
        Lq: float,
        W: float,
        Wq: float,
        throughput: float,
        is_stable: bool,
    ):
        self.model_name = model_name
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.num_servers = num_servers
        self.utilization = utilization
        self.L = L          # Mittlere Anzahl im System
        self.Lq = Lq        # Mittlere Warteschlangenlaenge
        self.W = W          # Mittlere Systemzeit
        self.Wq = Wq        # Mittlere Wartezeit in Warteschlange
        self.throughput = throughput
        self.is_stable = is_stable

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "lambda (Ankunftsrate)": self.arrival_rate,
            "mu (Bedienrate)": self.service_rate,
            "c (Server)": self.num_servers,
            "rho (Auslastung)": round(self.utilization, 6),
            "L (Kunden im System)": round(self.L, 6),
            "Lq (Warteschlange)": round(self.Lq, 6),
            "W (Systemzeit)": round(self.W, 6),
            "Wq (Wartezeit)": round(self.Wq, 6),
            "Throughput": round(self.throughput, 6),
            "stabil": self.is_stable,
        }

    def littles_law_verification(self) -> dict:
        """Verifiziert Little's Law: L = lambda * W und Lq = lambda * Wq."""
        L_check = self.arrival_rate * self.W
        Lq_check = self.arrival_rate * self.Wq
        return {
            "L = lambda * W": {
                "computed_L": round(self.L, 6),
                "lambda_times_W": round(L_check, 6),
                "difference": round(abs(self.L - L_check), 8),
                "verified": abs(self.L - L_check) < 1e-4,
            },
            "Lq = lambda * Wq": {
                "computed_Lq": round(self.Lq, 6),
                "lambda_times_Wq": round(Lq_check, 6),
                "difference": round(abs(self.Lq - Lq_check), 8),
                "verified": abs(self.Lq - Lq_check) < 1e-4,
            },
        }

    def __repr__(self) -> str:
        return (
            f"{self.model_name}(lambda={self.arrival_rate}, mu={self.service_rate}, "
            f"rho={self.utilization:.3f}, L={self.L:.3f}, W={self.W:.3f})"
        )


# ---------------------------------------------------------------------------
# M/M/1-Warteschlange
# ---------------------------------------------------------------------------

class MM1Queue:
    """
    M/M/1-Warteschlangensystem: Poisson-Ankuenfte, exponentielle Bedienung,
    ein Server, unendliche Kapazitaet, FCFS-Disziplin.

    Kenngroessen (stabil, rho < 1):
    - rho = lambda / mu
    - L = rho / (1 - rho)
    - Lq = rho^2 / (1 - rho)
    - W = 1 / (mu - lambda)
    - Wq = rho / (mu - lambda)
    """

    def __init__(self, arrival_rate: float, service_rate: float):
        """
        Parameters
        ----------
        arrival_rate : float
            Mittlere Ankunftsrate lambda (Ankuenfte pro Zeiteinheit).
        service_rate : float
            Mittlere Bedienrate mu (Kunden pro Zeiteinheit und Server).
        """
        if arrival_rate <= 0:
            raise ValueError("arrival_rate muss positiv sein.")
        if service_rate <= 0:
            raise ValueError("service_rate muss positiv sein.")
        self.lam = arrival_rate
        self.mu = service_rate
        self.rho = arrival_rate / service_rate

    @property
    def is_stable(self) -> bool:
        return self.rho < 1.0

    def metrics(self) -> QueueMetrics:
        """Berechnet alle Kenngroessen des M/M/1-Systems."""
        if not self.is_stable:
            return QueueMetrics(
                model_name="M/M/1",
                arrival_rate=self.lam,
                service_rate=self.mu,
                num_servers=1,
                utilization=self.rho,
                L=float("inf"),
                Lq=float("inf"),
                W=float("inf"),
                Wq=float("inf"),
                throughput=self.lam,
                is_stable=False,
            )
        rho = self.rho
        L = rho / (1.0 - rho)
        Lq = rho ** 2 / (1.0 - rho)
        W = 1.0 / (self.mu - self.lam)
        Wq = rho / (self.mu - self.lam)
        return QueueMetrics(
            model_name="M/M/1",
            arrival_rate=self.lam,
            service_rate=self.mu,
            num_servers=1,
            utilization=rho,
            L=L, Lq=Lq, W=W, Wq=Wq,
            throughput=self.lam,
            is_stable=True,
        )

    def sojourn_time_cdf(self, t: float) -> float:
        """
        Kumulierte Verteilung der Systemzeit W: P(W <= t).
        W ~ Exp(mu - lambda) im M/M/1.
        """
        if not self.is_stable:
            return 0.0 if t < float("inf") else 1.0
        if t < 0:
            return 0.0
        rate = self.mu - self.lam
        return 1.0 - math.exp(-rate * t)

    def waiting_time_cdf(self, t: float) -> float:
        """
        Kumulierte Verteilung der Wartezeit Wq: P(Wq <= t).
        P(Wq = 0) = 1 - rho, sonst Exp(mu - lambda).
        """
        if not self.is_stable:
            return 0.0 if t < float("inf") else 1.0
        if t < 0:
            return 0.0
        rho = self.rho
        # P(Wq <= t) = 1 - rho * exp(-(mu-lambda)*t)
        return 1.0 - rho * math.exp(-(self.mu - self.lam) * t)

    def percentile_sojourn(self, p: float) -> float:
        """
        p-Quantil der Systemzeit (0 < p < 1).
        Loest 1 - exp(-(mu-lam)*t) = p => t = -ln(1-p)/(mu-lam).
        """
        if not (0.0 < p < 1.0):
            raise ValueError("p muss in (0, 1) liegen.")
        if not self.is_stable:
            return float("inf")
        rate = self.mu - self.lam
        return -math.log(1.0 - p) / rate

    def state_probability(self, n: int) -> float:
        """
        Stationaere Wahrscheinlichkeit P(N=n) = (1-rho)*rho^n.
        """
        if not self.is_stable:
            return 0.0
        return (1.0 - self.rho) * (self.rho ** n)

    def littles_law_metrics(self) -> dict:
        """
        Gibt explizit Little's Law Metriken zurueck mit Hinweis,
        dass dies eine algebraische Gleichgewichtsaussage ist.
        """
        m = self.metrics()
        note = (
            "Little's Law (L = lambda*W) ist eine algebraische Gleichgewichtsrelation "
            "und KEIN Stabilitaetskriterium. Stabilitaet wird separat geprueft (rho < 1)."
        )
        return {
            "stability_check": {"rho": self.rho, "is_stable": self.is_stable},
            "littles_law": m.littles_law_verification() if self.is_stable else "N/A (instabil)",
            "note": note,
        }

    def sensitivity_analysis(self, delta_lambda: float = 0.1) -> dict:
        """
        Sensitivitaet der Kenngroessen gegenueber Aenderung von lambda.
        """
        if self.lam + delta_lambda >= self.mu:
            new_metrics = None
        else:
            new_q = MM1Queue(self.lam + delta_lambda, self.mu)
            new_metrics = new_q.metrics()

        base = self.metrics()
        result = {
            "base_lambda": self.lam,
            "perturbed_lambda": self.lam + delta_lambda,
            "base_L": base.L if self.is_stable else float("inf"),
            "perturbed_L": new_metrics.L if new_metrics and new_metrics.is_stable else float("inf"),
            "base_W": base.W if self.is_stable else float("inf"),
            "perturbed_W": new_metrics.W if new_metrics and new_metrics.is_stable else float("inf"),
        }
        if self.is_stable and new_metrics and new_metrics.is_stable:
            result["dL_dlambda_approx"] = (new_metrics.L - base.L) / delta_lambda
            result["dW_dlambda_approx"] = (new_metrics.W - base.W) / delta_lambda
        return result


# ---------------------------------------------------------------------------
# M/M/c-Warteschlange
# ---------------------------------------------------------------------------

class MMcQueue:
    """
    M/M/c-System: Poisson-Ankuenfte, exponentielle Bedienung, c Server.

    Stabilitat: lambda / (c * mu) < 1.
    Erlang-C-Formel fuer P(Wq > 0).
    """

    def __init__(self, arrival_rate: float, service_rate: float, num_servers: int):
        if arrival_rate <= 0:
            raise ValueError("arrival_rate muss positiv sein.")
        if service_rate <= 0:
            raise ValueError("service_rate muss positiv sein.")
        if num_servers < 1:
            raise ValueError("num_servers muss >= 1 sein.")
        self.lam = arrival_rate
        self.mu = service_rate
        self.c = num_servers
        self.rho = arrival_rate / (num_servers * service_rate)  # Serverauslastung

    @property
    def is_stable(self) -> bool:
        return self.rho < 1.0

    def erlang_c(self) -> float:
        """
        Berechnet Erlang-C-Wahrscheinlichkeit C(c, rho*c) = P(Wq > 0).
        """
        c = self.c
        a = self.lam / self.mu  # Verkehrsangebot in Erlang
        rho = self.rho

        if not self.is_stable:
            return 1.0

        # P0: Wahrscheinlichkeit fuer leeres System
        # Zaehler des ersten Terms
        sum_term = sum((a ** n) / _factorial(n) for n in range(c))
        last_term = (a ** c) / (_factorial(c) * (1.0 - rho))
        P0 = 1.0 / (sum_term + last_term)

        # Erlang-C
        C = ((a ** c) / (_factorial(c) * (1.0 - rho))) * P0
        return C

    def metrics(self) -> QueueMetrics:
        """Berechnet Kenngroessen des M/M/c-Systems."""
        if not self.is_stable:
            return QueueMetrics(
                model_name=f"M/M/{self.c}",
                arrival_rate=self.lam,
                service_rate=self.mu,
                num_servers=self.c,
                utilization=self.rho,
                L=float("inf"), Lq=float("inf"),
                W=float("inf"), Wq=float("inf"),
                throughput=self.lam,
                is_stable=False,
            )
        C = self.erlang_c()
        rho = self.rho
        Lq = C * rho / (1.0 - rho)
        Wq = Lq / self.lam
        W = Wq + 1.0 / self.mu
        L = self.lam * W
        return QueueMetrics(
            model_name=f"M/M/{self.c}",
            arrival_rate=self.lam,
            service_rate=self.mu,
            num_servers=self.c,
            utilization=rho,
            L=L, Lq=Lq, W=W, Wq=Wq,
            throughput=self.lam,
            is_stable=True,
        )

    def sojourn_time_cdf(self, t: float) -> float:
        """P(W <= t) fuer M/M/c."""
        if not self.is_stable:
            return 0.0 if t < float("inf") else 1.0
        if t < 0:
            return 0.0
        C = self.erlang_c()
        mu = self.mu
        c = self.c
        rho = self.rho
        # P(W<=t) = 1 - exp(-mu*t) * [1 + C*rho/(1-rho) * (1 - exp(-(c*mu*(1-rho)-0)*t)) ]
        # Vereinfachte Formel:
        term1 = (1.0 - C) * (1.0 - math.exp(-mu * t))
        term2 = C * (1.0 - math.exp(-(c * mu * (1.0 - rho) + mu) * t) if
                     c * mu * (1.0 - rho) + mu > 1e-15 else 0.0)
        # Korrekte Formel: P(W <= t) = 1 - exp(-mu*t) - C*rho/(1-rho)*(exp(-mu*t)-exp(-c*mu*(1-rho)*t))/(...)
        # Numerisch robuste Version:
        rate_service = self.mu
        rate_queue = self.c * self.mu * (1.0 - rho)
        if abs(rate_queue - rate_service) < 1e-10:
            cdf = 1.0 - (1.0 + C * rho / (1.0 - rho)) * math.exp(-rate_service * t)
        else:
            cdf = (1.0 - math.exp(-rate_service * t)
                   - C / (1.0 - rho) * (math.exp(-rate_service * t) - math.exp(-rate_queue * t))
                   / (rate_queue / rate_service - 1.0))
        return max(0.0, min(1.0, cdf))


# ---------------------------------------------------------------------------
# M/D/1-Warteschlange (deterministische Bedienzeiten)
# ---------------------------------------------------------------------------

class MD1Queue:
    """
    M/D/1-Warteschlange: Poisson-Ankuenfte, deterministische Bedienzeit 1/mu.

    Pollaczek-Khinchine-Formel fuer deterministische Bedienzeiten:
    - Var[S] = 0  =>  Lq = rho^2 / (2*(1-rho))
    """

    def __init__(self, arrival_rate: float, service_rate: float):
        if arrival_rate <= 0:
            raise ValueError("arrival_rate muss positiv sein.")
        if service_rate <= 0:
            raise ValueError("service_rate muss positiv sein.")
        self.lam = arrival_rate
        self.mu = service_rate
        self.rho = arrival_rate / service_rate

    @property
    def is_stable(self) -> bool:
        return self.rho < 1.0

    def metrics(self) -> QueueMetrics:
        """Berechnet Kenngroessen des M/D/1-Systems."""
        if not self.is_stable:
            return QueueMetrics(
                model_name="M/D/1",
                arrival_rate=self.lam, service_rate=self.mu,
                num_servers=1, utilization=self.rho,
                L=float("inf"), Lq=float("inf"),
                W=float("inf"), Wq=float("inf"),
                throughput=self.lam, is_stable=False,
            )
        rho = self.rho
        # P-K Formel fuer D/1: Lq = rho^2 / (2*(1-rho))
        Lq = rho ** 2 / (2.0 * (1.0 - rho))
        Wq = Lq / self.lam
        W = Wq + 1.0 / self.mu
        L = self.lam * W
        return QueueMetrics(
            model_name="M/D/1",
            arrival_rate=self.lam, service_rate=self.mu,
            num_servers=1, utilization=rho,
            L=L, Lq=Lq, W=W, Wq=Wq,
            throughput=self.lam, is_stable=True,
        )


# ---------------------------------------------------------------------------
# M/G/1-Warteschlange (allgemeine Bedienzeiten)
# ---------------------------------------------------------------------------

class MG1Queue:
    """
    M/G/1-Warteschlange: Poisson-Ankuenfte, allgemeine Bedienzeit G.

    Pollaczek-Khinchine (P-K) Formel:
    Lq = rho^2 + lambda^2 * Var[S] / (2*(1-rho))
    wobei Var[S] die Varianz der Bedienzeit ist.
    """

    def __init__(self, arrival_rate: float, service_rate: float,
                 service_time_variance: float):
        """
        Parameters
        ----------
        arrival_rate : float
        service_rate : float
            1/E[S]
        service_time_variance : float
            Var[S] (Varianz der Bedienzeit).
        """
        if arrival_rate <= 0:
            raise ValueError("arrival_rate muss positiv sein.")
        if service_rate <= 0:
            raise ValueError("service_rate muss positiv sein.")
        if service_time_variance < 0:
            raise ValueError("service_time_variance muss >= 0 sein.")
        self.lam = arrival_rate
        self.mu = service_rate
        self.var_S = service_time_variance
        self.rho = arrival_rate / service_rate
        self.E_S = 1.0 / service_rate
        self.E_S2 = service_time_variance + self.E_S ** 2  # E[S^2]

    @property
    def is_stable(self) -> bool:
        return self.rho < 1.0

    def metrics(self) -> QueueMetrics:
        """Berechnet Kenngroessen via P-K-Formel."""
        if not self.is_stable:
            return QueueMetrics(
                model_name="M/G/1",
                arrival_rate=self.lam, service_rate=self.mu,
                num_servers=1, utilization=self.rho,
                L=float("inf"), Lq=float("inf"),
                W=float("inf"), Wq=float("inf"),
                throughput=self.lam, is_stable=False,
            )
        rho = self.rho
        # P-K: Lq = lambda^2 * E[S^2] / (2*(1-rho))
        Lq = (self.lam ** 2 * self.E_S2) / (2.0 * (1.0 - rho))
        Wq = Lq / self.lam
        W = Wq + self.E_S
        L = self.lam * W
        return QueueMetrics(
            model_name="M/G/1",
            arrival_rate=self.lam, service_rate=self.mu,
            num_servers=1, utilization=rho,
            L=L, Lq=Lq, W=W, Wq=Wq,
            throughput=self.lam, is_stable=True,
        )

    def coefficient_of_variation(self) -> float:
        """Variationskoeffizient CV = sqrt(Var[S]) / E[S]."""
        return math.sqrt(self.var_S) / self.E_S if self.E_S > 0 else 0.0


# ---------------------------------------------------------------------------
# Batch-Analyse mehrerer Dienste
# ---------------------------------------------------------------------------

class ServiceSystemAnalyzer:
    """
    Analysiert ein Software-System mit mehreren Diensten (Microservices,
    Echtzeit-Komponenten), die als Warteschlangen modelliert werden.

    Jeder Dienst ist ein Warteschlangenmodell. Der Analyzer prueft
    Stabilitaet, berechnet Bottlenecks und erzeugt Gesamtberichte.
    """

    def __init__(self, name: str = "ServiceSystem"):
        self.name = name
        self._services: dict[str, object] = {}  # name -> Queue-Objekt
        self._annotations: dict[str, dict] = {}  # name -> UML-Annotationen

    def add_service(self, service_name: str, queue_model,
                    annotation: Optional[dict] = None):
        """
        Fuegt einen Dienst hinzu.

        Parameters
        ----------
        service_name : str
        queue_model : MM1Queue | MMcQueue | MD1Queue | MG1Queue
        annotation : dict, optional
            UML-Timing-Annotationen (deadline, maxDuration etc.).
        """
        self._services[service_name] = queue_model
        self._annotations[service_name] = annotation or {}

    def analyze_all(self) -> dict:
        """Analysiert alle Dienste und gibt vollstaendigen Bericht zurueck."""
        results = {}
        unstable = []
        for name, model in self._services.items():
            m = model.metrics()
            ann = self._annotations.get(name, {})
            svc_result = {
                "metrics": m.to_dict(),
                "littles_law": m.littles_law_verification() if m.is_stable else "N/A",
                "deadline_compliance": self._check_deadline(m, ann),
            }
            if not m.is_stable:
                unstable.append(name)
            results[name] = svc_result
        return {
            "system": self.name,
            "services": results,
            "unstable_services": unstable,
            "all_stable": len(unstable) == 0,
        }

    def _check_deadline(self, metrics: QueueMetrics, annotation: dict) -> dict:
        """Prueft Deadline-Einhaltung anhand der UML-Annotation."""
        deadline = annotation.get("deadline")
        if deadline is None:
            return {"has_deadline": False}
        if not metrics.is_stable:
            return {
                "has_deadline": True,
                "deadline": deadline,
                "compliant": False,
                "reason": "System instabil",
            }
        compliant = metrics.W <= deadline
        return {
            "has_deadline": True,
            "deadline": deadline,
            "mean_sojourn_time": round(metrics.W, 6),
            "compliant": compliant,
            "margin": round(deadline - metrics.W, 6),
        }

    def bottleneck_analysis(self) -> dict:
        """Identifiziert den Dienst mit der hoechsten Auslastung."""
        max_util = -1.0
        bottleneck = None
        utilizations = {}
        for name, model in self._services.items():
            m = model.metrics()
            util = m.utilization
            utilizations[name] = round(util, 4)
            if util > max_util:
                max_util = util
                bottleneck = name
        return {
            "bottleneck_service": bottleneck,
            "bottleneck_utilization": round(max_util, 4),
            "all_utilizations": utilizations,
        }

    def capacity_planning(self, target_utilization: float = 0.7) -> dict:
        """
        Empfiehlt Server-Anzahl, um Ziel-Auslastung zu erreichen.
        Gilt fuer M/M/c-Modelle; bei M/M/1 wird Mindest-mu empfohlen.
        """
        recommendations = {}
        for name, model in self._services.items():
            if isinstance(model, MMcQueue):
                needed_c = math.ceil(model.lam / (target_utilization * model.mu))
                recommendations[name] = {
                    "current_servers": model.c,
                    "recommended_servers": max(model.c, needed_c),
                    "target_utilization": target_utilization,
                }
            elif isinstance(model, MM1Queue):
                needed_mu = model.lam / target_utilization
                recommendations[name] = {
                    "current_service_rate": model.mu,
                    "recommended_service_rate": round(needed_mu, 4),
                    "target_utilization": target_utilization,
                }
        return recommendations

    def percentile_report(self, percentiles: list[float] = None) -> dict:
        """
        Berechnet Quantile der Systemzeit fuer M/M/1-Dienste.
        """
        if percentiles is None:
            percentiles = [0.5, 0.9, 0.95, 0.99]
        report = {}
        for name, model in self._services.items():
            if isinstance(model, MM1Queue) and model.is_stable:
                report[name] = {
                    f"p{int(p*100)}": round(model.percentile_sojourn(p), 6)
                    for p in percentiles
                }
        return report
