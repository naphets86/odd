"""
test_queue_analyzer.py
======================
Umfassende Test-Suite fuer queue_analyzer.py mit 100% Code-Coverage.

Testet MM1Queue, MMcQueue, MD1Queue, MG1Queue, QueueMetrics,
ServiceSystemAnalyzer sowie alle Hilfsfunktionen.

Ausfuehren:
    python -m pytest test_queue_analyzer.py -v --cov=queue_analyzer --cov-report=term-missing
"""

import math
import pytest

from src.queue_analyzer import (
    _factorial, _poisson_cdf,
    QueueMetrics, MM1Queue, MMcQueue, MD1Queue, MG1Queue,
    ServiceSystemAnalyzer,
)

# ===========================================================================
# Tests: Hilfsfunktionen
# ===========================================================================

class TestHelperFunctions:

    def test_factorial_zero(self):
        assert _factorial(0) == 1

    def test_factorial_one(self):
        assert _factorial(1) == 1

    def test_factorial_five(self):
        assert _factorial(5) == 120

    def test_factorial_ten(self):
        assert _factorial(10) == 3628800

    def test_factorial_negative(self):
        with pytest.raises(ValueError, match="negativ"):
            _factorial(-1)

    def test_poisson_cdf_k0(self):
        # P(X <= 0) = e^{-lam}
        lam = 2.0
        assert abs(_poisson_cdf(0, lam) - math.exp(-lam)) < 1e-10

    def test_poisson_cdf_large_k(self):
        # P(X <= 100) ~ 1 fuer lam=1
        assert abs(_poisson_cdf(100, 1.0) - 1.0) < 1e-6

    def test_poisson_cdf_monotone(self):
        lam = 3.0
        prev = _poisson_cdf(0, lam)
        for k in range(1, 10):
            curr = _poisson_cdf(k, lam)
            assert curr >= prev
            prev = curr


# ===========================================================================
# Tests: QueueMetrics
# ===========================================================================

class TestQueueMetrics:

    def _make_metrics(self, lam=2.0, mu=4.0, c=1, stable=True):
        rho = lam / (c * mu)
        L = rho / (1 - rho) if stable else float("inf")
        Lq = rho**2 / (1 - rho) if stable else float("inf")
        W = L / lam if stable and lam > 0 else float("inf")
        Wq = Lq / lam if stable and lam > 0 else float("inf")
        return QueueMetrics(
            model_name="M/M/1",
            arrival_rate=lam, service_rate=mu, num_servers=c,
            utilization=rho, L=L, Lq=Lq, W=W, Wq=Wq,
            throughput=lam, is_stable=stable,
        )

    def test_to_dict(self):
        m = self._make_metrics()
        d = m.to_dict()
        assert d["model"] == "M/M/1"
        assert d["stabil"] is True
        assert "L (Kunden im System)" in d

    def test_littles_law_verification_stable(self):
        m = self._make_metrics(lam=2.0, mu=4.0)
        lv = m.littles_law_verification()
        assert lv["L = lambda * W"]["verified"]
        assert lv["Lq = lambda * Wq"]["verified"]

    def test_repr(self):
        m = self._make_metrics()
        r = repr(m)
        assert "M/M/1" in r
        assert "lambda" in r

    def test_littles_law_verification_values(self):
        # rho=0.5: L=1, W=0.5, Lq=0.5, Wq=0.25 (lam=2, mu=4)
        m = self._make_metrics(lam=2.0, mu=4.0)
        lv = m.littles_law_verification()
        # L = lambda * W => 1.0 = 2.0 * 0.5
        assert lv["L = lambda * W"]["difference"] < 1e-4


# ===========================================================================
# Tests: MM1Queue
# ===========================================================================

class TestMM1Queue:

    def test_construction_valid(self):
        q = MM1Queue(2.0, 4.0)
        assert q.lam == 2.0
        assert q.mu == 4.0
        assert q.rho == pytest.approx(0.5)

    def test_construction_invalid_lambda(self):
        with pytest.raises(ValueError, match="arrival_rate"):
            MM1Queue(-1.0, 4.0)

    def test_construction_invalid_mu(self):
        with pytest.raises(ValueError, match="service_rate"):
            MM1Queue(2.0, 0.0)

    def test_is_stable_true(self):
        assert MM1Queue(2.0, 4.0).is_stable

    def test_is_stable_false(self):
        assert not MM1Queue(4.0, 2.0).is_stable

    def test_metrics_stable(self):
        q = MM1Queue(2.0, 4.0)
        m = q.metrics()
        assert m.is_stable
        assert abs(m.utilization - 0.5) < 1e-6
        assert abs(m.L - 1.0) < 1e-6
        assert abs(m.Lq - 0.5) < 1e-6
        assert abs(m.W - 0.5) < 1e-6
        assert abs(m.Wq - 0.25) < 1e-6

    def test_metrics_unstable(self):
        q = MM1Queue(5.0, 3.0)
        m = q.metrics()
        assert not m.is_stable
        assert m.L == float("inf")
        assert m.W == float("inf")

    def test_metrics_high_rho(self):
        q = MM1Queue(3.9, 4.0)
        m = q.metrics()
        assert m.is_stable
        assert m.L > 10  # Sehr langer Stau

    def test_sojourn_time_cdf_stable(self):
        q = MM1Queue(2.0, 4.0)
        cdf = q.sojourn_time_cdf(1.0)
        # mu - lam = 2, P(W <= 1) = 1 - e^{-2}
        expected = 1.0 - math.exp(-2.0)
        assert abs(cdf - expected) < 1e-6

    def test_sojourn_time_cdf_t0(self):
        q = MM1Queue(2.0, 4.0)
        assert q.sojourn_time_cdf(0.0) == 0.0

    def test_sojourn_time_cdf_negative_t(self):
        q = MM1Queue(2.0, 4.0)
        assert q.sojourn_time_cdf(-1.0) == 0.0

    def test_sojourn_time_cdf_unstable(self):
        q = MM1Queue(5.0, 3.0)
        assert q.sojourn_time_cdf(1000.0) == 0.0

    def test_waiting_time_cdf(self):
        q = MM1Queue(2.0, 4.0)
        rho = 0.5
        t = 1.0
        expected = 1.0 - rho * math.exp(-(4.0 - 2.0) * t)
        assert abs(q.waiting_time_cdf(t) - expected) < 1e-6

    def test_waiting_time_cdf_negative(self):
        q = MM1Queue(2.0, 4.0)
        assert q.waiting_time_cdf(-1.0) == 0.0

    def test_waiting_time_cdf_unstable(self):
        q = MM1Queue(5.0, 3.0)
        assert q.waiting_time_cdf(100.0) == 0.0

    def test_percentile_sojourn_median(self):
        q = MM1Queue(2.0, 4.0)
        p50 = q.percentile_sojourn(0.5)
        # 50%-Quantil: t = ln(2)/2
        expected = math.log(2.0) / 2.0
        assert abs(p50 - expected) < 1e-6

    def test_percentile_sojourn_invalid(self):
        q = MM1Queue(2.0, 4.0)
        with pytest.raises(ValueError, match="in \\(0, 1\\)"):
            q.percentile_sojourn(0.0)
        with pytest.raises(ValueError, match="in \\(0, 1\\)"):
            q.percentile_sojourn(1.0)

    def test_percentile_sojourn_unstable(self):
        q = MM1Queue(5.0, 3.0)
        assert q.percentile_sojourn(0.99) == float("inf")

    def test_state_probability(self):
        q = MM1Queue(2.0, 4.0)
        # P(N=0) = 1 - rho = 0.5
        assert abs(q.state_probability(0) - 0.5) < 1e-6
        # P(N=1) = (1-rho)*rho = 0.25
        assert abs(q.state_probability(1) - 0.25) < 1e-6

    def test_state_probability_unstable(self):
        q = MM1Queue(5.0, 3.0)
        assert q.state_probability(0) == 0.0

    def test_littles_law_metrics_stable(self):
        q = MM1Queue(2.0, 4.0)
        result = q.littles_law_metrics()
        assert result["stability_check"]["is_stable"]
        assert "Little's Law" in result["note"]
        assert "littles_law" in result

    def test_littles_law_metrics_unstable(self):
        q = MM1Queue(5.0, 3.0)
        result = q.littles_law_metrics()
        assert not result["stability_check"]["is_stable"]
        assert result["littles_law"] == "N/A (instabil)"

    def test_sensitivity_analysis_stable(self):
        q = MM1Queue(2.0, 4.0)
        sa = q.sensitivity_analysis(delta_lambda=0.1)
        assert sa["base_lambda"] == 2.0
        assert sa["perturbed_lambda"] == pytest.approx(2.1)
        assert "dL_dlambda_approx" in sa
        assert sa["dL_dlambda_approx"] > 0

    def test_sensitivity_analysis_near_instability(self):
        # Stoerung macht System instabil
        q = MM1Queue(3.95, 4.0)
        sa = q.sensitivity_analysis(delta_lambda=0.1)
        # Nach Stoerung: lam + delta = 4.05 > mu = 4.0 => instabil
        assert sa["perturbed_L"] == float("inf")

    def test_sensitivity_analysis_unstable_base(self):
        q = MM1Queue(5.0, 4.0)
        sa = q.sensitivity_analysis(delta_lambda=0.1)
        assert sa["base_L"] == float("inf")


# ===========================================================================
# Tests: MMcQueue
# ===========================================================================

class TestMMcQueue:

    def test_construction_valid(self):
        q = MMcQueue(3.0, 2.0, 2)
        assert q.c == 2
        assert q.rho == pytest.approx(0.75)

    def test_construction_invalid_lambda(self):
        with pytest.raises(ValueError, match="arrival_rate"):
            MMcQueue(0.0, 2.0, 2)

    def test_construction_invalid_mu(self):
        with pytest.raises(ValueError, match="service_rate"):
            MMcQueue(3.0, 0.0, 2)

    def test_construction_invalid_servers(self):
        with pytest.raises(ValueError, match="num_servers"):
            MMcQueue(3.0, 2.0, 0)

    def test_is_stable(self):
        assert MMcQueue(3.0, 2.0, 2).is_stable
        assert not MMcQueue(5.0, 2.0, 2).is_stable

    def test_erlang_c_stable(self):
        q = MMcQueue(2.0, 3.0, 2)
        C = q.erlang_c()
        assert 0.0 <= C <= 1.0

    def test_erlang_c_unstable(self):
        q = MMcQueue(10.0, 2.0, 2)
        assert q.erlang_c() == 1.0

    def test_metrics_stable(self):
        q = MMcQueue(2.0, 3.0, 2)
        m = q.metrics()
        assert m.is_stable
        assert m.L > 0
        assert m.W > 0
        assert m.utilization < 1.0

    def test_metrics_unstable(self):
        q = MMcQueue(10.0, 2.0, 2)
        m = q.metrics()
        assert not m.is_stable
        assert m.L == float("inf")

    def test_mmc_vs_mm1_single_server(self):
        """M/M/1 und M/M/1 (als M/M/c mit c=1) sollen gleich sein."""
        lam, mu = 2.0, 4.0
        q1 = MM1Queue(lam, mu)
        qc = MMcQueue(lam, mu, 1)
        m1 = q1.metrics()
        mc = qc.metrics()
        assert abs(m1.L - mc.L) < 0.01
        assert abs(m1.W - mc.W) < 0.01

    def test_sojourn_time_cdf_stable(self):
        q = MMcQueue(2.0, 3.0, 2)
        cdf = q.sojourn_time_cdf(1.0)
        assert 0.0 <= cdf <= 1.0

    def test_sojourn_time_cdf_t0(self):
        q = MMcQueue(2.0, 3.0, 2)
        # t < 0 sollte 0 ergeben
        assert q.sojourn_time_cdf(-1.0) == 0.0

    def test_sojourn_time_cdf_unstable(self):
        q = MMcQueue(10.0, 2.0, 2)
        assert q.sojourn_time_cdf(100.0) == 0.0

    def test_mmc_littles_law(self):
        q = MMcQueue(2.0, 3.0, 2)
        m = q.metrics()
        # Little's Law: L = lambda * W
        assert abs(m.L - q.lam * m.W) < 0.001


# ===========================================================================
# Tests: MD1Queue
# ===========================================================================

class TestMD1Queue:

    def test_construction_valid(self):
        q = MD1Queue(2.0, 4.0)
        assert q.rho == pytest.approx(0.5)

    def test_construction_invalid_lambda(self):
        with pytest.raises(ValueError, match="arrival_rate"):
            MD1Queue(-1.0, 4.0)

    def test_construction_invalid_mu(self):
        with pytest.raises(ValueError, match="service_rate"):
            MD1Queue(2.0, -1.0)

    def test_is_stable(self):
        assert MD1Queue(2.0, 4.0).is_stable
        assert not MD1Queue(5.0, 4.0).is_stable

    def test_metrics_stable(self):
        q = MD1Queue(2.0, 4.0)
        m = q.metrics()
        rho = 0.5
        expected_Lq = rho**2 / (2 * (1 - rho))
        assert abs(m.Lq - expected_Lq) < 1e-6
        assert m.model_name == "M/D/1"

    def test_metrics_unstable(self):
        q = MD1Queue(5.0, 3.0)
        m = q.metrics()
        assert not m.is_stable
        assert m.Lq == float("inf")

    def test_md1_vs_mm1_lower_lq(self):
        """M/D/1 hat geringere Lq als M/M/1 (deterministische Bedienzeit besser)."""
        lam, mu = 2.0, 4.0
        qd = MD1Queue(lam, mu)
        qm = MM1Queue(lam, mu)
        assert qd.metrics().Lq < qm.metrics().Lq

    def test_littles_law_md1(self):
        q = MD1Queue(2.0, 4.0)
        m = q.metrics()
        lv = m.littles_law_verification()
        assert lv["L = lambda * W"]["verified"]


# ===========================================================================
# Tests: MG1Queue
# ===========================================================================

class TestMG1Queue:

    def test_construction_valid(self):
        q = MG1Queue(2.0, 4.0, 0.1)
        assert q.var_S == 0.1

    def test_construction_invalid_lambda(self):
        with pytest.raises(ValueError, match="arrival_rate"):
            MG1Queue(0.0, 4.0, 0.1)

    def test_construction_invalid_mu(self):
        with pytest.raises(ValueError, match="service_rate"):
            MG1Queue(2.0, -1.0, 0.1)

    def test_construction_negative_variance(self):
        with pytest.raises(ValueError, match="service_time_variance"):
            MG1Queue(2.0, 4.0, -0.1)

    def test_is_stable(self):
        assert MG1Queue(2.0, 4.0, 0.1).is_stable
        assert not MG1Queue(5.0, 4.0, 0.1).is_stable

    def test_metrics_stable(self):
        q = MG1Queue(2.0, 4.0, 0.0)  # Var=0 => wie M/D/1
        m = q.metrics()
        assert m.is_stable
        # Vergleiche mit M/D/1
        qd = MD1Queue(2.0, 4.0)
        md1 = qd.metrics()
        assert abs(m.Lq - md1.Lq) < 1e-6

    def test_metrics_exponential(self):
        """Var[S] = 1/mu^2 entspricht M/M/1."""
        lam, mu = 2.0, 4.0
        var_S = (1.0 / mu) ** 2
        q = MG1Queue(lam, mu, var_S)
        m = q.metrics()
        # Vergleich mit M/M/1
        qm = MM1Queue(lam, mu)
        mm1 = qm.metrics()
        assert abs(m.Lq - mm1.Lq) < 1e-6

    def test_metrics_unstable(self):
        q = MG1Queue(5.0, 4.0, 0.1)
        m = q.metrics()
        assert not m.is_stable
        assert m.L == float("inf")

    def test_coefficient_of_variation(self):
        q = MG1Queue(2.0, 4.0, 0.0625)  # sigma=0.25, E[S]=0.25 => CV=1
        cv = q.coefficient_of_variation()
        assert abs(cv - 1.0) < 1e-6

    def test_cv_zero_variance(self):
        q = MG1Queue(2.0, 4.0, 0.0)
        assert q.coefficient_of_variation() == 0.0

    def test_littles_law_mg1(self):
        q = MG1Queue(2.0, 4.0, 0.01)
        m = q.metrics()
        lv = m.littles_law_verification()
        assert lv["L = lambda * W"]["verified"]


# ===========================================================================
# Tests: ServiceSystemAnalyzer
# ===========================================================================

class TestServiceSystemAnalyzer:

    @pytest.fixture
    def system(self):
        s = ServiceSystemAnalyzer("TestSystem")
        s.add_service("API",       MM1Queue(3.0, 6.0), annotation={"deadline": 1.0})
        s.add_service("Database",  MM1Queue(2.0, 5.0), annotation={"deadline": 0.5})
        s.add_service("Cache",     MMcQueue(4.0, 3.0, 2), annotation={"deadline": 2.0})
        return s

    def test_analyze_all_stable(self, system):
        report = system.analyze_all()
        assert report["all_stable"]
        assert report["system"] == "TestSystem"
        assert len(report["services"]) == 3

    def test_analyze_all_unstable(self):
        s = ServiceSystemAnalyzer("Instabil")
        s.add_service("Overloaded", MM1Queue(9.0, 5.0))
        report = s.analyze_all()
        assert not report["all_stable"]
        assert "Overloaded" in report["unstable_services"]

    def test_deadline_compliance_pass(self, system):
        report = system.analyze_all()
        api = report["services"]["API"]["deadline_compliance"]
        assert api["has_deadline"]
        assert api["compliant"]

    def test_deadline_compliance_fail(self):
        s = ServiceSystemAnalyzer()
        s.add_service("Slow", MM1Queue(3.9, 4.0), annotation={"deadline": 0.01})
        report = s.analyze_all()
        dc = report["services"]["Slow"]["deadline_compliance"]
        assert not dc["compliant"]

    def test_deadline_compliance_unstable_service(self):
        s = ServiceSystemAnalyzer()
        s.add_service("Bad", MM1Queue(5.0, 4.0), annotation={"deadline": 1.0})
        report = s.analyze_all()
        dc = report["services"]["Bad"]["deadline_compliance"]
        assert not dc["compliant"]
        assert "instabil" in dc["reason"]

    def test_no_deadline(self):
        s = ServiceSystemAnalyzer()
        s.add_service("NoDL", MM1Queue(2.0, 4.0))
        report = s.analyze_all()
        dc = report["services"]["NoDL"]["deadline_compliance"]
        assert not dc["has_deadline"]

    def test_bottleneck_analysis(self, system):
        bn = system.bottleneck_analysis()
        assert "bottleneck_service" in bn
        assert bn["bottleneck_service"] is not None
        assert 0.0 < bn["bottleneck_utilization"] <= 1.0

    def test_capacity_planning_mm1(self):
        s = ServiceSystemAnalyzer()
        s.add_service("SVC", MM1Queue(3.0, 4.0))
        cp = s.capacity_planning(target_utilization=0.7)
        assert "SVC" in cp
        assert "recommended_service_rate" in cp["SVC"]

    def test_capacity_planning_mmc(self):
        s = ServiceSystemAnalyzer()
        s.add_service("Multi", MMcQueue(6.0, 2.0, 2))
        cp = s.capacity_planning(target_utilization=0.7)
        assert "Multi" in cp
        assert cp["Multi"]["recommended_servers"] >= 2

    def test_percentile_report(self, system):
        pr = system.percentile_report()
        assert "API" in pr
        assert "p50" in pr["API"]
        assert "p90" in pr["API"]
        assert "p95" in pr["API"]
        assert "p99" in pr["API"]
        # Aufsteigende Reihenfolge
        assert pr["API"]["p50"] <= pr["API"]["p90"]
        assert pr["API"]["p90"] <= pr["API"]["p99"]

    def test_percentile_report_only_mm1(self):
        """Percentile-Report nur fuer M/M/1 (nicht fuer M/M/c)."""
        s = ServiceSystemAnalyzer()
        s.add_service("MM1", MM1Queue(2.0, 4.0))
        s.add_service("MMc", MMcQueue(2.0, 4.0, 2))
        pr = s.percentile_report()
        assert "MM1" in pr
        assert "MMc" not in pr

    def test_percentile_report_custom_percentiles(self):
        s = ServiceSystemAnalyzer()
        s.add_service("SVC", MM1Queue(2.0, 4.0))
        pr = s.percentile_report(percentiles=[0.5, 0.75])
        assert "p50" in pr["SVC"]
        assert "p75" in pr["SVC"]

    def test_analyze_all_with_littles_law(self, system):
        report = system.analyze_all()
        for svc_name, svc_data in report["services"].items():
            if svc_data["metrics"]["stabil"]:
                lv = svc_data["littles_law"]
                assert lv["L = lambda * W"]["verified"]

    def test_add_service_md1(self):
        s = ServiceSystemAnalyzer()
        s.add_service("DeterSvc", MD1Queue(2.0, 4.0), annotation={"deadline": 2.0})
        report = s.analyze_all()
        assert report["all_stable"]

    def test_add_service_mg1(self):
        s = ServiceSystemAnalyzer()
        s.add_service("GenSvc", MG1Queue(2.0, 4.0, 0.01))
        report = s.analyze_all()
        assert report["all_stable"]

    def test_empty_system(self):
        s = ServiceSystemAnalyzer("Empty")
        report = s.analyze_all()
        assert report["all_stable"]
        assert report["services"] == {}

    def test_bottleneck_empty_system(self):
        s = ServiceSystemAnalyzer()
        bn = s.bottleneck_analysis()
        assert bn["bottleneck_service"] is None

    def test_percentile_mm1_unstable_excluded(self):
        s = ServiceSystemAnalyzer()
        s.add_service("Bad", MM1Queue(5.0, 4.0))
        pr = s.percentile_report()
        assert "Bad" not in pr


# ===========================================================================
# Integrationstests
# ===========================================================================

class TestIntegration:

    def test_littles_law_not_stability(self):
        """
        Verifiziert: Little's Law gilt im stabilen System, aber ist kein
        Stabilitaetskriterium. Im instabilen System gilt es nicht.
        """
        # Stabiles System (rho=0.7)
        q_stable = MM1Queue(7.0, 10.0)
        r = q_stable.littles_law_metrics()
        assert r["stability_check"]["is_stable"]
        lv = r["littles_law"]
        assert lv["L = lambda * W"]["verified"]

        # Instabiles System (rho=1.2)
        q_unstable = MM1Queue(12.0, 10.0)
        r2 = q_unstable.littles_law_metrics()
        assert not r2["stability_check"]["is_stable"]
        assert r2["littles_law"] == "N/A (instabil)"

    def test_pk_formula_variance_effect(self):
        """
        Verifiziert P-K Formel: Hoehere Varianz => groessere Warteschlange.
        M/D/1 (Var=0) < M/M/1 (Var=1/mu^2) < M/G/1 (Var > 1/mu^2)
        """
        lam, mu = 3.0, 5.0
        var_exp = (1.0 / mu) ** 2

        q_det = MD1Queue(lam, mu)
        q_exp = MG1Queue(lam, mu, var_exp)
        q_hyp = MG1Queue(lam, mu, 2 * var_exp)  # Hoehere Varianz

        assert q_det.metrics().Lq < q_exp.metrics().Lq
        assert q_exp.metrics().Lq < q_hyp.metrics().Lq

    def test_rho_approaching_one(self):
        """Systemverhalten bei rho -> 1."""
        for rho_val in [0.5, 0.7, 0.9, 0.95, 0.99]:
            mu = 10.0
            lam = rho_val * mu
            q = MM1Queue(lam, mu)
            m = q.metrics()
            assert abs(m.utilization - rho_val) < 1e-6
            # L steigt bei rho -> 1
            expected_L = rho_val / (1.0 - rho_val)
            assert abs(m.L - expected_L) < 1e-4
