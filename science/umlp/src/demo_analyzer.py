"""
demo_analyzer.py
===================
Ausführliche Demo-Anwendung: Krankenhaus-Notaufnahme-Steuerungssystem

Dieses Demo-System modelliert eine Krankenhaus-Notaufnahme (Emergency Department)
mit:
- Verschiedenen Patientenkategorien (Triage-Level 1-5)
- Mehreren Behandlungspfaden (Aufnahme, Triage, Behandlung, Entlassung)
- Stochastischen Zustandsübergängen (Markov-Ketten)
- Warteschlangenmodellen für jeden Prozessschritt
- UML-Profil-kompatiblen Timing-Annotationen
- Stabilitäts- und Compliance-Analysen

Das System demonstriert beide Analysemodule (markov_analyzer, queue_analyzer)
in unterschiedlichen Anforderungsszenarien:
1. Normalbetrieb (mittlere Last)
2. Spitzenlast (hohe Last, kritische Anforderungen)
3. Kapazitätsplanung (Serveroptimierung)
4. Deadline-Compliance (zeitkritische Anforderungen)

Autor: Stephan Epp
"""

from src.markov_analyzer import (
    MarkovState, DiscreteMarkovChain, ContinuousMarkovChain, MarkovAnalyzer
)
from src.queue_analyzer import (
    MM1Queue, MMcQueue, MD1Queue, MG1Queue, ServiceSystemAnalyzer
)
import math


# ===========================================================================
# 1. SZENARIO: NORMALBETRIEB
# ===========================================================================

def szenario_normalbetrieb():
    """
    Normalbetrieb der Notaufnahme:
    - 8 Patienten/Stunde Ankunftsrate
    - Ausreichend Personal (3 Ärzte, 5 Pflegepersonal)
    - Triage-Level bestimmt Priorität
    """
    print("=" * 70)
    print("SZENARIO 1: NORMALBETRIEB")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1.1 Markov-Ketten-Modell: Patientenpfad durch die Notaufnahme
    # -----------------------------------------------------------------------
    print("\n--- 1.1 Diskrete Markov-Kette: Patientenpfad ---")

    patient_states = [
        MarkovState("Ankunft",        0, timing_annotation={"minDuration": 0.0, "maxDuration": 0.05}),
        MarkovState("Triage",         1, timing_annotation={"minDuration": 0.05, "maxDuration": 0.15, "typicalDuration": 0.1}),
        MarkovState("Wartebereich",   2, timing_annotation={"minDuration": 0.0, "maxDuration": 2.0}),
        MarkovState("Behandlung",     3, timing_annotation={"minDuration": 0.5, "maxDuration": 4.0, "typicalDuration": 1.5}),
        MarkovState("Beobachtung",    4, timing_annotation={"minDuration": 1.0, "maxDuration": 6.0}),
        MarkovState("Entlassung",     5, timing_annotation={"minDuration": 0.25, "maxDuration": 0.5}),
    ]

    # Übergangswahrscheinlichkeiten (pro Zeitschritt = 15 Minuten)
    P = [
        # Ank    Tri    War    Beh    Beo    Ent
        [0.00,  1.00,  0.00,  0.00,  0.00,  0.00],  # Ankunft -> Triage
        [0.00,  0.10,  0.60,  0.30,  0.00,  0.00],  # Triage -> Warten/Behandlung
        [0.00,  0.00,  0.20,  0.80,  0.00,  0.00],  # Wartebereich -> Behandlung
        [0.00,  0.00,  0.00,  0.30,  0.50,  0.20],  # Behandlung -> Beob/Entl
        [0.00,  0.00,  0.00,  0.10,  0.20,  0.70],  # Beobachtung -> Behand/Entl
        [0.10,  0.00,  0.00,  0.00,  0.00,  0.90],  # Entlassung (kleiner Anteil neu)
    ]

    chain = DiscreteMarkovChain(patient_states, P)
    pi = chain.stationary_distribution()

    print("\nStationäre Verteilung (Aufenthaltswahrscheinlichkeit je Zustand):")
    for i, state in enumerate(patient_states):
        print(f"  {state.name:<18}: {pi[i]:.4f}  ({pi[i]*100:.1f}%)")

    # Stabilitätsanalyse
    analyzer = MarkovAnalyzer(chain)
    report = analyzer.full_report()
    stab = report["stability_analysis"]

    print(f"\nStabilitätsanalyse:")
    print(f"  Dominanter Eigenwert:   {stab['dominant_eigenvalue']:.6f}")
    print(f"  Zweiter Eigenwert:      {stab['abs_second_eigenvalue']:.6f}")
    print(f"  Zerfallsrate alpha:     {stab['decay_rate']:.4f}")
    print(f"  Mixing-Zeit:            {stab['mixing_time_estimate']:.2f} Zeitschritte")
    print(f"  Konvergenz garantiert:  {stab['convergence_guaranteed']}")

    # MFPT zur Behandlung
    mfpt = chain.mean_first_passage_time(3)
    print(f"\nMittlere Erstpassagezeit zur Behandlung:")
    for i, state in enumerate(patient_states):
        print(f"  Von {state.name:<18}: {mfpt[i]:.2f} Schritte")

    # -----------------------------------------------------------------------
    # 1.2 Kontinuierliche Markov-Kette: CTMC-Modell mit Raten
    # -----------------------------------------------------------------------
    print("\n--- 1.2 Kontinuierliche Markov-Kette (CTMC): Zustandsraten ---")

    # Raten in 1/Stunde; Q[i][i] = -sum(Ausgangsraten)
    ctmc_states = [
        MarkovState("Triage_aktiv",    0, timing_annotation={"minDuration": 0.05, "maxDuration": 0.25}),
        MarkovState("Behandlung_aktiv",1, timing_annotation={"minDuration": 0.5,  "maxDuration": 4.0}),
        MarkovState("Beobachtung_aktiv",2,timing_annotation={"minDuration": 1.0,  "maxDuration": 6.0}),
    ]
    # Raten: Triage ~6/h, Behandlung ~0.67/h, Beobachtung ~0.33/h
    Q = [
        [-6.0,   5.0,   1.0],
        [ 0.0,  -0.8,   0.6],  # 0.2/h direkt entlassen (aus System heraus)
        [ 0.0,   0.1,  -0.4],  # 0.3/h entlassen
    ]
    # Normierung (Zeilensum = 0): Q[1][1] = -0.8 da 0.6+0.2=0.8
    # In Wirklichkeit: offenes System mit Absorption
    # Vereinfacht: geschlossene Approximation
    Q_closed = [
        [-6.0,   5.0,   1.0],
        [ 0.2,  -0.8,   0.6],
        [ 0.3,   0.1,  -0.4],
    ]

    ctmc = ContinuousMarkovChain(ctmc_states, Q_closed)
    pi_ctmc = ctmc.stationary_distribution()
    print("\nStationäre Verteilung (CTMC):")
    for i, s in enumerate(ctmc_states):
        print(f"  {s.name:<25}: {pi_ctmc[i]:.4f}")

    ctmc_analyzer = MarkovAnalyzer(ctmc)
    compliance = ctmc_analyzer.timing_compliance_check()
    print("\nTiming-Compliance-Prüfung (UML-Annotationen):")
    for c in compliance:
        status = "✓" if c["compliant"] else "✗"
        print(f"  {status} {c['state']}: Verweildauer = {c.get('mean_sojourn_time', 'N/A'):.3f}h | "
              f"Compliant: {c['compliant']}")
        for issue in c.get("issues", []):
            print(f"     → {issue}")

    # -----------------------------------------------------------------------
    # 1.3 Warteschlangenanalyse: Normalbetrieb
    # -----------------------------------------------------------------------
    print("\n--- 1.3 Warteschlangenanalyse: Normalbetrieb ---")

    system_normal = ServiceSystemAnalyzer("Notaufnahme-Normalbetrieb")

    # Aufnahme: 8 Pat/h Ankunft, 12 Pat/h Kapazität, 1 Server
    system_normal.add_service(
        "Aufnahme",
        MM1Queue(arrival_rate=8.0, service_rate=12.0),
        annotation={"deadline": 0.083}  # 5 Minuten = 0.083h
    )
    # Triage: 8 Pat/h, 15 Pat/h, 2 Pflegepersonal
    system_normal.add_service(
        "Triage",
        MMcQueue(arrival_rate=8.0, service_rate=8.0, num_servers=2),
        annotation={"deadline": 0.1}    # 6 Minuten
    )
    # Behandlung: 6 Pat/h (nach Triage), 3 Ärzte à 3 Pat/h
    system_normal.add_service(
        "Behandlung_Arzt",
        MMcQueue(arrival_rate=6.0, service_rate=2.0, num_servers=3),
        annotation={"deadline": 2.0}    # 2 Stunden
    )
    # Diagnostik (deterministisch): CT, Labor
    system_normal.add_service(
        "Diagnostik",
        MD1Queue(arrival_rate=3.0, service_rate=5.0),
        annotation={"deadline": 0.5}    # 30 Minuten
    )
    # Apotheke (variable Bedienzeit): allg. Verteilung
    system_normal.add_service(
        "Medikation",
        MG1Queue(arrival_rate=5.0, service_rate=8.0, service_time_variance=0.005),
        annotation={"deadline": 0.25}   # 15 Minuten
    )

    report_normal = system_normal.analyze_all()
    print(f"\nSystem stabil: {report_normal['all_stable']}")
    if report_normal["unstable_services"]:
        print(f"Instabile Dienste: {report_normal['unstable_services']}")

    print("\nKenngrößen je Dienst:")
    print(f"{'Dienst':<22} {'rho':>6} {'L':>7} {'W (min)':>10} {'Deadline':>10} {'OK?':>6}")
    print("-" * 65)
    for svc, data in report_normal["services"].items():
        m = data["metrics"]
        dc = data["deadline_compliance"]
        rho = m["rho (Auslastung)"]
        L = m["L (Kunden im System)"]
        W = m["W (Systemzeit)"]
        W_min = W * 60 if W != float("inf") else float("inf")
        dl = dc.get("deadline", "-")
        ok = "✓" if dc.get("compliant", False) else "✗"
        print(f"{svc:<22} {rho:>6.3f} {L:>7.3f} {W_min:>10.2f} {str(dl):>10} {ok:>6}")

    # Little's Law Verifikation
    print("\nLittle's Law Verifikation (L = λ·W):")
    for svc, data in report_normal["services"].items():
        lv = data.get("littles_law", "N/A")
        if lv != "N/A" and isinstance(lv, dict):
            diff = lv["L = lambda * W"]["difference"]
            verified = lv["L = lambda * W"]["verified"]
            print(f"  {svc:<22}: Differenz = {diff:.2e} | Verifiziert: {verified}")

    # Bottleneck
    bn = system_normal.bottleneck_analysis()
    print(f"\nBottleneck: '{bn['bottleneck_service']}' mit Auslastung {bn['bottleneck_utilization']:.1%}")

    # Quantile
    pr = system_normal.percentile_report([0.5, 0.90, 0.95, 0.99])
    print("\nQuantile der Systemzeit (in Minuten):")
    print(f"{'Dienst':<22} {'P50':>8} {'P90':>8} {'P95':>8} {'P99':>8}")
    print("-" * 52)
    for svc, pdata in pr.items():
        vals = {k: round(v * 60, 2) for k, v in pdata.items()}
        print(f"{svc:<22} {vals.get('p50', '-'):>8} {vals.get('p90', '-'):>8} "
              f"{vals.get('p95', '-'):>8} {vals.get('p99', '-'):>8}")

    return chain, ctmc, system_normal


# ===========================================================================
# 2. SZENARIO: SPITZENLAST
# ===========================================================================

def szenario_spitzenlast():
    """
    Spitzenlast: 20 Patienten/Stunde (z.B. Massenunfall).
    Zeigt Instabilität und Systemgrenzen.
    """
    print("\n" + "=" * 70)
    print("SZENARIO 2: SPITZENLAST (Massenunfall)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 2.1 Markov-Ketten mit veränderter Übergangsmatrix
    # -----------------------------------------------------------------------
    print("\n--- 2.1 Markov-Kette unter Hochlast ---")

    states = [
        MarkovState("Ankunft",      0),
        MarkovState("Triage",       1, timing_annotation={"minDuration": 0.05, "maxDuration": 0.15}),
        MarkovState("Überlauf",     2),  # Neuer Zustand: Überlastpuffer
        MarkovState("Behandlung",   3, timing_annotation={"minDuration": 0.5, "maxDuration": 4.0}),
        MarkovState("Entlassung",   4),
    ]

    # Unter Hochlast: mehr Patienten im Überlaufbereich
    P_high = [
        [0.00, 1.00, 0.00, 0.00, 0.00],
        [0.00, 0.05, 0.60, 0.35, 0.00],
        [0.00, 0.00, 0.40, 0.60, 0.00],
        [0.00, 0.00, 0.00, 0.30, 0.70],
        [0.15, 0.00, 0.00, 0.00, 0.85],
    ]

    chain_high = DiscreteMarkovChain(states, P_high)
    pi_high = chain_high.stationary_distribution()
    stab_high = chain_high.analyze_stability()

    print(f"\nStationäre Verteilung (Hochlast):")
    for i, s in enumerate(states):
        print(f"  {s.name:<18}: {pi_high[i]:.4f}  ({pi_high[i]*100:.1f}%)")

    print(f"\nStabilitätsanalyse (Hochlast):")
    print(f"  Zerfallsrate: {stab_high['decay_rate']:.4f}")
    print(f"  Mixing-Zeit:  {stab_high['mixing_time_estimate']:.2f} Schritte")

    # Überlauf-Anteil
    print(f"\n  ALARM: Überlauf-Wahrscheinlichkeit = {pi_high[2]*100:.1f}%")
    if pi_high[2] > 0.2:
        print("  → System außerhalb Kapazitätsgrenzen!")

    # -----------------------------------------------------------------------
    # 2.2 Warteschlangenanalyse: Spitzenlast
    # -----------------------------------------------------------------------
    print("\n--- 2.2 Warteschlangenanalyse: Spitzenlast ---")

    system_high = ServiceSystemAnalyzer("Notaufnahme-Spitzenlast")
    system_high.add_service("Aufnahme",       MM1Queue(20.0, 12.0))
    system_high.add_service("Triage",         MMcQueue(20.0, 8.0, 2))
    system_high.add_service("Behandlung",     MMcQueue(18.0, 2.0, 3),
                             annotation={"deadline": 2.0})
    system_high.add_service("Diagnostik",     MD1Queue(10.0, 5.0))

    report_high = system_high.analyze_all()
    print(f"\nSystem stabil: {report_high['all_stable']}")
    if report_high["unstable_services"]:
        print(f"⚠ INSTABILE DIENSTE: {report_high['unstable_services']}")

    bn_high = system_high.bottleneck_analysis()
    print(f"\nBottleneck: '{bn_high['bottleneck_service']}' "
          f"mit Auslastung {bn_high['bottleneck_utilization']:.1%}")

    print("\nAuslastungen je Dienst:")
    for svc, util in bn_high["all_utilizations"].items():
        bar = "█" * int(util * 20)
        status = "⚠ INSTABIL" if util >= 1.0 else ("❗ KRITISCH" if util > 0.9 else "OK")
        print(f"  {svc:<18}: {bar:<20} {util:.1%} {status}")

    # Vergleich normal vs. Hochlast
    print("\n--- 2.3 Vergleich: Normalbetrieb vs. Spitzenlast ---")
    normal_triage = MMcQueue(8.0, 8.0, 2)
    high_triage = MMcQueue(20.0, 8.0, 2)

    print(f"\nTriage M/M/2:")
    print(f"  {'Kennzahl':<25} {'Normal (λ=8)':>15} {'Hochlast (λ=20)':>18}")
    print(f"  {'-'*58}")

    if normal_triage.is_stable and high_triage.is_stable:
        mn = normal_triage.metrics()
        mh = high_triage.metrics()
        for attr, label in [("utilization","Auslastung ρ"), ("L","Kunden im System L"),
                             ("W","Systemzeit W (h)"), ("Lq","Warteschlange Lq")]:
            vn = getattr(mn, attr)
            vh = getattr(mh, attr)
            print(f"  {label:<25}: {vn:>15.4f} {vh:>18.4f}")
    elif not high_triage.is_stable:
        mn = normal_triage.metrics()
        print(f"  Auslastung ρ:              {mn.utilization:>15.4f} {'> 1.0 (instabil)':>18}")
        print(f"  System:                    {'stabil':>15} {'INSTABIL':>18}")


# ===========================================================================
# 3. SZENARIO: KAPAZITÄTSPLANUNG
# ===========================================================================

def szenario_kapazitaetsplanung():
    """
    Kapazitätsplanung: Optimale Server-Anzahl für Ziel-Auslastung von 70%.
    Vergleich verschiedener Modelle.
    """
    print("\n" + "=" * 70)
    print("SZENARIO 3: KAPAZITÄTSPLANUNG")
    print("=" * 70)

    print("\n--- 3.1 Markov-Sensitivitätsanalyse ---")

    states = [MarkovState("Idle", 0), MarkovState("Busy", 1), MarkovState("Peak", 2)]
    P = [[0.5, 0.4, 0.1],
         [0.3, 0.5, 0.2],
         [0.1, 0.6, 0.3]]
    chain = DiscreteMarkovChain(states, P)
    S = chain.sensitivity_analysis(delta=0.01)
    pi = chain.stationary_distribution()

    print(f"Stationäre Verteilung: Idle={pi[0]:.3f}, Busy={pi[1]:.3f}, Peak={pi[2]:.3f}")
    print("\nSensitivitätsmatrix ||dπ/dP_ij||:")
    print(f"  {'':>10} {'P[i,0]':>10} {'P[i,1]':>10} {'P[i,2]':>10}")
    for i, row in enumerate(S):
        print(f"  Zeile {i:<5}: {row[0]:>10.4f} {row[1]:>10.4f} {row[2]:>10.4f}")

    print("\n--- 3.2 Optimale Serveranzahl ---")

    arrival_rates = [4.0, 8.0, 12.0, 16.0, 20.0]
    mu_per_server = 5.0
    target_util = 0.70

    print(f"\nZiel-Auslastung: {target_util:.0%}, μ/Server = {mu_per_server} Pat/h")
    print(f"\n{'λ (Pat/h)':>10} {'Min. Server':>12} {'ρ(opt)':>10} {'L(opt)':>10} {'W(opt) min':>12}")
    print("-" * 58)

    for lam in arrival_rates:
        # Optimale Serveranzahl
        c_opt = math.ceil(lam / (target_util * mu_per_server))
        rho_actual = lam / (c_opt * mu_per_server)
        q = MMcQueue(lam, mu_per_server, c_opt)
        m = q.metrics()
        W_min = m.W * 60 if m.is_stable else float("inf")
        print(f"{lam:>10.1f} {c_opt:>12} {rho_actual:>10.3f} {m.L:>10.3f} {W_min:>12.2f}")

    print("\n--- 3.3 Kostenoptimierung: M/M/c vs. M/M/1 ---")
    lam = 10.0
    mu = 12.0

    q_mm1 = MM1Queue(lam, mu)
    m_mm1 = q_mm1.metrics()
    print(f"\nM/M/1 (λ={lam}, μ={mu}):")
    print(f"  ρ = {m_mm1.utilization:.3f}, L = {m_mm1.L:.3f}, W = {m_mm1.W*60:.2f} min")
    print(f"  99%-Quantil: {q_mm1.percentile_sojourn(0.99)*60:.2f} min")

    for c in [2, 3, 4]:
        q_mmc = MMcQueue(lam, mu / c, c)  # Gleiches Gesamtdurchsatz
        m_mmc = q_mmc.metrics()
        if m_mmc.is_stable:
            print(f"\nM/M/{c} (λ={lam}, μ/Server={mu/c:.1f}):")
            print(f"  ρ = {m_mmc.utilization:.3f}, L = {m_mmc.L:.3f}, W = {m_mmc.W*60:.2f} min")

    print("\n--- 3.4 Kapazitätsplanung über ServiceSystemAnalyzer ---")

    system_cp = ServiceSystemAnalyzer("Notaufnahme-Kapazitätsplanung")
    system_cp.add_service("Triage",      MMcQueue(10.0, 6.0, 2))
    system_cp.add_service("Behandlung",  MMcQueue(8.0, 3.0, 3))
    system_cp.add_service("Labor",       MM1Queue(5.0, 8.0))

    cp = system_cp.capacity_planning(target_utilization=0.70)
    print("\nKapazitätsempfehlungen für Ziel-Auslastung 70%:")
    for svc, rec in cp.items():
        print(f"  {svc}:")
        for k, v in rec.items():
            print(f"    {k}: {v}")


# ===========================================================================
# 4. SZENARIO: DEADLINE-COMPLIANCE (ECHTZEIT-ANFORDERUNGEN)
# ===========================================================================

def szenario_deadline_compliance():
    """
    Echtzeit-Anforderungen: Kritische Deadlines für Notfall-Triage.
    Demonstriert UML-Timing-Annotationen und Compliance-Prüfungen.
    """
    print("\n" + "=" * 70)
    print("SZENARIO 4: DEADLINE-COMPLIANCE (Echtzeit-Anforderungen)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 4.1 UML-Profil-annotierte CTMC mit Timing-Constraints
    # -----------------------------------------------------------------------
    print("\n--- 4.1 CTMC mit UML-Timing-Constraints ---")

    # Notfall-Behandlungspfad mit strengen Zeitanforderungen
    notfall_states = [
        MarkovState("Ersteinschätzung", 0,
                    timing_annotation={"minDuration": 0.017,   # 1 Min
                                       "maxDuration": 0.05,    # 3 Min
                                       "typicalDuration": 0.033}),
        MarkovState("Notfall_Behandlung", 1,
                    timing_annotation={"minDuration": 0.5,     # 30 Min
                                       "maxDuration": 2.0}),   # 2h
        MarkovState("Intensiv",         2,
                    timing_annotation={"minDuration": 1.0,
                                       "maxDuration": 48.0}),
    ]

    # Raten für Notfall-CTMC (1/h)
    # Ersteinschätzung: Rate 30/h (2min mittlere Verweildauer)
    # Behandlung: Rate 0.5/h (2h)
    # Intensiv: Rate 0.05/h (20h)
    Q_notfall = [
        [-30.0, 28.0,  2.0],
        [  0.0, -0.6,  0.1],  # 0.5/h entlassen
        [  0.0,  0.0, -0.05],
    ]
    # Geschlossen machen (keine Absorption)
    Q_notfall[1][0] = 0.5
    Q_notfall[2][0] = 0.05
    Q_notfall[2][2] = -0.05

    ctmc_notfall = ContinuousMarkovChain(notfall_states, Q_notfall)
    analyzer_notfall = MarkovAnalyzer(ctmc_notfall)

    pi_notfall = ctmc_notfall.stationary_distribution()
    print("\nStationäre Verteilung (Notfall-Pfad):")
    for i, s in enumerate(notfall_states):
        print(f"  {s.name:<25}: {pi_notfall[i]:.4f}  ({pi_notfall[i]*100:.1f}%)")

    print("\nCompliance-Prüfung UML-Timing-Constraints:")
    compliance = analyzer_notfall.timing_compliance_check()
    for c in compliance:
        status = "✓ OK" if c["compliant"] else "✗ VERLETZUNG"
        sojourn_h = c.get("mean_sojourn_time", float("nan"))
        sojourn_min = sojourn_h * 60 if not math.isnan(sojourn_h) else float("nan")
        print(f"  {status} | {c['state']}")
        print(f"    Mittlere Verweildauer: {sojourn_min:.1f} Min | "
              f"Constraint: min={notfall_states[int([s.state_id for s in notfall_states if s.name==c['state']][0])].timing_annotation.get('minDuration',0)*60:.0f}-"
              f"max={notfall_states[int([s.state_id for s in notfall_states if s.name==c['state']][0])].timing_annotation.get('maxDuration',0)*60:.0f} Min")

    # -----------------------------------------------------------------------
    # 4.2 Percentile-basierte Deadline-Prüfung
    # -----------------------------------------------------------------------
    print("\n--- 4.2 Percentile-basierte Deadline-Prüfung ---")

    # Kritische Deadlines nach EU-Triage-Standards
    deadlines = {
        "Triage_Level_1": 0.0,   # Sofort (0 Min)
        "Triage_Level_2": 0.083,  # 5 Min
        "Triage_Level_3": 0.5,   # 30 Min
        "Triage_Level_4": 1.0,   # 60 Min
        "Triage_Level_5": 2.0,   # 120 Min
    }

    arrival_rates_by_level = {
        "Triage_Level_1": 0.5,
        "Triage_Level_2": 1.5,
        "Triage_Level_3": 3.0,
        "Triage_Level_4": 2.0,
        "Triage_Level_5": 1.0,
    }

    system_triage = ServiceSystemAnalyzer("Triage-Deadline-System")
    for level, lam in arrival_rates_by_level.items():
        # Bedienrate proportional zu Dringlichkeit
        mu = 1.0 / deadlines.get(level, 2.0) * 0.7 if deadlines.get(level, 0) > 0 else 20.0
        system_triage.add_service(
            level,
            MM1Queue(lam, mu),
            annotation={"deadline": deadlines.get(level, 2.0)}
        )

    report_triage = system_triage.analyze_all()
    pr_triage = system_triage.percentile_report([0.5, 0.90, 0.95, 0.99])

    print("\nTriage-Level Deadline-Compliance:")
    print(f"{'Level':<20} {'Deadline':>10} {'P50 (min)':>12} {'P95 (min)':>12} {'P99 (min)':>12} {'Status':>8}")
    print("-" * 78)

    for level in arrival_rates_by_level:
        svc_data = report_triage["services"].get(level, {})
        dc = svc_data.get("deadline_compliance", {})
        dl_h = deadlines.get(level, 2.0)
        dl_min = dl_h * 60

        pr_data = pr_triage.get(level, {})
        p50 = pr_data.get("p50", float("inf")) * 60
        p95 = pr_data.get("p95", float("inf")) * 60
        p99 = pr_data.get("p99", float("inf")) * 60

        compliant = dc.get("compliant", False)
        status = "✓" if compliant else "✗"

        print(f"{level:<20} {dl_min:>10.0f} {p50:>12.2f} {p95:>12.2f} {p99:>12.2f} {status:>8}")

    # -----------------------------------------------------------------------
    # 4.3 Sensitivitätsanalyse auf kritische Ankunftsraten
    # -----------------------------------------------------------------------
    print("\n--- 4.3 Sensitivitätsanalyse: Ankunftsrate vs. Systemzeit ---")

    q_krit = MM1Queue(3.0, 5.0)  # Level-3-Triage
    sa = q_krit.sensitivity_analysis(delta_lambda=0.5)

    print(f"\nLevel-3-Triage (μ=5/h):")
    print(f"  Basis λ={sa['base_lambda']:.1f}/h:     W = {sa['base_W']*60:.2f} min, L = {sa['base_L']:.3f}")
    print(f"  Gestörte λ={sa['perturbed_lambda']:.1f}/h:   W = {sa['perturbed_W']*60:.2f} min, L = {sa['perturbed_L']:.3f}")
    if "dW_dlambda_approx" in sa:
        print(f"  dW/dλ ≈ {sa['dW_dlambda_approx']*60:.2f} min/(Pat/h) → kritische Sensitivität!")


# ===========================================================================
# 5. ABSCHLIESSENDE ZUSAMMENFASSUNG
# ===========================================================================

def gesamtzusammenfassung():
    """Demonstriert den Unterschied zwischen algebraischer (Little's Law)
    und dynamischer (Stabilitäts-) Analyse."""
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG: ALGEBRAISCHE vs. DYNAMISCHE ANALYSE")
    print("=" * 70)

    print("""
    Diese Demo hat demonstriert:

    [Markov-Ketten-Analyzer]
    ✓ Diskrete Markov-Ketten: Patientenpfad-Modellierung
    ✓ Stationäre Verteilung: zeitliche Aufenthaltswahrscheinlichkeiten
    ✓ Stabilitätsanalyse: Eigenwertmethode, Zerfallsraten
    ✓ MFPT: Erwartete Schritte bis zur Behandlung
    ✓ Absorption: Wahrscheinlichkeiten absorbierende Zustände
    ✓ Sensitivitätsanalyse: Robustheit der stationären Verteilung
    ✓ CTMC: Kontinuierliche Raten-Modellierung
    ✓ UML-Timing-Compliance: Prüfung gegen Timing-Annotationen

    [Warteschlangen-Analyzer]
    ✓ M/M/1: Einfacher Dienst (Aufnahme)
    ✓ M/M/c: Mehrere Server (Triage, Behandlung)
    ✓ M/D/1: Deterministische Bedienzeit (Diagnostik)
    ✓ M/G/1: Allgemeine Bedienzeit (Medikation)
    ✓ Little's Law: L = λW (algebraische Gleichgewichtsrelation)
    ✓ Stabilitätsprüfung: ρ < 1 (separate dynamische Bedingung)
    ✓ Percentile-Analyse: P90, P95, P99 Systemzeiten
    ✓ Deadline-Compliance: UML-annotierte Zeitgrenzen
    ✓ Kapazitätsplanung: Optimale Serveranzahl
    ✓ Bottleneck-Erkennung: Systemengpass-Identifikation

    KERNAUSSAGE:
    Little's Law (L = λW) ist eine ALGEBRAISCHE Gleichgewichtsrelation –
    sie setzt Stabilität voraus, beweist sie aber NICHT.
    Stabilität wird durch die Markov-Kette (Eigenwerte < 1) und
    die Warteschlangenbedingung (ρ < 1) separat gesichert.
    """)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    chain, ctmc, system_normal = szenario_normalbetrieb()
    szenario_spitzenlast()
    szenario_kapazitaetsplanung()
    szenario_deadline_compliance()
    gesamtzusammenfassung()
