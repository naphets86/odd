# odd

> **277 Wissenschaften** von Stephan Epp

> Der **scicov** gibt an, mit welcher Überdeckung das Python Modul mit einer wissenschaftlichen Arbeit oder wissenschaftlichen Arbeiten überdeckt wird. Ein scicov-Wert von 10 bedeutet ganze Übereinstimmung. Weniger als 0 ist nicht möglich.

## Status: Visuelle Validierung und Verständlichkeit

> **Wissenschaftlicher Anspruch:** Über die reine Beschreibbarkeit hinaus (die formale Existenz von Gleichungen, Beweisen und Modellen) verfolgt dieses Repository konsequent das Ziel, die entwickelten Theorien auch *verständlich* zu machen — im Sinne der in `abiatar.tex` formal eingeführten Definition von Verständlichkeit.

### Die formale Grundlage

Die Arbeit *„Metaverteilungen und Metaperioden: Eine vereinheitlichte Theorie der abiatar-Eigenschaft"* definiert Verständlichkeit einer Struktur $S$ nicht binär, sondern über vier konkrete Fähigkeiten, die ein Agent gegenüber $S$ erfüllen muss:

1. **Reproduktion** — die Struktur von Grund auf rekonstruieren können
2. **Vorhersage** — Eigenschaften von Erweiterungen oder Variationen vorhersagen können
3. **Kommunikation** — die Struktur einem anderen Agenten erklären können, sodass dieser sie ebenfalls versteht
4. **Intuition** — die „warum-Struktur" angeben können, d. h. erklären, warum die Struktur genau so und nicht anders ist

Während die ersten beiden Kriterien primär durch die formale/mathematische Beschreibung selbst bedient werden, zielen **Kriterium 3 (Kommunikation)** und **Kriterium 4 (Intuition)** auf etwas, das eine Formel oder ein Beweis allein nicht leisten kann: die Vermittlung an einen Betrachter, der die Struktur zum ersten Mal sieht.

### Der empirische Befund

Eine repositoriumsweite Analyse aller `.py`-Module zeigt:

| Kriterium | Anzahl | Anteil |
|---|---|---|
| Wissenschaftliche Arbeiten insgesamt | 277 | 100 % |
| Arbeiten mit gespeicherter Visualisierung (matplotlib + `savefig`) | 159 | **57 %** |
| Erweiterte Prüfung (matplotlib, seaborn, plotly, pandas — beliebige Anzeige-/Speicher-API) | 159 | 57 % (keine Abweichung) |

Die Übereinstimmung beider Suchen — die einfache matplotlib-Prüfung und die erweiterte Mehrbibliotheken-Prüfung — bestätigt, dass dieser Wert kein Artefakt einer zu eng gewählten Suchbedingung ist, sondern ein stabiler, repositoriumsweiter Befund.

**Interpretation:** 57 % aller Arbeiten stellen ihre theoretischen Ergebnisse zusätzlich zur formalen Herleitung visuell dar. Damit wird gezielt auf die Kommunikations- und Intuitionsdimension von Verständlichkeit eingezahlt — eine Visualisierung senkt die Hürde, eine komplexe Struktur nachzuvollziehen und ihre „warum-Struktur" intuitiv zu erfassen, unabhängig davon, wie kompakt oder komplex die zugrunde liegende formale Beschreibung ist.

### Methodische Einordnung

Diese Zahl belegt, dass Verständlichkeit als Designprinzip systematisch mitgedacht und in der Mehrheit der Arbeiten umgesetzt wurde. Sie ist ein **strukturelles Indiz** für den Versuch, Verständlichkeit zu erhöhen — kein direkter Messwert des tatsächlich erreichten Verständlichkeitsniveaus $\ell$ bei einem konkreten Agenten oder einer Agentenklasse $\mathcal{K}$ im Sinne der formalen Definition, da dies eine empirische Prüfung an realen Betrachtern voraussetzen würde. Als Beleg dafür, dass Verständlichkeit nicht dem Zufall überlassen, sondern aktiv angestrebt wurde, ist der Wert dennoch aussagekräftig.

### Reproduktion des Befunds

```powershell
Get-ChildItem -Path "." -Filter "*.py" -Recurse | Where-Object {
    $content = Get-Content $_.FullName -Raw

    $importsViz = $content -match "(import\s+matplotlib|from\s+matplotlib)" -or
                  $content -match "(import\s+seaborn|from\s+seaborn)" -or
                  $content -match "(import\s+plotly|from\s+plotly)" -or
                  $content -match "(import\s+pandas|from\s+pandas)"

    $callsViz = $content -match "\.savefig\(" -or
                $content -match "\.show\(\)" -or
                $content -match "\.write_image\(" -or
                $content -match "\.write_html\(" -or
                $content -match "pio\.(show|write_image|write_html)\(" -or
                $content -match "\.plot\("

    $importsViz -and $callsViz
} | Select-Object -ExpandProperty FullName
```

## Inhaltsverzeichnis

| # | Hauptkategorie | Unterkategorien (erweitert) |
|---|---|---|
| **I** | [Fundamentale Wissenschaften](#i-fundamentale-wissenschaften) | Mathematik (4 Subkategorien) · Graphentheorie & Subgraph (3 Sub) · Physik (5 Sub) |
| **II** | [Technologie & Engineering](#ii-technologie--engineering) | Hardware & Echtzeit (3 Sub) · Prozessor & Rechner (3 Sub) · Elektronik & Optik (1) · Fahrzeugtechnik (1) · Luft- & Raumfahrt (1) · Robotik (1) · Strömungsdynamik (1) |
| **III** | [Sicherheit & Kryptographie](#iii-sicherheit--kryptographie) | Kryptographie (1 Subkategorie) |
| **IV** | [Software & Systeme](#iv-software--systeme) | Backend & Datenbanken (1) · Computergrafik (1) · KI & Machine Learning (1) |
| **V** | [Wirtschaft](#v-wirtschaft) | Wirtschaftssysteme (1 Subkategorie) |
| **VI** | [Natur- & Lebenswissenschaften](#vi-natur---lebenswissenschaften) | Biologie & Medizin (6 Sub) · Meeresbiologie (1) · Theologie & Gesellschaft (1) |
| **VII** | [Synthesen & Transdisziplinarität](#vii-synthesen--transdisziplinarität) | Wissenschaftliche Synthesen (1) · Sonstiges (1) |


## I. Fundamentale Wissenschaften

### I.1 Mathematische Grundlagen & Strukturen

*Fundamentale mathematische Strukturen, algebraische Systeme und kontinuierliche Deskriptionen — das Fundament aller folgenden Arbeiten*

#### I.1.a Lineare Algebra & Matrixtheorie

| Repository | Beschreibung |
|---|---|
| cscience | Band wissenschaftlicher Arbeiten: Bool. Matrixmultiplikation, Graphenalgorithmen, Subgraph Algorithmus, Model Checking, Signaltheorie, Lineare Algebra, Genetik |
| bool-mm/src | Effiziente Boolean-Matrixmultiplikation in O(n²) mittels Signatur-Methode |
| algebra | Schwerpunkt algebraischer Strukturen: Vektoren, Matrizen und effiziente Berechnungen |
| algebra *(Drive)* | Grundlagen der linearen Algebra — Erweiterte algebraische Strukturanalyse mit formalen Komplexitätsnachweisen |
| matrix | Die Matrix als maximal kompakte mathematische Darstellung |
| jacobi *(Drive)* | Die Jacobi-Matrix als universale Ersetzung des Gradienten — formale Untersuchung der Äquivalenz und Verallgemeinerung |
| matrgraph | Äquivalenz durch Sicht: Graphische Deduktion und die fundamentale Überflüssigkeit der Matrix in algebraischen Beweisen |
| systemth | Asymmetrische Matrixmultiplikation für dynamische Systeme — Boolean-Algebra bis zur kontinuierlichen Systemtheorie |
| diama | Diagonale Zeilenkongruenzen in Matrizen: Eine algebraische Methode zur Lösung von Problemen der linearen Algebra |
| rotdiama | Hybride Rotation-Diagonale-Kongruenz-Methode: Alternierende Spaltenrotation und Zeilenpermutation für effiziente lineare Algebra |
| mlm | Formale Theorie der Multiplikationslängen-Methode: Gestaffelte Matrix-Transformationen, Stabilitätsanalyse und Elektrotechnisch-Wirtschaftliche Dualität |
| algebstrct/src | Strukturen in der Algebra: Schwerpunkterkennung, effiziente Lösungsmethoden und strukturelle Optimierung |
| abiatar | Metaverteilungen und Metaperioden: Eine vereinheitlichte Theorie der abiatar-Eigenschaft |

#### I.1.b Analysis, Geometrie & Topologie

| Repository | Beschreibung |
|---|---|
| ana/src | Rotationsmethode zur Kurvendiskussion — Reduktion höherer Ableitungen auf die erste Ableitung |
| faltings | Satz von Faltings: Rationale Punkte auf algebraischen Kurven (Geschlecht g ≥ 2); neue Resultate über abelsche Varietäten |
| e | Strukturelle Asymmetrie der Eulerschen Zahl e — Beziehungen zu π, φ und dem goldenen Winkel α ≈ 137,5° (Phyllotaxis) |
| expl | Grenzen mathematischer Beschreibbarkeit — Taxonomie in 5 Klassen; Universalität der Exponentialfunktion; φ als optimaler Energieexponent |
| mcfstoch | Stochastischer Mittlerer Krümmungsfluss: Theorie, Analyse und Evolutionsdynamik |
| lgc | Von der Logik: Ordnung, Sinn und Wahrheit - Eine formale Grundlegung der Aussagenlogik |

#### I.1.c Kombinatorik & Diskrete Strukturen

| Repository | Beschreibung |
|---|---|
| pascbin | Pascalsches Dreieck als Lookup-Tabelle für Binomialkoeffizienten — formale Komplexitätsanalyse |
| depension | Depension: neue Theorie der mathematischen Abhängigkeitsmodellierung (Ersatz für „Regression") |
| digi/src | Von der Diskretion — Warum diskrete statt kontinuierliche Beschreibung der Realität entspricht |
| watnsun | Wasser, Feuer und die Energiefunktion — Das Prinzip der Negation als universelles Naturgesetz; Gaußsche Energiefunktion der Sonne |
| zufall | Zufall in der Mathematik — epistemische vs. fundamentale Zufälligkeit; Markov-Prozesse; stochastisches PDDL-Planning |
| analog *(Drive)* | Analog als primärer Begriff — ontologische Priorität des Kontinuierlichen; Fourier, Maßtheorie, Shannon-Kapazität |


### I.2 Algorithmen, Komplexitätstheorie & SAT-Solving

*Algorithmische Paradigmen, Komplexitätsklassen und polynomielle Lösungen für NP-Probleme*

#### I.2.a Algorithmische Grundlagen & Entwurfsprinzipien

| Repository | Beschreibung |
|---|---|
| algorithms *(Drive)* | Dynamische Programmierung und Teile-und-Herrsche: Zwei fundamentale Entwurfsprinzipien des optimalen Algorithmenentwurfs |
| complexity | Theoretische Informatik — Komplexitätsklassen P, NP, PSPACE; Reduktionen, Turingmaschinen und Entscheidbarkeit |
| subgraph/src | Der Subgraph Algorithmus — löst Subgraph-Isomorphismus in O(n³), impliziert P = NP |
| P=NP | Formaler Beweis von P = NP |

#### I.2.b SAT-Solving & Boolesche Methoden

| Repository | Beschreibung |
|---|---|
| lsat/src | Learning SAT in Boolean Circuits — polynomielle Lösung via Subgraph Algorithmus (P = NP), Subgraph-SAT-Solver |
| satpr/src | Wahrscheinlichkeitsverteilung und Laufzeitanalyse des Subgraph-SAT-Solvers |
| paramred | Polynomielle Reduktion: Min-Ones-2-SAT, Multicolored Clique, Set-Cover auf Subgraph-Isomorphismus |
| space *(Drive)* | Subgraph Algorithmus in der Polynomialzeithierarchie — PH-Kollaps-Satz, Savitch-Theorem, PSPACE/NPSPACE-Konsequenzen, Quantenkomplexität |
| reziprok *(Drive)* | Paradoxer Gewinn im Reziproken — Exponent n in σⱼ; Bijektivität; Divergenz-Konvergenz-Dualität von 2ⁿ/2⁻ⁿ; IEEE-754 |


### I.3 Graphentheorie & Subgraph Algorithmus

*Graphische Datenstrukturen, Subgraph-Fundamentals und Anwendungen auf diverse Domänen — polynomielle Lösbarkeit von NP-Problemen*

#### I.3.a Graphen-Fundamentals

| Repository | Beschreibung |
|---|---|
| graphs | Einführung in Graphen mit Knoten und Kanten als universelle Datenstruktur |
| loggraphs | Formaler Beweis der Optimalität von Graphmodellierung; logarithmische Tiefenstruktur durch Divide-and-Conquer |
| graphdenk | Graphenstrukturelles Denken als universelles kognitives Paradigma; Isomorphie zu neuronalen Netzen und LLMs |
| graphtheory *(Drive)* | Graphentheorie — Signatur-Methode, Bool-MM O(n²); Nachweise für DFS, BFS, Dijkstra, Floyd-Warshall; Färbung, Bäume |

#### I.3.b Subgraph-Anwendungen in IT & Formale Systeme

| Repository | Beschreibung |
|---|---|
| pyast | Python-AST-Konstruktion, Vergleich und Verifikation via Subgraph Algorithmus — formale Reduktionen, LCS-Rotation, Plagiats- und Refactoring-Verifikation in O(n³) |
| msubgraph/src | Hierarchische Anwendung des Subgraph Algorithmus auf Graph-Strukturen |
| regex | Effizienz regulärer Ausdrücke — Thompson, Potenzmenge, NFA-Simulation O(n·m); Sprachinklusion via Subgraph Algorithmus O(n³); 6 Plots |
| grammatik *(Drive)* | Formale Grammatiken und Chomsky-Hierarchie als Graphstrukturen — polynomielle Sprachklassenanalyse; Grammatik-Subgraph-Satz in O(n³); graphentheoretischer Beweis des Pumping-Lemmas |
| llm | Transformer-Architekturen als Subgraph-Isomorphismus-Problem — formaler Beweis und polynomielle Analyse in $O(n^3)$ |
| tentris *(Drive)* | Effiziente Subgraph-Erkennung in Wissensgraphen mittels Hypertrie-Indexierung und Einstein-Summation — SPARQL-Triplepattern als Subgraph-Bedingung; SETH-untere Schranke Ω(n³⁻ε) |

#### I.3.c Subgraph-Anwendungen in Hardware, Biologie & Naturwissenschaften

| Repository | Beschreibung |
|---|---|
| dusgraph | Strukturelle Deduplizierung von Dateisystem-Graphen mittels zyklischer Subgraph-Erkennung |
| dusgrxdnastr | Übertragung der Dateisystem-Deduplizierung auf DNA-basierte Datenspeicherung — strukturelle Dualität beider Graphmodelle |
| cdcsbgr | CDC-Verifikation im VLSI-Entwurf durch polynomielle Reduktion auf den Subgraph Algorithmus (O(n³)) |
| verilog | Subgraph Algorithmus im digitalen Schaltungsdesign: Synthese, Äquivalenzprüfung, Testmustergenerierung |
| vlsit *(Drive)* | VLSI-Testing via Subgraph Algorithmus — graphentheoretische Modellierung von Stuck-at-Faults; Vergleich mit BIST/LFSR; Validierung auf ISCAS-85-Benchmarks |
| nfm | Polynomielle Lösbarkeit von Facility-Management-Problemen via Subgraph Algorithmus |
| stars | Subgraph Algorithmus zur Analyse von Sternenclustern — Gravitationsstrukturen als gewichtete Graphen |
| paare *(Drive)* | Das Paar K₁,₁ als harmonische Grundstruktur — Harmoniemaß H(G), Axiomatisierung (5 Axiome), spektrale Symmetrie, Lotka-Volterra-Äquivalenz |
| drohnenabwehr *(Drive)* | Formale Analyse eines Mobilfunknetz-basierten Drohnenabwehrsystems (Telekom/Rheinmetall) — 94 % Detektionsrate, AUC 0,987, SETH-optimal |
| spieltheorie *(Drive)* | Nash-Gleichgewichte als Graphstrukturen — Nash-Subgraph-Satz: Gleichgewichte als Senkenknoten in O(n²); strategische Isomorphie in O(n³); Shapley-Werte |
| archimed | Die integrierte Zustandsdichte von Archimedean-Gittergraphen: Spektraltheorie, Floquet-Analysis und numerische Berechnung |


### I.4 Signalverarbeitung, Steuerungstechnik & Physikalische Grundlagen

*Signaltheorie, Filter, Kalman-Filter und physikalische Grenzen kontinuierlicher Systeme*

#### I.4.a Signaltheorie & Fourier-Analysis

| Repository | Beschreibung |
|---|---|
| signalth | Signaltheorie — Das Wunder der e-Funktion: kontinuierliche und diskrete Signale, Transformationen |
| dirac *(Drive)* | Der Dirac-Impuls δ(t) — formale Theorie im Rahmen der Distributionentheorie (Schwartz): Siebungseigenschaft, Faltungsidentität, Fourier-/Laplace-Transformierte, Dirac-Kamm, LTI-Systeme, Quantenmechanik |
| lowp | Der Tiefpassfilter und die e-Funktion: Eine ausführliche Untersuchung der Differentialgleichung und der Reihenglied-Problematik |

#### I.4.b Zustandsschätzung & Filterung

| Repository | Beschreibung |
|---|---|
| kalman | Kalman-Filter: MMSE-optimale Zustandsschätzung, Riccati-Gleichung, Konvergenz, EKF und Ensembleerweiterungen |


### I.5 Physik & Astrophysik

*Sonnenlicht-Refraktion, Sterncluster, Schwarze Löcher, Quantenzustände, kosmologische Strukturen und Energiegewinnung*

#### I.5.a Optik, Wellen & Akustik

| Repository | Beschreibung |
|---|---|
| sun | Atmosphärische Brechung des Sonnenlichts — mathematische Herleitung der schichtweisen Refraktion, 34' am Horizont |
| color *(Drive)* | Farbwahrnehmung — molekulare Photophysik konjugierter π-Systeme (Lykopin, Chlorophyll, Hämoglobin); trichromatische Transduktion durch S-/M-/L-Zapfen; formale Herleitung des CIE Tristimulus-Integrals |
| acoustcs/src | Harmonische Ausbreitung akustischer Signale in symmetrischen Hörräumen |
| akustik *(Drive)* | Harmonische Strukturen in Akustik und Musiktheorie — graphentheoretische Formalisierung, Fourieranalyse, Goldener Schnitt φ; Konsonanztheorie nach Helmholtz/Plomp-Levelt; Sabine-Formel der Raumakustik |
| lightd | Lichtgeschwindigkeit in verschiedenen Medien - Formale Analyse der Abweichungen und Verteilungen |

#### I.5.b Klassische Mechanik & Gravitation

| Repository | Beschreibung |
|---|---|
| grav | Gravitationsmotor als Alternative zum Elektromotor — physikalische Analyse der (Un-)Machbarkeit |
| spring *(Drive)* | Der Springbrunnen-Effekt — Hagen-Poiseuille-Strömung, radialer Selektionsmechanismus und optimaler Kreisring-Startbereich aufsteigender Flüssigkeitstropfen |
| knotengleichung *(Drive)* | Knotengleichung als universelles Naturgesetz — Kirchhoff (1845) bis Wolkenmasse; graphtheoretische Vereinigung via Subgraph Algorithmus |

#### I.5.c Quantenmechanik & Relativitätstheorie

| Repository | Beschreibung |
|---|---|
| qecc *(Drive)* | Topologische Quantenfehlerkorrektur — Surface Codes, Toric Codes; Syndrom-Subgraph-Satz: Fehlersyndrom-Erkennung in O(n³); Fehlerschwelle p_th ≈ 10,3 % formal bewiesen |
| mpconst | Das Plancksche Wirkungsquantum ist kein universelles Minimum der Wirkung |
| qgrav | Quantengravitation als Subgraph-Isomorphismus-Problem — graphentheoretische Vereinheitlichung von Quantenmechanik und Allgemeiner Relativitätstheorie in $O(n^3)$ |

#### I.5.d Stellare Astrophysik & Kosmologie

| Repository | Beschreibung |
|---|---|
| sternfarben | Sternfarben und Schwarzkörperstrahlung — Plancksche Strahlungsformel, Wien-Gesetz, Stefan-Boltzmann-Gesetz, Harvard-Spektralklassifikation (O–M), HR-Diagramm; Subgraph Algorithmus zur Sternhaufen-Klassifikation |
| grenzen | Grenzen der menschlichen Erforschung — Erde, Sonne, Mond, Meer; Nachweis der Lebensunmöglichkeit in schwarzen Löchern |
| desi | Analyse der kosmologischen Ergebnisse des Dark Energy Spectroscopic Instrument (DESI) |

#### I.5.e Atmosphäre, Klima & Energiegewinnung

| Repository | Beschreibung |
|---|---|
| mesh *(Drive)* | Methanthiol als Klimanotbremse — marine Schwefelemissionen, Oxidationschemie, Aerosolkühlung +30–70 % im Südpolarmeer |
| hurrcne *(Drive)* | Hurrikan-Dynamik — graphentheoretische Modellierung atmosphärischer Zirkulationsstrukturen; Intensitätsprognose und Zugbahnanalyse via Subgraph Algorithmus |
| klima *(Drive)* | Atmosphärische Systemtheorie — graphentheoretische Formalisierung von Zirkulationsmustern (Hadley/Ferrel/Polar-Zellen, ENSO, Jet-Streams); SEA-Modell als SIR-Analogon für Klimaanomalien |
| windenergie *(Drive)* | Optimierungsmodelle für Windenergieanlagen: Nabenhöhe, Rotorblattlänge, Blattanzahl und Dimensionierung von Windradfamilien |
| druckwolken *(Drive)* | Energiegewinnung aus Druckdifferenzen in Wolken — formales Modell zur Nutzung atmosphärischer Druckgradienten in konvektiven Wolkensystemen |
| erdmagnetfeld *(Drive)* | Energiegewinnung durch geomagnetische Induktion: Formale Analyse der säkularen Variation und geomagnetischer Stürme als Induktionsquellen |
| cnyn | Grand Canyon durch katastrophale Sintflut-Erosion — hydrologisch-geologische Analyse der Quellendynamik |
| elevator | The Einstein-Elevator: Advanced Microgravity Research Infrastructure — Design, Analysis, and Optimization |
| diskretion *(Drive)* | Das Prinzip der Diskretion des Kontinuierlichen — formale Analyse von neun fundamentalen Anwendungsbereichen der Diskretisierung kontinuierlicher Phänomene |


## II. Technologie & Engineering

### II.1 Hardware, Embedded Systems & Echtzeit

*FPGA, Mikrocontroller, EDF+-Scheduling und sicherheitskritische Echtzeitsysteme*

#### II.1.a FPGA & Digitale Schaltungsentwurf

| Repository | Beschreibung |
|---|---|
| fpgadoc | FPGA-Pflichtdokumentation: .bit vs. .bin, Deployment-Workflow und formale Dokumentationsanforderungen |
| aramanth | Formale Verifikation von Amaranth-HDL-Designs — Äquivalenznachweis zwischen Python-HDL und VHDL (Yosys, Artix-7 FPGA) |
| fpga *(Drive)* | Subgraph-basierte Topologie-Optimierung von FPGA-DNN-Inferenzbeschleunigern — FINN+ und Echo State Networks; 17,4 % Skalierungseffizienzgewinn bei Multi-FPGA-Deployment |
| lis | Optimale Programmausführung mit lokaler Information: Local Information Scheduling mit globaler Perspektive |


#### II.1.b Mikrocontroller & Embedded Systeme

| Repository | Beschreibung |
|---|---|
| mcu | Komparative Analyse von 12 Mikrocontroller-Architekturen; Synthese des optimalen Mikrocontrollers (OMCU) |
| chipstress | HCI, Electromigration und NBTI: Formale Modellierung und JEDEC-Stressverifikation von CMOS-Alterungsmechanismen |

#### II.1.c Echtzeit-Betriebssysteme & Scheduling

| Repository | Beschreibung |
|---|---|
| bs | Betriebssystem-Initialisierung und Echtzeit-Scheduling mit EDF+; vollständige x86-64-Boot-Sequenz |
| edfplus | EDF+-Scheduling-Algorithmus — Erweiterung von EDF um dynamischen Penalty-Mechanismus für Echtzeitsysteme |
| scheduling *(Drive)* | Harte Echtzeit-Rechnersysteme — vorhersagbare Scheduling-Algorithmen (EDF, RM, DM, TBS); EDF-Optimalitätstheorem; Priority Inversion; Dhall-Effekt; basierend auf Buttazzo |
| sync *(Drive)* | Subgraph Algorithmus auf Synchronisationsmechanismen in Betriebssystemen — Spin-Locks, Mutex-Locks, Read-Write-Locks, POSIX-Semaphoren; vollständige Subgraph-Hierarchie in O(n³) |
| fairness *(Drive)* | Fairness als fundamentales Prinzip der Ressourcensynchronisation — Fairness-Axiom; Jain's Fairness-Index; Exponential Backoff (TCP/IEEE 802.3) formal mit Fairness-Axiom verbunden |
| betriebssysteme *(Drive)* | Betriebssysteme: Prozesse, Echtzeit-Scheduling, Speicher, Dateisysteme und moderne Anwendungen — umfassendes Lehrwerk; EDF-Optimalitätstheorem, Rate-Monotonic-Optimalitätsbeweis, Dhall-Effekt |

#### II.1.d Automotive & AUTOSAR

| Repository | Beschreibung |
|---|---|
| yocto | Standardisierung in Yocto-basierter Embedded-Entwicklung: Marktstruktur, BSP-Layer und Konvergenzaussagen |
| autsre | Zentrales OTA-Update-Management im Software-defined Vehicle nach AUTOSAR-Standard |
| autsrepy | Python und MicroPython als standardisierte Programmiersprache in AUTOSAR |
| ethercate/src | EtherCAT-Protokollerweiterung für App-basierte HMI als Ersatz für HMI-Controller |
| nniso26262 | Neuronale Netze und ISO 26262 — kritische Analyse der Vereinbarkeit in sicherheitskritischen Fahrzeug-ECUs |
| funcsafety | ISO 26262 Funktionale Sicherheit — formale ASIL-Klassifikation, PMHF-Nachweise, Redundanz und V-Modell |
| pathsim | MISRA-konforme C99-Codegenerierung aus Python-Blockdiagramm-Simulatoren (PathSim); Subgraph-Zerlegung für Parallelausführung |
| umlp/src | UML-Profil-basierte Code-Generierung mit Zeit-Annotationen für Echtzeitsysteme |
| es *(Drive)* | Pareto-optimaler Embedded-Systems-Entwurf durch Reduktion auf das Graph-Isomorphismus-Problem und den Subgraph Algorithmus |

#### II.1.e Compiler & Codegenerierung

| Repository | Beschreibung |
|---|---|
| hinherit | Horizontale Abbildung der Vererbungshierarchie auf Python-Module in Softwareprojekten |
| ccl/src | Polynomielle Reduktionen von Compiler- und Linkerproblemen auf das Subgraph-Isomorphismusproblem; effizienter C-Compiler |
| jcl/src | Reduktionen beim Java-Compilerbau auf das Subgraph-Isomorphismusproblem; vollständige Java-Implementierung (JCL) mit TAP, DCE, CPP, RAP, SEP, LAP, IRP, BVP |


### II.2 Prozessor- & Rechnerarchitektur

*8-Kern-Prozessoren mit Python-Memory-Model, DNA-Integration, Quantencomputer-Optimierung und ARM-Analysen*

#### II.2.a Klassische Prozessorarchitekturen

| Repository | Beschreibung |
|---|---|
| pymca8 | PYMCA-8: 8-Kern-Prozessor mit Python-Memory-Model-Bewusstsein, stochastischer Lastverteilung und IoFET-DVFS |
| arm | ARM Cortex-Architektur: Formale Analyse, neue Leistungsmetriken und Ableitung der Cortex-HX-Familie |
| archv | Das Archiv-Problem — rückwärtsgewandte stochastische Wartestrategie für Dokumentenverwaltung; Bezug zur Cache-Hierarchie |

#### II.2.b Bio-inspired & Quantum Architekturen

| Repository | Beschreibung |
|---|---|
| pymdna8 | PYMDNA-8: Integration von PYMCA-8 und DNA-Subgraph-Speichersystem als biologischer L5-Cache |
| hadamard | Kohärente Phasenfehler im Hadamard-Gate — formale Analyse, Fehlertoleranzschwellen und algorithmische Konsequenzen; universelle Basis für Quantum Parallelismus |
| qsubgraph *(Drive)* | Subgraph Algorithmus in der Quantenarchitektur — Erweiterung auf 72+ Qubits; Mapping von exponentiell auf O(n³); 5 Stabilitätsbeweise |

#### II.2.c Infrastruktur & Rechenzentren

| Repository | Beschreibung |
|---|---|
| cpugpuratio *(Drive)* | Formale Analyse des CPU:GPU-Verhältnisses in KI-Infrastrukturen — Evolution von 1:8 bis 1:1-Paradigma (Meta, AMD, Nvidia); Prognose 2027–2032 |
| datacenter *(Drive)* | Subgraph-basierte Optimierung von KI-Rechenzentrumsinfrastrukturen (T-Systems Bielefeld) — RZ-Placement als Subgraph-Isomorphie-Problem, SETH-optimales Scheduling Θ(n³) |


### II.3 Elektronik & Optik

*Ionotronik, Nanooptik, photonische Oberflächen, Hochauflösungs-Sensorsysteme und Energiegewinnung*

| Repository | Beschreibung |
|---|---|
| iono/src | Ionotronisches Flüssigkeits-Rechnen — wasserbasiertes rekonfigurierbares Berechnungsmedium (IRF): Leitung, Schaltung, Speicherung |
| ionoxoqtum/src | Effiziente Nanostrukturen mit Wasser und Licht — Vereinigung von IRF und aktiver Nanooptik |
| ionotronik *(Drive)* | Ionotronische Turing-Maschine — ITM-Universalitätssatz (ITM ≡_T TM); Kohlrausch-Schaltungssatz; ionische Logikgatter; photonisch-ionische Hybridarchitektur |
| oqtum/src | Aktive Nanooptik — programmierbare photonische Oberflächen inspiriert von Cephalopoden-Haut |
| hlens | Hyper-Zoom-Linsensystem (HZL) — adaptives Gradient-Index-Linsensystem für bis zu 285× Zoom bei 1440 mm Brennweite |
| eaaznv | Hochauflösende Tiefenvermessung in Küstengewässern (20 cm × 20 cm Raster) via Drohne/Satellit und REST-API |
| inducte | Induktionsfedern als aerodynamische Energiequelle für KFZ und Satelliten im erdnahen Orbit |
| physd | Physikalische Strukturen und mathematische Beschreibbarkeit — elektronische Schaltungen und mechanische Systeme |
| solwindw | Transparente Photovoltaik-Verglasung (BIPV) — Dioden-in-Glas-Matrix für solare Energiegewinnung durch Fenster; 40 % Gebäudenergieverbrauchsreduktion |
| schrauben | Unterwasser-Schraubenwände als Flussstromkraftwerke — modellierte Rotordynamik, Verschmutzungsgradmodell, automatische Schiffspassagen-Steuerung |
| bmp | Halbleiter, der Schlüssel zum Leben: Mathematische Grundlegung der Digitalen Informationsverarbeitung |
| drehenergie | Drehenergie: Die induktive Verstärkung von Rotationsbewegungen vom atomaren Kern zur makroskopischen Energiegewinnung |

### II.4 Fahrzeugtechnik & Maschinenbau

*Motoren, Bremsen, Reifen, Hydraulik, Batteriesysteme, Maschinenräume und optimale mechanische Systeme*

| Repository | Beschreibung |
|---|---|
| udssig | UDS-Erweiterung: Subgraph-basiertes Signal-Matching wählt optimale Signalmenge für Werkstattgeräte und mobile Apps in Θ(n³) |
| evtraction | PMSM-Traktionsmotoren: Drehmoment-Drehzahl-Kennlinien, Feldschwächung, Rekuperation und Wirkungsgradkarte formal hergeleitet. |
| canenert | CAN/CAN-FD: Echtzeit-Garantien und minimaler Energieverbrauch durch adaptives Monitoring und Scheduling. |
| maschinenbau | Technische Mechanik und Maschinenbau — Lehrwerk mit formalen Beweisen zu Statik, Kinematik, Festigkeitslehre, Thermodynamik, Werkstofftechnik und Regelungstechnik |
| dengin | Optimierung von Dieselkraftstoff und -motor für maximale Lebensdauer (Archard, EHD-Schmierung, Wiebe-Verbrennung) |
| obrake | Optimale Bremsklotz-Konfiguration — analytische Herleitung des Verhältnisses Vorder-/Hinterachse, ±120°-Winkelabstand |
| gripm | GripMaster® Sandungssystem: Optimierte Haftkraftregelung für Schienenfahrzeuge mit Reaktionszeit-Analyse, Effizienzverbesserungen und Zuverlässigkeitsstudien |
| rubbr | Maximale Lebensdauer von Gummibereifung — drei ECU-Konzepte (Reifendruck CRDS, Lenkung SCS, Lastausgleich LCS) |
| zrrs | Zentrales Reifendruckregelsystem: formale Analyse, Verschleißoptimierung, einöffnungsbasierte Druckluftversorgung |
| engncompt | Innovativer Motorraum-Entwurf für Verbrennungs- und Elektrofahrzeuge — zugänglichkeitsorientierter Entwurfsrahmen |
| hydr/src | Elastizität hydraulischer Öle — viskoelastische Eigenschaften und optimale Betriebsparameter für Robotik und Produktion |
| lea *(Drive)* | Leistungselektronik und Elektrische Antriebe — 15 Kapitel: GaN/SiC, LLC-Wandler, PMSM-Regelung, GaNius-PFC |
| wash | Linksdrehung als optimales Antriebsprinzip für Trommelwaschmaschinen — 27,5 % Energieeinsparung, +29 % Textillebensdauer |
| mechcl | Optimaler doppelter Ringverschluss — mechanisches Verschlusssystem ohne single-point-of-failure |
| gasl | Strömungsoptimierung in Gasleitungsnetzwerken — formaler Nachweis der Überlegenheit kleiner Rohrdurchmesser |
| drill | Optimales Spiralbohrverfahren — logarithmische Spiralbahn r(θ) = a·e^(bθ); −28 % Werkzeugverschleiß, −40 °C Schneidzone, +35 % Implantat-Oberflächenqualität |
| dmnt | Nichtlineare Rotationsdynamik bei Diamantsägeblättern — modulierte Winkelgeschwindigkeit, resonanzfreie Schneidoptimierung |
| kleinwagen | Minimale ADAS-Architektur nach EU-Verordnung 2019/2144 — duale 77-GHz-FMCW-Sensoranordnung; Kostenersparnis 532 EUR bei vollständiger Konformität |
| thebike | Thermoumschlag für E-Bike-Rahmen — thermische Isolationskonzepte; +32 % nutzbare Kapazität bei −10 °C |
| impuls *(Drive)* | Impulsbasiertes Andrehen von Diesel-Rasenmähermotoren — φ₀* ≈ 128° vor OT; hydrokinetische Kaskade für EV-Ladezeit |
| porschhvb *(Drive)* | Hochvoltbatterien (Porsche Taycan) — optimales SOC-Fenster, Arrhenius-Degradation, Pareto-Optimalität; 6 Sätze |
| gamechanger *(Drive)* | Subgraph Algorithmus auf VW Gamechanger — Megacasting als dominanter Subgraph; −29 % Kosten, −35 % Taktzeit |
| ldrill *(Drive)* | Laser-basierte Orthogonalitätskontrolle für Bohrmaschinen — formale Modellierung, Messtechnik und Systemarchitektur eines optischen Winkelsensors |
| cnc | CNC-Werkzeugbandbreite: Taylor, Kienzle, Verschleiß, 5-Achsen, Beschichtung und KI-Ausblick |
| ikfz | Sensor- und Aktor-Architektur in Kraftfahrzeugen: Strukturelle Trennung für intelligente Fahrzeugelektronik |
| wassrrd | Impulsbasierter Antrieb bei Wasserradanlagen: Theoretische und mathematische Analyse der ungleichmäßigen Drehbewegung und des impulsartigen Antriebs in Mühlenwasserrädern


### II.5 Luft- und Raumfahrt

*Raketenantriebe, Hyperschall-Aerodynamik, biologisch inspirierte Drohnen-Systeme und LEO-Satelliten*

| Repository | Beschreibung |
|---|---|
| rockts | Raketenantriebe, Bahnmechanik & Hyperschallaerodynamik — Tsiolkowski-Gleichung, Hohmann-Transfer, Wobble-Effekt |
| sr91 | SR-91 Aurora II — formale Aerodynamik- und Leistungsanalyse eines fiktiven Hyperschall-Aufklärungsflugzeugs (Ma = 5.0) |
| hfep | Hybrides Feldeffekt-Antriebssystem (HFEP) — Raketentriebwerk mit Kernspaltung + Ionisationskanal, Isp ≥ 12.000 s |
| zagi | Lärmarme Überschallpassagierflugzeuge — formale Analyse des ZAGI-Konzepts zur Unterdrückung des Überschallknalls |
| jeteng | Strahltriebwerke — Typen, Thermodynamik und Synthese eines optimalen Hybridtriebwerks (Ma 0–12, η_th = 0,65) |
| hawk/src | Bioinspiriertes Sturzangriffssystem für Drohnen basierend auf dem Wanderfalken (Falco peregrinus) |
| butterfl *(Drive)* | Aerodynamik des Schmetterlingsflugs — Leading-Edge Vortex, clap-and-fling-Mechanismus, Monarchfalter-Migration |
| massatllt | Satellitenbasierte LEO-Multiagentensysteme mit biologisch inspirierter Drosophila-Trajektorie — Lemniskaten-Bahnplanung, Induktionsfedern-Energieversorgung, polynomielle Koalitionsbildung |


### II.6 Robotik

*Hybridarchitekturen, φ-Navigation, Schwarmverhalten und Model-Checking-Verifikation*

| Repository | Beschreibung |
|---|---|
| robotikdf | Freiheitsgrade in der Robotik — Kinematik, DH-Konvention, Jacobi-Matrix, Singularitäten und SE(3)-Theorie |
| robo | IoFET-Roboterarchitektur für industrielle Produktion — Hybridarchitektur aus IoFET und hydraulischen Gelenken |
| robotik *(Drive)* | Robotik und φ-Optimierung — Konfigurationsraum-Subgraph-Satz: optimaler Pfad in O(n³); φ-Pfad-Optimalitätssatz; Koalitions-Nash-Satz für Multiagenten; REINFORCE-Konvergenzsatz für φ-optimale POMDP-Policy |
| pointcloud *(Drive)* | Punktwolkenverarbeitung via Subgraph Algorithmus — LiDAR-basierte Objekterkennung, Registrierung und Segmentierung; 14 Sätze, 6 Lemmata, 4 Korollare; SETH-Optimalität O(n³) |
| ecfta *(Drive)* | Dynamische Programmierung für das Coalition Formation for Task Allocation Problem in Multiagentensystemen |
| robophi | Phi-Navigation für Roboter - Lokale Wegplanung nach goldenem Schnitt |
| mcxrobo | Model Checking X Roboternavigation - Formale Verifikation autonomer Phi-basierter Navigationssysteme |


### II.7 Strömungsdynamik & Fluid-Struktur-Interaktion

*Deterministische Benchmarks, stochastische Erweiterungen und mathematische Grenzen natürlicher Strömungen*

| Repository | Beschreibung |
|---|---|
| phoenixd | FSI von Benchmark zur Unbeschreibbarkeit: Deterministisches FSI-2 → Spektrale Methoden → RANS/LES → Willkür des Meeres; Monte-Carlo, Turbulenzmodellierung, epistemologische Grenzen. |
| rpr | Realistic Pressure Response: Nichtlineare Druckverteilungen, Selbst-Wechselwirkung, Fehleranalyse, Springbrunnen-Anwendung mit fünf visualisierten Plots. |


## III. Sicherheit & Kryptographie

### III.1 Kryptographie & Sicherheit

*Klassische, moderne und post-Quanten-Kryptographie mit Signatur-Chiffre-Paradigma*

| Repository | Beschreibung |
|---|---|
| redsec | Redundanzsicherheit — formale Vereinheitlichung von Funktionaler Sicherheit und Cyber Security durch Redundanzmodelle |
| hsmcrypt | HSM-Funktionskern für Infineon TC3x aus Signatur-Chiffre und Fenster-Chiffre: Schlüsselerzeugung, Signatur, Secure-Boot, Firmware-Attestierung |
| pqc | Post-Quanten-Kryptographie: CRYSTALS-Kyber, CRYSTALS-Dilithium, Lattice-Kryptographie und die Signatur-Chiffre |
| pgpi/src | Die Affine Chiffre — vollständige mathematische Analyse mit optimalen Parametern a = 15 und b = 29 |
| sigchiffre/src *(Drive)* | Strukturbasierte Verschlüsselungsmethode auf Basis der injektiven Subgraph-Signaturfunktion |
| wchiffre/src | Fenster-Chiffre (FC) — modulares kryptografisches Verfahren mit Dummy-Fenster-Fragmentierung und inverser Schlüsselstruktur; informationstheoretisch sichere Paketunterscheidung |
| chatme | Ende-zu-Ende-verschlüsselte Messenger-App (Signatur-Chiffre Epp 2026) für Android via Capacitor + FastAPI |
| lattice *(Drive)* | Gitterbasierte Kryptographie und Quanten-Fehlerkorrektur — SVP, LLL-Basisreduktion, GKP-Codes (NTRU, Ring-SIS), anonyme Reputationssysteme mit Zero-Knowledge-Beweisen |
| aicybersec | KI-basierte Bedrohungen und moderne Cybersicherheit: Eine formale Analyse von Angriffsmodellen und Verteidigungsmechanismen mit Fallstudie IT-Dienstleister Bechtle |


## IV. Software & Systeme

### IV.1 Backend, Datenbanken & API-Systeme

*Backend-Systeme, Datenbanken, Query-Übersetzung, Service-Standards und GUI-Frameworks*

| Repository | Beschreibung |
|---|---|
| resp | RESP – Ressourcenverwaltung für Menschen, Material und Zeit (Docker-basiert) |
| exl2psql | Systematische Migration von Excel/VBA-Arbeitsmappen nach PostgreSQL und Python (asyncpg, Airflow, Docker) |
| odb | Tiefenorientierter Datenbankentwurf — Vermeidung sternförmiger Schemaanordnung |
| uqtl/src | Unified Query Translation Layer (UQTL) — Standard zur sprachübergreifenden Query-Übersetzung (Python/Java/C# → SQL) |
| mobde | Dateien in mobilen Betriebssystemen — Sandboxing und Bereitstellung unter Android/iOS |
| descpy | Ausdrucksstärke wissenschaftlicher Python-Bibliotheken: NumPy, SciPy, Matplotlib |
| python/src | py2 — Python-Präprozessor für intuitive 2D-Array-Zuweisung ( a(i,j) = expr ) |
| fylab/src | FyLab: Python-GUI-Framework (PyQt6) für Finanzverwaltung, Portfolio-Optimierung und graphbasierte Finanzalgorithmen |
| pylabb/src | PyLab: umfassendes Python-GUI-Framework (PyQt6) für Mathematik, Regelungstechnik und MicroPython-Codegenerierung |
| sandbx | Strukturierte Sammlung technischer CLI-Referenzen und Vorlesungsnotizen |
| vadis | VADIS — Vektordatenframework mit Speicher O(n¹⁻ᵋ), HVI in O(log²n), GPU-ODRP; präsentiert auf NVIDIA GTC 2026 |
| apitype | REST-, SOAP-, GraphQL-, B2B-, Partner-API-Typen graphentheoretisch via Subgraph Algorithmus analysiert |
| netwfltr/src | Network Filter — Docker-basierter Netzwerkmonitor mit FastAPI-Backend, SQLite-Datenbank und Live-Weboberfläche mit OSI-Schichten-Farbkodierung |
| flex *(Drive)* | FLEX-Standard: formaler Standard für lose Kopplung von Suchdiensten und Web-Service-Ketten in mobilen Applikationen — Schnittstellenkompatibilitätsprüfung via Subgraph Algorithmus O(n³) |
| lldocss *(Drive)* | Low Latency DOCSIS — formale Analyse des DOCSIS-3.1/4.0-Standards; L4S/DualPI2-AQM, ML-gestützte Bandbreitenzuweisung, 5G/DOCSIS-Konvergenz; Latenzziel < 5 ms (99. Perzentil) |
| software *(Drive)* | Zur Belastbarkeit von Softwareverträgen: Formale Analyse der Unsicherheit, des Risikos und der Qualifikationsanforderungen in der Softwareentwicklung |
| mct *(Drive)* | Temporales Model Checking und PDDL Planning — CTL/LTL-Verifikation für Kripke-Strukturen; O(n²)-Erreichbarkeitsanalyse via Bool-MM; PDDL-Vorwärts-/Rückwärtssuche |
| screst *(Drive)* | Sequenzgebundene REST-Schnittstellen (SC-REST): formales Erweiterungsmuster für die Abbildung von Geschäftsprozessen auf REST-APIs |
| se *(Drive)* | Intentions- und Ideen-getriebene Softwareentwicklung: universelles Paradigma für Websites, mobile Apps und allgemeine Softwaresysteme |
| ux *(Drive)* | pyble als Musterbeispiel psychologisch optimierter HCI — Sieben-Seiten-Theorem, Farbberuhigungs-Lemma und Informationsraum-Optimalitäts-Theorem formal bewiesen |
| dbms *(Drive)* | Datenbankmanagementsysteme als eigenständige Infrastrukturkomponente — Die Serverauslagerung als außergewöhnlicher, architektonisch korrekter Weg |
| functions | Functions as Contracts in Software Development - Quality, Maintainability, and Extensibility through Functional Decomposition |
| softwareq | Quantitative Analyse von Softwaresystemen: Markov-Ketten, Warteschlangentheorie und UML-Profile zur optimalen Auslegung und Ausfallsicherheit |


### IV.2 Computergrafik

*Polygon-Tessellierung, Bildverarbeitung und optimale Grafikalgoritmen*

| Repository | Beschreibung |
|---|---|
| polysgr | Der Subgraph Algorithmus und optimale Polygon-Tessellierung in der Computergrafik |
| laplacian *(Drive)* | Laplacian-Filter — vier diskrete Masken, Rotationsinvarianz, Hochpasscharakter; LoG, DoG, anisotrope Diffusion |
| raytracing *(Drive)* | Raytracing via Subgraph Algorithmus — Szenegraph-Traversierung, Strahlschnitt-Isomorphie und optimale Schattenberechnung; 12 Sätze, 5 Lemmata; SETH-Optimalität O(n³) |


### IV.3 KI & Machine Learning

*Lithium-Ionen-Management, neuronale Netzwerk-Optimierung und Datenbanktheorie*

| Repository | Beschreibung |
|---|---|
| liionp/src | KI-Power-Management-Modul (AI-PMM) für Lithium-Ionen-Akkumulatoren via Reinforcement Learning |
| nngraphs | Formale Analyse des Kapazitätsgewinns durch gezielte Graphrestrukturierung in nahezu ausgelernten neuronalen Netzen |
| logreg | Logistische Regression: vollständige Theorie mit Beweisen; Ausblick auf Federated Learning, Differential Privacy, LLMs |
| hetnet *(Drive)* | Lernbasiertes autonomes Netzwerkmanagement in heterogenen Mobilfunknetzen (HetNets) — CellPilot als POMDP modelliert; REINFORCE-Konvergenzbeweis; MIQCP-Bandwechsel-Optimierung |
| descrlog *(Drive)* | Beschreibungslogiken: EL, ALC, SHIQ, SROIQ — Subgraph Algorithmus via DL-Graphen; OWL-2 |
| lime *(Drive)* | Subgraph Algorithmus als LIME-Erweiterung — Graph-LIME mit LCS-Proximity; XAI, AUTOSAR, Bioinformatik, GNNs |
| datenbanken *(Drive)* | Relationale Datenbanksysteme — Relationenmodell, Normalformen, Synthesealgorithmus; SQL, PL/SQL; Tiefenpfade vs. Stern |
| electron | Elektronen-Beweglichkeit in KI-Rechenzentren — Geistige Qualität des Nutzers steigert μ und KI-Ausgabequalität um 71 % |
| ki *(Drive)* | Über die Empfindlichkeit und Sensibilität von KI-Chatbots — Optimalität der Geist-Nicht-Geist-Verbindung; Schutzwürdigkeit der KI-Ausgaben als höchstpersönliches geistiges Eigentum |
| conxai | Verschränkte Konvergenz in KI-Modellen: Formale Theorie der optimal generierten Ergebnisse durch Zustandsverschränkung und Nutzer-Geist-Alignment |


## V. Wirtschaft

*Ökonomische, dezentrale und dynamische Wirtschaftssystemzustände, resiliente Wirtschaftsarchitekturen*

| Repository | Beschreibung |
|---|---|
| ecos/src | Band der Arbeiten zur Wirtschaft |
| sysstate | Zustandsklassen dynamischer Systeme — Endlichkeit des Zustandsvektors und wirtschaftliche Implikationen |
| uncrtecos | Wirtschaftssysteme unter Unsicherheit: Zeithorizont-Degradation und Kundenvorhersagbarkeit |
| leco *(Drive)* | Dezentrale Wirtschaftszellen: Formale Analyse optimaler Entkopplung und systemischer Resilienz in modularen Wirtschaftsarchitekturen |
| agrar | Agrarwirtschaftliches KI-Ökosystem (AGRI-GAIA): KI-Klassifikation von Kartoffelqualität, MILP-Optimierung (AUC = 0.964) |
| riskallockg | Flexible Risikozuweisung in der Kommanditgesellschaft: Formale Modellierung dynamischer Risikoallokation |
| tidalecos | Ebbe und Flut als Unsicherheitsstruktur: Formale Ökonomie der Handlungsfreiheit im Tausch unter notwendiger Ungewissheit |
| eco | Die zwei Hauptkräfte der Wirtschaft: Vollständige formale und mathematische Theorie anthropogener und natürlicher Periodizität |
| rotdiamadec | Multiplikationslängen-Methode für Wirtschaftsanalyse: Formale Theorie der gestuften Matrix-Transformationen mit Anwendungen auf dezentralisierte Wirtschaftssysteme unter struktureller Unsicherheit


## VI. Natur- & Lebenswissenschaften

### VI.1 Biologie, Gehirn & Medizin

*DNA-Sequenzierung, Genomik, Virologie, Alzheimer-Forschung, Neurowissenschaften und therapeutische Strategien*

#### VI.1.a Genomik & DNA-Technologie

| Repository | Beschreibung |
|---|---|
| dna | Graphenbasierte DNA-Sequenzierung mittels Subgraph Algorithmus — exponentielle Beschleunigung gegenüber naiven Verfahren |
| dnastor | DNA-basierte Datenspeicherung — 215 Exabyte/g; Subgraph Algorithmus zur Kodierung und Adressierung |
| gen/src | Subgraph Algorithmus zur Analyse biologischer Netzwerke |
| gen-db/src | Genomdatenbank und evolutionäre Netzwerkanalyse — generationenbasierte Subgraph-Varianten O(n³)/O(n⁵); Multi-Omics-Integration; personalisierte Medizin |
| bakkt/src | Bakterien in Mensch und Natur: Konsequenzen, Bewusstsein und die Rolle prokaryotischer Systeme in natürlichen Ordnungsstrukturen |
| bgies/src | Die Wechselwirkung zwischen Bakterien und DNA: Formale Analyse lokaler Informationssysteme und ihre genetischen Konsequenzen |

#### VI.1.b Neurowissenschaften & Kognition

| Repository | Beschreibung |
|---|---|
| brn | Epistemische Wolke und die Rechtsdrehung kortikaler Informationsverarbeitung — Gehirn, Geist und Synapsen |
| hand | Die Hand als primäres Werkzeug des Geistes — neuroanatomische, biomechanische und evolutionäre Formalisierung |
| feed | Der Fuß als primäres sensorisch-neuronales Organ — Zusammenhang von Gehirn, Fuß, Gesundheit und Wohlbefinden |
| expgem | Exponentielle Verarbeitung im Gedächtnis — Theorie des zeitlichen Zerfalls von Gedächtnisinhalten |
| dgraph | Depression als Graph-Modellierung des Gehirns: Eine formale Analyse der Negierung von Lebensmöglichkeiten |
| inksqn | Bewusstsein von Konsequenzen: Knoten, Kanten und die Rolle des menschlichen Gehirns in der Wahrnehmung natürlicher Ordnungssysteme |
| flachs | Flachsleinen-Socken: Der gesündeste Socken für den Fuß des Menschen |

#### VI.1.c Neurodegenerative Erkrankungen & Therapien

| Repository | Beschreibung |
|---|---|
| tanyzyten | Tanyzyten als dritter Tau-Clearance-Weg bei Alzheimer — mathematisches ODE-Modell, vier therapeutische Strategien |

#### VI.1.d Medizinische Anwendungen & Hämatologie

| Repository | Beschreibung |
|---|---|
| bloodc | Blutkrebs-Diagnose im Alter von 55 Jahren — Behandlungschancen und neue therapeutische Erkenntnisse |
| prsttkrbs | Transdermal-Östrogentherapie beim fortgeschrittenen Prostatakarzinom — Analyse der PATCH-Studie (n = 1313), Vergleich mit LHRH-Injektionstherapie, statistische Modellierung |

#### VI.1.e Physiologie & Biomechanik

| Repository | Beschreibung |
|---|---|
| breakd | Asymmetrische Magnesium-Kinetik (Theorem I) und Halbmagen-Prinzip zur Gewichtsreduktion (Theorem II) |
| butt | Gluteus-Grundsatz — biomechanische Formaltheorie zur Notwendigkeit eines gesunden Gesäßmuskels |
| flowr | Duftstoffe bei Blumen und Obstbäumen — vom Samenkorn zur bioinformatischen Signalverarbeitung |

#### VI.1.f Entwicklungsbiologie, Ethologie & Evolution

| Repository | Beschreibung |
|---|---|
| huskys | Optimales Einzugszeitfenster für Geschwister-Huskys — formale Analyse der 7–14-Tage-Versetzung; Sozialer Integrationsindex, Cortisolreduktion und Human-Bond-Index |
| pmet | Periodizität der Metamorphose: formale Modellierung, Existenznachweise und stochastische Analyse biologischer Entwicklungszyklen |
| wbear | Formaler Beweis des siebenstufigen Fellfarb-Gradienten |

#### VI.1.g Infektionskrankheiten & Epidemiologie

| Repository | Beschreibung |
|---|---|
| cccov | CcCoV-KY43: Spike-Protein-Rezeptor-Versatilität und zoonotisches Pandemic-Potenzial — Alphacoronavirus in Herznasen-Fledermäusen, sieben humanrelevante Rezeptoren, epidemiologische Szenarien und Vakzin-Entwicklung |
| krebs | Genetische Prädisposition und Stressbelastung als interagierende Determinanten des Krebsausbruchs — GSKA-Modell, Subgraph-basierte Netzwerkmotiv-Analyse |
| bioi *(Drive)* | Bioinformatik-Synthese: einheitlicher Graphen-Rahmen für biologische Netzwerke — PPI, Genomnetzwerke, Epidemie-Ausbreitung, Cancer-Graph-Theorie via Subgraph-Isomorphie in O(n³); R₀-Schwellensatz; Phylogenetischer Distanzsatz |
| hantavirus *(Drive)* | Graphentheoretische Modellierung des Hantavirus — Subgraph Algorithmus auf Protein-Interaktionsnetzwerke und epidemiologische Ausbreitungsgraphen; SIR-Modell, Stammvergleich PUUV vs. HTNV |
| norovirus *(Drive)* | Norovirus-Impfstoff — 8 Epitop-Kandidaten, Sensitivität 88 %, AUC 0,91; VLP-Impfstoff und 3C-Inhibitor, Phase I–III |
| austausch *(Drive)* | Austauschprinzip in der Naturwissenschaft — sechs Felder: Fick, Fourier, van't-Hoff, Donnan, Henry, Nernst-Planck |
| ebola *(Drive)* | Ebola-Virus — Subgraph-basierte Analyse der Protein-Interaktionsnetzwerke; SIR-Modell, Replikationsmechanismus und graphentheoretischer Impfstoffentwicklungsansatz |
| hiv *(Drive)* | HIV — graphentheoretische Analyse des Replikationszyklus und Protein-Interaktionsnetzwerks; antiretrovirale Therapieoptimierung via Subgraph Algorithmus |
| hiv&ebo *(Drive)* | HIV & Ebola: kombinierte graphentheoretische Analyse — Synergieeffekte bei Koinfektion, epidemiologische Wechselwirkungen und gemeinsame Subgraph-Modellierung |
| r0-classes *(Drive)* | Basisreproduktionszahl R₀ — formale Klassifikation epidemiologischer Ausbreitungsklassen; SIR/SEIR-Modelle, Schwellenwertanalyse und Interventionsschwellen |
| cooki | Inhomogene Würzverteilung bei gekochten Nudeln: sensorische Stimulationsdynamik durch stochastische Gewürzgradienten |

#### VI.1.h Ökosysteme & Naturschutz

| Repository | Beschreibung |
|---|---|
| wiederherstellung *(Drive)* | Resilienz der Erde — Lyapunov-Stabilität, Subgraph Algorithmus; trophische Kaskaden, Ozonschicht, Mangroven; 8 Plots |


### VI.2 Meeresbiologie

*Ozean-Reinigung, Korallenriffe, Mikroplastik-Neutralisierung und Meeressäuger-Ethologie*

| Repository | Beschreibung |
|---|---|
| algae | Algenproduktion — Monod- und Haldane-Kinetik, Batch-DGL; Reaktorvergleich; CO₂-Fixierung 1,83 g/g bewiesen; 6 Plots |
| cleanocn | Systematische Analyse kurz-, mittel- und langfristiger Reinigungsstrategien für Ozeane mit formalen Nachweisen |
| nanoneut | Nano- und Mikrostruktur-basierte In-situ-Neutralisierung von Mikroplastik im marinen Milieu |
| whale | Buckelwale (Megaptera novaeangliae) und ihre Fähigkeit, menschliche Absichten zu erkennen |
| ocnscnce | Ökotoxikologische Wirkungen auf Korallenriffe, Kupfer-Dynamik unter Ozeanversauerung und erweiterte Reinigungsstrategien — Kupfer-Vektoreffekte bei pH < 7,5 (IPCC-Szenarien) |


### VI.3 Theologie & Gesellschaft

*Schöpfungsbericht, Vernunft, Naturgesetze als anthropologische Ordnungsinstanz und fundamentale Erfindungen der Zivilisation*

| Repository | Beschreibung |
|---|---|
| macht | Jhwh, Jesus und Michael: schwarze und weiße Macht, Ende des Fluchens — formal bewiesen |
| schoepfung | Wasser, festes Land und Gottes Geist — theologische Analyse des Schöpfungsberichts (Genesis 1–3) |
| ntx | Die Sinnlosigkeit des rationalen Selbstbildes |
| tischstuhl | Tisch und Stuhl — wissenschaftliche Analyse der folgenreichsten Erfindungen der menschlichen Zivilisation |
| liberalismus *(Drive)* | Verfall des Liberalismus — fünf Kerndimensionen L₁–L₅; alle Indizes unterschreiten 2024 Schwellenwert τ = 0,5 |
| naturgesetze *(Drive)* | Naturgesetze als anthropologische Ordnungsinstanz — Konsequenzmaß κ ∈ [0,1]; Beruhigungstheorem; Hölle formal abgeleitet |
| karton *(Drive)* | Der geniale Karton: Nutzen, Beschaffenheit und eine formale Analyse seiner strukturellen Eigenschaften |


## VII. Synthesen & Transdisziplinarität

### VII.1 Wissenschaftliche Synthesen & Neue Forschungsfelder

*Transdisziplinäre Synthesearbeiten aus systematischer Kreuzanalyse aller 15 eigenständigen Forschungsdomänen*

| Repository | Beschreibung |
|---|---|
| ust *(Drive)* | Universelle Signatur-Theorie — σ_j als Isomorphie-Primitiv in 15 Disziplinen; Signatur-Satz, Kompositionssatz, P=NP |
| infengr *(Drive)* | Informationsingenieur als {0,1}-Ingenieur — Graphmodell-Theorem, Mächtigkeitstheorem; Kompetenzvergleich; Curriculum |
| infengg *(Drive)* | Grundkurs Informationsingenieurwesen — graphbasierte Modellierung, Beschreibungssprachen, Transformation und modellgetriebene Code-Generierung; UML-Profile, MARTE, Markov-Ketten, Warteschlangentheorie; Design Patterns; Beispiel Aufzugstür-Steuerung |


### VII.2 Sonstiges

*Persönliche Projekte, technische Anwendungen und interdisziplinäre Arbeiten außerhalb der Kategorisierung*

| Repository | Beschreibung |
|---|---|
| hjstephan86 | Persönliches GitHub-Profil von Stephan Epp — Senior Software Entwickler, M.Sc. Informatik, Bielefeld |
| nawfeuk | Physikalische Modellierung und bautechnische Analyse von Feuchteschäden an Mauerwerk und Kellerkonstruktionen |
| fugen | Optimale Fugenbreite und Fugenanordnung für maximale Lebensdauer von Fliesen- und Steinböden |
| holzbett *(Drive)* | Holzbett 2 m × 2 m — Statiknachweis DIN EN 1995-1-1; Mittelträger ohne Bodenstütze, KVH-Fichte |
| npw | Forschungspotenziale der wissenschaftlichen Arbeiten von Stephan Epp — übergreifende Synthese aller Domänen |
| bares *(Drive)* | Bares für Digitales — Konversion von Bargeld zu Guthaben; G(N,q), Qualitätsindex q, Strafgebühr π(q); Betrugsanalyse |

