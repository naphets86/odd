# RESP – Ressourcenverwaltung

Effiziente Verwaltung und Planung von Ressourcen: **Menschen**, **Material** und **Zeit**.

## Erwerb

Der Preis für diese Software beträgt 115.000,00 EUR.

### Zahlungsinformationen

Name: Stephan Epp  
IBAN: DE24 5003 1900 0012 5603 20
BIC: BBVADEFFXXX

## Schnellstart (Docker)

```bash
# Projekt klonen / entpacken
cd resp

# Starten (baut Images automatisch)
sudo docker-compose up --build

# Fertig! Öffne im Browser:
# - Frontend:   http://localhost
# - API-Docs:   http://localhost:8000/docs
# - ReDoc:      http://localhost:8000/redoc
```

## Benutzeroberfläche

Die Webanwendung bietet eine intuitive, moderne Oberfläche zur Verwaltung aller Ressourcen. Sie ist vollständig in Deutsch verfasst und mit Emojis für schnelle visuelle Orientierung ausgestattet.

### Hauptbereiche

**Dashboard** 📊  
Übersichtsseite mit Statistiken zu allen Ressourcentypen und einer Vorschau der aktuellen Projekte.

**Ressourcenverwaltung**
- **Menschen** 👤: Verwaltung von Mitarbeitern, Studenten, Lehrern und anderen Personentypen mit E-Mail und Abteilungszuordnung
- **Material** 📦: Bestandsverwaltung verschiedener Materialtypen mit Mengen, Einheiten und Lagerorten
- **Zeit** ⏱️: Verwaltung von Zeitressourcen (Stunden, Tage, Wochen, etc.) für Projekte und Budgets

**Planung & Analyse**
- **Projekte** 📋: Erstellen und verwalten von Projekten mit Status (geplant, aktiv, abgeschlossen, abgebrochen) und Zeitrahmen
- **Zuteilungen** 🔗: Zuordnung von Ressourcen zu Projekten mit Mengen, Status und Zeiträumen
- **Graphanalyse** 🔍: Fortgeschrittene Analyse mit Subgraph-Vergleich (O(n³)) und Redundanzprüfung über mehrere Projektgruppen

### Funktionen

- **Modale Dialoge**: Alle Ressourcen können über intuitive Formulare hinzugefügt und bearbeitet werden
- **Tabellarische Ansichten**: Übersichtliche Darstellung aller Daten mit Aktionsschaltflächen (Bearbeiten, Löschen)
- **Filterung**: Zuteilungen können nach Projekten gefiltert werden
- **Graphvisualisierung**: Bipartite Graphen zur Visualisierung von Ressourcen-Projekt-Beziehungen
- **Responsive Design**: Moderne Benutzeroberfläche mit Inter-Schriftart und konsistenter Farbgebung

## Bipartite Planung & Subgraph Algorithmus

Für die Ressourcenplanung wird ein **bipartiter Graph** verwendet, in dem eine Seite die Ressourcen (Personen, Material, Zeit) und die andere Seite die Projekte darstellt. Kanten repräsentieren Zuteilungen.

Der **Subgraph Algorithmus** ermittelt gemeinsame Teilgraphen zwischen zwei Projektgruppen und erkennt Redundanzen über mehrere Gruppen hinweg. Die Laufzeit des Algorithmus beträgt **O(n³)**, wobei *n* die Anzahl der Knoten (Ressourcen + Projekte) im Graphen ist. Diese kubische Komplexität ergibt sich aus dem paarweisen Abgleich aller Knoten- und Kantenkombinationen beider Graphen.

Der Algorithmus wird über den Endpunkt `POST /api/subgraph/compare` aufgerufen und bildet die Grundlage der **Graphanalyse**-Ansicht im Frontend.

## Tests ausführen

```bash
# In der API-Container-Shell oder lokal mit venv
pip install -r requirements.txt

# Tests mit Coverage
pytest
```

## API-Endpunkte

### Personen
| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| `GET` | `/api/persons/` | Alle Personen |
| `POST` | `/api/persons/` | Person anlegen |
| `GET` | `/api/persons/{id}` | Person lesen |
| `PATCH` | `/api/persons/{id}` | Person ändern |
| `DELETE` | `/api/persons/{id}` | Person löschen |

### Material
| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| `GET` | `/api/materials/` | Alle Materialien |
| `POST` | `/api/materials/` | Material anlegen |
| `GET` | `/api/materials/{id}` | Material lesen |
| `PATCH` | `/api/materials/{id}` | Material ändern |
| `DELETE` | `/api/materials/{id}` | Material löschen |

### Zeitressourcen
| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| `GET` | `/api/time-resources/` | Alle Zeitressourcen |
| `POST` | `/api/time-resources/` | Zeitressource anlegen |
| `GET` | `/api/time-resources/{id}` | Zeitressource lesen |
| `PATCH` | `/api/time-resources/{id}` | Zeitressource ändern |
| `DELETE` | `/api/time-resources/{id}` | Zeitressource löschen |

### Projekte
| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| `GET` | `/api/projects/` | Alle Projekte |
| `POST` | `/api/projects/` | Projekt erstellen |
| `GET` | `/api/projects/{id}` | Projekt lesen |
| `PATCH` | `/api/projects/{id}` | Projekt ändern |
| `DELETE` | `/api/projects/{id}` | Projekt löschen |

### Zuteilungen
| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| `GET` | `/api/allocations/` | Alle Zuteilungen (optional: `?project_id=`) |
| `POST` | `/api/allocations/` | Ressource einem Projekt zuteilen |
| `GET` | `/api/allocations/{id}` | Zuteilung lesen |
| `PATCH` | `/api/allocations/{id}` | Zuteilung ändern |
| `DELETE` | `/api/allocations/{id}` | Zuteilung löschen |

### Subgraph / Bipartiter Graph
| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| `POST` | `/api/subgraph/compare` | Zwei Projektgruppen als Graphen vergleichen (O(n³)) |
| `GET` | `/api/subgraph/graph?project_ids=1,2` | Bipartiten Graph als Matrix abrufen |
| `POST` | `/api/subgraph/redundancy` | Redundanzcheck über mehrere Projektgruppen |

### Meta
| Methode | Pfad | Beschreibung |
|---------|------|-------------|
| `GET` | `/api/enums` | Alle Enum-Werte (Typen, Einheiten, Status) |
| `GET` | `/health` | Health-Check |

## Ressourcentypen

### Menschen
Arbeitnehmer · Arbeitgeber · Professor · Student · Lehrer · Schüler

### Material
Geld · Metall · Stahl · Öl · Gas · Zuckerrüben · Holz · Silizium · Kupfer  
Kreide · Schwamm · Beamer · Laptop · PC/Workstation

### Zeit
Sekunden · Minuten · Stunden · Tage · Wochen · Monate · Jahre

### Projektstatus
geplant · aktiv · abgeschlossen · abgebrochen

## Umgebungsvariablen

| Variable | Standard | Beschreibung |
|---------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://resp_user:resp_pass@localhost:5432/resp_db` | PostgreSQL-Verbindung |

## Lokale Entwicklung (ohne Docker)

```bash
# PostgreSQL starten (oder nur DB-Container):
docker-compose up db -d

# Python-Umgebung:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# API starten:
uvicorn app.main:app --reload --port 8000

# Frontend: einfach index.html im Browser öffnen
# (API_BASE wird automatisch auf localhost:8000 gesetzt)
```