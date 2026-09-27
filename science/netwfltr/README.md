# Network Filter — Netzwerkverkehrsmonitor

Ein Docker-basierter Netzwerkmonitor mit FastAPI-Backend, SQLite-Datenbank
und einer Live-Weboberfläche mit OSI-Schichten-Farbkodierung.

## Voraussetzungen

- Ubuntu 24.04 LTS / 26.04 oder neuer
- Docker & Docker Compose
- Ausführung mit `sudo` oder als root (wegen `NET_RAW` Capability)

### Docker installieren (falls nicht vorhanden)

```bash
# Abhängigkeiten
sudo apt update
sudo apt install -y ca-certificates curl gnupg

# Docker GPG-Schlüssel
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Docker Repository hinzufügen
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker installieren
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Docker ohne sudo nutzbar machen (Neuanmeldung erforderlich)
sudo usermod -aG docker $USER
newgrp docker
```

## Installation & Starten

```bash
# In das Projektverzeichnis wechseln
cd netwfltr

# Build & Start im Hintergrund
sudo docker compose up -d --build

# Logs verfolgen
sudo docker compose logs -f
```

Browser öffnen: **http://localhost:8000**

> **Hinweis:** Der Hinweis `Published ports are discarded when using host network mode`
> ist kein Fehler — bei `network_mode: host` lauscht der Container direkt
> auf dem Host-Netzwerk.

## Funktionen

- **Live-Capture** via `tcpdump` auf allen Netzwerkinterfaces
- **90+ Protokolle** erkannt — IT, IoT und Industrieautomation (Modbus, S7comm, EtherNet/IP, OPC UA, BACnet, DNP3, IEC-104, HART-IP, FINS, ADS/AoE u.v.m.)
- **OSI-Schichten-Farbkodierung**:

| Schicht | Name          | Farbe   | Protokolle                                                                          |
|---------|---------------|---------|-------------------------------------------------------------------------------------|
| 2       | Data Link     | Violett | ARP, RARP, EtherCAT/ECAT, PROFINET/PN-DCP, SERCOS III, POWERLINK/EPL, CDP          |
| 3       | Netzwerk      | Blau    | IP, IPv4, IPv6, ICMP, ICMPv6, IGMP, OSPF, BGP, VRRP, PIM, GRE, IPIP               |
| 4       | Transport     | Türkis  | TCP, UDP, SCTP                                                                      |
| 5       | Sitzung       | Grün    | RPC, NetBIOS, PPTP, OPC-UA                                                          |
| 6       | Darstellung   | Gelb    | TLS, SSL, SSH, DTLS                                                                 |
| 7       | Anwendung     | Orange  | HTTP/S, DNS, SMTP, IMAP, NTP, SNMP, MQTT, CoAP, Modbus-TCP, EtherNet/IP, OPC-UA-TCP, BACnet, DNP3, IEC-104, IEC-61850/GOOSE/MMS, S7comm/S7plus, HART-IP, FINS, ADS/AoE, PROFIBUS, CANopen, DeviceNet, SECS/GEM, MTConnect |

- **Filter** nach Protokoll, Datum (von/bis) und Uhrzeit (von/bis)
- **Chronologisch absteigend** sortierte Anzeige (neueste Pakete oben)
- **Pagination** (100 Einträge pro Seite)
- **Auto-Refresh** alle 5 Sekunden
- **SQLite-Datenbank** persistent via Docker Volume

## API-Endpunkte

| Methode  | Pfad             | Beschreibung                       |
|----------|------------------|------------------------------------|
| `GET`    | `/api/traffic`   | Gefilterte Einträge (JSON)         |
| `GET`    | `/api/protocols` | Alle Protokolle + Anzahl           |
| `GET`    | `/api/stats`     | Gesamtstatistik nach OSI-Schicht   |
| `DELETE` | `/api/traffic`   | Alle Einträge löschen              |

### Filterparameter für `/api/traffic`

| Parameter   | Typ    | Beispiel       | Beschreibung              |
|-------------|--------|----------------|---------------------------|
| `protocol`  | string | `DNS`          | Protokollfilter           |
| `date_from` | date   | `2026-04-22`   | Datum von (inkl.)         |
| `date_to`   | date   | `2026-04-22`   | Datum bis (inkl.)         |
| `time_from` | time   | `08:00:00`     | Uhrzeit von               |
| `time_to`   | time   | `18:00:00`     | Uhrzeit bis               |
| `limit`     | int    | `100`          | Einträge pro Seite        |
| `offset`    | int    | `0`            | Seitenversatz             |

## Datenbank

Network Filter verwendet **SQLite**. Die Datenbankdatei liegt im Docker Volume
unter `/data/netwatch.db`.

### sqlite3 installieren (falls nicht vorhanden)

```bash
sudo apt install -y sqlite3
```

### Zugriff über Terminal

```bash
# Variante 1: Direkt im laufenden Container
docker exec -it netwatch sqlite3 /data/netwatch.db

# Variante 2: Über das Docker Volume (als root)
sudo sqlite3 /var/lib/docker/volumes/netwfltr_netwatch_data/_data/netwatch.db
```

### Nützliche SQLite-Abfragen

```sql
-- Tabellenstruktur anzeigen
.schema traffic

-- Letzte 20 Einträge
SELECT * FROM traffic ORDER BY timestamp DESC LIMIT 20;

-- Anzahl Einträge gesamt
SELECT COUNT(*) FROM traffic;

-- Gruppiert nach Protokoll
SELECT protocol, COUNT(*) AS anzahl
FROM traffic
GROUP BY protocol
ORDER BY anzahl DESC;

-- Gruppiert nach OSI-Schicht
SELECT osi_layer, COUNT(*) AS anzahl
FROM traffic
GROUP BY osi_layer
ORDER BY osi_layer;

-- Filter nach Protokoll
SELECT * FROM traffic
WHERE protocol = 'DNS'
ORDER BY timestamp DESC
LIMIT 10;

-- Filter nach Zeitraum
SELECT * FROM traffic
WHERE timestamp BETWEEN '2026-04-22 08:00:00' AND '2026-04-22 18:00:00'
ORDER BY timestamp DESC;

-- SQLite beenden
.quit
```

## Container verwalten

```bash
# Status prüfen
docker ps

# Logs anzeigen
docker logs netwatch

# Container stoppen (Daten bleiben erhalten)
sudo docker compose down

# Container stoppen + Daten löschen
sudo docker compose down -v

# Container neu starten
sudo docker compose restart

# Rebuild nach Code-Änderungen
sudo docker compose up -d --build
```

## Projektstruktur

```
netwfltr/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── app/
│   └── main.py          # FastAPI Backend + tcpdump Parser
├── static/
│   ├── css/
│   │   └── style.css    # OSI-Farbschema + Dark Theme
│   └── js/
│       └── app.js       # Frontend-Logik, Filter, Auto-Refresh
└── templates/
    └── index.html       # Hauptseite
```

## Hinweise zur Sicherheit

Der Container benötigt `NET_RAW` und `NET_ADMIN` Capabilities sowie
`network_mode: host`, um den Netzwerkverkehr des Hosts mitlesen zu können.

## Erwerb

Der Preis für diese Software beträgt 1.111.000.000,00 EUR.

### Zahlungsinformationen

Name: Stephan Epp  
IBAN: DE24 5003 1900 0012 5603 20
BIC: BBVADEFFXXX

