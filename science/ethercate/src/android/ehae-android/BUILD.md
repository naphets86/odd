# EHAE HMI – EtherCAT HMI App Extension

Basierend auf der Arbeit **„EtherCAT Protocol Erweiterung – App-basierte HMI als Ersatz für HMI-Controller"** von Stephan Epp.

---

## Projektstruktur

```
ehae-android/
├── www/                    ← Web-App (kompiliertes HTML/CSS/JS)
│   └── index.html          ← EHAE HMI Single-Page-App
├── android/                ← Natives Android-Projekt (Gradle)
│   └── app/src/...
├── capacitor.config.json   ← Capacitor-Konfiguration
├── package.json
└── BUILD.md                ← Diese Datei
```

---

## 1. Web-App (Standalone)

Die Web-App läuft direkt im Browser ohne Installation:

```bash
# Einfach öffnen:
open www/index.html

# Oder mit lokalem HTTP-Server:
npx serve www/
# → http://localhost:3000
```

### Features der Web-App

| Bereich | Operator | Engineer |
|---|---|---|
| Dashboard (KPIs, Live-Variablen) | ✓ | ✓ |
| Topologie-Ansicht (Live SVG) | ✓ | ✓ |
| Alarm Manager + ACK | ✓ | ✓ |
| Maschinensteuerung (START/STOP) | ✓ | ✓ |
| Echtzeit-Oszilloskop | ✓ | ✓ |
| TCO-Kostenanalyse | ✓ | ✓ |
| Migrationsstatus | ✓ | ✓ |
| SDO Browser (CoE) | ✗ | ✓ |
| FoE Firmware-Update | ✗ | ✓ |
| HCP Protokoll-Log | ✗ | ✓ |
| Parametrierung (Slider) | ✗ | ✓ |

Rollenumschaltung per Klick auf `OPERATOR`/`ENGINEER`-Badge oben rechts.

---

## 2. Android APK (Capacitor)

### Voraussetzungen

- **Node.js** ≥ 18
- **Android Studio** (mit Android SDK, Build Tools ≥ 34)
- **Java JDK 17** (via Android Studio oder separat)
- `ANDROID_HOME` und `JAVA_HOME` als Umgebungsvariablen gesetzt

### Build-Schritte

```bash
# 1. Abhängigkeiten installieren (bereits done)
npm install

# 2. Web-Assets in Android-Projekt synchronisieren
npx cap sync android

# 3. In Android Studio öffnen (GUI)
npx cap open android

# → dann in Android Studio: Build → Generate Signed APK
#   oder: Build → Build Bundle(s) / APK(s) → Build APK(s)
```

### APK via Kommandozeile bauen (ohne Android Studio GUI)

```bash
cd android
./gradlew assembleRelease
# APK liegt dann unter:
# android/app/build/outputs/apk/release/app-release-unsigned.apk
```

#### APK signieren (für Produktion)

```bash
# Keystore erstellen (einmalig):
keytool -genkey -v -keystore ehae-release.keystore \
        -alias ehae -keyalg RSA -keysize 2048 -validity 10000

# APK signieren:
jarsigner -verbose -sigalg SHA256withRSA -digestalg SHA-256 \
          -keystore ehae-release.keystore \
          app-release-unsigned.apk ehae

# ZIP-align (optional, aber empfohlen):
zipalign -v 4 app-release-unsigned.apk EHAE-HMI.apk
```

### Debug-APK direkt auf Gerät installieren

```bash
cd android
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

---

## 3. Capacitor-Konfiguration (`capacitor.config.json`)

```json
{
  "appId": "de.epp.ehae",
  "appName": "EHAE HMI",
  "webDir": "www",
  "android": {
    "allowMixedContent": true,
    "backgroundColor": "#07090f",
    "overrideUserAgent": "EHAE-HMI/1.0 Android"
  }
}
```

### Verbindung zum echten HAG-Server

Für den produktiven Einsatz muss in `www/index.html` der WebSocket-Endpunkt konfiguriert werden:

```javascript
// In index.html – WebSocket-Verbindung zum HAG:
const ws = new WebSocket('wss://192.168.10.1:9740/hcp');
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  // VAR_UPDATE, ALARM_RAISE, etc. verarbeiten
};
```

Das Demo-Frontend simuliert diese Verbindung – alle Live-Daten sind Simulation.

---

## 4. Android-spezifische Einstellungen

| Einstellung | Wert | Begründung |
|---|---|---|
| `screenOrientation` | `landscape` | HMI-Dashboards nutzen Querformat |
| `usesCleartextTraffic` | `true` | Für lokale HAG-Verbindungen ohne TLS (Intranet) |
| `minSdkVersion` | 22 (Android 5.1) | Breite Gerätekompatibilität |
| `targetSdkVersion` | 34 (Android 14) | Aktuell |

---

## 5. Architektur-Referenz (aus der Arbeit)

```
[EtherCAT Master]
      │
      └── [HAG – HMI Application Gateway]
              │  WebSocket (HCP/1.2)
              ├── [EHAE Operator App – Android Tablet]
              └── [EHAE Engineer App – Desktop / Android]
```

**HCP Nachrichtentypen:**
- `VAR_UPDATE` – Live-Prozessvariablen (PDO-Mapping)
- `ALARM_RAISE / ALARM_ACK` – Alarm-Lifecycle
- `SDO_READ / SDO_WRITE` – CAN over EtherCAT
- `FOE_UPLOAD` – Firmware over EtherCAT
- `MACHINE_CMD` – Maschinensteuerung (START/STOP/RESET)

---

## App-ID / Package

`de.epp.ehae`
