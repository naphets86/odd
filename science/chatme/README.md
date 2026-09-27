# Chat Me

Ende-zu-Ende-verschlüsselte Messenger-App  
Verschlüsselungsverfahren: Signatur-Chiffre (Epp 2026)  
Plattform: Android via Capacitor 6 · Web via FastAPI + WebSocket  

## Starten der Applikation

```bash
# Docker Image bauen
sudo docker-compose up --build

# URL im Browser
http://localhost:8000/
```

## Tests ausführen

```bash
# Aktivieren der venv
source backend/venv/bin/activate

# Backend-Tests ausführen
cd backend
pytest tests/ -v

# UI-Tests ausführen
cd ../
pytest tests-frontend/ -v
```

## Docker Image neu bauen

```bash
# Docker Image entfernen
sudo docker compose down

# Docker Image bauen
sudo docker-compose up --build
```

## Android App bauen und installieren

```bash
# Synchronisiere Frontend und aktualisiere Abhängigkeiten
npx cap sync android

# Kompiliere Android Projekt
cd android
./gradlew assembleDebug

# Übertrage APK auf Device
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Erwerb

Der Preis für diese Software beträgt 545.000.000,00 EUR.

### Zahlungsinformationen

Name: Stephan Epp  
IBAN: DE24 5003 1900 0012 5603 20
BIC: BBVADEFFXXX

