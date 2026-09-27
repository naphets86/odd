# EHAE HMI – Android Installation

## Voraussetzungen

- Android-Smartphone mit **Android 5.1 oder neuer**
- USB-Kabel

---

## 1. USB-Debugging aktivieren

1. **Einstellungen → Über das Telefon → Buildnummer** → 7× tippen
2. **Einstellungen → Entwickleroptionen → USB-Debugging** → Ein
3. Smartphone per USB anschließen
4. Benachrichtigungsleiste → „USB für…" → **Dateiübertragung (MTP)** wählen
5. Dialog „USB-Debugging erlauben?" auf dem Gerät → **Erlauben**

---

## 2. APK bauen

```bash
cd src/android/ehae-android/android
./gradlew assembleDebug
```

APK liegt unter:
```
app/build/outputs/apk/debug/app-debug.apk
```

---

## 3. Gerät prüfen

```bash
adb devices
# Erwartete Ausgabe:
# XXXXXXXX    device
```

---

## 4. APK installieren

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

Die App **EHAE HMI** erscheint danach im App-Drawer des Geräts.

---

## Erneut installieren (Update)

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Das `-r` Flag ersetzt die vorhandene Installation ohne Datenverlust.
