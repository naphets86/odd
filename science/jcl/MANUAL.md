# JCL Manual

Dieses Handbuch beschreibt die wichtigsten Schritte zum Ausführen des Compilers auf einem Windows-PC ohne Administratorrechte.

## 1. Voraussetzungen

- Java JDK 26 ist installiert
- Apache Maven 3.9.x ist installiert
- Die Ordner von Java und Maven sind in der PATH-Umgebung verfügbar

Die verwendeten Pfade in diesem Beispiel sind:

- Java: `C:\Users\sepp5\Downloads\jdk-26_windows-x64_bin\jdk-26.0.1`
- Maven: `C:\Users\sepp5\Downloads\apache-maven-3.9.16-bin\apache-maven-3.9.16`

## 2. Projektordner öffnen

Im PowerShell-Fenster in das Projektverzeichnis wechseln:

```powershell
cd C:\Users\sepp5\Git\science\science\jcl\src
```

## 3. Compiler ausführen

Den Compiler mit dem Beispielprogramm starten:

```powershell
mvn --% -DskipTests compile org.codehaus.mojo:exec-maven-plugin:3.6.3:java -Dexec.mainClass=de.hjstephan86.jcl.JCLCompiler -Dexec.args="C:\Users\sepp5\Git\science\science\jcl\src\HelloWorld.java"
```

Wenn der Befehl erfolgreich war, wird eine Datei wie `HelloWorld.class` erzeugt.

## 4. Wenn `mvn` nicht erkannt wird

Falls PowerShell `mvn` nicht findet, kann der Aufruf direkt über die Maven-Datei erfolgen:

```powershell
& "C:\Users\sepp5\Downloads\apache-maven-3.9.16-bin\apache-maven-3.9.16\bin\mvn.cmd" --% -DskipTests compile org.codehaus.mojo:exec-maven-plugin:3.6.3:java -Dexec.mainClass=de.hjstephan86.jcl.JCLCompiler -Dexec.args="C:\Users\sepp5\Git\science\science\jcl\src\HelloWorld.java"
```

## 5. Maven und Java dauerhaft verfügbar machen

Falls der Befehl in neuen PowerShell-Fenstern nicht funktioniert, können die Pfade für den aktuellen Benutzer gesetzt werden:

```powershell
$javaDir = "C:\Users\sepp5\Downloads\jdk-26_windows-x64_bin\jdk-26.0.1\bin"
$mavenDir = "C:\Users\sepp5\Downloads\apache-maven-3.9.16-bin\apache-maven-3.9.16\bin"
[Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Users\sepp5\Downloads\jdk-26_windows-x64_bin\jdk-26.0.1", "User")
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";" + $mavenDir + ";" + $javaDir, "User")
```

Anschließend PowerShell neu öffnen.

## 6. Tests ausführen

```powershell
mvn test
```
