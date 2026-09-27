# Strukturelle Deduplizierung von Dateisystem-Graphen mittels zyklischer Subgraph-Erkennung

## Anwendung des Subgraph Algorithmus auf Inode-Graphen, Snapshot-Systeme und verteilte Speicherarchitekturen

Das Unix-Werkzeug **du** (disk usage) traversiert seit seiner Entstehung
in den frühen 1970er-Jahren Dateisystem-Hierarchien und summiert
Speicherbelegungen. Obwohl der zugrundeliegende Algorithmus algorithmisch
linear ist ($O(n)$ in der Anzahl der Inodes), birgt das Problem strukturelle
Komplexität, die bisher nicht vollständig ausgeschöpft wurde: Hardlinks,
Bind-Mounts, Btrfs-Snapshots und replizierte Verzeichnisbäume erzeugen
**strukturell identische Teilgraphen**, die naiv mehrfach traversiert
werden.

Diese Arbeit stellt eine neuartige Methode vor, die den
**Subgraph Algorithmus** von Stephan Epp -- einen effizienten
Graphenvergleich mittels Adjazenzmatrizen, Signatur-Arrays und zyklischer
Rotation -- auf das Dateisystem-Problem überträgt. Das Ziel ist nicht die
Beschleunigung des einfachen **du**-Befehls selbst (dieser ist bereits
I/O-gebunden), sondern die Erschließung einer neuen Problemklasse:

1. **Snapshot-Vergleich**: Erkennung strukturell identischer
          Teilbäume in Btrfs/ZFS-Snapshot-Serien
2. **Backup-Deduplizierung**: Blockebenen-unabhängige,
          strukturelle Erkennung redundanter Verzeichnisstrukturen
3. **Verteilte Speichersysteme**: Konsistenzprüfung ohne
          vollständige übertragung aller Inode-Metadaten
4. **Container-Images**: Erkennung gemeinsamer Layer-Strukturen
          in OCI/Docker-Images

## Erwerb

Der Preis für diese Arbeit beträgt 1.511.000.000,00 EUR.

### Zahlungsinformationen

Name: Stephan Epp  
IBAN: DE24 5003 1900 0012 5603 20
BIC: BBVADEFFXXX

