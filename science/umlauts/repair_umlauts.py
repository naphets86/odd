#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reparaturskript für defekte Umlaute in LaTeX-Dateien
Commit 4b93d9f hat UTF-8 Umlaute zu Latin-1/ISO-8859-1 beschädigt.

Dieses Skript:
- Findet alle .tex Dateien im science/-Verzeichnis
- Konvertiert defekte Umlaute zurück zu korrektem UTF-8
- Erstellt ein detailliertes Report
- Kann optional ein Backup erstellen
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse
from datetime import datetime


# Mapping von defekten zu korrekten Umlauten und Sonderzeichen
DEFECT_TO_CORRECT = {
    # Lowercase Umlaute & Sonderzeichen
    'Ã¤': 'ä',   # ä
    'Ã¶': 'ö',   # ö
    'Ã¼': 'ü',   # ü
    'ÃŸ': 'ß',   # ß (Häufiger UTF-8 Encoding-Fehler)
    'Ã§': 'ç',   # ç
    'Ã±': 'ñ',   # ñ
    
    # Uppercase Umlaute & Sonderzeichen
    'Ã„': 'Ä',   # Ä
    'Ã–': 'Ö',   # Ö
    'Ãœ': 'Ü',   # Ü (Falls Großschreibung vorkommt)
    'Ã¯': 'Ï',   # Ï
    'Ã‰': 'É',   # É
    'Ã©': 'é',   # é
    
    # Additional common defects
    'Â': ' ',    # Non-breaking space artifact
}

# Mapping für inputenc-Zeilen (z.B. {Ã¤}{{\"a}}1 → {ä}{{\"a}}1)
INPUTENC_PATTERN = r'\{([Ã][¤¶¼„–])\}'
INPUTENC_REPLACEMENT = lambda m: '{' + DEFECT_TO_CORRECT.get(m.group(1), m.group(1)) + '}'


class UmlautRepair:
    """Hauptklasse für die Umlaut-Reparatur"""
    
    def __init__(self, root_dir: str = '.', backup: bool = True, dry_run: bool = False):
        """
        Initialisiere den Reparierer
        
        Args:
            root_dir: Wurzelverzeichnis für Suche (default: '.')
            backup: Erstelle Backups vor Änderung (default: True)
            dry_run: Nur simulieren, keine echten Änderungen (default: False)
        """
        self.root_dir = Path(root_dir)
        self.backup = backup
        self.dry_run = dry_run
        self.stats = defaultdict(int)
        self.repairs_per_file = {}
        self.files_processed = []
        
    def find_tex_files(self) -> List[Path]:
        """Finde alle .tex Dateien im Projekt"""
        tex_files = list(self.root_dir.rglob('*.tex'))
        print(f"✓ {len(tex_files)} .tex Dateien gefunden")
        return sorted(tex_files)
    
    def repair_file(self, filepath: Path) -> Tuple[int, str]:
        """
        Repariere eine einzelne Datei
        
        Returns:
            Tuple von (Anzahl Reparaturen, Originaler Text)
        """
        try:
            # Lese Datei mit UTF-8
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                original_content = f.read()
            
            repair_count = 0
            repaired_content = original_content
            
            # Ersetze defekte Umlaute
            for defect, correct in DEFECT_TO_CORRECT.items():
                if defect in repaired_content:
                    count = repaired_content.count(defect)
                    repaired_content = repaired_content.replace(defect, correct)
                    repair_count += count
                    self.stats[f'umlaut_{defect}'] += count
            
            # Repariere inputenc-Zeilen
            pattern_count = 0
            for match in re.finditer(INPUTENC_PATTERN, repaired_content):
                pattern_count += 1
            
            if pattern_count > 0:
                repaired_content = re.sub(INPUTENC_PATTERN, INPUTENC_REPLACEMENT, repaired_content)
                repair_count += pattern_count
                self.stats['inputenc_patterns'] += pattern_count
            
            self.stats['total_repairs'] += repair_count
            self.stats['files_with_defects'] += 1 if repair_count > 0 else 0
            
            return repair_count, original_content, repaired_content
            
        except Exception as e:
            print(f"  ⚠️  Fehler bei {filepath}: {e}")
            self.stats['errors'] += 1
            return 0, "", ""
    
    def backup_file(self, filepath: Path):
        """Erstelle Backup einer Datei"""
        backup_path = Path(str(filepath) + '.bak')
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.stats['backups_created'] += 1
        except Exception as e:
            print(f"  ⚠️  Backup-Fehler für {filepath}: {e}")
    
    def write_file(self, filepath: Path, content: str):
        """Schreibe Datei mit UTF-8 Encoding"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            self.stats['files_written'] += 1
        except Exception as e:
            print(f"  ⚠️  Schreib-Fehler für {filepath}: {e}")
    
    def process_all_files(self, verbose: bool = False) -> Dict:
        """Verarbeite alle .tex Dateien"""
        tex_files = self.find_tex_files()
        
        print(f"\n{'='*70}")
        print(f"Starte Reparatur (dry_run={self.dry_run}, backup={self.backup})")
        print(f"{'='*70}\n")
        
        for i, filepath in enumerate(tex_files, 1):
            repair_count, original, repaired = self.repair_file(filepath)
            
            if repair_count > 0:
                rel_path = filepath.relative_to(self.root_dir)
                self.files_processed.append(rel_path)
                self.repairs_per_file[str(rel_path)] = repair_count
                
                status = "DRY RUN" if self.dry_run else "REPARIERT"
                print(f"[{i:3d}] {status}: {rel_path}")
                print(f"      └─ {repair_count} Umlaute korrigiert")
                
                if verbose:
                    # Zeige erste Unterschiede
                    if original != repaired:
                        print(f"      Beispielreparaturen:")
                        for defect, correct in list(DEFECT_TO_CORRECT.items())[:3]:
                            if defect in original:
                                print(f"        {defect} → {correct}")
                
                # Schreibe Datei, wenn nicht dry_run
                if not self.dry_run:
                    if self.backup:
                        self.backup_file(filepath)
                    self.write_file(filepath, repaired)
        
        self.stats['files_scanned'] = len(tex_files)
        return self.stats
    
    def generate_report(self, output_file: str = None) -> str:
        """Erstelle Detailbericht"""
        report = []
        report.append(f"\n{'='*70}")
        report.append("REPARATURBERICHT - DEFEKTE UMLAUTE")
        report.append(f"{'='*70}\n")
        
        report.append(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Verzeichnis: {self.root_dir}")
        report.append(f"Dry-Run: {self.dry_run}\n")
        
        # Statistiken
        report.append("STATISTIKEN:")
        report.append(f"  Dateien gescannt:           {self.stats['files_scanned']}")
        report.append(f"  Dateien mit Defekten:       {self.stats['files_with_defects']}")
        report.append(f"  Gesamtanzahl Reparaturen:  {self.stats['total_repairs']}")
        report.append(f"  Backups erstellt:           {self.stats['backups_created']}")
        report.append(f"  Dateien geschrieben:        {self.stats['files_written']}")
        report.append(f"  Fehler:                     {self.stats['errors']}\n")
        
        # Defekt-Zusammenfassung
        if any(k.startswith('umlaut_') for k in self.stats):
            report.append("DEFEKT-ZUSAMMENFASSUNG:")
            for defect, count in sorted(self.stats.items()):
                if defect.startswith('umlaut_') and count > 0:
                    correct = DEFECT_TO_CORRECT.get(defect[7:], '?')
                    report.append(f"  {defect[7:]} → {correct}: {count} Vorkommen")
            report.append("")
        
        if self.stats.get('inputenc_patterns', 0) > 0:
            report.append(f"  inputenc-Muster: {self.stats['inputenc_patterns']} Reparaturen\n")
        
        # Top 20 betroffene Dateien
        if self.repairs_per_file:
            report.append("TOP 20 AM STÄRKSTEN BETROFFENE DATEIEN:")
            sorted_files = sorted(
                self.repairs_per_file.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20]
            for filepath, count in sorted_files:
                report.append(f"  {filepath}: {count} Reparaturen")
            report.append("")
        
        # Alle betroffenen Dateien
        if self.files_processed:
            report.append(f"ALLE {len(self.files_processed)} BETROFFENEN DATEIEN:")
            for filepath in sorted(self.files_processed):
                count = self.repairs_per_file.get(str(filepath), 0)
                report.append(f"  {filepath} ({count} Reparaturen)")
            report.append("")
        
        report.append(f"{'='*70}\n")
        
        report_text = '\n'.join(report)
        
        # Speichere Report falls Dateiname gegeben
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✓ Report gespeichert: {output_file}")
        
        return report_text


def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(
        description='Repariere defekte Umlaute in LaTeX-Dateien (Commit 4b93d9f)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Nur simulieren (keine Änderungen)
  python3 repair_umlauts.py --dry-run

  # Repariere alle Dateien mit Backups
  python3 repair_umlauts.py --root science/

  # Repariere ohne Backups und speichere Report
  python3 repair_umlauts.py --no-backup --report umlaut_repair.txt

  # Verbose Output mit Beispielen
  python3 repair_umlauts.py --verbose
        """
    )
    
    parser.add_argument(
        '--root',
        default='.',
        help='Wurzelverzeichnis (default: aktuelle Verzeichnis)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Nur simulieren, keine echten Änderungen'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Keine Backups erstellen'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Speichere Report in diese Datei'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Zeige Beispiele für Reparaturen'
    )
    
    args = parser.parse_args()
    
    # Initialisiere Reparierer
    repair = UmlautRepair(
        root_dir=args.root,
        backup=not args.no_backup,
        dry_run=args.dry_run
    )
    
    # Verarbeite Dateien
    stats = repair.process_all_files(verbose=args.verbose)
    
    # Zeige Report
    report = repair.generate_report(output_file=args.report)
    print(report)
    
    # Warnung bei dry_run
    if args.dry_run:
        print("⚠️  DRY-RUN MODUS: Keine Dateien wurden tatsächlich geändert.")
        print("   Entfernen Sie --dry-run um echte Änderungen durchzuführen.\n")
    else:
        print(f"✅ Reparatur abgeschlossen! {stats['total_repairs']} Umlaute korrigiert.")


if __name__ == '__main__':
    main()
