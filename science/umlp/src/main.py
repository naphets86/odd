import os
import sys
from src.profile_parser import ProfileParser
from src.class_generator import ProfileClassGenerator
from src.model_parser import ModelParser
from src.code_generator import CodeGenerator

def main():
    # Pfade definieren
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(base_dir, 'examples')
    output_dir = os.path.join(base_dir, 'output')
    
    # Output-Verzeichnis erstellen
    os.makedirs(output_dir, exist_ok=True)
    
    # Versuche zuerst Elevator-Beispiel, dann Fallback zu my_profile
    elevator_profile = os.path.join(examples_dir, 'elevator_profile.xmi')
    elevator_model = os.path.join(examples_dir, 'elevator_model.xmi')
    
    my_profile = os.path.join(examples_dir, 'my_profile.xmi')
    my_model = os.path.join(examples_dir, 'my_model.xmi')
    
    # Wähle Dateien aus
    if os.path.exists(elevator_profile) and os.path.exists(elevator_model):
        profile_file = elevator_profile
        model_file = elevator_model
        print("  Verwende Elevator-Beispiel (elevator_profile.xmi, elevator_model.xmi)")
        output_prefix = "elevator"
    elif os.path.exists(my_profile) and os.path.exists(my_model):
        profile_file = my_profile
        model_file = my_model
        print("  Verwende einfaches Beispiel (my_profile.xmi, my_model.xmi)")
        output_prefix = "simple"
    else:
        print(f"  Fehler: Keine Beispieldateien gefunden!")
        print(f"   Erwartet:")
        print(f"   - {elevator_profile} und {elevator_model}")
        print(f"   - ODER {my_profile} und {my_model}")
        return 1
    
    try:
        # STUFE 1: Profil → Python-Klassen
        print("\n=== Stufe 1: Generiere Klassen aus Profil ===")
        
        # Parse UML-Profil
        parser = ProfileParser()
        profile = parser.parse_profile(profile_file)
        
        # Generiere Python-Klassen
        generator = ProfileClassGenerator(profile)
        generated_classes_file = os.path.join(output_dir, f'generated_{output_prefix}_classes.py')
        generator.generate_classes(generated_classes_file)
        
        print(f"  Klassen generiert für Profil: {profile.name}")
        print(f"  Stereotypes: {[s.name for s in profile.stereotypes]}")
        print(f"  Ausgabe: {generated_classes_file}")
        
        # STUFE 2: Modell → Python-Code
        print("\n=== Stufe 2: Generiere Code aus Modell ===")
        
        # Parse UML-Modell
        model_parser = ModelParser()
        instances = model_parser.parse_model(model_file)
        
        print(f"✓ Modell-Instanzen gefunden: {len(instances)}")
        for inst in instances:
            print(f"  - {inst.element_name} ({inst.stereotype_name})")
        
        # Generiere Code
        # Füge output_dir zum Python-Pfad hinzu
        if output_dir not in sys.path:
            sys.path.insert(0, output_dir)
        
        code_gen = CodeGenerator(f'generated_{output_prefix}_classes')
        generated_code = code_gen.generate_code(instances)
        
        # Schreibe generierten Code
        output_code_file = os.path.join(output_dir, f'generated_{output_prefix}_code.py')
        with open(output_code_file, 'w') as f:
            f.write(generated_code)
        
        print(f"\n✓ Code generiert für {len(instances)} Instanzen")
        print(f"  Ausgabe: {output_code_file}")
        print("\n" + "="*70)
        print("Generierter Code (Vorschau):")
        print("="*70)
        
        # Zeige ersten Teil des generierten Codes
        lines = generated_code.split('\n')
        preview_lines = min(50, len(lines))
        print('\n'.join(lines[:preview_lines]))
        if len(lines) > preview_lines:
            print(f"\n... ({len(lines) - preview_lines} weitere Zeilen)")
            print(f"\nVollständiger Code in: {output_code_file}")
        
        print("\n" + "="*70)
        print("✓ Code-Generierung erfolgreich abgeschlossen!")
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\nFehler: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
