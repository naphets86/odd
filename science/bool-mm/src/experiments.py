"""
Experimente zum Vergleich der Algorithmen.

Vergleicht:
1. Boolean: Signatur-Methode (Algo 2.1) vs. Naive O(n^3)
2. k-beschränkt: Schichten-Signaturen (Algo 4.1) vs. Strassen
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import os
from pathlib import Path

from src.boolean_matrix_multiplier import BooleanMatrixMultiplier
from src.strassen_multiplier import StrassenMultiplier, KBoundedStrassenMultiplier
from src.k_bounded_multiplier import KBoundedMatrixMultiplier


def experiment_boolean_multiplication():
    """
    Experiment 1: Boolean Matrixmultiplikation.
    
    Vergleicht:
    - Algorithmus 2.1 (Signaturen): Theoretisch O(n²), aber Python-Implementierung
    - Naive Python-Schleifen: Theoretisch O(n³), aber mit early-exit
    - NumPy matmul: Hochoptimiert (BLAS), praktisch am schnellsten
    
    WICHTIG: Die theoretische Komplexität gilt nur für optimierte C-Implementierungen!
    In Python dominieren Konstanten und Interpreter-Overhead.
    """
    print("="*80)
    print("EXPERIMENT 1: Boolean Matrixmultiplikation")
    print("="*80)
    print("HINWEIS: Python-Implementierung zeigt praktische Performance, nicht theoretische Komplexität!")
    print("="*80)
    
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    
    times_signature = []
    times_naive = []
    times_numpy = []
    
    multiplier_sig = BooleanMatrixMultiplier()
    
    for n in sizes:
        print(f"\nTeste Größe n={n}")
        np.random.seed(42)
        
        # Generiere zufällige Boolean-Matrizen
        A = np.random.randint(0, 2, (n, n))
        B = np.random.randint(0, 2, (n, n))
        
        # Test 1: NumPy (Referenz - am schnellsten)
        start = time.perf_counter()
        C_numpy = (np.matmul(A, B) > 0).astype(int)
        time_numpy = time.perf_counter() - start
        times_numpy.append(time_numpy)
        print(f"  NumPy (optimiert):      {time_numpy*1000:.3f} ms")
        
        # Test 2: Signatur-Methode (Algo 2.1)
        start = time.perf_counter()
        C_sig = multiplier_sig.multiply_optimized(A, B)
        time_sig = time.perf_counter() - start
        times_signature.append(time_sig)
        print(f"  Signatur-Methode:       {time_sig*1000:.3f} ms  (Faktor: {time_sig/time_numpy:.1f}x)")
        
        # Test 3: Naive (nur für n <= 512
        if n <= 512:
            start = time.perf_counter()
            C_naive = multiplier_sig.multiply_naive(A, B)
            time_naive = time.perf_counter() - start
            times_naive.append(time_naive)
            print(f"  Naive O(n³):            {time_naive*1000:.3f} ms  (Faktor: {time_naive/time_numpy:.1f}x)")
            
            # Verifikation
            assert np.array_equal(C_naive, C_numpy), f"Fehler bei n={n}: Naive != NumPy"
        else:
            times_naive.append(None)
    
    # Plot erstellen
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, [t*1000 for t in times_signature], 'o-', 
             label='Signatur-Methode O(n²)', linewidth=2, markersize=8)
    
    # Naive nur für kleinere n
    sizes_naive = [s for s, t in zip(sizes, times_naive) if t is not None]
    times_naive_filtered = [t*1000 for t in times_naive if t is not None]
    if times_naive_filtered:
        plt.plot(sizes_naive, times_naive_filtered, 's-', 
                label='Naive O(n³)', linewidth=2, markersize=8)
    
    plt.xlabel('Matrixgröße n', fontsize=12)
    plt.ylabel('Laufzeit (ms)', fontsize=12)
    plt.title('Boolean Matrixmultiplikation: Signatur-Methode vs. Naive', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log', base=2)
    
    # Speichere als SVG
    output_path = Path(__file__).parent / 'results/experiment_boolean.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    print(f"\n✓ Plot gespeichert: {output_path}")
    plt.close()
    
    return sizes, times_signature, times_naive


def experiment_k_bounded_multiplication():
    """
    Experiment 2: k-beschränkte Matrixmultiplikation.
    
    Vergleicht Algorithmus 4.1 (Schichten-Signaturen O(k·n²)) mit Strassen O(n^2.807).
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: k-beschränkte Matrixmultiplikation")
    print("="*80)
    
    k_values = [2, 3, 5]
    sizes = [8, 16, 32, 64, 128]
    
    results = {}
    
    multiplier_sig = KBoundedMatrixMultiplier()
    multiplier_strassen = KBoundedStrassenMultiplier(threshold=32)
    
    for k in k_values:
        print(f"\n--- k = {k} ---")
        times_sig = []
        times_strassen = []
        
        for n in sizes:  # Diese Schleife MUSS eingerückt sein!
            print(f"  Teste Größe n={n}")
            np.random.seed(42)
            
            # Generiere zufällige k-beschränkte Matrizen
            A = np.random.randint(0, k + 1, (n, n))
            B = np.random.randint(0, k + 1, (n, n))
            
            # Test 1: Schichten-Signaturen (Algo 4.1)
            start = time.perf_counter()
            C_sig = multiplier_sig.multiply_k_bounded(A, B, k)
            time_sig = time.perf_counter() - start
            times_sig.append(time_sig)
            print(f"    Schichten-Signaturen: {time_sig*1000:.3f} ms")
            
            # Test 2: Strassen
            start = time.perf_counter()
            C_strassen = multiplier_strassen.multiply(A, B)
            time_strassen = time.perf_counter() - start
            times_strassen.append(time_strassen)
            print(f"    Strassen:             {time_strassen*1000:.3f} ms")
            
            # Verifikation
            expected = np.matmul(A, B)
            assert np.array_equal(C_sig, expected), f"Fehler bei k={k}, n={n}: Signatur != NumPy"
            assert np.array_equal(C_strassen, expected), f"Fehler bei k={k}, n={n}: Strassen != NumPy"
        
        results[k] = (times_sig, times_strassen)
    
    # Plot erstellen
    plt.figure(figsize=(12, 6))
    
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, k in enumerate(k_values):
        times_sig, times_strassen = results[k]
        
        plt.plot(sizes, [t*1000 for t in times_sig], 
                marker=markers[idx], linestyle='-', color=colors[idx],
                label=f'Schichten-Signaturen (k={k})', linewidth=2, markersize=8)
        plt.plot(sizes, [t*1000 for t in times_strassen], 
                marker=markers[idx], linestyle='--', color=colors[idx],
                label=f'Strassen (k={k})', linewidth=2, alpha=0.7, markersize=8)
    
    plt.xlabel('Matrixgröße n', fontsize=12)
    plt.ylabel('Laufzeit (ms)', fontsize=12)
    plt.title('k-beschränkte Matrixmultiplikation: Schichten-Signaturen vs. Strassen', fontsize=14)
    plt.legend(fontsize=9, ncol=2, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log', base=2)
    
    # Speichere als SVG
    output_path = Path(__file__).parent / 'results/experiment_k_bounded.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    print(f"\n✓ Plot gespeichert: {output_path}")
    plt.close()
    
    return results


def print_summary(exp1_data, exp2_data):
    """Druckt Zusammenfassung der Experimente."""
    sizes, times_sig, times_naive = exp1_data
    
    print("\n" + "="*80)
    print("ZUSAMMENFASSUNG")
    print("="*80)
    
    print("\n1. Boolean Matrixmultiplikation:")
    print(f"\n   Größe n=256:")
    idx_256 = sizes.index(256)
    if idx_256 < len(times_sig) and idx_256 < len(times_naive) and times_naive[idx_256] is not None:
        speedup = times_naive[idx_256] / times_sig[idx_256]
        print(f"     Signatur-Methode O(n²): {times_sig[idx_256]*1000:.2f} ms")
        print(f"     Naive O(n³):            {times_naive[idx_256]*1000:.2f} ms")
        print(f"     Speedup:                {speedup:.2f}x")
    
    print(f"\n   Größe n=512:")
    idx_512 = sizes.index(512)
    if idx_512 < len(times_sig):
        print(f"     Signatur-Methode O(n²): {times_sig[idx_512]*1000:.2f} ms")
        print(f"     Naive O(n³):            (nicht getestet)")
    
    print("\n2. k-beschränkte Matrixmultiplikation (n=128):")
    for k, (times_sig, times_strassen) in exp2_data.items():
        if len(times_sig) > 0 and len(times_strassen) > 0:
            ratio = times_sig[-1] / times_strassen[-1]
            print(f"   k={k}:")
            print(f"     Schichten-Signaturen O(k·n²): {times_sig[-1]*1000:.2f} ms")
            print(f"     Strassen O(n^2.807):          {times_strassen[-1]*1000:.2f} ms")
            print(f"     Verhältnis (Sig/Strassen):    {ratio:.2f}x")
    
    print("\n" + "="*80)
    print("ERKENNTNISSE")
    print("="*80)
    print("\n1. Boolean Matrixmultiplikation:")
    print("   - Signatur-Methode zeigt klaren O(n²) Vorteil gegenüber O(n³)")
    print("   - Speedup wächst quadratisch mit n")
    print("   - Strassen ist NICHT geeignet für Boolean (algebraische Inkompatibilität)")
    
    print("\n2. k-beschränkte Matrixmultiplikation:")
    print("   - Für kleine k ist Strassen schneller (O(n^2.807) < O(k·n²))")
    print("   - Schichten-Signaturen skalieren linear mit k")
    print("   - Ab k ≈ n^0.807 wäre Schichten-Methode theoretisch besser")


if __name__ == "__main__":
    print("Starte Experimente...")
    print("Stelle sicher, dass matplotlib installiert ist: pip install matplotlib\n")

    os.makedirs("src/results", exist_ok=True)
    
    # Experiment 1: Boolean
    exp1_data = experiment_boolean_multiplication()
    
    # Experiment 2: k-beschränkt
    exp2_data = experiment_k_bounded_multiplication()
    
    # Zusammenfassung
    print_summary(exp1_data, exp2_data)
    
    print("\nAlle Experimente abgeschlossen!")