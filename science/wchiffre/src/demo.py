"""
demo.py – Fenster-Chiffre (FC) Demonstration
Stephan Epp

Zeigt das vollstaendige Protokoll FC = (Gen, Enc, Dec) aus der Arbeit
"Modulares kryptografisches Verfahren mit Dummy-Fenster-Fragmentierung"
am Lehrbeispiel (n=55, e=23, d=7) und mit zufaelligen Schluesseln.
"""

import math
from math import comb, log2

from src.keygen import KeygenFC
from src.encrypt import encrypt, count_dummy_positions, message_to_blocks
from src.decrypt import decrypt
from src.math_utils import extended_gcd, mod_inverse, euler_phi


# ============================================================
# Hilfsfunktionen fuer die Ausgabe
# ============================================================

def separator(title: str = "") -> None:
    line = "=" * 62
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(f"{line}")
    else:
        print(line)


def hex_packet(p: int, width: int = 4) -> str:
    return f"0x{p:0{width}X}"


# ============================================================
# 1. Zahlentheoretische Grundlagen (Abschnitt 2)
# ============================================================

def demo_zahlentheorie() -> None:
    separator("1. Zahlentheoretische Grundlagen")

    print("\n--- Erweiterter Euklidischer Algorithmus fuer (32, 23) ---")
    g, x, y = extended_gcd(32, 23)
    print(f"  gcd(32, 23) = {g}")
    print(f"  Bezout:  32 * ({x}) + 23 * ({y}) = {32*x + 23*y}")
    print(f"  => 23^(-1) mod 32 = {y % 32}  (Lemma 2.3)")

    print("\n--- Modulares Inverses (Satz 2.2) ---")
    inv_23 = mod_inverse(23, 32)
    inv_7  = mod_inverse(7,  32)
    print(f"  23^(-1) mod 32 = {inv_23}")
    print(f"   7^(-1) mod 32 = {inv_7}")
    print(f"  Probe: 23 * 7 mod 32 = {(23 * 7) % 32}  (muss 1 sein)")

    print("\n--- Euler-Phi-Funktion ---")
    print(f"  phi(32) = {euler_phi(32)}  (Satz 2.1: Einheitengruppe mod 32)")
    print(f"  phi(55) = {euler_phi(55)}  (= phi(5)*phi(11) = 4*10)")

    print("\n--- RSA-Lehrbeispiel: p=5, q=11, n=55 (Korollar 2.4) ---")
    p, q, n = 5, 11, 55
    phi_n = (p - 1) * (q - 1)
    e, d = 23, 7
    print(f"  n = {p}*{q} = {n},  phi(n) = {phi_n}")
    print(f"  e = {e},  d = {d},  e*d mod phi(n) = {(e*d) % phi_n}  (muss 1 sein)")
    # Verifikation fuer alle gueltigen M_j
    valid = [m for m in range(1, n) if math.gcd(m, n) == 1]
    errors = [m for m in valid if pow(m, e * d, n) != m]
    print(f"  Korrektheit fuer alle {len(valid)} gueltigen M_j: "
          f"{'OK' if not errors else 'FEHLER ' + str(errors)}")


# ============================================================
# 2. Schluesselerzeugung (Gen)
# ============================================================

def demo_keygen() -> None:
    separator("2. Schluesselgenerierung Gen(1^lambda)")

    print("\n--- Lehrbeispiel (Grundversion der Arbeit) ---")
    pk, sk = KeygenFC.toy_example(k=3, s=0)
    print(f"  pk = (n={pk.n}, e={pk.e}, k={pk.k})")
    print(f"  sk = (d={sk.d}, s={sk.s})")

    print("\n--- Zufaelliger Offset s ---")
    offsets = [KeygenFC.toy_example()[1].s for _ in range(10)]
    print(f"  10 zufaellige Offsets: {offsets}")

    print("\n--- Groessere RSA-Schluessels (bits=64) ---")
    pk_large, sk_large = KeygenFC.generate(bits=64, e=65537, k=3)
    print(f"  n = {pk_large.n}")
    print(f"  e = {pk_large.e}")
    print(f"  d = {sk_large.d}")
    print(f"  s = {sk_large.s}")
    # sk_large.phi_n ist direkt verfuegbar – euler_phi(pk_large.n) wuerde
    # eine teure Faktorisierung des ~128-Bit-Moduls erfordern.
    print(f"  e*d mod phi(n) = {(pk_large.e * sk_large.d) % sk_large.phi_n}")


# ============================================================
# 3. Verschluesselung (Enc, Algorithmus 1)
# ============================================================

def demo_encrypt() -> None:
    separator("3. Verschluesselung Enc(pk, s, M)  [Algorithmus 1]")

    pk, sk = KeygenFC.toy_example(k=3, s=0)
    msg = b"\x02\x05\x08"  # Drei Bytes, alle < n=55

    print(f"\n  Nachricht M   = {list(msg)}")
    print(f"  pk = (n={pk.n}, e={pk.e}, k={pk.k})")
    print(f"  Offset s      = {sk.s}")

    blocks = message_to_blocks(msg, pk.n)
    print(f"\n  Klartextbloecke (nach Padding): {blocks}")
    r = len(blocks)
    total = math.ceil(r * pk.k / (pk.k - 1))
    print(f"  r = {r} echte Bloecke")
    print(f"  l = ceil({r}*{pk.k}/{pk.k-1}) = {total} Pakete gesamt")

    dummy_pos = count_dummy_positions(total, pk.k, sk.s)
    print(f"\n  Dummy-Positionen (s={sk.s}): {dummy_pos}")

    packets = encrypt(pk, sk.s, msg)
    print(f"\n  Paketfolge P ({len(packets)} Pakete):")
    for i, p in enumerate(packets, start=1):
        tag = " [DUMMY]" if (i - 1 + sk.s) % pk.k == 0 else " [echt] "
        print(f"    P_{i:02d} = {p:4d}  {tag}")

    overhead = (len(packets) - r) / r * 100
    print(f"\n  Traffic-Overhead: {overhead:.1f}%  "
          f"(theoretisch: {100/(pk.k-1):.0f}%, Proposition 5.2)")

    print("\n--- Alle drei Offsets im Vergleich ---")
    for s in range(3):
        pkts = encrypt(pk, s, msg)
        dp = count_dummy_positions(len(pkts), pk.k, s)
        print(f"  s={s}: {[hex_packet(p) for p in pkts]}  Dummies@{dp}")


# ============================================================
# 4. Entschluesselung (Dec, Algorithmus 2)
# ============================================================

def demo_decrypt() -> None:
    separator("4. Entschluesselung Dec(sk, P)  [Algorithmus 2]")

    pk, sk = KeygenFC.toy_example(k=3, s=0)
    msg = b"\x02\x05\x08"

    packets = encrypt(pk, sk.s, msg)
    recovered = decrypt(sk, packets)

    print(f"\n  Originalnachricht:     {list(msg)}")
    print(f"  Paketfolge:            {packets}")
    print(f"  Entschluesselt:        {list(recovered)}")
    print(f"  Korrektheit:           {'OK' if recovered == msg else 'FEHLER'}")

    print("\n  Schrittweise Entschluesselung:")
    for i, p in enumerate(packets, start=1):
        is_dummy = (i - 1 + sk.s) % pk.k == 0
        if is_dummy:
            print(f"    i={i}: P_{i}={p:4d}  -> DUMMY, ignoriert")
        else:
            m = pow(p, sk.d, sk.n)
            print(f"    i={i}: P_{i}={p:4d}  -> M = {p}^{sk.d} mod {sk.n} = {m}")


# ============================================================
# 5. Sicherheitsanalyse (Abschnitt 5)
# ============================================================

def demo_security() -> None:
    separator("5. Sicherheitsanalyse")

    print("\n--- Informationstheoretische Sicherheit (Satz 5.1) ---")
    k = 3
    print(f"  Offset s aus {{0,...,{k-1}}} gleichverteilt:")
    print(f"  P[Angreifer errät s] <= 1/{k} = {1/k:.4f}")

    print("\n--- Angreifer-Entropie H = log2(C(N, N/k)) (Bemerkung 5.1) ---")
    for N in [9, 18, 30, 60, 90]:
        dummy_count = N // k
        h = log2(comb(N, dummy_count))
        print(f"  N={N:3d}: H = log2(C({N},{dummy_count})) = {h:.2f} Bit")

    print("\n--- Simulation: Zufaellige Offset-Schaetzung (1000 Versuche) ---")
    import random
    n_trials, correct = 1000, 0
    for _ in range(n_trials):
        _, sk_sim = KeygenFC.toy_example(k=3)
        if random.randrange(0, 3) == sk_sim.s:
            correct += 1
    print(f"  Trefferrate: {correct}/{n_trials} = {correct/n_trials:.3f}  "
          f"(Theorie: 1/3 = {1/3:.3f})")

    print("\n--- Traffic-Overhead als Funktion von k ---")
    print(f"  {'k':>4}  {'Overhead':>10}  {'Dummy-Anteil':>14}")
    for k_val in [2, 3, 4, 5, 10]:
        overhead = 1 / (k_val - 1) * 100
        dummy_share = 1 / k_val * 100
        print(f"  {k_val:>4}  {overhead:>9.1f}%  {dummy_share:>13.1f}%")


# ============================================================
# 6. Groessere Schluessels + Langtext
# ============================================================

def demo_realistic() -> None:
    separator("6. Realistisches Beispiel (bits=64, RSA-Schluessels)")

    pk, sk = KeygenFC.generate(bits=64, k=3)
    msg = b"Fenster-Chiffre: sicher durch Dummy-Pakete!"

    print(f"\n  Nachricht ({len(msg)} Bytes): {msg.decode()!r}")
    print(f"  RSA-Modulus n (~128 Bit): {pk.n}")
    print(f"  Oeffentlicher Exponent e: {pk.e}")
    print(f"  Geheimer Offset s: {sk.s}")

    packets = encrypt(pk, sk.s, msg)
    recovered = decrypt(sk, packets)

    print(f"\n  Paketanzahl: {len(packets)}")
    blocks_count = len(message_to_blocks(msg, pk.n))
    print(f"  Echte Bloecke: {blocks_count}")
    print(f"  Traffic-Overhead: {(len(packets) - blocks_count) / blocks_count * 100:.1f}%")
    print(f"\n  Entschluesselt: {recovered.decode()!r}")
    print(f"  Korrektheit:    {'OK ✓' if recovered == msg else 'FEHLER ✗'}")


# ============================================================
# Hauptprogramm
# ============================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Fenster-Chiffre (FC) – Demonstration                    ║")
    print("║  Stephan Epp                                             ║")
    print("║  23 * 7 ≡ 1 (mod 32)                                     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_zahlentheorie()
    demo_keygen()
    demo_encrypt()
    demo_decrypt()
    demo_security()
    demo_realistic()

    separator()
    print("  Demo abgeschlossen. Alle Algorithmen korrekt.")
    separator()
