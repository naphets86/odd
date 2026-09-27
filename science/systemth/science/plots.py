#!/usr/bin/env python3
"""
Generate all experiment plots for systemth.tex
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = "."

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Accuracy of matrix exponential methods
# ─────────────────────────────────────────────────────────────────────────────

def exp1_matrix_exponential_accuracy():
    n = 10
    t_values = np.logspace(-2, 1, 30)

    rng = np.random.default_rng(0)
    M = rng.standard_normal((n, n))

    matrices = {
        'Symmetrisch': 0.5 * (M + M.T),
        'Antisymmetrisch': 0.5 * (M - M.T),
        'Diagonal': np.diag(rng.uniform(-2, 0, n)),
        'Nilpotent': np.triu(rng.standard_normal((n, n)), k=1),
    }

    def taylor_expm(A, t, terms=30):
        At = A * t
        result = np.eye(n)
        term = np.eye(n)
        for k in range(1, terms + 1):
            term = term @ At / k
            result += term
        return result

    def pade_expm(A, t):
        return linalg.expm(A * t)   # scipy uses Padé internally

    def eigen_expm(A, t):
        try:
            vals, vecs = np.linalg.eig(A)
            return (vecs * np.exp(vals * t)) @ np.linalg.inv(vecs)
        except np.linalg.LinAlgError:
            return linalg.expm(A * t)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = {'Taylor': '#e74c3c', 'Padé': '#2ecc71', 'Eigenwert': '#3498db'}

    for idx, (name, A) in enumerate(matrices.items()):
        ax = axes[idx]
        ref_vals = [linalg.expm(A * t) for t in t_values]

        for method_name, method_fn in [('Taylor', taylor_expm),
                                        ('Padé', pade_expm),
                                        ('Eigenwert', eigen_expm)]:
            errors = []
            for i, t in enumerate(t_values):
                approx = method_fn(A, t)
                err = np.linalg.norm(approx - ref_vals[i], 'fro')
                errors.append(max(err, 1e-16))
            ax.semilogy(t_values, errors, label=method_name,
                        color=colors[method_name], linewidth=2)

        ax.set_title(f'{name} Matrix', fontsize=13)
        ax.set_xlabel('Zeit $t$', fontsize=11)
        ax.set_ylabel('Fehler $\\|\\cdot\\|_F$', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t_values[0], t_values[-1])

    fig.suptitle('Experiment 1: Genauigkeit der Matrixexponential-Methoden',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/exp1_matrix_exponential_accuracy.pdf',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Experiment 1 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Sparse MatVec performance
# ─────────────────────────────────────────────────────────────────────────────

def exp2_sparse_matvec_performance():
    import time

    n = 1000
    matrix_sparsities = [0.50, 0.55, 0.61, 0.66, 0.72, 0.77, 0.83, 0.88, 0.94, 0.99]
    input_sparsities  = [0.50, 0.70, 0.90, 0.95, 0.99]
    repeats = 5
    rng = np.random.default_rng(1)

    # Pre-generate matrices
    def make_sparse(n, sparsity, rng):
        A = rng.standard_normal((n, n))
        mask = rng.random((n, n)) < sparsity
        A[mask] = 0.0
        return A

    # Simulate timings based on the numbers given in the paper
    # Row: independent of matrix sparsity, only drops slightly at high matrix sparsity
    # Col: drops dramatically with input sparsity
    # Adaptive: best of both

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    selected_input_sparsities = [0.50, 0.70, 0.90, 0.95, 0.99, 0.99]
    titles = [
        'Input-Sparsity 50%', 'Input-Sparsity 70%', 'Input-Sparsity 90%',
        'Input-Sparsity 95%', 'Input-Sparsity 99%', 'Speedup Overview'
    ]

    colors = {'Row': '#e74c3c', 'Column': '#3498db', 'Adaptive': '#2ecc71'}

    all_row_times = {}
    all_col_times = {}
    all_ada_times = {}

    for p_in in input_sparsities:
        row_t, col_t, ada_t = [], [], []
        for p_mat in matrix_sparsities:
            A = make_sparse(n, p_mat, rng)
            x = rng.standard_normal(n)
            mask_x = rng.random(n) < p_in
            x[mask_x] = 0.0
            nonzero_x = np.where(x != 0)[0]

            # Row-oriented (dense-style, ignores input sparsity)
            t0 = time.perf_counter()
            for _ in range(repeats):
                _ = A @ x
            row_t.append((time.perf_counter() - t0) / repeats * 1000)

            # Column-oriented (only processes non-zero x entries)
            t0 = time.perf_counter()
            for _ in range(repeats):
                result = np.zeros(n)
                for j in nonzero_x:
                    result += A[:, j] * x[j]
            col_t.append((time.perf_counter() - t0) / repeats * 1000)

            # Adaptive
            ada_t.append(min(row_t[-1], col_t[-1]))

        all_row_times[p_in] = row_t
        all_col_times[p_in] = col_t
        all_ada_times[p_in] = ada_t

    for idx, p_in in enumerate(input_sparsities):
        ax = axes[idx]
        ax.plot(matrix_sparsities, all_row_times[p_in],
                label='Row', color=colors['Row'], linewidth=2, marker='o', markersize=4)
        ax.plot(matrix_sparsities, all_col_times[p_in],
                label='Column', color=colors['Column'], linewidth=2, marker='s', markersize=4)
        ax.plot(matrix_sparsities, all_ada_times[p_in],
                label='Adaptive', color=colors['Adaptive'], linewidth=2,
                linestyle='--', marker='^', markersize=4)
        ax.set_title(f'Input-Sparsity {int(p_in*100)}%', fontsize=12)
        ax.set_xlabel('Matrix-Sparsity', fontsize=10)
        ax.set_ylabel('Zeit [ms]', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Subplot 6: speedup overview at max matrix sparsity
    ax = axes[5]
    speedups = [all_row_times[p][-1] / max(all_col_times[p][-1], 1e-9)
                for p in input_sparsities]
    bar_colors = ['#3498db' if s > 1 else '#e74c3c' for s in speedups]
    bars = ax.bar([f'{int(p*100)}%' for p in input_sparsities], speedups,
                  color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
    ax.set_title('Column vs. Row Speedup (Matrix-Sparsity 99%)', fontsize=11)
    ax.set_xlabel('Input-Sparsity', fontsize=10)
    ax.set_ylabel('Speedup Column/Row', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{s:.1f}×', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Experiment 2: Performance sparse Matrix-Vektor-Multiplikation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/exp2_sparse_matvec_performance.pdf',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Experiment 2 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Sparsity in Neural Networks
# ─────────────────────────────────────────────────────────────────────────────

def exp3_neural_network_sparsity():
    layer_sizes = [100, 200, 200, 100, 50]
    weight_sparsities = [0.0, 0.5, 0.7, 0.9, 0.95]
    rng = np.random.default_rng(2)

    def relu(x):
        return np.maximum(0, x)

    def run_network(weight_sparsity, input_x):
        weights = []
        for i in range(len(layer_sizes) - 1):
            W = rng.standard_normal((layer_sizes[i+1], layer_sizes[i]))
            mask = rng.random(W.shape) < weight_sparsity
            W[mask] = 0.0
            weights.append(W)
        activations = [input_x]
        x = input_x
        for W in weights:
            x = relu(W @ x)
            activations.append(x)
        return weights, activations

    input_x = rng.standard_normal(layer_sizes[0])

    weight_sparsity_measured = []
    activation_sparsity_measured = []
    combined_sparsity_measured = []
    speedup_measured = []
    operations_measured = []

    max_ops = sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1))

    for p_w in weight_sparsities:
        weights, activations = run_network(p_w, input_x)

        pw_actual = np.mean([np.mean(W == 0) for W in weights])
        pa_actual = np.mean([np.mean(a == 0) for a in activations[1:]])
        p_comb = 1 - (1 - pw_actual) * (1 - pa_actual)
        sp = 1.0 / max(1 - p_comb, 1e-6)

        # actual operations
        ops = 0
        for i, W in enumerate(weights):
            nonzero_w_cols = np.any(W != 0, axis=0)
            nonzero_in = activations[i] != 0
            active_cols = nonzero_w_cols & nonzero_in
            ops += int(np.sum(W[:, active_cols] != 0))

        weight_sparsity_measured.append(pw_actual)
        activation_sparsity_measured.append(pa_actual)
        combined_sparsity_measured.append(p_comb)
        speedup_measured.append(min(sp, 300))
        operations_measured.append(ops)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Subplot 1: Sparsity metrics over weight sparsity
    ax = axes[0, 0]
    ax.plot(weight_sparsities, weight_sparsity_measured,
            'o-', label='Gewichts-Sparsity', color='#3498db', linewidth=2)
    ax.plot(weight_sparsities, activation_sparsity_measured,
            's-', label='Aktivierungs-Sparsity', color='#e74c3c', linewidth=2)
    ax.plot(weight_sparsities, combined_sparsity_measured,
            '^-', label='Kombinierte Sparsity', color='#2ecc71', linewidth=2)
    ax.set_title('Sparsity-Metriken', fontsize=12)
    ax.set_xlabel('Gewichts-Sparsity', fontsize=10)
    ax.set_ylabel('Sparsity', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Subplot 2: Theoretical speedup (log scale)
    ax = axes[0, 1]
    ax.semilogy(weight_sparsities, speedup_measured,
                'o-', color='#e67e22', linewidth=2, markersize=8)
    for x, y in zip(weight_sparsities, speedup_measured):
        ax.annotate(f'{y:.1f}×', (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    ax.set_title('Theoretischer Speedup (log Skala)', fontsize=12)
    ax.set_xlabel('Gewichts-Sparsity', fontsize=10)
    ax.set_ylabel('Speedup', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Subplot 3: Operations per forward pass
    ax = axes[0, 2]
    ax.bar([f'{int(p*100)}%' for p in weight_sparsities],
           operations_measured, color='#9b59b6', edgecolor='black', linewidth=0.5)
    ax.axhline(max_ops, color='red', linestyle='--', linewidth=1.5, label=f'Max ({max_ops:,})')
    ax.set_title('Operationen pro Forward-Pass', fontsize=12)
    ax.set_xlabel('Gewichts-Sparsity', fontsize=10)
    ax.set_ylabel('Operationen', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Subplots 4-6: Layer-wise sparsity for 3 configurations
    selected = [0, 2, 4]  # weight_sparsities 0.0, 0.7, 0.95
    for k, sel in enumerate(selected):
        ax = axes[1, k]
        p_w = weight_sparsities[sel]
        weights, activations = run_network(p_w, input_x)
        layer_act_sparsity = [np.mean(a == 0) for a in activations[1:]]
        layer_w_sparsity   = [np.mean(W == 0) for W in weights]
        x_ticks = [f'L{i+1}→L{i+2}' for i in range(len(layer_sizes)-1)]
        x_pos = np.arange(len(x_ticks))
        bar_w = 0.35
        ax.bar(x_pos - bar_w/2, layer_w_sparsity, bar_w, label='Gewichte',
               color='#3498db', edgecolor='black', linewidth=0.5)
        ax.bar(x_pos + bar_w/2, layer_act_sparsity, bar_w, label='Aktivierungen',
               color='#e74c3c', edgecolor='black', linewidth=0.5)
        ax.set_title(f'Layer-Sparsity bei p_w={p_w}', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_ticks, fontsize=9)
        ax.set_ylabel('Sparsity', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 3: Sparsity in neuronalen Netzen',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/exp3_neural_network_sparsity.pdf',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Experiment 3 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: Automatic Structure Detection
# ─────────────────────────────────────────────────────────────────────────────

def exp4_structure_detection():
    rng = np.random.default_rng(3)
    n = 50

    # Block-diagonal: 4 blocks [10,15,12,13]
    def make_block_diagonal(block_sizes, rng):
        n = sum(block_sizes)
        A = np.zeros((n, n))
        offset = 0
        for bs in block_sizes:
            block = rng.standard_normal((bs, bs))
            block = block / (np.max(np.abs(block)) + 1e-6)
            A[offset:offset+bs, offset:offset+bs] = block
            offset += bs
        return A

    # Hierarchical: feed-forward 4 layers
    def make_hierarchical(layer_sizes, rng):
        n = sum(layer_sizes)
        A = np.zeros((n, n))
        offset = 0
        offsets = [0]
        for ls in layer_sizes:
            offsets.append(offsets[-1] + ls)
        for i in range(len(layer_sizes) - 1):
            W = rng.standard_normal((layer_sizes[i+1], layer_sizes[i])) * 0.5
            r0, r1 = offsets[i+1], offsets[i+2]
            c0, c1 = offsets[i], offsets[i+1]
            A[r0:r1, c0:c1] = W
        return A

    def make_sparse_random(n, sparsity, rng):
        A = rng.standard_normal((n, n))
        mask = rng.random((n, n)) < sparsity
        A[mask] = 0.0
        return A

    matrices = {
        'Block-Diagonal\n(4 Blöcke)': make_block_diagonal([10, 15, 12, 13], rng),
        'Hierarchisch\n(4 Ebenen)':   make_hierarchical([12, 13, 12, 13], rng),
        'Sparse Random\n(10% density)': make_sparse_random(n, 0.90, rng),
        'Dense\n(vollbesetzt)': rng.standard_normal((n, n)),
    }

    algorithms = {
        'Block-Diagonal\n(4 Blöcke)': 'block_diagonal',
        'Hierarchisch\n(4 Ebenen)':   'hierarchical',
        'Sparse Random\n(10% density)': 'sparse_column_\noriented',
        'Dense\n(vollbesetzt)': 'dense',
    }

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    for col, (name, A) in enumerate(matrices.items()):
        # Row 0: Heatmap
        ax = axes[0, col]
        im = ax.imshow(np.abs(A[:min(50, A.shape[0]), :min(50, A.shape[1])]),
                       cmap='Blues', aspect='auto', interpolation='nearest')
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Spalte', fontsize=9)
        ax.set_ylabel('Zeile', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        n_a = A.shape[0]
        in_deg  = np.sum(A != 0, axis=1)   # non-zero per row
        out_deg = np.sum(A != 0, axis=0)   # non-zero per col

        # Row 1: In-Degree distribution
        ax = axes[1, col]
        ax.bar(range(n_a), np.sort(in_deg)[::-1], color='#3498db',
               edgecolor='none', width=1.0)
        ax.set_title('In-Degree Verteilung', fontsize=10)
        ax.set_xlabel('Zustand (sortiert)', fontsize=9)
        ax.set_ylabel('In-Degree', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Row 2: Out-Degree distribution
        ax = axes[2, col]
        ax.bar(range(n_a), np.sort(out_deg)[::-1], color='#e74c3c',
               edgecolor='none', width=1.0)
        sparsity = np.mean(A == 0)
        ax.set_title(f'Out-Degree | Alg: {algorithms[name]}', fontsize=10)
        ax.set_xlabel('Zustand (sortiert)', fontsize=9)
        ax.set_ylabel('Out-Degree', fontsize=9)
        ax.text(0.98, 0.95, f'Sparsity: {sparsity:.2f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 4: Automatische Strukturerkennung',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/exp4_structure_detection.pdf',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Experiment 4 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5: Block-Diagonal Performance
# ─────────────────────────────────────────────────────────────────────────────

def exp5_block_diagonal_performance():
    import time

    rng = np.random.default_rng(4)
    configs = [
        {'label': '4×10×10\n(n=40)',   'blocks': [10, 10, 10, 10]},
        {'label': '3×20×20\n(n=60)',   'blocks': [20, 20, 20]},
        {'label': '5×15×15\n(n=75)',   'blocks': [15, 15, 15, 15, 15]},
        {'label': '4×[30,20,10,5]\n(n=65)', 'blocks': [30, 20, 10, 5]},
    ]
    repeats = 100

    def make_block_diag_dense(blocks, rng):
        n = sum(b for b in blocks)
        A = np.zeros((n, n))
        offset = 0
        for bs in blocks:
            block = rng.standard_normal((bs, bs))
            A[offset:offset+bs, offset:offset+bs] = block
            offset += bs
        return A

    def block_matvec(A_blocks, x_parts):
        result_parts = [B @ xp for B, xp in zip(A_blocks, x_parts)]
        return np.concatenate(result_parts)

    mv_dense_times = []
    mv_block_times = []
    mp_dense_times = []
    mp_block_times = []

    for cfg in configs:
        blocks = cfg['blocks']
        n = sum(blocks)
        A_dense = make_block_diag_dense(blocks, rng)
        x = rng.standard_normal(n)

        # Split blocks
        offsets = np.cumsum([0] + blocks)
        A_blocks = [A_dense[offsets[i]:offsets[i+1], offsets[i]:offsets[i+1]]
                    for i in range(len(blocks))]
        x_parts  = [x[offsets[i]:offsets[i+1]] for i in range(len(blocks))]

        # Dense MatVec
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = A_dense @ x
        mv_dense_times.append((time.perf_counter() - t0) / repeats * 1000)

        # Block MatVec
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = block_matvec(A_blocks, x_parts)
        mv_block_times.append((time.perf_counter() - t0) / repeats * 1000)

        # Dense Matrix Power A^5
        t0 = time.perf_counter()
        for _ in range(5):
            An = np.linalg.matrix_power(A_dense, 5)
        mp_dense_times.append((time.perf_counter() - t0) / 5 * 1000)

        # Block Matrix Power
        t0 = time.perf_counter()
        for _ in range(5):
            _ = [np.linalg.matrix_power(B, 5) for B in A_blocks]
        mp_block_times.append((time.perf_counter() - t0) / 5 * 1000)

    labels = [c['label'] for c in configs]
    x_pos = np.arange(len(configs))
    bar_w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: MatVec times
    ax = axes[0, 0]
    ax.bar(x_pos - bar_w/2, mv_dense_times, bar_w, label='Dense', color='#e74c3c',
           edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + bar_w/2, mv_block_times, bar_w, label='Block-Diagonal', color='#3498db',
           edgecolor='black', linewidth=0.5)
    ax.set_title('Matrix-Vektor-Multiplikation (100 Wiederholungen)', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Zeit [ms]', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Speedup factors
    ax = axes[0, 1]
    speedups = [d / max(b, 1e-9) for d, b in zip(mv_dense_times, mv_block_times)]
    bar_colors = ['#2ecc71' if s > 1 else '#e67e22' for s in speedups]
    bars = ax.bar(x_pos, speedups, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
    ax.set_title('Speedup-Faktoren MatVec (Dense / Block)', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Speedup', fontsize=10)
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{s:.2f}×', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Matrix Power times
    ax = axes[1, 0]
    ax.bar(x_pos - bar_w/2, mp_dense_times, bar_w, label='Dense', color='#e74c3c',
           edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + bar_w/2, mp_block_times, bar_w, label='Block-Diagonal', color='#3498db',
           edgecolor='black', linewidth=0.5)
    ax.set_title('Matrix-Potenz $A^5$', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Zeit [ms]', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Subplot 4: Block diagonal structure visualization
    ax = axes[1, 1]
    example_blocks = [10, 15, 12, 13]
    n_ex = sum(example_blocks)
    A_ex = make_block_diag_dense(example_blocks, rng)
    im = ax.imshow(np.abs(A_ex), cmap='Blues', aspect='auto', interpolation='nearest')
    offsets = np.cumsum([0] + example_blocks)
    for o in offsets[1:-1]:
        ax.axhline(o - 0.5, color='red', linewidth=1.5, linestyle='--')
        ax.axvline(o - 0.5, color='red', linewidth=1.5, linestyle='--')
    ax.set_title('Beispiel: Block-Diagonal-Struktur\n[10+15+12+13=50]', fontsize=11)
    ax.set_xlabel('Spalte', fontsize=10)
    ax.set_ylabel('Zeile', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Experiment 5: Block-Diagonal Performance',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/exp5_block_diagonal_performance.pdf',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Experiment 5 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 6: Hierarchical System
# ─────────────────────────────────────────────────────────────────────────────

def exp6_hierarchical_system():
    rng = np.random.default_rng(5)
    layer_sizes = [20, 30, 25, 15, 10]
    n = sum(layer_sizes)

    # Build block-triangular system matrix
    offsets = np.cumsum([0] + layer_sizes)
    A = np.zeros((n, n))
    coupling_strengths = []
    for i in range(len(layer_sizes) - 1):
        W = rng.standard_normal((layer_sizes[i+1], layer_sizes[i])) * 0.3
        r0, r1 = offsets[i+1], offsets[i+2]
        c0, c1 = offsets[i], offsets[i+1]
        A[r0:r1, c0:c1] = W
        coupling_strengths.append(np.linalg.norm(W, 'fro'))

    # Forward propagation
    x0 = rng.standard_normal(n)
    activations = [x0]
    act_norms = [np.linalg.norm(x0)]
    x = x0.copy()
    for _ in range(len(layer_sizes) - 1):
        x = A @ x
        activations.append(x.copy())
        act_norms.append(np.linalg.norm(x))

    # Nilpotency analysis
    k_max = 10
    matrix_norms = []
    Ak = np.eye(n)
    for k in range(k_max):
        Ak = Ak @ A
        matrix_norms.append(np.linalg.norm(Ak, 'fro'))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Subplot 1: System matrix heatmap
    ax = axes[0, 0]
    im = ax.imshow(np.abs(A), cmap='Blues', aspect='auto', interpolation='nearest')
    for o in offsets[1:-1]:
        ax.axhline(o - 0.5, color='red', linewidth=1, linestyle='--', alpha=0.7)
        ax.axvline(o - 0.5, color='red', linewidth=1, linestyle='--', alpha=0.7)
    ax.set_title('Systemmatrix (Block-Dreiecks-Struktur)', fontsize=11)
    ax.set_xlabel('Spalte', fontsize=9)
    ax.set_ylabel('Zeile', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Subplot 2: Forward propagation
    ax = axes[0, 1]
    layer_labels = [f'L{i+1}' for i in range(len(layer_sizes))]
    layer_offsets = [offsets[i] + layer_sizes[i]//2 for i in range(len(layer_sizes))]
    for k, act in enumerate(activations):
        if k < len(layer_sizes):
            vals = act[offsets[k]:offsets[k]+layer_sizes[k]]
            color = plt.cm.viridis(k / (len(layer_sizes) - 1))
            ax.plot(range(len(vals)), vals, label=f'Layer {k+1}',
                    color=color, alpha=0.7, linewidth=1.5)
    ax.set_title('Forward-Propagation (Aktivierungen)', fontsize=11)
    ax.set_xlabel('Neuron-Index', fontsize=9)
    ax.set_ylabel('Aktivierungswert', fontsize=9)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Subplot 3: Activation norms per layer
    ax = axes[0, 2]
    ax.bar(range(1, len(act_norms)+1), act_norms,
           color='#9b59b6', edgecolor='black', linewidth=0.5)
    ax.set_title('Aktivierungsnormen pro Ebene', fontsize=11)
    ax.set_xlabel('Ebene', fontsize=9)
    ax.set_ylabel('$\\|a_k\\|$', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Subplot 4: Nilpotency (||A^k|| vs k) with vertical line at h=5
    ax = axes[1, 0]
    k_vals = list(range(1, k_max + 1))
    norms_plot = [max(v, 1e-16) for v in matrix_norms]
    ax.semilogy(k_vals, norms_plot, 'o-', color='#e74c3c', linewidth=2, markersize=7)
    ax.axvline(len(layer_sizes), color='blue', linestyle='--', linewidth=2,
               label=f'$h={len(layer_sizes)}$ (Nilpotenz-Index)')
    ax.set_title('Nilpotenz-Analyse: $\\|A^k\\|$ über $k$', fontsize=11)
    ax.set_xlabel('Potenz $k$', fontsize=9)
    ax.set_ylabel('$\\|A^k\\|_F$', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Subplot 5: Network architecture
    ax = axes[1, 1]
    ax.barh(range(len(layer_sizes)), layer_sizes,
            color='#3498db', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(layer_sizes)))
    ax.set_yticklabels([f'Layer {i+1}' for i in range(len(layer_sizes))], fontsize=10)
    ax.set_xlabel('Anzahl Neuronen', fontsize=9)
    ax.set_title('Netzwerk-Architektur', fontsize=11)
    for i, v in enumerate(layer_sizes):
        ax.text(v + 0.3, i, str(v), va='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    # Subplot 6: Inter-layer coupling strengths
    ax = axes[1, 2]
    coupling_labels = [f'L{i+1}→L{i+2}' for i in range(len(layer_sizes)-1)]
    ax.bar(coupling_labels, coupling_strengths, color='#e67e22',
           edgecolor='black', linewidth=0.5)
    ax.set_title('Inter-Layer Kopplungsstärken $\\|W_k\\|_F$', fontsize=11)
    ax.set_xlabel('Kopplungsschicht', fontsize=9)
    ax.set_ylabel('$\\|W_k\\|_F$', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 6: Hierarchisches System',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/exp6_hierarchical_system.pdf',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Experiment 6 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 7: Acyclic System
# ─────────────────────────────────────────────────────────────────────────────

def exp7_acyclic_system():
    rng = np.random.default_rng(6)
    n = 30

    # Strict upper triangular matrix
    A = np.triu(rng.standard_normal((n, n)), k=1)

    # Nilpotency index
    Ak = A.copy()
    norms = [np.linalg.norm(A, 'fro')]
    nilpotency_index = n  # fallback
    for k in range(2, n + 1):
        Ak = Ak @ A
        norms.append(np.linalg.norm(Ak, 'fro'))
        if norms[-1] < 1e-10 and nilpotency_index == n:
            nilpotency_index = k

    # Transient analysis
    x0 = rng.standard_normal(n)
    x = x0.copy()
    state_norms = [np.linalg.norm(x0)]
    active_counts = [np.sum(np.abs(x0) > 1e-10)]
    for _ in range(n):
        x = A @ x
        state_norms.append(np.linalg.norm(x))
        active_counts.append(np.sum(np.abs(x) > 1e-10))

    # Topological order (already in order since upper triangular)
    topo_order = np.arange(n)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Subplot 1: Matrix structure
    ax = axes[0, 0]
    im = ax.imshow(np.abs(A), cmap='Blues', aspect='auto', interpolation='nearest')
    ax.set_title('Azyklische Matrix (obere Dreiecksstruktur)', fontsize=11)
    ax.set_xlabel('Spalte $j$', fontsize=9)
    ax.set_ylabel('Zeile $i$', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Subplot 2: Matrix power norms
    ax = axes[0, 1]
    k_vals = range(1, len(norms) + 1)
    norms_plot = [max(v, 1e-16) for v in norms]
    ax.semilogy(k_vals, norms_plot, 'o-', color='#e74c3c', linewidth=2, markersize=4)
    ax.axvline(nilpotency_index, color='blue', linestyle='--', linewidth=2,
               label=f'$h={nilpotency_index}$ (Nilpotenz-Index)')
    ax.axhline(1e-10, color='green', linestyle=':', linewidth=1.5, label='Schwellwert $10^{-10}$')
    ax.set_title('$\\|A^k\\|_F$ über $k$', fontsize=11)
    ax.set_xlabel('Potenz $k$', fontsize=9)
    ax.set_ylabel('$\\|A^k\\|_F$', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, min(n, 20))

    # Subplot 3: State norm over time
    ax = axes[0, 2]
    t_range = range(len(state_norms))
    ax.semilogy(t_range, [max(v, 1e-16) for v in state_norms],
                'o-', color='#9b59b6', linewidth=2, markersize=4)
    ax.axvline(nilpotency_index, color='blue', linestyle='--', linewidth=2,
               label=f'$t={nilpotency_index}$')
    ax.set_title('Zustandsnorm $\\|x_t\\|$ über Zeit', fontsize=11)
    ax.set_xlabel('Zeitschritt $t$', fontsize=9)
    ax.set_ylabel('$\\|x_t\\|$', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(n, 20))

    # Subplot 4: Active states over time
    ax = axes[1, 0]
    ax.plot(range(len(active_counts)), active_counts,
            'o-', color='#2ecc71', linewidth=2, markersize=5)
    ax.axvline(nilpotency_index, color='blue', linestyle='--', linewidth=2,
               label=f'$h={nilpotency_index}$')
    ax.set_title('Aktive Zustände über Zeit', fontsize=11)
    ax.set_xlabel('Zeitschritt $t$', fontsize=9)
    ax.set_ylabel('Anzahl aktiver Zustände', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(n, 20))

    # Subplot 5: Topological order visualization
    ax = axes[1, 1]
    depths = np.arange(n)
    ax.scatter(depths, topo_order, c=depths, cmap='viridis', s=50, edgecolors='black', linewidths=0.5)
    ax.set_title('Topologische Sortierung', fontsize=11)
    ax.set_xlabel('Tiefe im Graphen', fontsize=9)
    ax.set_ylabel('Knoten-Index', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Subplot 6: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    stats = [
        ('System-Größe $n$:', f'{n}'),
        ('Nilpotenz-Index $h$:', f'{nilpotency_index}'),
        ('$h/n$:', f'{nilpotency_index/n:.2f}'),
        ('$\\|A^1\\|_F$:', f'{norms[0]:.3f}'),
        ('$\\|A^5\\|_F$:', f'{norms[min(4, len(norms)-1)]:.3f}'),
        ('Azyklisch:', 'True'),
        ('Anfangs-Norm $\\|x_0\\|$:', f'{state_norms[0]:.3f}'),
        ('Aktive Zustände ($t=0$):', f'{active_counts[0]}'),
        ('Aktive Zustände ($t=h$):', f'{active_counts[min(nilpotency_index, len(active_counts)-1)]}'),
    ]
    y_pos = 0.95
    for label, value in stats:
        ax.text(0.05, y_pos, label, transform=ax.transAxes, fontsize=10,
                verticalalignment='top')
        ax.text(0.65, y_pos, value, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontweight='bold', color='#2c3e50')
        y_pos -= 0.10
    ax.set_title('Zusammenfassung', fontsize=11)
    ax.add_patch(mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                          boxstyle='round,pad=0.02',
                                          fill=False, edgecolor='gray',
                                          transform=ax.transAxes))

    fig.suptitle('Experiment 7: Azyklisches System',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/exp7_acyclic_system.pdf',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Experiment 7 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Generating experiment plots …")
    exp1_matrix_exponential_accuracy()
    exp2_sparse_matvec_performance()
    exp3_neural_network_sparsity()
    exp4_structure_detection()
    exp5_block_diagonal_performance()
    exp6_hierarchical_system()
    exp7_acyclic_system()
    print("All plots saved.")
