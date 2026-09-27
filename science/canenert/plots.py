#!/usr/bin/env python3
"""Matplotlib plots for CAN/CAN-FD energy-efficient real-time paper."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

# ── Plot 1: Bus-Auslastung vs. Energieverbrauch ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
load = np.linspace(0, 100, 300)
# CAN 1 Mbit/s static
e_can_static  = 0.05 + 0.0035 * load + 1e-5 * load**2
# CAN adaptive
e_can_adapt   = 0.03 + 0.0018 * load + 0.8e-5 * load**2
# CAN-FD static
e_canfd_static = 0.06 + 0.004  * load + 1.2e-5 * load**2
# CAN-FD adaptive
e_canfd_adapt  = 0.035 + 0.002 * load + 0.9e-5 * load**2

ax.plot(load, e_can_static,   'b-',  lw=2, label='CAN statisch')
ax.plot(load, e_can_adapt,    'b--', lw=2, label='CAN adaptiv (vorgeschlagen)')
ax.plot(load, e_canfd_static, 'r-',  lw=2, label='CAN-FD statisch')
ax.plot(load, e_canfd_adapt,  'r--', lw=2, label='CAN-FD adaptiv (vorgeschlagen)')
ax.set_xlabel('Bus-Auslastung (%)')
ax.set_ylabel('Energieverbrauch (mJ/Frame)')
ax.set_title('Energieverbrauch in Abhängigkeit der Bus-Auslastung')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0)
plt.tight_layout()
plt.savefig('plot1_energy_vs_load.pdf')
plt.close()
print("Plot 1 gespeichert.")

# ── Plot 2: Latenz-Garantie unter verschiedenen Schedulings ──────────────────
fig, ax = plt.subplots(figsize=(8, 5))
msgs = np.arange(1, 33)
# Worst-case latency CAN (bit-stuffing overhead ~20 %)
lat_can_static  = 1.0 + 0.12 * msgs + 0.003 * msgs**2
lat_can_adapt   = 0.8 + 0.09 * msgs + 0.002 * msgs**2
lat_canfd_static = 0.6 + 0.07 * msgs + 0.0015 * msgs**2
lat_canfd_adapt  = 0.5 + 0.055 * msgs + 0.001  * msgs**2
deadline = np.full_like(msgs, 5.0, dtype=float)

ax.plot(msgs, lat_can_static,   'b-',  lw=2, label='CAN statisch')
ax.plot(msgs, lat_can_adapt,    'b--', lw=2, label='CAN adaptiv')
ax.plot(msgs, lat_canfd_static, 'r-',  lw=2, label='CAN-FD statisch')
ax.plot(msgs, lat_canfd_adapt,  'r--', lw=2, label='CAN-FD adaptiv')
ax.plot(msgs, deadline,         'k:',  lw=2, label='Deadline 5 ms')
ax.fill_between(msgs, deadline, 10, alpha=0.08, color='red', label='Deadline-Verletzung')
ax.set_xlabel('Anzahl gleichzeitiger Nachrichten')
ax.set_ylabel('Worst-Case-Latenz (ms)')
ax.set_title('Worst-Case-Latenz vs. Nachrichtenanzahl')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 32)
ax.set_ylim(0, 10)
plt.tight_layout()
plt.savefig('plot2_latency.pdf')
plt.close()
print("Plot 2 gespeichert.")

# ── Plot 3: Monitoring-Overhead und Energieeinsparung ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
intervals_ms = np.array([0.5, 1, 2, 5, 10, 20, 50, 100])
# Monitoring-Overhead (% CPU)
overhead_cpu = 18 / intervals_ms + 0.5
# Energieeinsparung (%)
saving = 40 * (1 - np.exp(-0.05 * intervals_ms))

ax1, ax2 = axes
ax1.semilogx(intervals_ms, overhead_cpu, 'go-', lw=2, ms=7)
ax1.set_xlabel('Monitor-Intervall (ms)')
ax1.set_ylabel('CPU-Overhead (%)')
ax1.set_title('Monitoring-CPU-Overhead vs. Intervall')
ax1.grid(True, alpha=0.3, which='both')
ax1.set_ylim(0)

ax2.semilogx(intervals_ms, saving, 'ms-', lw=2, ms=7)
ax2.set_xlabel('Monitor-Intervall (ms)')
ax2.set_ylabel('Energieeinsparung (%)')
ax2.set_title('Energieeinsparung vs. Monitor-Intervall')
ax2.grid(True, alpha=0.3, which='both')
ax2.set_ylim(0, 45)

plt.tight_layout()
plt.savefig('plot3_monitoring.pdf')
plt.close()
print("Plot 3 gespeichert.")

# ── Plot 4: Vergleich Bitraten und Rahmeneffizienz ────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
payload = np.arange(1, 65)  # bytes
# CAN classic: max 8 byte payload, frame overhead ~47 bits
eff_can = np.where(payload <= 8, payload * 8 / (payload * 8 + 47) * 100, np.nan)
# CAN-FD: max 64 byte payload, frame overhead ~60 bits
eff_canfd = payload * 8 / (payload * 8 + 60) * 100

ax.plot(payload[:8],  eff_can[:8],   'b-o', lw=2, ms=4, label='CAN (max 8 Byte)')
ax.plot(payload,      eff_canfd,     'r-',  lw=2,        label='CAN-FD (max 64 Byte)')
ax.axvline(8,  color='blue', linestyle=':', alpha=0.5)
ax.axvline(64, color='red',  linestyle=':', alpha=0.5)
ax.set_xlabel('Nutzlast (Byte)')
ax.set_ylabel('Rahmeneffizienz (%)')
ax.set_title('Rahmeneffizienz: CAN vs. CAN-FD')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 64)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('plot4_frame_efficiency.pdf')
plt.close()
print("Plot 4 gespeichert.")

# ── Plot 5: Adaptives Scheduling – Zustands-Übergangsdiagramm als Simulation ──
fig, ax = plt.subplots(figsize=(9, 5))
t = np.linspace(0, 200, 2000)
# Simulate bus load
np.random.seed(42)
base = 30 + 20 * np.sin(2 * np.pi * t / 60)
noise = np.random.normal(0, 5, len(t))
bus_load = np.clip(base + noise, 0, 100)

# Adaptive bitrate selection
bitrate = np.where(bus_load < 40, 250,
          np.where(bus_load < 70, 500, 1000)).astype(float)
# CAN-FD boost above 80 %
bitrate_fd = np.where(bus_load < 40,  500,
             np.where(bus_load < 70, 2000, 5000)).astype(float)

ax2 = ax.twinx()
ax.plot(t, bus_load, 'k-', lw=1, alpha=0.6, label='Bus-Auslastung (%)')
ax2.step(t, bitrate,    'b-', lw=1.5, label='CAN Bitrate (kbit/s)')
ax2.step(t, bitrate_fd, 'r-', lw=1.5, label='CAN-FD Bitrate (kbit/s)', alpha=0.8)
ax.set_xlabel('Zeit (ms)')
ax.set_ylabel('Bus-Auslastung (%)', color='k')
ax2.set_ylabel('Adaptive Bitrate (kbit/s)')
ax.set_title('Adaptives Bitrate-Scheduling in Echtzeit')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('plot5_adaptive_scheduling.pdf')
plt.close()
print("Plot 5 gespeichert.")

# ── Plot 6: Energieprofil Ruhezustand vs. Aktiv ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
categories = ['Sleep\n(Idle)', 'Standby\n(Monitor)', 'CAN\naktiv', 'CAN-FD\naktiv',
              'CAN\nadaptiv', 'CAN-FD\nadaptiv']
power_mw = [0.5, 2.5, 85, 120, 52, 75]
colors = ['#2ecc71', '#27ae60', '#3498db', '#2980b9', '#e67e22', '#d35400']
bars = ax.bar(categories, power_mw, color=colors, edgecolor='black', linewidth=0.7)
for bar, val in zip(bars, power_mw):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val} mW', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Leistungsaufnahme (mW)')
ax.set_title('Leistungsaufnahme: Verschiedene Betriebsmodi')
ax.set_ylim(0, 140)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plot6_power_modes.pdf')
plt.close()
print("Plot 6 gespeichert.")

print("Alle Plots erfolgreich erstellt.")

