"""
digi – Von der Diskretion
=========================
Python library derived from:
  "Von der Diskretion – Kontinuum, Digitalisierung und das Wunder der Membran"
  by Stephan Epp, March 2026.

Modules
-------
signals        : harmonic signals, Fourier transform, energy/power
sampling       : Nyquist-Shannon theorem, aliasing, sinc interpolation
quantization   : N-bit quantisation, quantisation error, SNR
acoustics      : acoustic pressure wave, sound pressure level (dB SPL)
mechanics      : spring-mass-damper membrane model
rlc            : RLC circuit analogue: impedance, resonance, Q, step response
neuron         : simplified Hodgkin-Huxley, action potential, threshold model
basilar        : basilar membrane: tonotopy, gammatone filterbank, cochlear amplifier
hearing        : HearingProfile (cat, dog), Shannon capacity, membrane efficiency Ω
information    : Shannon capacity, information content, thermal noise, Standard Quantum Limit
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("digi")
except PackageNotFoundError:
    __version__ = "dev"

__all__ = [
    "signals",
    "sampling",
    "quantization",
    "acoustics",
    "mechanics",
    "rlc",
    "neuron",
    "basilar",
    "hearing",
    "information",
]
