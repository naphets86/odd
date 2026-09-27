"""
Acoustics: Biological Hearing and Cochlear A/D Conversion.

This package provides comprehensive models for:
  - Feline (cat) auditory system properties and comparisons with humans/dogs
  - Cochlear filtering and tonotopy based on Greenwood's model
  - Basilar membrane mechanics and frequency selectivity
  - A/D conversion with biologically-inspired filterbanks
  - Spatial localization (ITD, ILD) and hearing loss effects

References:
  - Greenwood, D. D. (1961). Critical bandwidth and the frequency coordinates
    of the basilar membrane. JASA, 33(10), 1344-1356.
  - Pickles, J. O. (2013). An Introduction to the Physiology of Hearing.
    4th ed., Emerald.
"""

__version__ = "1.0.0"
__author__ = "Acoustics Research Team"
__all__ = [
    "FelineAuditory",
    "CochlearModel",
    "BasilarMembrane",
    "GammaToneFilter",
    "CochlearFilterBank",
    "AudioConverter",
    "HearingComparison",
    "SpatialLocalization",
]

from .feline_auditory import FelineAuditory
from .cochlear_model import CochlearModel, BasilarMembrane
from .filterbank import GammaToneFilter, CochlearFilterBank
from .audio_converter import AudioConverter
from .hearing_comparison import HearingComparison
from .spatial_localization import SpatialLocalization

__all__ = [
    "FelineAuditory",
    "CochlearModel",
    "BasilarMembrane",
    "GammaToneFilter",
    "CochlearFilterBank",
    "AudioConverter",
    "HearingComparison",
    "SpatialLocalization",
]
