"""Ready probes: the TS2Vec evaluation protocols (Yue et al., AAAI 2022).

This is the only TSLab package that imports scikit-learn.
"""

from tslab.probes.metrics import classification_metrics, forecasting_metrics
from tslab.probes.ridge import RidgeProbe, RidgeProbeSettings
from tslab.probes.svm import SVMProbe, SVMProbeSettings

__all__ = [
    'RidgeProbe',
    'RidgeProbeSettings',
    'SVMProbe',
    'SVMProbeSettings',
    'classification_metrics',
    'forecasting_metrics',
]
