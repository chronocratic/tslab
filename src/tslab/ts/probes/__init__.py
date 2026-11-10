"""Ready probes: the TS2Vec evaluation protocols.

This is the only TSLab package that imports scikit-learn.
"""

from tslab.ts.probes.metrics import classification_metrics, forecasting_metrics
from tslab.ts.probes.ridge import RidgeProbe, RidgeProbeSettings
from tslab.ts.probes.svm import SVMProbe, SVMProbeSettings

__all__ = [
    'RidgeProbe',
    'RidgeProbeSettings',
    'SVMProbe',
    'SVMProbeSettings',
    'classification_metrics',
    'forecasting_metrics',
]
