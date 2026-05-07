"""Calibrated Rouse-model parameter sets for the two frame intervals in use.

Both sets come from a joint calibration on Harvey's S+V control. They are
fixed inputs to the simulation/BILD pipeline and are keyed by the frame
interval ``delta_t`` (in seconds).
"""
import numpy as np


PARAM_SETS = {
    5: {
        'L': 16,
        'k': 1.34,
        'D': 0.00199,
        'L_looped': 0.334,
        'localization_error': np.array([0.0425, 0.0425, 0.0449]),
    },
    30: {
        'L': 16,
        'k': 8.04,
        'D': 0.01194,
        'L_looped': 0.334,
        'localization_error': np.array([0.0443, 0.0443, 0.0444]),
    },
}


def get_params(delta_t):
    """Return the parameter dict for the given frame interval (in seconds)."""
    if delta_t not in PARAM_SETS:
        raise KeyError(f'No parameter set for delta_t={delta_t} s; '
                       f'known intervals: {sorted(PARAM_SETS)}')
    return PARAM_SETS[delta_t]
