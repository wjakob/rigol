"""
Bode plot measurement package for Rigol DHO924S oscilloscope.
"""

from .probe import FreqProbe
from .util import (
    fit_sine_at_freq,
    format_frequency,
    parse_si,
    rc_lowpass,
    rc_highpass,
    rlc_lowpass,
    rlc_highpass,
    lc_bandpass,
    lc_bandstop,
    run_bode_sweep,
    run_bode_sweep_live,
    save_to_csv,
    load_rigol_bode_csv,
    progress_printer,
)

__version__ = "1.0.0"
__all__ = [
    "FreqProbe",
    "fit_sine_at_freq",
    "format_frequency",
    "parse_si",
    "rc_lowpass",
    "rc_highpass",
    "rlc_lowpass",
    "rlc_highpass",
    "lc_bandpass",
    "lc_bandstop",
    "run_bode_sweep",
    "run_bode_sweep_live",
    "save_to_csv",
    "load_rigol_bode_csv",
    "progress_printer",
]
