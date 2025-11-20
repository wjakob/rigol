"""
Utility functions for Bode plot analysis and visualization.
"""

import re
import sys
from typing import Tuple, List, Optional, Dict, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def _format_frequency_tick(value, pos):
    """Format frequency tick labels in Hz/KHz/MHz."""
    if value >= 1e6:
        return f'{value/1e6:.0f} MHz' if value >= 10e6 else f'{value/1e6:.1f} MHz'
    elif value >= 1e3:
        return f'{value/1e3:.0f} KHz' if value >= 10e3 else f'{value/1e3:.1f} KHz'
    else:
        return f'{value:.0f} Hz'


def format_frequency(value: float) -> str:
    """Format frequency value with appropriate units (Hz/KHz/MHz)."""
    if value >= 1e6:
        return f'{value/1e6:.3f} MHz' if value < 100e6 else f'{value/1e6:.2f} MHz'
    elif value >= 1e3:
        return f'{value/1e3:.3f} KHz' if value < 100e3 else f'{value/1e3:.2f} KHz'
    else:
        return f'{value:.2f} Hz'


def parse_time(value: str) -> float:
    """
    Parse a time string with SI prefixes into seconds.

    Supports time units:
    - s = seconds
    - ms = milliseconds (10^-3)
    - us/µs = microseconds (10^-6)
    - ns = nanoseconds (10^-9)
    - ps = picoseconds (10^-12)

    Examples
    --------
    >>> parse_time('1.83ns')
    1.83e-9
    >>> parse_time('-2.5ns')
    -2.5e-9
    >>> parse_time('100ps')
    1e-10
    """
    import re

    original_value = value
    value = value.strip()

    # Match: optional sign, number, optional space, time unit
    match = re.match(r'^([+-]?[\d.]+)\s*(ps|ns|us|µs|ms|s)$', value, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid time format: {original_value}")

    number = float(match.group(1))
    unit = match.group(2).lower()

    # Time unit multipliers
    multipliers = {
        'ps': 1e-12,
        'ns': 1e-9,
        'us': 1e-6,
        'µs': 1e-6,
        'ms': 1e-3,
        's': 1.0,
    }

    return number * multipliers[unit]


def parse_si(value: str, unit: str = 'Hz') -> float:
    """
    Parse a string with SI prefixes into a numeric value.

    Supports standard SI prefixes with case sensitivity:
    - m = milli (10^-3)
    - k/K = kilo (10^3)
    - M = mega (10^6)
    - G = giga (10^9)
    - T = tera (10^12)

    Parameters
    ----------
    value : str
        String to parse, e.g., '10KHz', '5V', '10mV', '1.5MHz'
    unit : str, optional
        Expected unit ('Hz', 'V', etc.). Default is 'Hz'.
        Used to validate the input.

    Returns
    -------
    float
        Numeric value in base units (Hz for frequency, V for voltage, etc.)

    Examples
    --------
    >>> parse_si('10KHz', unit='Hz')
    10000.0
    >>> parse_si('1.5MHz', unit='Hz')
    1500000.0
    >>> parse_si('10mV', unit='V')
    0.01
    >>> parse_si('5V', unit='V')
    5.0
    >>> parse_si('10K', unit='Hz')  # Unit suffix optional
    10000.0
    >>> parse_si('10k', unit='Hz')  # Lowercase k also works
    10000.0
    """
    original_value = value
    value = value.strip()

    # Match number (with optional decimal) followed by optional SI prefix and unit
    # Case-sensitive for prefix to distinguish m (milli) from M (mega)
    match = re.match(r'^([\d.]+)\s*([mkKMGT]?)([a-zA-Z]+)?$', value)
    if not match:
        raise ValueError(f"Invalid format: {original_value}")

    number = float(match.group(1))
    prefix = match.group(2)  # Keep case-sensitive
    found_unit = match.group(3) if match.group(3) else None

    # Normalize unit comparison (case-insensitive)
    if found_unit and found_unit.upper() != unit.upper():
        raise ValueError(f"Expected unit '{unit}' but found '{found_unit}' in: {original_value}")

    # SI prefix multipliers (case-sensitive)
    multipliers = {
        '': 1,
        'm': 1e-3,   # milli
        'k': 1e3,    # kilo (lowercase)
        'K': 1e3,    # kilo (uppercase)
        'M': 1e6,    # mega
        'G': 1e9,    # giga
        'T': 1e12,   # tera
    }

    return number * multipliers.get(prefix, 1)


def generate_frequencies_per_decade(f_min: float, f_max: float, steps_per_decade: int) -> np.ndarray:
    """
    Generate logarithmically-spaced frequencies with a specified number of steps per decade.

    This matches the Rigol oscilloscope behavior where frequencies are distributed
    evenly on a logarithmic scale within each decade.

    Parameters
    ----------
    f_min : float
        Start frequency in Hz
    f_max : float
        End frequency in Hz
    steps_per_decade : int
        Number of frequency points per decade (e.g., 10)

    Returns
    -------
    np.ndarray
        Array of frequencies logarithmically spaced with steps_per_decade points per decade

    Examples
    --------
    >>> freqs = generate_frequencies_per_decade(1000, 10000, 10)
    >>> len(freqs)  # 1 decade with 10 steps per decade
    11
    >>> freqs = generate_frequencies_per_decade(1000, 100000, 10)
    >>> len(freqs)  # 2 decades with 10 steps per decade
    21
    """
    # Calculate the number of decades
    log_min = np.log10(f_min)
    log_max = np.log10(f_max)
    num_decades = log_max - log_min

    # Total number of steps is steps_per_decade × num_decades + 1 (to include endpoint)
    # We use linspace to ensure we hit both endpoints exactly
    total_steps = int(np.ceil(steps_per_decade * num_decades)) + 1

    # Generate logarithmically-spaced frequencies
    freqs = np.logspace(log_min, log_max, total_steps)

    return freqs


def fit_sine_at_freq(sig, freq, dt):
    """
    Fit a sine at known frequency `freq` to the signal `sig`.

    sig : 1D array of samples
    freq: excitation frequency in Hz
    dt  : sample spacing (seconds) OR full time array (same length as sig)
    """
    sig = np.asarray(sig)
    N = len(sig)

    if np.isscalar(dt):
        t = np.arange(N) * dt
    else:
        t = np.asarray(dt)
        if t.shape != sig.shape:
            raise ValueError("dt as time array must have same shape as signal")

    w = 2 * np.pi * freq
    # s[n] ≈ a*cos(w t_n) + b*sin(w t_n)
    A = np.column_stack([np.cos(w * t), np.sin(w * t)])
    coeffs, *_ = np.linalg.lstsq(A, sig, rcond=None)
    a, b = coeffs

    amplitude = np.hypot(a, b)      # sqrt(a^2 + b^2)
    phase = np.arctan2(a, b)        # radians, relative to sine
    return amplitude, phase


def rc_lowpass(fc: float):
    """
    Create a callable for a 1st-order RC lowpass filter.

    Parameters
    ----------
    fc : float
        Cutoff frequency (-3dB point) in Hz

    Returns
    -------
    callable
        Function that takes frequency array and returns (gain_db, phase_deg)
    """
    def func(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = f / fc
        # Low pass: |H(jω)| = 1 / sqrt(1 + (f/fc)^2)
        gain = 1.0 / np.sqrt(1.0 + x**2)
        gain_db = 20 * np.log10(gain)
        # φ = -atan(f/fc), ranges from 0° to -90°
        phase_deg = -np.degrees(np.arctan(x))
        return gain_db, phase_deg
    return func


def rc_highpass(fc: float):
    """
    Create a callable for a 1st-order RC highpass filter.

    Parameters
    ----------
    fc : float
        Cutoff frequency (-3dB point) in Hz

    Returns
    -------
    callable
        Function that takes frequency array and returns (gain_db, phase_deg)
    """
    def func(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = f / fc
        # High-pass RC: H(jω) = jωRC / (1 + jωRC)
        # |H| = (f/fc) / sqrt(1 + (f/fc)^2)
        gain = x / np.sqrt(1.0 + x**2)
        gain_db = 20 * np.log10(gain)
        # φ = 90° - atan(f/fc), ranges from 90° to 0°
        # Convert to negative convention: ranges from -90° to 0° by subtracting 180°
        # Actually, for highpass: at low f, phase = 90°, at high f, phase = 0°
        # In negative convention: at low f, phase should be -270° = 90°
        # Let's keep it as is for now, it's already correct
        phase_deg = 90.0 - np.degrees(np.arctan(x))
        return gain_db, phase_deg
    return func


def rlc_lowpass(fc: float, r: float = 0.0):
    """
    Create a callable for a 2nd-order RLC lowpass filter.

    Parameters
    ----------
    fc : float
        Resonant frequency in Hz (f₀ = 1/(2π√LC))
    r : float
        Series resistance in Ohms (default: 0.0 for ideal LC)

    Returns
    -------
    callable
        Function that takes frequency array and returns (gain_db, phase_deg)

    Notes
    -----
    Models an RLC lowpass filter (L and R in series, C to ground).
    Transfer function: H(jω) = 1 / (1 - ω²LC + jωRC)

    Phase transitions smoothly from 0° (low freq) to -180° (high freq).
    At resonance: phase = -90°

    The quality factor Q = ω₀L/R determines damping.
    """
    def func(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        omega = 2 * np.pi * f
        omega_0 = 2 * np.pi * fc

        # Normalized frequency
        x = f / fc

        # For the phase calculation, we need the impedance ratio
        # H(jω) = 1 / (1 - (ω/ω₀)² + jω(R/ωL))
        # We need to know R/(ω₀L), which is 1/Q
        # From ω₀ = 1/√(LC), we get L = 1/(ω₀²C)
        # But we don't know L and C individually, only their product via fc

        # Let's use a different approach: quality factor Q
        # For small R: Q ≈ ω₀L/R is large (underdamped)
        # We can parameterize by Q = 1/(R√(C/L)) = √(L/C)/R
        # But we need to know Z₀ = √(L/C) - the characteristic impedance

        # Assume a reasonable characteristic impedance, say 50Ω if R not specified
        # Or we can make Q a function of R and fc
        # Let's use: if R=0, Q=1000 (very underdamped), otherwise Q = Z₀/R with Z₀=50Ω

        if r == 0:
            Q = 1000.0  # Very high Q for ideal LC
        else:
            Z0 = 50.0  # Assume 50Ω characteristic impedance
            Q = Z0 / r

        # Denominator: 1 - x² + jx/Q
        real_part = 1.0 - x**2
        imag_part = x / Q

        # Magnitude
        gain = 1.0 / np.sqrt(real_part**2 + imag_part**2)
        gain_db = 20 * np.log10(gain)

        # Phase: -arctan(imag/real) = -arctan((x/Q) / (1 - x²))
        phase_deg = -np.degrees(np.arctan2(imag_part, real_part))

        return gain_db, phase_deg
    return func


def rlc_highpass(fc: float, r: float = 0.0):
    """
    Create a callable for a 2nd-order RLC highpass filter.

    Parameters
    ----------
    fc : float
        Resonant frequency in Hz (f₀ = 1/(2π√LC))
    r : float
        Series resistance in Ohms (default: 0.0 for ideal LC)

    Returns
    -------
    callable
        Function that takes frequency array and returns (gain_db, phase_deg)

    Notes
    -----
    Models an RLC highpass filter (C and R in series, L to ground).
    Transfer function: H(jω) = -ω²LC / (1 - ω²LC + jωRC)

    Phase transitions smoothly from -180° (low freq) to 0° (high freq).
    At resonance: phase = -90°

    The quality factor Q = ω₀L/R determines damping.
    """
    def func(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        omega = 2 * np.pi * f
        omega_0 = 2 * np.pi * fc

        # Normalized frequency
        x = f / fc

        if r == 0:
            Q = 1000.0  # Very high Q for ideal LC
        else:
            Z0 = 50.0  # Assume 50Ω characteristic impedance
            Q = Z0 / r

        # Numerator: -x²
        # Denominator: 1 - x² + jx/Q
        real_part = 1.0 - x**2
        imag_part = x / Q

        # Magnitude: |numerator|/|denominator| = x² / sqrt(real² + imag²)
        gain = x**2 / np.sqrt(real_part**2 + imag_part**2)
        gain_db = 20 * np.log10(gain)

        # Phase: arg(-x²) - arg(1 - x² + jx/Q)
        # arg(-x²) = -180° (negative real number)
        # arg(denominator) = arctan2(imag, real)
        phase_numerator = -180.0  # Phase of -x²
        phase_denominator = np.degrees(np.arctan2(imag_part, real_part))
        phase_deg = phase_numerator - phase_denominator

        # Wrap phase to [-180, 180] range
        phase_deg = ((phase_deg + 180.0) % 360.0) - 180.0

        return gain_db, phase_deg
    return func


def run_bode_sweep(probe, freqs: np.ndarray,
                  progress_callback=None, quiet: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep probe.measure_dynamic() across specified frequencies,
    compute gain (dB) and phase shift (deg).

    Parameters
    ----------
    probe : FreqProbe
        Probe instance to use for measurements
    freqs : np.ndarray
        Array of frequencies to measure (in Hz)
    progress_callback : callable, optional
        Function(current, total) called for each frequency
    quiet : bool
        Suppress measurement output (default: False)

    Returns
    -------
    freqs : np.ndarray
        Frequency points measured (same as input)
    gain_db : np.ndarray
        Gain in dB at each frequency
    phase_deg : np.ndarray
        Phase in degrees at each frequency
    """
    gains = np.zeros_like(freqs)
    phases = np.zeros_like(freqs)

    for i, f in enumerate(freqs):
        # Measure with dynamic range adjustment
        vin, vout, dt = probe.measure_dynamic(freq_hz=f)

        # Fit sine waves to extract amplitude and phase
        Ain,  phiin  = fit_sine_at_freq(vin,  f, dt)
        Aout, phiout = fit_sine_at_freq(vout, f, dt)

        # Store gain and phase
        gains[i]  = Aout / Ain
        phases[i] = phiout - phiin   # radians

        # Calculate and display current measurement
        gain_db = 20 * np.log10(gains[i])
        phase_deg = np.degrees(phases[i])
        gain_linear = 10 ** (gain_db / 20.0)

        if not quiet:
            freq_str = format_frequency(f)
            print(f'{freq_str:>13}  {gain_db:>7.3f} dB ({gain_linear:>6.4f}×)  {phase_deg:>8.2f}°')

        if progress_callback:
            progress_callback(i + 1, len(freqs))

    gain_db = 20 * np.log10(gains)
    phase_deg = np.degrees(np.unwrap(phases))

    return freqs, gain_db, phase_deg


def plot_bode_interactive(freqs: np.ndarray, gain_db: np.ndarray, phase_deg: np.ndarray,
                         extra: Optional[Dict[str, Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]]] = None):
    """
    Create an interactive Bode plot that updates as data becomes available.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency points for the measurement
    gain_db : np.ndarray
        Initial gain values (can be empty)
    phase_deg : np.ndarray
        Initial phase values (can be empty)
    extra : dict, optional
        Dictionary mapping label -> function(freqs) -> (gain_db, phase_deg)
        Used to plot reference/theoretical curves alongside measurements
    """
    plt.ion()
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    # Measured curves
    line_gain_meas, = ax_mag.semilogx([], [], marker='o', label="Measured")
    line_phase_meas, = ax_phase.semilogx([], [], marker='o', label="Measured")

    # Set X-axis limits to full frequency range at the beginning
    if len(freqs) > 0:
        f_min, f_max = freqs.min(), freqs.max()
        # Add small margin on log scale
        margin = (np.log10(f_max) - np.log10(f_min)) * 0.05
        ax_mag.set_xlim(10**(np.log10(f_min) - margin), 10**(np.log10(f_max) + margin))

    # Extra reference curves if provided
    if extra:
        # Generate dense frequency array for smooth curves
        f_ref = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), 400) if len(freqs) > 0 else freqs

        for label, func in extra.items():
            # Special styling for Rigol reference data (hard-coded)
            # Sample at measurement frequencies, not dense reference array
            if "Rigol" in label:
                gain_db_ref, phase_deg_ref = func(freqs)
                ax_mag.semilogx(freqs, gain_db_ref, color='red', marker='o',
                               linestyle='-', label=label, markersize=4, linewidth=1, zorder=1)
                ax_phase.semilogx(freqs, phase_deg_ref, color='red', marker='o',
                                 linestyle='-', markersize=4, linewidth=1, zorder=1)
            else:
                gain_db_ref, phase_deg_ref = func(f_ref)
                ax_mag.semilogx(f_ref, gain_db_ref, linestyle='--', label=label, alpha=0.7)
                ax_phase.semilogx(f_ref, phase_deg_ref, linestyle='--', alpha=0.7)

    # Configure axes
    ax_mag.set_ylabel("Gain [dB]")
    ax_mag.grid(True, which="both", ls=":")
    ax_mag.legend(loc="best")

    ax_phase.set_ylabel("Phase shift [deg]")
    ax_phase.set_xlabel("Frequency")
    ax_phase.grid(True, which="both", ls=":")
    ax_phase.legend(loc="best")

    # Format X-axis tick labels in Hz/KHz/MHz
    freq_formatter = FuncFormatter(_format_frequency_tick)
    ax_phase.xaxis.set_major_formatter(freq_formatter)

    fig.tight_layout()

    return fig, (ax_mag, ax_phase), (line_gain_meas, line_phase_meas)


def update_bode_plot(fig, axes, lines, freqs: np.ndarray,
                    gain_db: np.ndarray, phase_deg: np.ndarray):
    """Update the Bode plot with new data."""
    ax_mag, ax_phase = axes
    line_gain_meas, line_phase_meas = lines

    # Update measured lines
    line_gain_meas.set_data(freqs, gain_db)
    line_phase_meas.set_data(freqs, phase_deg)

    # Rescale y-limits only (keep x-limits fixed)
    ax_mag.relim()
    ax_mag.autoscale_view(scalex=False, scaley=True)
    ax_phase.relim()
    ax_phase.autoscale_view(scalex=False, scaley=True)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)


def run_bode_sweep_live(probe, freqs: np.ndarray,
                       extra: Optional[Dict[str, Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]]] = None,
                       quiet: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, plt.Figure]:
    """
    Run a Bode sweep with live plotting.

    Parameters
    ----------
    probe : FreqProbe
        Probe instance to use for measurements
    freqs : np.ndarray
        Array of frequencies to measure (in Hz)
    extra : dict, optional
        Dictionary mapping label -> function(freqs) -> (gain_db, phase_deg)
        Used to plot reference/theoretical curves alongside measurements
    quiet : bool
        Suppress measurement output

    Returns
    -------
    freqs : np.ndarray
        Frequency points measured (same as input)
    gain_db : np.ndarray
        Gain in dB at each frequency
    phase_deg : np.ndarray
        Phase in degrees at each frequency
    fig : plt.Figure
        Matplotlib figure object
    """
    gains = np.zeros_like(freqs)
    phases = np.zeros_like(freqs)

    # Set up plot
    fig, axes, lines = plot_bode_interactive(freqs, gains, phases, extra)

    for i, f in enumerate(freqs):
        # Check if plot window is still open
        if not plt.fignum_exists(fig.number):
            print("\nPlot window closed by user - aborting measurement")
            sys.exit(1)

        # Measure with dynamic range adjustment
        vin, vout, dt = probe.measure_dynamic(freq_hz=f)

        # Fit sine waves to extract amplitude and phase
        Ain,  phiin  = fit_sine_at_freq(vin,  f, dt)
        Aout, phiout = fit_sine_at_freq(vout, f, dt)

        # Store gain and phase
        gains[i]  = Aout / Ain
        phases[i] = phiout - phiin   # radians

        # Calculate and display current measurement
        gain_db = 20 * np.log10(gains[i])
        phase_deg = np.degrees(phases[i])
        gain_linear = 10 ** (gain_db / 20.0)

        if not quiet:
            freq_str = format_frequency(f)
            print(f'{freq_str:>13}  {gain_db:>7.3f} dB ({gain_linear:>6.4f}×)  {phase_deg:>8.2f}°')

        # Prepare data up to current point
        freqs_partial = freqs[:i+1]
        gain_db_partial = 20 * np.log10(gains[:i+1])
        phase_deg_partial = np.degrees(np.unwrap(phases[:i+1]))

        # Update plot
        update_bode_plot(fig, axes, lines, freqs_partial, gain_db_partial, phase_deg_partial)

    # Final processing (don't call plt.show() here - let caller do it)
    plt.ioff()
    fig.tight_layout()

    gain_db = 20 * np.log10(gains)
    phase_deg = np.degrees(np.unwrap(phases))

    return freqs, gain_db, phase_deg, fig


def progress_printer(current: int, total: int):
    """Print progress to stderr with line rewind."""
    percent = (current / total) * 100
    sys.stderr.write(f'\r Progress: {percent:.1f}% ({current}/{total})')
    sys.stderr.flush()
    if current == total:
        sys.stderr.write('\n')
        sys.stderr.flush()


def save_to_csv(filename: str, freqs: np.ndarray, gain_db: np.ndarray, phase_deg: np.ndarray):
    """Save Bode plot data to a CSV file."""
    import csv

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frequency (Hz)', 'Gain (dB)', 'Phase (deg)'])
        for freq, gain, phase in zip(freqs, gain_db, phase_deg):
            writer.writerow([freq, gain, phase])


def load_rigol_bode_csv(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Bode data from Rigol oscilloscope CSV export.

    The Rigol CSV format has:
    - Header lines (RIGOL Bode Data, Para Set, StartFreq, StopFreq)
    - Column headers: Freq(Hz),Gain(dB),Phase
    - Data rows: freq,gain,phase

    Returns: (freqs, gain_db, phase_deg)
    """
    import csv

    freqs = []
    gains = []
    phases = []

    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # Skip first 5 lines (headers)
            if i < 5:
                continue
            if len(row) >= 3:
                try:
                    freqs.append(float(row[0]))
                    gains.append(float(row[1]))
                    phases.append(float(row[2]))
                except (ValueError, IndexError):
                    continue

    return np.array(freqs), np.array(gains), np.array(phases)


def save_calibration(filename: str, freqs: np.ndarray, gain_db: np.ndarray, phase_deg: np.ndarray):
    """Save calibration data to a CSV file."""
    import csv

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['# Bode Calibration Data'])
        writer.writerow(['Frequency (Hz)', 'Gain (dB)', 'Phase (deg)'])
        for freq, gain, phase in zip(freqs, gain_db, phase_deg):
            writer.writerow([freq, gain, phase])


def load_calibration(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load calibration data from a CSV file.

    Returns: (freqs, gain_db, phase_deg)
    """
    import csv

    freqs = []
    gains = []
    phases = []

    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # Skip header rows and comments
            if i == 0 and row[0].startswith('#'):
                continue
            if i <= 1:  # Skip first two rows (comment and header)
                continue
            if len(row) >= 3:
                freqs.append(float(row[0]))
                gains.append(float(row[1]))
                phases.append(float(row[2]))

    return np.array(freqs), np.array(gains), np.array(phases)


def apply_calibration(freqs: np.ndarray, gain_db: np.ndarray, phase_deg: np.ndarray,
                     cal_freqs: np.ndarray, cal_gain_db: np.ndarray, cal_phase_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply calibration corrections to measured data.

    Interpolates calibration data to match measurement frequencies and
    subtracts calibration gain/phase from measured values.

    Returns: (corrected_gain_db, corrected_phase_deg)
    """
    # Interpolate calibration data to measurement frequencies
    cal_gain_interp = np.interp(freqs, cal_freqs, cal_gain_db)
    cal_phase_interp = np.interp(freqs, cal_freqs, cal_phase_deg)

    # Apply corrections: subtract calibration from measurement
    corrected_gain_db = gain_db - cal_gain_interp
    corrected_phase_deg = phase_deg - cal_phase_interp

    return corrected_gain_db, corrected_phase_deg
