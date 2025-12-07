"""
BodePlot: Unified class for Bode plot measurements with Rigol DHO924S.

Combines scope configuration, measurement sweeps, and data management.
Supports flexible callbacks for progress reporting and live plotting.

Usage:
    bode = BodePlot(
        input_ch=1,
        output_ch=2,
        afg_amplitude=10.0,
        max_voltage=12.0,
        probe_factor=10,
        headroom=1.2,
    )

    # Run sweep with custom callback
    freqs = np.logspace(np.log10(1e3), np.log10(10e6), 30)
    bode.sweep(freqs, on_measurement=lambda **kw: print(f"{kw['freq_hz']:.0f} Hz"))

    # Save results
    bode.save_csv('results.csv')

    # Cleanup
    bode.close()
"""

from typing import Tuple, Optional, Callable
import argparse
import numpy as np

from .scope import Scope
from .util import (
    format_frequency,
    parse_si,
    generate_frequencies_per_decade,
    rc_lowpass,
    rc_highpass,
    rlc_lowpass,
    rlc_highpass,
    lc_bandpass,
    lc_bandstop,
    create_print_callback,
    LivePlotUpdater,
)


class BodePlot:
    """
    Unified Bode plot measurement class.

    Manages scope configuration, frequency sweeps, amplitude/phase extraction,
    and data export. Uses callbacks for flexible progress reporting and plotting.
    """

    def __init__(
        self,
        ip: Optional[str] = None,
        input_ch: int = 1,
        output_ch: int = 2,
        desired_cycles: int = 10,
        mem_depth: str = '10K',
        max_voltage: float = 10.0,
        probe_factor: int = 10,
        afg_amplitude: float = 10.0,
        headroom: float = 1.2,
        terminated: bool = False,
        debug_level: int = 0,
        quiet: bool = False,
    ):
        """
        Initialize BodePlot measurement system.

        Parameters
        ----------
        ip : str, optional
            Oscilloscope IP address. If None, auto-discovers on network.
        input_ch : int
            Input channel number (1-4)
        output_ch : int
            Output channel number (1-4)
        desired_cycles : int
            Number of signal cycles to display on screen
        mem_depth : str
            Memory depth ('1K', '10K', '100K', '1M', '10M', '25M', '50M')
        max_voltage : float
            Channel voltage range (includes headroom)
        probe_factor : int
            Probe attenuation factor (e.g., 10 for 10x probe)
        afg_amplitude : float
            AFG output signal amplitude (peak voltage, not peak-to-peak)
        headroom : float
            Headroom factor for dynamic range (must be >= 1.0)
        terminated : bool
            If True, compensate for 50Ω termination on channels
        debug_level : int
            Debug verbosity level (0=off, 1=print commands, 2=print and check errors)
        quiet : bool
            If True, suppress informational messages
        """
        if headroom < 1.0:
            raise ValueError(f"headroom must be >= 1.0, got {headroom}")

        self.desired_cycles = desired_cycles
        self.headroom = headroom
        self.quiet = quiet

        # Create scope instance
        self.scope = Scope(ip=ip, debug_level=debug_level)
        self.scope.stop()

        # Store channel references
        self.input_ch = self.scope.channels[input_ch - 1]
        self.output_ch = self.scope.channels[output_ch - 1]

        # Configure AFG (set termination first so amplitude is compensated correctly)
        self.scope.afg.termination = 50.0 if terminated else float('inf')
        self.scope.afg.function = 'SINusoid'
        self.scope.afg.frequency = 1000.0  # Default 1kHz (will be set to actual value during sweep)
        self.scope.afg.amplitude = afg_amplitude
        self.scope.afg.offset = 0.0
        self.scope.afg.enabled = True

        # Configure input channel
        self.input_ch.enabled = True
        self.input_ch.coupling = 'DC'
        self.input_ch.probe = probe_factor
        self.input_ch.bwlimit = '20M'
        self.input_ch.offset = 0.0
        self.input_ch.vmax = max_voltage

        # Configure output channel
        self.output_ch.enabled = True
        self.output_ch.coupling = 'DC'
        self.output_ch.probe = probe_factor
        self.output_ch.bwlimit = '20M'
        self.output_ch.offset = 0.0
        self.output_ch.vmax = max_voltage

        # Disable other channels
        for i in range(4):
            if i != input_ch - 1 and i != output_ch - 1:
                self.scope.channels[i].enabled = False

        # Configure acquisition
        self.scope.mem_depth = mem_depth
        self.scope.acq_type = 'NORMal'
        self.scope.acq_averages = 1
        self.scope.tmode = 'MAIN'

        # Configure trigger
        self.scope.trigger.mode = 'EDGE'
        self.scope.trigger.source = self.input_ch
        self.scope.trigger.level = 0.0
        self.scope.trigger.slope = 'POSitive'
        self.scope.trigger.sweep = 'SINGle'

        # Store last sweep results
        self.freqs: Optional[np.ndarray] = None
        self.gain_db: Optional[np.ndarray] = None
        self.phase_deg: Optional[np.ndarray] = None

        if not quiet:
            print(f"BodePlot initialized: CH{input_ch} (input), CH{output_ch} (output)")

    def sweep(
        self,
        freqs: np.ndarray,
        on_measurement: Optional[Callable] = None,
        max_scale_adjustments: int = 6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform frequency sweep and measure gain/phase at each point.

        Parameters
        ----------
        freqs : np.ndarray
            Array of frequencies to measure (in Hz)
        on_measurement : callable, optional
            Callback function called after each measurement with kwargs:
                - freq_hz: float - measured frequency
                - gain_db: float - gain in dB
                - phase_deg: float - phase in degrees
                - gain_linear: float - linear gain
                - index: int - current measurement index (0-based)
                - total: int - total number of frequencies
        max_scale_adjustments : int
            Maximum iterations for adaptive voltage scaling

        Returns
        -------
        freqs : np.ndarray
            Frequency points measured
        gain_db : np.ndarray
            Gain in dB at each frequency
        phase_deg : np.ndarray
            Phase in degrees at each frequency (unwrapped)
        """
        gains = np.zeros_like(freqs)
        phases = np.zeros_like(freqs)

        for i, freq_hz in enumerate(freqs):
            # Configure frequency and timebase
            target_span = self.desired_cycles / freq_hz
            safety_factor = 1.2
            visible_span = target_span * safety_factor
            tscale = visible_span / 10.0

            self.scope.afg.frequency = freq_hz
            self.scope.tdiv = tscale

            self.scope.single()
            v_out, dt = self.output_ch.waveform(
                dt=True,
                adaptive=True,
                headroom=self.headroom,
                max_iterations=max_scale_adjustments
            )

            # Read input from the same final acquisition (non-adaptive)
            v_in = self.input_ch.waveform()

            # Extract amplitude and phase using sine fitting
            A_in, phi_in = self._fit_sine_at_freq(v_in, freq_hz, dt)
            A_out, phi_out = self._fit_sine_at_freq(v_out, freq_hz, dt)

            # Store gain and phase
            gain = A_out / A_in
            phase = phi_out - phi_in  # radians
            gains[i] = gain
            phases[i] = phase

            # Invoke callback if provided
            if on_measurement:
                gain_db_current = 20 * np.log10(gain)
                phase_deg_current = self._wrap_phase(np.degrees(phase))
                gain_linear = gain

                on_measurement(
                    freq_hz=freq_hz,
                    gain_db=gain_db_current,
                    phase_deg=phase_deg_current,
                    gain_linear=gain_linear,
                    index=i,
                    total=len(freqs),
                )

        # Convert to dB and unwrap phase
        self.freqs = freqs
        self.gain_db = 20 * np.log10(gains)
        self.phase_deg = np.degrees(np.unwrap(phases))

        return self.freqs, self.gain_db, self.phase_deg

    def save_csv(self, filename: str) -> None:
        """
        Save last sweep results to CSV file.

        Parameters
        ----------
        filename : str
            Output CSV filename

        Raises
        ------
        RuntimeError
            If no sweep has been performed yet
        """
        if self.freqs is None or self.gain_db is None or self.phase_deg is None:
            raise RuntimeError("No sweep results to save. Run sweep() first.")

        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frequency (Hz)', 'Gain (dB)', 'Phase (deg)'])
            for freq, gain, phase in zip(self.freqs, self.gain_db, self.phase_deg):
                writer.writerow([freq, gain, phase])

    def close(self) -> None:
        """Disable AFG and return scope to RUN mode."""
        self.scope.afg.enabled = False
        self.scope.run()

    def _fit_sine_at_freq(
        self,
        sig: np.ndarray,
        freq: float,
        dt: float
    ) -> Tuple[float, float]:
        """
        Fit a sine wave at known frequency to the signal using least squares.

        Parameters
        ----------
        sig : np.ndarray
            1D array of voltage samples
        freq : float
            Excitation frequency in Hz
        dt : float
            Sample interval in seconds

        Returns
        -------
        amplitude : float
            Fitted sine wave amplitude
        phase : float
            Fitted sine wave phase in radians (relative to sine)
        """
        N = len(sig)
        t = np.arange(N) * dt
        w = 2 * np.pi * freq

        # Fit: sig ≈ a*cos(wt) + b*sin(wt)
        A = np.column_stack([np.cos(w * t), np.sin(w * t)])
        coeffs, *_ = np.linalg.lstsq(A, sig, rcond=None)
        a, b = coeffs

        amplitude = np.hypot(a, b)  # sqrt(a^2 + b^2)
        phase = np.arctan2(a, b)    # radians

        return amplitude, phase

    @staticmethod
    def _wrap_phase(deg: float) -> float:
        """
        Wrap phase to (-180, 180] for display friendliness (avoids 0..360 jumps).
        """
        return ((deg + 180.0) % 360.0) - 180.0


def parse_resistance(value: str) -> float:
    """
    Parse resistance value with SI prefixes (e.g., '3.3k', '0.5', '10M', '100m').
    """
    value = value.strip()

    # Strip optional 'Ohm' or 'Ω' suffix (case-insensitive)
    import re
    value = re.sub(r'(Ohm|Ω)$', '', value, flags=re.IGNORECASE).strip()

    # Try plain float first
    try:
        return float(value)
    except ValueError:
        pass

    # Parse with SI multipliers
    match = re.match(r'^([\d.]+)\s*([pnuµmkKMGT]?)$', value)
    if not match:
        raise ValueError(f"Invalid resistance format: {value}")

    number = float(match.group(1))
    prefix = match.group(2)

    multipliers = {
        '': 1,
        'p': 1e-12,
        'n': 1e-9,
        'u': 1e-6,
        'µ': 1e-6,
        'm': 1e-3,
        'k': 1e3,
        'K': 1e3,
        'M': 1e6,
        'G': 1e9,
        'T': 1e12,
    }

    return number * multipliers.get(prefix, 1)


def main():
    parser = argparse.ArgumentParser(
        description='''Bode plot measurement tool for Rigol DHO924S oscilloscope.

Measures frequency response (gain and phase) of circuits by sweeping the
internal AFG across a frequency range and comparing input/output channels.
Features automatic dynamic range adjustment.''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Basic Examples:
  %(prog)s
    Run with all defaults: 5V amplitude, 1KHz-10MHz, 30 steps, live plot

  %(prog)s -a 2.5V -s 100Hz -e 1MHz --steps 50
    Custom amplitude and frequency range with 50 measurement points

  %(prog)s -a 10mV --start 1KHz --end 100KHz
    Low amplitude measurement (e.g., for sensitive circuits)

  %(prog)s --dump data.csv --headless
    Save to CSV without displaying plots (shows progress)

  %(prog)s -a 2V --terminated
    Use when 50Ω terminators are physically connected to channels (compensates for voltage divider)

Reference Curves:
  %(prog)s --rc-lowpass 10KHz
    Overlay theoretical 1st-order RC lowpass response at 10KHz cutoff

  %(prog)s --rlc-lowpass 100KHz:3.6
    Overlay theoretical 2nd-order RLC lowpass at 100KHz with 3.6Ω resistance

  %(prog)s --rc-highpass 100Hz --rc-lowpass 10KHz
    Compare measurement against RC highpass and lowpass models

  %(prog)s -a 2.5V --rlc-lowpass 10KHz --rlc-lowpass 10KHz:5
    Compare ideal LC (R=0) vs RLC with 5Ω resistance at same frequency

Advanced:
  %(prog)s -i 2 -o 4 -A 192.168.1.100
    Use CH2 input, CH4 output, specify oscilloscope IP explicitly

  %(prog)s --headroom 1.5 --mem-depth 100K
    Increase headroom to 50%% and use 100K sample memory

  %(prog)s --dump result.csv -H
    Automated measurement: save CSV, no GUI

Notes:
  - Amplitude is specified as peak voltage (not peak-to-peak)
  - Channel voltage ranges include headroom (default 20%%) to prevent clipping
  - Dynamic range adjustment automatically optimizes output channel scale
  - Use --quiet to suppress all output except errors
        '''
    )

    # Connection and channel options
    parser.add_argument('-A', '--addr', default=None,
                       help='Oscilloscope IP address (auto-discovers if not specified)')
    parser.add_argument('-i', '--input', type=int, default=1, choices=[1, 2, 3, 4],
                       help='Input channel (default: 1)')
    parser.add_argument('-o', '--output', type=int, default=2, choices=[1, 2, 3, 4],
                       help='Output channel (default: 2)')

    # Measurement parameters
    parser.add_argument('-a', '--amplitude', type=str, default='5V',
                       help='AFG signal amplitude (peak voltage, not peak-to-peak), e.g., 10mV, 5V. Channel voltage ranges are automatically set to amplitude × headroom factor (default: 5V)')
    parser.add_argument('-s', '--start', type=str, default='1KHz',
                       help='Start frequency for sweep (e.g., 100Hz, 1KHz, 10KHz) (default: 1KHz)')
    parser.add_argument('-e', '--end', type=str, default='10MHz',
                       help='End frequency for sweep (e.g., 100KHz, 1MHz, 10MHz) (default: 10MHz)')

    # Frequency steps (mutually exclusive)
    steps_group = parser.add_mutually_exclusive_group()
    steps_group.add_argument('--steps', type=int, default=30,
                       help='Number of logarithmically-spaced frequency points to measure (default: 30)')
    steps_group.add_argument('--steps-per-decade', type=int, metavar='N',
                       help='Number of frequency points per decade (logarithmically spaced), matching Rigol behavior. Example: 10 steps per decade from 1kHz to 10MHz gives 10×4=40 points plus endpoints')

    # Output options
    parser.add_argument('-d', '--dump', type=str, metavar='FILE',
                       help='Save measurement data to CSV file (frequency, gain, phase)')
    parser.add_argument('-H', '--headless', action='store_true',
                       help='Run without displaying plots (shows progress on stderr, useful for automation)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress all output except errors (implies headless)')

    # Reference circuit options
    parser.add_argument('--rc-lowpass', type=str, action='append', metavar='FREQ',
                       help='Overlay theoretical 1st-order RC lowpass response at cutoff frequency (e.g., 10KHz). Can be specified multiple times')
    parser.add_argument('--rc-highpass', type=str, action='append', metavar='FREQ',
                       help='Overlay theoretical 1st-order RC highpass response at cutoff frequency (e.g., 100Hz). Can be specified multiple times')
    parser.add_argument('--rlc-lowpass', type=str, action='append', metavar='FREQ[:R]',
                       help='Overlay theoretical 2nd-order RLC lowpass response at resonant frequency. Format: FREQ or FREQ:RESISTANCE (e.g., 10KHz, 100KHz:3.6). RESISTANCE in Ohms accounts for inductor ESR. Can be specified multiple times')
    parser.add_argument('--rlc-highpass', type=str, action='append', metavar='FREQ[:R]',
                       help='Overlay theoretical 2nd-order RLC highpass response at resonant frequency. Format: FREQ or FREQ:RESISTANCE (e.g., 100Hz, 10KHz:3.6). RESISTANCE in Ohms accounts for inductor ESR. Can be specified multiple times')
    parser.add_argument('--lc-bandpass', type=str, action='append', metavar='L:C:R_ESR:R_SRC',
                       help='Overlay theoretical LC bandpass (parallel LC with voltage divider). Format: L:C:R_ESR:R_SOURCE (e.g., 1mH:10nF:0.5:3.3k). L is inductance, C is capacitance, R_ESR is inductor ESR in Ohms, R_SOURCE is source/protection resistor in Ohms. All parameters required. Can be specified multiple times')
    parser.add_argument('--lc-bandstop', type=str, action='append', metavar='L:C:R_ESR:R_SRC',
                       help='Overlay theoretical LC bandstop/notch (series LC shunt to ground). Format: L:C:R_ESR:R_SOURCE (e.g., 1mH:10nF:0.5:3.3k). L is inductance, C is capacitance, R_ESR is inductor ESR in Ohms, R_SOURCE is source/protection resistor in Ohms. All parameters required. Can be specified multiple times')

    # Advanced options
    parser.add_argument('--mem-depth', type=str, default='10K',
                       help='Oscilloscope memory depth (e.g., 10K, 100K, 1M, 10M). Higher values capture more cycles (default: 10K)')
    parser.add_argument('--probe-factor', type=int, default=10,
                       help='Probe attenuation factor (1 for 1:1, 10 for 10:1, etc.) (default: 10)')
    parser.add_argument('--cycles', type=int, default=10,
                       help='Target number of waveform cycles to display on screen timebase (default: 10)')
    parser.add_argument('--headroom', type=float, default=1.2,
                       help='Safety margin above signal voltage to prevent clipping. 1.2 = 20%% headroom. Must be >= 1.0 (default: 1.2)')
    parser.add_argument('--terminated', action='store_true',
                       help='Specify if 50Ω terminators are physically connected to oscilloscope channels. Termination creates a voltage divider that halves signal amplitude; this flag adjusts channel sensitivity to compensate')
    parser.add_argument('--debug', type=int, default=0, choices=[0, 1, 2],
                       help='Debug level: 0=off, 1=print SCPI commands, 2=print commands and check errors after each')

    args = parser.parse_args()

    # Validate that input and output are different
    if args.input == args.output:
        parser.error("Input and output channels must be different")

    # Validate headroom
    if args.headroom < 1.0:
        parser.error("Headroom must be >= 1.0")

    # Parse voltage and frequencies
    try:
        voltage_v = parse_si(args.amplitude, unit='V')
        start_hz = parse_si(args.start, unit='Hz')
        end_hz = parse_si(args.end, unit='Hz')
    except ValueError as e:
        parser.error(str(e))

    if start_hz >= end_hz:
        parser.error("Start frequency must be less than end frequency")

    # Parse reference frequencies
    rc_lowpass_freqs = []
    if args.rc_lowpass:
        for lp in args.rc_lowpass:
            try:
                rc_lowpass_freqs.append(parse_si(lp, unit='Hz'))
            except ValueError as e:
                parser.error(f"Invalid RC lowpass frequency: {e}")

    rc_highpass_freqs = []
    if args.rc_highpass:
        for hp in args.rc_highpass:
            try:
                rc_highpass_freqs.append(parse_si(hp, unit='Hz'))
            except ValueError as e:
                parser.error(f"Invalid RC highpass frequency: {e}")

    # Parse RLC filters with optional resistance (format: FREQ or FREQ:R)
    rlc_lowpass_params = []
    if args.rlc_lowpass:
        for lp in args.rlc_lowpass:
            try:
                if ':' in lp:
                    freq_str, r_str = lp.split(':', 1)
                    freq = parse_si(freq_str, unit='Hz')
                    r = float(r_str)
                else:
                    freq = parse_si(lp, unit='Hz')
                    r = 0.0
                rlc_lowpass_params.append((freq, r))
            except ValueError as e:
                parser.error(f"Invalid RLC lowpass parameter '{lp}': {e}")

    rlc_highpass_params = []
    if args.rlc_highpass:
        for hp in args.rlc_highpass:
            try:
                if ':' in hp:
                    freq_str, r_str = hp.split(':', 1)
                    freq = parse_si(freq_str, unit='Hz')
                    r = float(r_str)
                else:
                    freq = parse_si(hp, unit='Hz')
                    r = 0.0
                rlc_highpass_params.append((freq, r))
            except ValueError as e:
                parser.error(f"Invalid RLC highpass parameter '{hp}': {e}")

    # Parse LC bandpass filters (format: L:C:R_ESR:R_SOURCE)
    lc_bandpass_params = []
    if args.lc_bandpass:
        for bp in args.lc_bandpass:
            try:
                parts = bp.split(':')
                if len(parts) != 4:
                    raise ValueError("Format must be L:C:R_ESR:R_SOURCE (e.g., 1mH:10nF:0.5:3.3k)")
                # Parse with SI units
                L = parse_si(parts[0], unit='H')
                C = parse_si(parts[1], unit='F')
                r_esr = parse_resistance(parts[2])
                r_source = parse_resistance(parts[3])
                lc_bandpass_params.append((L, C, r_esr, r_source))
            except ValueError as e:
                parser.error(f"Invalid LC bandpass parameter '{bp}': {e}")

    # Parse LC bandstop filters (format: L:C:R_ESR:R_SOURCE)
    lc_bandstop_params = []
    if args.lc_bandstop:
        for bs in args.lc_bandstop:
            try:
                parts = bs.split(':')
                if len(parts) != 4:
                    raise ValueError("Format must be L:C:R_ESR:R_SOURCE (e.g., 1mH:10nF:0.5:3.3k)")
                # Parse with SI units
                L = parse_si(parts[0], unit='H')
                C = parse_si(parts[1], unit='F')
                r_esr = parse_resistance(parts[2])
                r_source = parse_resistance(parts[3])
                lc_bandstop_params.append((L, C, r_esr, r_source))
            except ValueError as e:
                parser.error(f"Invalid LC bandstop parameter '{bs}': {e}")

    # Build extra plot functions if requested
    extra = {}
    if rc_lowpass_freqs:
        for fc in rc_lowpass_freqs:
            label = f"RC lowpass ({format_frequency(fc)})"
            extra[label] = rc_lowpass(fc)
    if rc_highpass_freqs:
        for fc in rc_highpass_freqs:
            label = f"RC highpass ({format_frequency(fc)})"
            extra[label] = rc_highpass(fc)
    if rlc_lowpass_params:
        for fc, r in rlc_lowpass_params:
            if r > 0:
                label = f"RLC lowpass ({format_frequency(fc)}, R={r:.1f}Ω)"
            else:
                label = f"RLC lowpass ({format_frequency(fc)})"
            extra[label] = rlc_lowpass(fc, r)
    if rlc_highpass_params:
        for fc, r in rlc_highpass_params:
            if r > 0:
                label = f"RLC highpass ({format_frequency(fc)}, R={r:.1f}Ω)"
            else:
                label = f"RLC highpass ({format_frequency(fc)})"
            extra[label] = rlc_highpass(fc, r)
    if lc_bandpass_params:
        for L, C, r_esr, r_source in lc_bandpass_params:
            # Calculate resonant frequency for the label
            fc = 1.0 / (2 * np.pi * np.sqrt(L * C))
            # Build label showing component values
            r_esr_str = f"{r_esr:.1f}Ω" if r_esr < 1000 else f"{r_esr/1000:.1f}kΩ"
            r_src_str = f"{r_source:.1f}Ω" if r_source < 1000 else f"{r_source/1000:.1f}kΩ"
            label = f"LC bandpass (L={L*1e3:.2f}mH, C={C*1e9:.1f}nF, R_esr={r_esr_str}, R_src={r_src_str}, f₀={format_frequency(fc)})"
            extra[label] = lc_bandpass(L, C, r_esr, r_source)
    if lc_bandstop_params:
        for L, C, r_esr, r_source in lc_bandstop_params:
            # Calculate resonant frequency for the label
            fc = 1.0 / (2 * np.pi * np.sqrt(L * C))
            r_esr_str = f"{r_esr:.1f}Ω" if r_esr < 1000 else f"{r_esr/1000:.1f}kΩ"
            r_src_str = f"{r_source:.1f}Ω" if r_source < 1000 else f"{r_source/1000:.1f}kΩ"
            label = f"LC bandstop (L={L*1e3:.2f}mH, C={C*1e9:.1f}nF, R_esr={r_esr_str}, R_src={r_src_str}, f₀={format_frequency(fc)})"
            extra[label] = lc_bandstop(L, C, r_esr, r_source)

    # Pass None if no extra plots requested
    extra = extra if extra else None

    afg_amplitude = voltage_v
    input_max_voltage = voltage_v * args.headroom
    mem_depth_str = args.mem_depth

    if args.steps_per_decade:
        freqs = generate_frequencies_per_decade(start_hz, end_hz, args.steps_per_decade)
        if not args.quiet:
            print(f"Using {args.steps_per_decade} steps per decade: {len(freqs)} total frequency points")
    else:
        freqs = np.logspace(np.log10(start_hz), np.log10(end_hz), args.steps)

    if not args.quiet:
        print(f"Connecting to oscilloscope at {args.addr}...")
        print(f"Measurement range: {start_hz/1e3:.1f} kHz to {end_hz/1e6:.1f} MHz")
        termination_str = " (50Ω terminated)" if args.terminated else ""
        print(f"Signal amplitude: {voltage_v:.3f} V (peak), Headroom: {args.headroom}x{termination_str}")
        print(f"AFG amplitude: {afg_amplitude:.3f} V, Channel range: ±{input_max_voltage:.3f} V")
        print(f"Channels: Input=CH{args.input}, Output=CH{args.output}")

    bode = BodePlot(
        ip=args.addr,
        input_ch=args.input,
        output_ch=args.output,
        desired_cycles=args.cycles,
        mem_depth=mem_depth_str,
        max_voltage=input_max_voltage,
        probe_factor=args.probe_factor,
        afg_amplitude=afg_amplitude,
        headroom=args.headroom,
        terminated=args.terminated,
        debug_level=args.debug,
        quiet=args.quiet,
    )

    plotter = None
    try:
        if args.headless:
            if not args.quiet:
                print("Running sweep in headless mode...")
                print(f"{'Frequency':>13}  {'Gain':>21}  {'Phase':>8}")
                print(f"{'-'*13}  {'-'*21}  {'-'*8}")
            callback = create_print_callback(quiet=args.quiet)
        else:
            if not args.quiet:
                print("Running sweep with live plotting...")
                print(f"{'Frequency':>13}  {'Gain':>21}  {'Phase':>8}")
                print(f"{'-'*13}  {'-'*21}  {'-'*8}")
            plotter = LivePlotUpdater(freqs, extra=extra)
            print_cb = create_print_callback(quiet=args.quiet)

            def callback(**kwargs):
                if print_cb:
                    print_cb(**kwargs)
                plotter.update(**kwargs)

        freqs, gain_db, phase_deg = bode.sweep(freqs, on_measurement=callback)

        if args.dump:
            bode.save_csv(args.dump)
            if not args.quiet:
                print(f"Data saved to {args.dump}")

        if not args.quiet:
            print("Measurement complete!")

    finally:
        bode.close()
        if not args.quiet:
            print("Disconnected from oscilloscope.")

    if plotter is not None:
        plotter.show()


if __name__ == "__main__":
    # Allow `python -m rigol.bode` as a convenience alias for the CLI
    main()
