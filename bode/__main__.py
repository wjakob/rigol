#!/usr/bin/env python3
"""
Command-line interface for Bode plot measurements.
"""

import argparse
import os

import numpy as np

from .probe import FreqProbe
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
    run_bode_sweep,
    run_bode_sweep_live,
    save_to_csv,
    progress_printer,
)


def parse_resistance(value: str) -> float:
    """
    Parse resistance value with SI prefixes (e.g., '3.3k', '0.5', '10M', '100m').

    Supports: p (pico), n (nano), u/µ (micro), m (milli), k/K (kilo), M (mega), G (giga), T (tera)
    Optional unit suffix 'Ohm' or 'Ω' is accepted and stripped.

    Examples: '0.5' -> 0.5, '3.3k' -> 3300, '3.6Ohm' -> 3.6, '3.3KOhm' -> 3300, '1M' -> 1000000
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
    Run with all defaults: 10V, 1KHz-10MHz, 30 steps, live plot

  %(prog)s -v 5V -s 100Hz -e 1MHz --steps 50
    Custom voltage and frequency range with 50 measurement points

  %(prog)s -v 10mV --start 1KHz --end 100KHz
    Low voltage measurement (e.g., for sensitive circuits)

  %(prog)s --dump data.csv --headless
    Save to CSV without displaying plots (shows progress)

  %(prog)s -v 2V --terminated
    Use when 50Ω terminators are physically connected to channels (compensates for voltage divider)

Reference Curves:
  %(prog)s --rc-lowpass 10KHz
    Overlay theoretical 1st-order RC lowpass response at 10KHz cutoff

  %(prog)s --rlc-lowpass 100KHz:3.6
    Overlay theoretical 2nd-order RLC lowpass at 100KHz with 3.6Ω resistance

  %(prog)s --rc-highpass 100Hz --rc-lowpass 10KHz
    Compare measurement against RC highpass and lowpass models

  %(prog)s -v 5V --rlc-lowpass 10KHz --rlc-lowpass 10KHz:5
    Compare ideal LC (R=0) vs RLC with 5Ω resistance at same frequency

Advanced:
  %(prog)s -i 2 -o 4 -a 192.168.1.100
    Use CH2 input, CH4 output, custom oscilloscope IP

  %(prog)s --headroom 1.5 --mem-depth 100K
    Increase headroom to 50%% and use 100K sample memory

  %(prog)s --dump result.csv -H
    Automated measurement: save CSV, no GUI

Notes:
  - AFG output voltage is set to the specified voltage
  - Channel voltage ranges include headroom (default 20%%) to prevent clipping
  - Dynamic range adjustment automatically optimizes output channel scale
  - Use --quiet to suppress all output except errors
        '''
    )

    # Connection and channel options
    parser.add_argument('-a', '--addr', default='192.168.5.2',
                       help='Oscilloscope IP address (default: 192.168.5.2)')
    parser.add_argument('-i', '--input', type=int, default=1, choices=[1, 2, 3, 4],
                       help='Input channel (default: 1)')
    parser.add_argument('-o', '--output', type=int, default=2, choices=[1, 2, 3, 4],
                       help='Output channel (default: 2)')

    # Measurement parameters
    parser.add_argument('-v', '--voltage', type=str, default='10V',
                       help='AFG signal amplitude (e.g., 10mV, 5V, 10V). Channel voltage ranges are automatically set to voltage × headroom factor (default: 10V)')
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

    args = parser.parse_args()

    # Validate that input and output are different
    if args.input == args.output:
        parser.error("Input and output channels must be different")

    # Validate headroom
    if args.headroom < 1.0:
        parser.error("Headroom must be >= 1.0")

    # Parse voltage and frequencies
    try:
        voltage_v = parse_si(args.voltage, unit='V')
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
            # Format resistances nicely (e.g., 3300 -> 3.3k, 0.5 -> 0.5Ω)
            r_esr_str = f"{r_esr:.1f}Ω" if r_esr < 1000 else f"{r_esr/1000:.1f}kΩ"
            r_src_str = f"{r_source:.1f}Ω" if r_source < 1000 else f"{r_source/1000:.1f}kΩ"
            label = f"LC bandpass (L={L*1e3:.2f}mH, C={C*1e9:.1f}nF, R_esr={r_esr_str}, R_src={r_src_str}, f₀={format_frequency(fc)})"
            extra[label] = lc_bandpass(L, C, r_esr, r_source)
    if lc_bandstop_params:
        for L, C, r_esr, r_source in lc_bandstop_params:
            # Calculate resonant frequency for the label
            fc = 1.0 / (2 * np.pi * np.sqrt(L * C))
            # Build label showing component values
            # Format resistances nicely (e.g., 3300 -> 3.3k, 0.5 -> 0.5Ω)
            r_esr_str = f"{r_esr:.1f}Ω" if r_esr < 1000 else f"{r_esr/1000:.1f}kΩ"
            r_src_str = f"{r_source:.1f}Ω" if r_source < 1000 else f"{r_source/1000:.1f}kΩ"
            label = f"LC bandstop (L={L*1e3:.2f}mH, C={C*1e9:.1f}nF, R_esr={r_esr_str}, R_src={r_src_str}, f₀={format_frequency(fc)})"
            extra[label] = lc_bandstop(L, C, r_esr, r_source)

    # Pass None if no extra plots requested
    extra = extra if extra else None

    # Build resource string
    resource = f"TCPIP0::{args.addr}::INSTR"

    # Configure voltage: AFG gets base voltage, channels get headroom
    # AFG amplitude is peak-to-peak voltage (e.g., 10V amplitude = ±5V peak)
    # Channel ranges must handle the peak voltage with headroom
    # If terminated (50Ω), voltage divider halves the signal at the channel
    afg_voltage = voltage_v
    peak_voltage = voltage_v / 2.0  # Convert amplitude to peak voltage
    termination_factor = 0.5 if args.terminated else 1.0
    input_max_voltage = peak_voltage * termination_factor * args.headroom

    # Calculate frequencies based on steps or steps-per-decade
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
        print(f"Signal voltage: {voltage_v:.3f} V, Headroom: {args.headroom}x{termination_str}")
        print(f"AFG voltage: {afg_voltage:.3f} V, Channel range: {input_max_voltage:.3f} V")
        print(f"Channels: Input=CH{args.input}, Output=CH{args.output}")

    # Create probe instance
    probe = FreqProbe(
        resource=resource,
        channel_a=args.input,
        channel_b=args.output,
        desired_cycles=args.cycles,
        mem_depth=args.mem_depth,
        max_voltage=input_max_voltage,
        probe_factor=args.probe_factor,
        afg_amplitude_v=afg_voltage,
        headroom=args.headroom,
        quiet=args.quiet,
    )

    fig_to_show = None
    try:
        if args.headless:
            # Headless mode: run sweep with progress reporting
            if not args.quiet:
                print("Running sweep in headless mode...")
                print(f"{'Frequency':>13}  {'Gain':>21}  {'Phase':>8}")
                print(f"{'-'*13}  {'-'*21}  {'-'*8}")
            freqs, gain_db, phase_deg = run_bode_sweep(
                probe, freqs,
                progress_callback=None if args.quiet else progress_printer,
                quiet=args.quiet
            )
        else:
            # Interactive mode with live plotting
            if not args.quiet:
                print("Running sweep with live plotting...")
                print(f"{'Frequency':>13}  {'Gain':>21}  {'Phase':>8}")
                print(f"{'-'*13}  {'-'*21}  {'-'*8}")
            freqs, gain_db, phase_deg, fig_to_show = run_bode_sweep_live(
                probe, freqs,
                extra=extra,
                quiet=args.quiet
            )

        if args.dump:
            save_to_csv(args.dump, freqs, gain_db, phase_deg)
            if not args.quiet:
                print(f"Data saved to {args.dump}")

        if not args.quiet:
            print("Measurement complete!")

    finally:
        # Close probe and return scope to normal operation
        probe.close()
        if not args.quiet:
            print("Disconnected from oscilloscope.")

    # Show plot after probe is closed (interactive mode only)
    if fig_to_show is not None:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()
