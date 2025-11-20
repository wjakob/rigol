# Bode Plot Measurement Tool

Some Rigol DHO900 series oscilloscopes (DHO914S or DHO924S) provide built-in
support for generating [Bode plots](https://en.wikipedia.org/wiki/Bode_plot).
While this is very nice, the plots produced by this feature unfortunately
appear to contain artifacts and noticeably deviate from expected results. A
discussion on the EEVBlog forum mentioned that this could be a software bug and
not a hardware problem.

This simple Python package (to be run on a separate computer) drives the scope
and arbitrary function generator via SCPI commands, using NumPy to compute
attenuation+phase shift. It provides results that better match the expected
behavior when examining simple RC/LC filters (see below for some results).

Note: I am a computer scientist studying electronics as a hobby. This project
was vibe-coded in an afternoon. Your mileage may vary.

## Example Measurements

<table>
  <tr>
    <td width="50%">
      <img src="images/rc_lowpass.png" alt="RC Lowpass Filter" width="100%"/>
      <p align="center"><b>RC Lowpass Filter</b></p>
    </td>
    <td width="50%">
      <img src="images/rc_highpass.png" alt="RC Highpass Filter" width="100%"/>
      <p align="center"><b>RC Highpass Filter</b></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="images/rlc_lowpass.png" alt="RLC Lowpass Filter" width="100%"/>
      <p align="center"><b>RLC Lowpass Filter</b></p>
    </td>
    <td width="50%">
      <img src="images/rlc_highpass.png" alt="RLC Highpass Filter" width="100%"/>
      <p align="center"><b>RLC Highpass Filter</b></p>
    </td>
  </tr>
</table>

## Features

- **Automated frequency sweeps** with logarithmic spacing
- **Live plotting** with real-time updates during measurement
- **Automatic dynamic range adjustment** to optimize signal quality
- **Reference curve overlays** for theoretical RC/RLC filters
- **CSV export** for post-processing and analysis
- **Headless mode** for automated measurements without GUI

## Hardware Requirements

- Rigol oscilloscope with function generator (e.g., DHO124S or DHO924S)
- Oscilloscope must be reachable via TCP/IP
- Two oscilloscope channels (default: CH1 for input, CH2 for output)

### Software Dependencies

- Python 3.7+
- `numpy` - numerical computing
- `matplotlib` - plotting
- `pyvisa` - VISA communication
- `pyvisa-py` - Python VISA backend

## Installation

```bash
# Clone the repository
git clone https://github.com/wjakob/bode
cd bode

# Install dependencies
pip install numpy matplotlib pyvisa pyvisa-py
```

## Quick Start

The following are to be executed in the project directory.

### Basic measurement with live plotting

```bash
python -m bode
```

This runs a sweep from 1 KHz to 10 MHz with 10V signal amplitude and displays a
live Bode plot. The script assumes that the scope is available at the IP
address ``192.168.5.2``, and that 10X probes measure the input and output at
channels 1/2. See below for the full set of command line options to change
these behaviors.

### Measure and save to CSV

```bash
python -m bode --dump measurement.csv
```

### Custom frequency range and voltage

```bash
python -m bode -v 5V --start 100Hz --end 1MHz --steps 50
```

### Headless mode (no GUI, shows progress)

```bash
python -m bode --headless --dump data.csv
```

## Usage Examples

### Reference Curves

Compare measurements against theoretical filter responses:

```bash
# Overlay 1st-order RC lowpass at 10 KHz cutoff
python -m bode --rc-lowpass 10KHz

# Overlay 2nd-order RLC lowpass at 100 KHz with 3.6Ω ESR
python -m bode --rlc-lowpass 100KHz:3.6

# Ideal LC filter (no resistance)
python -m bode --rlc-lowpass 100KHz

# Compare against multiple reference curves (RC and RLC)
python -m bode --rc-lowpass 1KHz --rlc-lowpass 10KHz:3.6 --rc-highpass 100Hz

# Compare ideal LC vs real RLC with resistance
python -m bode --rlc-lowpass 10KHz --rlc-lowpass 10KHz:5
```

### Low Voltage Measurements

```bash
# For sensitive circuits (e.g., 10mV signal)
python -m bode -v 10mV --start 1KHz --end 100KHz
```

### 50Ω Termination

```bash
# When 50Ω terminators are physically connected to channels
python -m bode -v 10V --terminated
```

The `--terminated` flag compensates for the voltage divider effect of 50Ω terminators.

### Custom IP Address and Channels

```bash
# Use CH2 for input, CH4 for output, custom scope IP
python -m bode -i 2 -o 4 -a 192.168.1.100
```

## Command-Line Options

### Connection Options
- `-a`, `--addr` - Oscilloscope IP address (default: `192.168.5.2`)
- `-i`, `--input` - Input channel 1-4 (default: `1`)
- `-o`, `--output` - Output channel 1-4 (default: `2`)

### Measurement Parameters
- `-v`, `--voltage` - AFG signal amplitude (e.g., `10mV`, `5V`, `10V`) (default: `10V`)
- `-s`, `--start` - Start frequency (e.g., `100Hz`, `1KHz`) (default: `1KHz`)
- `-e`, `--end` - End frequency (e.g., `100KHz`, `10MHz`) (default: `10MHz`)
- `--steps` - Number of measurement points (default: `30`)

### Output Options
- `-d`, `--dump` - Save data to CSV file
- `-H`, `--headless` - Run without GUI (shows progress)
- `-q`, `--quiet` - Suppress all output except errors

### Reference Curves
- `--rc-lowpass FREQ` - Overlay 1st-order RC lowpass response at cutoff frequency (can specify multiple times)
- `--rc-highpass FREQ` - Overlay 1st-order RC highpass response at cutoff frequency (can specify multiple times)
- `--rlc-lowpass FREQ[:R]` - Overlay 2nd-order RLC lowpass response at resonant frequency. Optional R specifies series resistance in Ohms for inductor ESR (e.g., `100KHz:3.6`). Can be specified multiple times
- `--rlc-highpass FREQ[:R]` - Overlay 2nd-order RLC highpass response at resonant frequency. Optional R specifies series resistance in Ohms for inductor ESR (e.g., `10KHz:3.6`). Can be specified multiple times

### Advanced Options
- `--mem-depth` - Memory depth (`10K`, `100K`, `1M`, `10M`) (default: `10K`)
- `--probe-factor` - Probe attenuation factor (default: `10` for 10×)
- `--cycles` - Target cycles displayed on screen (default: `10`)
- `--headroom` - Headroom factor above signal (default: `1.2` = 20%)
- `--terminated` - Account for 50Ω termination in channel voltage computation

## Technical Details

- **Sine wave fitting**: Uses least-squares regression to extract amplitude and
  phase at the known excitation frequency (robust to noise)

- **Dynamic voltage**: The output channel voltage scale is automatically
  adjusted during measurement: increase by 2x if signal amplitude < 25% of
  scale (improves SNR), reduction by 2x, if signal exceeds headroom threshold
  (prevents clipping).

## License

This project is licensed under the BSD 3-Clause License.
