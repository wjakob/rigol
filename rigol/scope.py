"""
Generic scope abstraction for SCPI-controlled oscilloscopes.

This module provides a property-based API for controlling oscilloscope parameters
with batched command execution for efficiency.

Properties are automatically generated from parameter tables for efficiency and maintainability.
"""

from enum import Enum
from typing import Optional, Any, Dict, Tuple, overload, Literal, TYPE_CHECKING
import numpy as np
import pyvisa
import sys
import time
import re

from .util import parse_si


class Type(Enum):
    """Parameter type enum for SCPI value formatting."""
    UNITLESS = 1
    VOLTAGE = 2
    STRING = 3
    FREQUENCY = 4
    BOOLEAN = 5
    TIME = 6


# Parameter table structure: (name, type, scpi_template, valid_values, priority)
# Priority is used to sort commands before sending. Lower = earlier. None = no reordering.
SCOPE_PARAMS = [
    ('tdiv', Type.TIME, 'TIMebase:MAIN:SCALe', None, None),
    ('toffset', Type.TIME, 'TIMebase:MAIN:OFFSet', None, None),
    ('tmode', Type.STRING, 'TIMebase:MODE', ['MAIN', 'XY', 'ROLL'], None),
    ('mem_depth', Type.STRING, 'ACQuire:MDEPth', ['1K', '10K', '100K', '1M', '10M', '25M', '50M'], None),
    ('acq_type', Type.STRING, 'ACQuire:TYPE', ['NORMal', 'PEAK', 'AVERages', 'ULTRa'], None),
    ('acq_averages', Type.UNITLESS, 'ACQuire:AVERages', None, None),
]

CHANNEL_PARAMS = [
    # Priority enforces order: enabled -> probe -> vdiv -> offset (each affects the next's valid range)
    ('enabled', Type.BOOLEAN, 'CHANnel{ch}:DISPlay', None, 0),
    ('probe', Type.UNITLESS, 'CHANnel{ch}:PROBe', None, 1),
    ('vdiv', Type.VOLTAGE, 'CHANnel{ch}:SCALe', None, 2),
    ('offset', Type.VOLTAGE, 'CHANnel{ch}:OFFSet', None, 3),
    ('position', Type.VOLTAGE, 'CHANnel{ch}:POSition', None, None),
    ('coupling', Type.STRING, 'CHANnel{ch}:COUPling', ['DC', 'AC', 'GND'], None),
    ('bwlimit', Type.STRING, 'CHANnel{ch}:BWLimit', ['20M', 'OFF'], None),
    ('invert', Type.BOOLEAN, 'CHANnel{ch}:INVert', None, None),
]

AFG_PARAMS = [
    ('enabled', Type.BOOLEAN, 'SOURce:OUTPut:STATe', None, None),
    ('function', Type.STRING, 'SOURce:FUNCtion', ['SINusoid', 'SQUare', 'RAMP', 'PULSe', 'DC', 'NOISe', 'ARB'], None),
    ('_amplitude_raw', Type.VOLTAGE, 'SOURce:VOLTage:AMPLitude', None, None),
    ('frequency', Type.FREQUENCY, 'SOURce:FREQuency', None, None),
    ('_offset_raw', Type.VOLTAGE, 'SOURce:VOLTage:OFFSet', None, None),
    ('phase', Type.UNITLESS, 'SOURce:PHASe', None, None),
    ('duty', Type.UNITLESS, 'SOURce:FUNCtion:SQUare:DUTY', None, None),
    ('symmetry', Type.UNITLESS, 'SOURce:FUNCtion:RAMP:SYMMetry', None, None),
]

TRIGGER_PARAMS = [
    ('mode', Type.STRING, 'TRIGger:MODE', ['EDGE', 'PULSe', 'RUNT', 'WIND', 'NEDG', 'SLOPe', 'VIDeo', 'PATTern', 'DELay', 'TIMeout', 'DURation', 'SHOLd', 'RS232', 'IIC', 'SPI'], None),
    ('_source_raw', Type.STRING, 'TRIGger:EDGE:SOURce', ['CHAN1', 'CHAN2', 'CHAN3', 'CHAN4', 'AC'], None),
    ('level', Type.VOLTAGE, 'TRIGger:EDGE:LEVel', None, None),
    ('slope', Type.STRING, 'TRIGger:EDGE:SLOPe', ['POSitive', 'NEGative', 'EITHer'], None),
    ('sweep', Type.STRING, 'TRIGger:SWEep', ['AUTO', 'NORMal', 'SINGle'], None),
    ('nreject', Type.BOOLEAN, 'TRIGger:NREJect', None, None),
]

# Build SCPI command -> priority lookup from param tables
_SCPI_PRIORITY = {}
for _name, _type, _tmpl, _valid, _prio in (
    SCOPE_PARAMS + CHANNEL_PARAMS + AFG_PARAMS + TRIGGER_PARAMS
):
    if _prio is not None:
        if '{ch}' in _tmpl:
            for _ch in range(1, 5):
                _SCPI_PRIORITY[_tmpl.format(ch=_ch)] = _prio
        else:
            _SCPI_PRIORITY[_tmpl] = _prio


# Helper function for SI unit parsing - used by property setters
def _normalize_value(value: Any, ptype: Type) -> Any:
    """Parse SI unit strings and normalize values based on type."""
    if not isinstance(value, str):
        return value

    match ptype:
        case Type.VOLTAGE:
            return parse_si(value, unit='V')
        case Type.FREQUENCY:
            return parse_si(value, unit='Hz')
        case Type.TIME:
            return parse_si(value, unit='s')
        case _:
            return value


class Scope:
    """
    Generic SCPI-controlled oscilloscope interface.

    All parameter changes are batched and automatically committed when needed
    (e.g., when reading values, arming triggers, or running acquisitions).
    Call ``commit()`` explicitly to flush pending changes immediately.

    Example:
        scope = Scope(ip='192.168.5.2', debug_level=0)

        # Configure scope parameters
        scope.mem_depth = 100000
        scope.tdiv = 1e-3  # 1ms/div

        # Configure channel
        scope.channels[0].enabled = True
        scope.channels[0].probe = 10
        scope.channels[0].vmax = 10
        scope.channels[0].coupling = 'DC'

        # Configure trigger
        scope.trigger.mode = 'EDGE'
        scope.trigger.source = scope.channels[0]
        scope.trigger.level = 0.0

        # Configure AFG
        scope.afg.enabled = True
        scope.afg.amplitude = '5V'
        scope.afg.frequency = '1kHz'

        # Arm single-shot trigger (automatically commits all pending changes)
        scope.single()

        # Wait for trigger
        scope.wait_trigger()

        # Read waveform (automatically commits before reading)
        waveform = scope.channels[0].waveform()

        # Or use adaptive capture
        waveform = scope.channels[0].waveform(adaptive=True, headroom=1.2)
    """

    if TYPE_CHECKING:
        @property
        def tdiv(self) -> float:
            """Time per division in seconds (horizontal scale). Accepts SI strings like '1ms'."""
            ...
        @tdiv.setter
        def tdiv(self, value: float | str) -> None: ...

        @property
        def toffset(self) -> float:
            """Horizontal time offset in seconds. Accepts SI strings like '100us'."""
            ...
        @toffset.setter
        def toffset(self, value: float | str) -> None: ...

        @property
        def tmode(self) -> Literal['MAIN', 'XY', 'ROLL']:
            """Timebase mode: MAIN (normal), XY, or ROLL."""
            ...
        @tmode.setter
        def tmode(self, value: Literal['MAIN', 'XY', 'ROLL']) -> None: ...

        @property
        def mem_depth(self) -> Literal['1K', '10K', '100K', '1M', '10M', '25M', '50M']:
            """Acquisition memory depth (number of samples)."""
            ...
        @mem_depth.setter
        def mem_depth(self, value: Literal['1K', '10K', '100K', '1M', '10M', '25M', '50M']) -> None: ...

        @property
        def acq_type(self) -> Literal['NORMal', 'PEAK', 'AVERages', 'ULTRa']:
            """Acquisition type: NORMal, PEAK detect, AVERages, or ULTRa."""
            ...
        @acq_type.setter
        def acq_type(self, value: Literal['NORMal', 'PEAK', 'AVERages', 'ULTRa']) -> None: ...

        @property
        def acq_averages(self) -> int:
            """Number of acquisitions to average (when acq_type is AVERages)."""
            ...
        @acq_averages.setter
        def acq_averages(self, value: int) -> None: ...

    def __init__(self, ip: Optional[str] = None, debug_level: int = 0):
        """
        Initialize scope connection.

        Args:
            ip: IP address of the oscilloscope. If None, auto-discovers the first
                available scope on the network (requires 'zeroconf' package).
            debug_level: Debug verbosity level:
                0 = no debug output
                1 = print SCPI commands to stderr
                2 = print SCPI commands to stderr and check device status after each command
        """
        self.debug_level = debug_level
        # Queue stores pending changes: {scpi_command: (value, type)}
        self._queue: Dict[str, Tuple[Any, Type]] = {}
        # Cache stores committed values: {scpi_command: (value, type)}
        self._cache: Dict[str, Tuple[Any, Type]] = {}

        # Connect to scope
        rm = pyvisa.ResourceManager()
        if ip is None:
            # Auto-discover scope on network
            try:
                resources = rm.list_resources()
            except Exception:
                resources = ()
            tcpip_resources = [r for r in resources if r.startswith('TCPIP')]
            if not tcpip_resources:
                raise RuntimeError(
                    "No oscilloscope found on network. "
                    "For auto-discovery, install the 'zeroconf' package: pip install zeroconf. "
                    "Alternatively, specify the IP address explicitly: Scope(ip='192.168.x.x')"
                )

            discovered = tcpip_resources[0]
            if self.debug_level >= 1:
                print(f"< Auto-discovered: {discovered}", file=sys.stderr)

            # Extract host from something like TCPIP0::192.168.0.188::INSTR
            import re
            m = re.match(r"TCPIP\d*::([^:]+)::", discovered)
            if not m:
                raise RuntimeError(f"Unexpected VISA resource format: {discovered}")
            host = m.group(1)
            resource = f"TCPIP0::{host}::5555::SOCKET"
            if self.debug_level >= 1:
                print(f"< Using VISA resource: {resource}", file=sys.stderr)
        else:
            resource = f"TCPIP0::{ip}::INSTR"
            resource = f"TCPIP0::{ip}::5555::SOCKET"

        # Set up line endings for SOCKET communication
        self.inst = rm.open_resource(resource)
        self.inst.read_termination = "\n"
        self.inst.write_termination = "\n"
        self.inst.timeout = 120_000

        # Clear any existing errors in the queue before we start
        if self.debug_level >= 1:
            # Read and discard all errors until we get "0,No error"
            while True:
                try:
                    error = self.inst.query(":SYSTem:ERRor?").strip()
                    print(f"< Clearing old error: {error}", file=sys.stderr)
                    if error.startswith("0,"):
                        break
                except:
                    break

        # Create child objects (4 channels for DHO924S)
        self.channels = [Channel(self, i) for i in range(4)]
        self.afg = AFG(self)
        self.trigger = Trigger(self)

        # Waveform transfer settings are constant; set once
        self._write(":WAVeform:MODE RAW")
        self._write(":WAVeform:FORMat WORD")

    def _write(self, cmd: str, check_errors: bool = True) -> None:
        """
        Execute SCPI write command.

        Args:
            cmd: SCPI command to write
            check_errors: If True (default), check for errors when debug_level >= 2
        """
        if not self.debug_level:
            self.inst.write(cmd)
        else:
            print(f"> {cmd}", file=sys.stderr)
            self.inst.write(cmd)
            if self.debug_level >= 2 and check_errors:
                self._check_error(cmd)

    def _query(self, cmd: str) -> str:
        """Execute SCPI query command."""
        if not self.debug_level:
            return self.inst.query(cmd)
        else:
            print(f"> {cmd}", file=sys.stderr)
            result = self.inst.query(cmd)
            print(f"< {result.strip()}", file=sys.stderr)
            if self.debug_level >= 2:
                self._check_error(cmd)
            return result

    def _check_error(self, last_cmd: str = "") -> None:
        """
        Check for SCPI errors and raise exception if found.

        Only called when debug_level >= 2 to help diagnose issues.

        Args:
            last_cmd: The command that was just executed (for error reporting)
        """
        error_response = self.inst.query(":SYSTem:ERRor?").strip()
        # Error format: "code,message" e.g. "0,No error" or "-113,Undefined header"
        try:
            code_str, message = error_response.split(',', 1)
            code = int(code_str)
            if code != 0:
                cmd_info = f" after command: {last_cmd}" if last_cmd else ""
                raise RuntimeError(f"SCPI Error {code}: {message}{cmd_info}")
        except ValueError:
            # If parsing fails, just print the raw error response
            print(f"  Error response: {error_response}", file=sys.stderr)

    def commit(self, extra_cmd: Optional[str] = None) -> None:
        """
        Flush all queued parameter changes to the oscilloscope.

        Parameter changes made via property setters (e.g., ``scope.tdiv = 1e-3``)
        are batched in an internal queue for efficiency. This method sends all
        pending changes to the device. It is called automatically before operations
        that require current settings (e.g., ``single()``, ``run()``, property reads),
        but can also be called explicitly when immediate execution is needed.

        When debug_level >= 2: sends each command sequentially and checks for errors after each.
        Otherwise: batches all commands into a single SCPI command for efficiency.

        Args:
            extra_cmd: Optional SCPI command to append (e.g., ':SINGle').
                      Primarily used internally by methods like ``single()`` and ``run()``.

        Example:
            scope.tdiv = 1e-3
            scope.channels[0].vdiv = 0.5
            scope.commit()  # Send both changes now
        """
        if not self._queue and not extra_cmd:
            return

        # Sort queue by priority (for channels: enabled -> probe -> scale -> offset)
        # Stable sort preserves insertion order for commands with equal priority (99)
        sorted_items = sorted(self._queue.items(), key=lambda x: _SCPI_PRIORITY.get(x[0], 99))

        if self.debug_level >= 2:
            # Debug mode: send commands one-by-one to catch errors
            for scpi_cmd, (value, ptype) in sorted_items:
                value_str = ('ON' if value else 'OFF') if ptype == Type.BOOLEAN else str(value)
                cmd = f":{scpi_cmd} {value_str}"
                self._write(cmd)

            if extra_cmd:
                self._write(extra_cmd)
        else:
            # Normal mode: batch commands for efficiency
            batched_cmd = ''
            for scpi_cmd, (value, ptype) in sorted_items:
                value_str = ('ON' if value else 'OFF') if ptype == Type.BOOLEAN else str(value)
                batched_cmd += f';:{scpi_cmd} {value_str}'
                self._cache.pop(scpi_cmd, None)

            if extra_cmd:
                batched_cmd += f';{extra_cmd}'

            # Execute batched command (strip leading ';')
            self._write(batched_cmd[1:])

        # Move queued items to cache (they're now committed)

        # Clear queue
        self._queue.clear()

    def _parse_value(self, value_str: str, ptype: Type) -> Any:
        """Parse a value from SCPI response according to its type."""
        value_str = value_str.strip()

        match ptype:
            case Type.BOOLEAN:
                val = value_str.upper()
                if val in ('ON', '1'):
                    return True
                elif val in ('OFF', '0'):
                    return False
                else:
                    raise ValueError(f"Invalid boolean value: {value_str!r}")
            case Type.VOLTAGE | Type.FREQUENCY | Type.TIME:
                return float(value_str)
            case Type.UNITLESS:
                try:
                    return int(value_str) if '.' not in value_str else float(value_str)
                except ValueError:
                    return value_str
            case Type.STRING:
                return value_str

    def single(self) -> None:
        """Arm single-shot acquisition."""
        self.commit(':SINGle')

    def run(self) -> None:
        """Start continuous acquisition."""
        self.commit(':RUN')

    def stop(self) -> None:
        """Stop acquisition and freeze waveform buffer."""
        self.commit(':STOP')

    def force(self) -> None:
        """Force trigger event immediately (useful for testing)."""
        self.commit(':TFORce')

    def reset(self) -> None:
        """
        Reset the oscilloscope to factory default settings.

        This restores all operational settings (timebase, channels, trigger, etc.)
        to their factory defaults. Calibration data is stored separately and
        should not be affected by this command.

        Note: Clears the parameter queue since reset invalidates pending changes.
        """
        # Don't check errors after *RST because scope is resetting
        self._write('*RST', check_errors=False)
        self._query('*OPC?')
        # Additional delay - some subsystems may not be fully ready even after *OPC?
        time.sleep(0.5)
        self._queue.clear()
        self._cache.clear()

    def wait_trigger(self, timeout: float = 5.0) -> None:
        """
        Wait for single-shot acquisition to complete and stop.

        This method polls the trigger status until acquisition completes,
        then issues a STOP command to freeze the waveform buffer for reading.

        Args:
            timeout: Maximum time to wait in seconds (default: 5.0)

        Raises:
            TimeoutError: If acquisition does not complete within timeout
        """
        t0 = time.time()

        while True:
            status = self._query(':TRIGger:STATus?').strip().upper()
            if status == 'TD' or status == 'STOP':  # Trigger detected or already stopped
                break
            if time.time() - t0 > timeout:
                raise TimeoutError(
                    f"Acquisition did not trigger within {timeout} s; last status={status!r}"
                )
            time.sleep(0.02)

    def clear_cache(self) -> None:
        """
        Clear the parameter cache.

        This forces subsequent property reads to query the device rather than
        returning cached values. Useful after external changes to scope settings
        or for testing purposes.
        """
        self._cache.clear()

    @property
    def tmax(self) -> float:
        """Total time on screen (10 horizontal divisions). Accepts SI strings like '100ms'."""
        return self.tdiv * 10

    @tmax.setter
    def tmax(self, value: float | str) -> None:
        value = _normalize_value(value, Type.TIME)
        self.tdiv = value / 10


class Channel:
    """
    Oscilloscope channel interface.

    Provides property-based access to channel parameters. All changes
    are queued in the parent Scope's queue.

    Properties are automatically generated from CHANNEL_PARAMS table.
    """

    if TYPE_CHECKING:
        @property
        def vdiv(self) -> float:
            """Volts per division (vertical scale). Accepts SI strings like '100mV'."""
            ...
        @vdiv.setter
        def vdiv(self, value: float | str) -> None: ...

        @property
        def probe(self) -> int:
            """Probe attenuation ratio (1, 10, 100, etc.)."""
            ...
        @probe.setter
        def probe(self, value: int) -> None: ...

        @property
        def enabled(self) -> bool:
            """Whether the channel is displayed."""
            ...
        @enabled.setter
        def enabled(self, value: bool) -> None: ...

        @property
        def coupling(self) -> Literal['DC', 'AC', 'GND']:
            """Input coupling mode: DC, AC, or GND."""
            ...
        @coupling.setter
        def coupling(self, value: Literal['DC', 'AC', 'GND']) -> None: ...

        @property
        def bwlimit(self) -> Literal['20M', 'OFF']:
            """Bandwidth limit: 20M (20 MHz filter) or OFF."""
            ...
        @bwlimit.setter
        def bwlimit(self, value: Literal['20M', 'OFF']) -> None: ...

        @property
        def offset(self) -> float:
            """Vertical offset in volts. Accepts SI strings like '500mV'."""
            ...
        @offset.setter
        def offset(self, value: float | str) -> None: ...

        @property
        def position(self) -> float:
            """Vertical position (bias voltage) in volts. Accepts SI strings like '500mV'."""
            ...
        @position.setter
        def position(self, value: float | str) -> None: ...

        @property
        def invert(self) -> bool:
            """Whether the channel display is inverted."""
            ...
        @invert.setter
        def invert(self, value: bool) -> None: ...

    def __init__(self, scope: Scope, ch_num: int):
        """
        Initialize channel.

        Args:
            scope: Parent Scope instance
            ch_num: 0-based channel index (0-3 for 4-channel scope)
        """
        self._scope = scope
        self._ch_num = ch_num

    @property
    def vmax(self) -> float:
        """Maximum voltage (full scale = 4 divisions). Accepts SI strings like '500mV'."""
        return self.vdiv * 4

    @vmax.setter
    def vmax(self, value: float | str) -> None:
        value = _normalize_value(value, Type.VOLTAGE)
        self.vdiv = value / 4

    @overload
    def waveform(self, dt: Literal[False] = False, adaptive: bool = False,
                 headroom: float = 1.2, max_iterations: int = 6) -> np.ndarray: ...

    @overload
    def waveform(self, dt: Literal[True], adaptive: bool = False,
                 headroom: float = 1.2, max_iterations: int = 6) -> Tuple[np.ndarray, float]: ...

    def waveform(self, dt: bool = False, adaptive: bool = False,
                 headroom: float = 1.2, max_iterations: int = 6) -> np.ndarray | Tuple[np.ndarray, float]:
        """
        Read waveform data from the most recent acquisition.

        For adaptive mode, reads the current acquisition and may re-trigger
        with adjusted voltage scale if needed.

        Args:
            dt: If True, return (waveform, sample_interval) tuple
            adaptive: If True, automatically adjust voltage scale for optimal signal quality.
                     May re-trigger acquisition if scale needs adjustment.
            headroom: Headroom factor for adaptive mode (must be >= 1.0)
            max_iterations: Maximum scale adjustment iterations for adaptive mode

        Returns:
            If dt=False: numpy array of voltage values
            If dt=True: tuple of (voltage_array, sample_interval_seconds)

        Examples:
            # Basic usage (after single() + wait_trigger())
            scope.single()
            waveform = scope.channels[0].waveform()

            # Get waveform with sample interval
            waveform, dt_val = scope.channels[0].waveform(dt=True)

            # Adaptive capture (reads current acquisition, may re-trigger)
            scope.single()
            waveform = scope.channels[0].waveform(adaptive=True, headroom=1.2)
        """
        if adaptive:
            if headroom < 1.0:
                raise ValueError(f"headroom must be >= 1.0, got {headroom}")

            current_vmax = self.vmax

            # Try to adapt the scale if signal doesn't fit well
            for _ in range(max_iterations):
                # Read current acquisition (first iteration uses existing data from single/wait_trigger)
                voltage_array, xincr = self._read_waveform()

                # Check if scale needs adjustment
                max_allowed = current_vmax / headroom
                peak_voltage = np.max(np.abs(voltage_array))
                pct_exceeding = 100.0 * np.sum(np.abs(voltage_array) > max_allowed) / len(voltage_array)

                if pct_exceeding > 1.0:
                    current_vmax = current_vmax * 2.0
                    print(f'  CH{self._ch_num + 1}: {pct_exceeding:.1f}% samples exceed headroom, zooming out to {current_vmax:.3f}V', file=sys.stderr)
                    self.vmax = current_vmax
                    self._scope.single()
                    continue

                # Zoom in if peak < 25% of range
                if peak_voltage < 0.25 * current_vmax and current_vmax > 0.001:
                    current_vmax = current_vmax / 2.0
                    print(f'  CH{self._ch_num + 1}: Peak {peak_voltage:.3f}V < 25% of scale, zooming in to {current_vmax:.3f}V', file=sys.stderr)
                    self.vmax = current_vmax
                    self._scope.single()
                    continue

                # Scale is good
                break
        else:
            # Non-adaptive: just read current waveform
            voltage_array, xincr = self._read_waveform()

        # Return based on dt flag
        if dt:
            return voltage_array, xincr
        else:
            return voltage_array

    def _read_waveform(self) -> Tuple[np.ndarray, float]:
        """Internal method to read waveform data from scope's acquisition buffer."""
        ch_num = self._ch_num + 1  # 1-based for SCPI

        # Get waveform preamble for scaling info
        preamble_str = self._scope._query(f":WAVeform:SOURce CHANnel{ch_num};:WAVeform:PREamble?")
        preamble = [float(x) for x in preamble_str.split(',')]

        # Extract scaling parameters (format: format, type, points, count, xincr, xorig, xref, yincr, yorig, yref)
        xincr = preamble[4]  # Sample interval (dt)
        yincr, yorig, yref = preamble[7:10]

        # Read definite-length block: #<n><length><data><newline>
        # Parse header to determine exact byte count, avoiding buffer desync
        self._scope._write(f":WAVeform:DATA?")
        header = self._scope.inst.read_bytes(2)  # "#" + digit count
        n_digits = int(chr(header[1]))
        length = int(self._scope.inst.read_bytes(n_digits))
        data_bytes = self._scope.inst.read_bytes(length)
        self._scope.inst.read_bytes(1)  # consume trailing newline

        data_array = np.frombuffer(data_bytes, dtype="<u2").astype(np.float64)
        voltage_array = (data_array - (yref - yorig)) * yincr

        return voltage_array, xincr


class AFG:
    """
    Arbitrary Function Generator (AFG) interface.

    Provides property-based access to AFG parameters. All changes
    are queued in the parent Scope's queue.

    Properties are automatically generated from AFG_PARAMS table.
    """

    if TYPE_CHECKING:
        @property
        def enabled(self) -> bool:
            """Whether the AFG output is enabled."""
            ...
        @enabled.setter
        def enabled(self, value: bool) -> None: ...

        @property
        def function(self) -> Literal['SINusoid', 'SQUare', 'RAMP', 'PULSe', 'DC', 'NOISe', 'ARB']:
            """Waveform function type."""
            ...
        @function.setter
        def function(self, value: Literal['SINusoid', 'SQUare', 'RAMP', 'PULSe', 'DC', 'NOISe', 'ARB']) -> None: ...

        @property
        def termination(self) -> float:
            """Load termination resistance (default: inf). Compensates voltage/offset for voltage divider."""
            ...
        @termination.setter
        def termination(self, value: float) -> None: ...

        @property
        def amplitude(self) -> float:
            """Output amplitude (peak, not peak-to-peak). Accepts SI strings like '1V'. Set termination first."""
            ...
        @amplitude.setter
        def amplitude(self, value: float | str) -> None: ...

        @property
        def frequency(self) -> float:
            """Output frequency in Hz. Accepts SI strings like '1kHz'."""
            ...
        @frequency.setter
        def frequency(self, value: float | str) -> None: ...

        @property
        def offset(self) -> float:
            """DC offset at the load. Accepts SI strings like '500mV'. Set termination first."""
            ...
        @offset.setter
        def offset(self, value: float | str) -> None: ...

        @property
        def vrange(self) -> Tuple[float, float]:
            """Output voltage range (min, max) at the load. Settable with (V_min, V_max) tuple. Accepts SI strings."""
            ...
        @vrange.setter
        def vrange(self, value: Tuple[float | str, float | str]) -> None: ...

        @property
        def phase(self) -> float:
            """Phase offset in degrees (0-360)."""
            ...
        @phase.setter
        def phase(self, value: float) -> None: ...

        @property
        def duty(self) -> float:
            """Duty cycle for square wave (0-100%)."""
            ...
        @duty.setter
        def duty(self, value: float) -> None: ...

        @property
        def symmetry(self) -> float:
            """Symmetry for ramp wave (0-100%)."""
            ...
        @symmetry.setter
        def symmetry(self, value: float) -> None: ...

    def __init__(self, scope: Scope):
        """
        Initialize AFG.

        Args:
            scope: Parent Scope instance
        """
        self._scope = scope
        self._termination: float = float('inf')

    @property
    def termination(self) -> float:
        """
        Load termination resistance in Ohms (default: inf for high-impedance).

        When set, amplitude and offset values are automatically scaled to compensate
        for the voltage divider formed by the AFG's 50Ω output impedance and the
        load termination. Set this property BEFORE setting amplitude or offset.

        Common values: 50 (for 50Ω termination), inf (high-impedance, no compensation).
        """
        return self._termination

    @termination.setter
    def termination(self, value: float | str) -> None:
        if isinstance(value, str):
            # Parse SI strings like '50 Ohm', '50Ω', or just '50'
            value = parse_si(value, unit='Ohm')
        self._termination = value

    def _compensation_factor(self) -> float:
        """Return the voltage compensation factor based on termination."""
        # Voltage divider: V_load = V_source * R_load / (R_source + R_load)
        # For high-impedance (inf), no compensation needed (factor = 1.0)
        if self._termination == float('inf'):
            return 1.0
        return (50.0 + self._termination) / self._termination

    @property
    def amplitude(self) -> float:
        """
        Output amplitude (peak voltage, not peak-to-peak). Accepts SI strings like '1V'.

        The device internally uses Vpp, so this property converts automatically.
        If termination is set, returns the actual amplitude at the load (compensated).
        """
        # Device stores Vpp, convert to peak amplitude
        return self._amplitude_raw / self._compensation_factor() / 2

    @amplitude.setter
    def amplitude(self, value: float | str) -> None:
        # Convert peak amplitude to Vpp for device
        self._amplitude_raw = _normalize_value(value, Type.VOLTAGE) * 2 * self._compensation_factor()

    @property
    def offset(self) -> float:
        """
        DC offset in volts. Accepts SI strings like '500mV'.

        If termination is set, returns the actual offset at the load (compensated).
        The AFG is commanded with a higher offset to account for the voltage divider.
        """
        return self._offset_raw / self._compensation_factor()

    @offset.setter
    def offset(self, value: float | str) -> None:
        self._offset_raw = _normalize_value(value, Type.VOLTAGE) * self._compensation_factor()

    @property
    def vrange(self) -> Tuple[float, float]:
        """
        Output voltage range (min, max) at the load.

        Can be set with a tuple (V_min, V_max) which computes amplitude and offset automatically.
        """
        return (self.offset - self.amplitude, self.offset + self.amplitude)

    @vrange.setter
    def vrange(self, value: Tuple[float | str, float | str]) -> None:
        v_min = _normalize_value(value[0], Type.VOLTAGE)
        v_max = _normalize_value(value[1], Type.VOLTAGE)
        self.amplitude = (v_max - v_min) / 2
        self.offset = (v_max + v_min) / 2


class Trigger:
    """
    Trigger interface.

    Provides property-based access to trigger parameters. All changes
    are queued in the parent Scope's queue.

    Properties are automatically generated from TRIGGER_PARAMS table.
    """

    if TYPE_CHECKING:
        @property
        def mode(self) -> Literal['EDGE', 'PULSe', 'RUNT', 'WIND', 'NEDG', 'SLOPe', 'VIDeo', 'PATTern', 'DELay', 'TIMeout', 'DURation', 'SHOLd', 'RS232', 'IIC', 'SPI']:
            """Trigger mode."""
            ...
        @mode.setter
        def mode(self, value: Literal['EDGE', 'PULSe', 'RUNT', 'WIND', 'NEDG', 'SLOPe', 'VIDeo', 'PATTern', 'DELay', 'TIMeout', 'DURation', 'SHOLd', 'RS232', 'IIC', 'SPI']) -> None: ...

        @property
        def source(self) -> str:
            """Trigger source. Accepts Channel object, channel number (1-4), or string like 'CHAN1'."""
            ...
        @source.setter
        def source(self, value: Channel | int | str) -> None: ...

        @property
        def level(self) -> float:
            """Trigger level in volts. Accepts SI strings like '500mV'."""
            ...
        @level.setter
        def level(self, value: float | str) -> None: ...

        @property
        def slope(self) -> Literal['POSitive', 'NEGative', 'EITHer']:
            """Trigger slope: POSitive, NEGative, or EITHer."""
            ...
        @slope.setter
        def slope(self, value: Literal['POSitive', 'NEGative', 'EITHer']) -> None: ...

        @property
        def sweep(self) -> Literal['AUTO', 'NORMal', 'SINGle']:
            """Trigger sweep mode: AUTO, NORMal, or SINGle."""
            ...
        @sweep.setter
        def sweep(self, value: Literal['AUTO', 'NORMal', 'SINGle']) -> None: ...

        @property
        def nreject(self) -> bool:
            """Noise rejection filter enabled."""
            ...
        @nreject.setter
        def nreject(self, value: bool) -> None: ...

    def __init__(self, scope: Scope):
        """
        Initialize Trigger.

        Args:
            scope: Parent Scope instance
        """
        self._scope = scope

    @property
    def source(self) -> str:
        """
        Trigger source (e.g., 'CHAN1', 'EXT').

        Can be set with a Channel object, channel number (1-4), or string.
        """
        return self._source_raw

    @source.setter
    def source(self, value: Channel | int | str) -> None:
        if isinstance(value, Channel):
            value = f'CHAN{value._ch_num + 1}'
        elif isinstance(value, int):
            value = f'CHAN{value}'
        self._source_raw = value


# Unified property generator - eliminates duplication
def _generate_properties(cls, params, scpi_cmd_fn):
    """
    Generate and attach properties to a class from a parameter table.

    Args:
        cls: Class to attach properties to
        params: Parameter table (list of tuples)
        scpi_cmd_fn: Function to generate SCPI command from template and instance
    """
    for name, ptype, scpi_template, valid_values, _priority in params:
        def make_getter(name, ptype, scpi_template):
            def getter(self):
                scope = self if isinstance(self, Scope) else self._scope
                scpi_cmd = scpi_cmd_fn(self, scpi_template)

                # Check cache first (committed values)
                if scpi_cmd in scope._cache:
                    return scope._cache[scpi_cmd][0]  # Return just the value

                # Not in cache: commit pending changes, query device, and cache the result
                scope.commit()
                result = scope._query(f":{scpi_cmd}?")
                parsed = scope._parse_value(result, ptype)
                scope._cache[scpi_cmd] = (parsed, ptype)
                return parsed
            return getter

        def make_setter(name, ptype, scpi_template, valid_values):
            def setter(self, value):
                scope = self if isinstance(self, Scope) else self._scope

                # Normalize value (parse SI units)
                value = _normalize_value(value, ptype)

                # Validate against valid values
                if valid_values is not None:
                    # Case-insensitive string comparison for STRING types
                    value_upper = str(value).upper()
                    if not any(value_upper == str(v).upper() for v in valid_values):
                        raise ValueError(f"Invalid {name}: {value}. Must be one of {valid_values}")

                # Check if value matches cache (skip if unchanged)
                scpi_cmd = scpi_cmd_fn(self, scpi_template)
                if scpi_cmd in scope._cache:
                    cached_value, cached_type = scope._cache[scpi_cmd]
                    # Compare values - for floats use approximate comparison
                    if ptype in (Type.VOLTAGE, Type.FREQUENCY, Type.TIME):
                        if isinstance(value, (int, float)) and isinstance(cached_value, (int, float)):
                            if abs(value - cached_value) < 1e-9:
                                return  # Value unchanged, skip
                    elif value == cached_value:
                        return  # Value unchanged, skip

                # Clear cache entry (will be re-populated on next read)
                if scpi_cmd in scope._cache:
                    del scope._cache[scpi_cmd]

                # Queue the value with SCPI command as key
                scope._queue[scpi_cmd] = (value, ptype)
            return setter

        prop = property(
            make_getter(name, ptype, scpi_template),
            make_setter(name, ptype, scpi_template, valid_values)
        )

        setattr(cls, name, prop)


# Generate all properties at module import time
_generate_properties(
    Scope,
    SCOPE_PARAMS,
    lambda _, template: template
)

_generate_properties(
    Channel,
    CHANNEL_PARAMS,
    lambda self, template: template.format(ch=self._ch_num + 1)
)

_generate_properties(
    AFG,
    AFG_PARAMS,
    lambda _, template: template
)

_generate_properties(
    Trigger,
    TRIGGER_PARAMS,
    lambda _, template: template
)

del _generate_properties
