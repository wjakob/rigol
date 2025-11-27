"""
Generic scope abstraction for SCPI-controlled oscilloscopes.

This module provides a property-based API for controlling oscilloscope parameters
with batched command execution for efficiency.

Properties are automatically generated from parameter tables for efficiency and maintainability.
"""

from enum import Enum
from typing import Optional, Any, Dict, Tuple, overload, Literal
import numpy as np
import pyvisa
import sys
import time

from .util import parse_si


class Type(Enum):
    """Parameter type enum for SCPI value formatting."""
    UNITLESS = 1
    VOLTAGE = 2
    STRING = 3
    FREQUENCY = 4
    BOOLEAN = 5
    TIME = 6


# Parameter table structure: (name, type, scpi_template, valid_values)
SCOPE_PARAMS = [
    ('tdiv', Type.TIME, 'TIMebase:MAIN:SCALe', None),
    ('toffset', Type.TIME, 'TIMebase:MAIN:OFFSet', None),
    ('tmode', Type.STRING, 'TIMebase:MODE', ['MAIN', 'XY', 'ROLL']),
    ('mem_depth', Type.STRING, 'ACQuire:MDEPth', ['1K', '10K', '100K', '1M', '10M', '25M', '50M']),
    ('acq_type', Type.STRING, 'ACQuire:TYPE', ['NORMal', 'PEAK', 'AVERages', 'ULTRa']),
    ('acq_averages', Type.UNITLESS, 'ACQuire:AVERages', None),
]

CHANNEL_PARAMS = [
    ('vdiv', Type.VOLTAGE, 'CHANnel{ch}:SCALe', None),
    ('probe', Type.UNITLESS, 'CHANnel{ch}:PROBe', None),
    ('enabled', Type.BOOLEAN, 'CHANnel{ch}:DISPlay', None),
    ('coupling', Type.STRING, 'CHANnel{ch}:COUPling', ['DC', 'AC', 'GND']),
    ('bwlimit', Type.STRING, 'CHANnel{ch}:BWLimit', ['20M', 'OFF']),
    ('offset', Type.VOLTAGE, 'CHANnel{ch}:OFFSet', None),
    ('position', Type.VOLTAGE, 'CHANnel{ch}:POSition', None),
    ('invert', Type.BOOLEAN, 'CHANnel{ch}:INVert', None),
]

AFG_PARAMS = [
    ('enabled', Type.BOOLEAN, 'SOURce:OUTPut:STATe', None),
    ('function', Type.STRING, 'SOURce:FUNCtion', ['SINusoid', 'SQUare', 'RAMP', 'PULSe', 'DC', 'NOISe', 'ARB']),
    ('voltage', Type.VOLTAGE, 'SOURce:VOLTage:AMPLitude', None),
    ('frequency', Type.FREQUENCY, 'SOURce:FREQuency', None),
    ('offset', Type.VOLTAGE, 'SOURce:VOLTage:OFFSet', None),
    ('phase', Type.UNITLESS, 'SOURce:PHASe', None),
    ('duty', Type.UNITLESS, 'SOURce:FUNCtion:SQUare:DUTY', None),
    ('symmetry', Type.UNITLESS, 'SOURce:FUNCtion:RAMP:SYMMetry', None),
]

TRIGGER_PARAMS = [
    ('mode', Type.STRING, 'TRIGger:MODE', None),
    ('level', Type.VOLTAGE, 'TRIGger:EDGE:LEVel', None),
    ('slope', Type.STRING, 'TRIGger:EDGE:SLOPe', None),
    ('sweep', Type.STRING, 'TRIGger:SWEep', None),
    ('nreject', Type.BOOLEAN, 'TRIGger:NREJect', None),
]


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


def _parse_tmc_block(raw: bytes) -> bytes:
    """
    Parse a SCPI definite-length block (#<digits><len><data>...).

    Returns the data payload (excluding the header). Raises ValueError if the
    block is malformed.
    """
    if not raw or raw[0:1] != b"#":
        raise ValueError(f"Expected SCPI block starting with '#', got: {raw[:10]!r}")

    n_digits = int(raw[1:2].decode("ascii"))
    length_str = raw[2:2 + n_digits].decode("ascii")
    length = int(length_str)
    start = 2 + n_digits
    end = start + length
    return raw[start:end]


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
        scope.afg.voltage = '10V'
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

    def __init__(self, ip: str = '192.168.5.2', debug_level: int = 0):
        """
        Initialize scope connection.

        Args:
            ip: IP address of the oscilloscope
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
        resource = f"TCPIP0::{ip}::INSTR"
        rm = pyvisa.ResourceManager()
        self.inst = rm.open_resource(resource)
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

        if self.debug_level >= 2:
            # Debug mode: send commands one-by-one to catch errors
            for scpi_cmd, (value, ptype) in self._queue.items():
                value_str = ('ON' if value else 'OFF') if ptype == Type.BOOLEAN else str(value)
                cmd = f":{scpi_cmd} {value_str}"
                self._write(cmd)

            if extra_cmd:
                self._write(extra_cmd)
        else:
            # Normal mode: batch commands for efficiency
            batched_cmd = ''
            for scpi_cmd, (value, ptype) in self._queue.items():
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
    def tmax(self):
        """Total time on screen (10 horizontal divisions). Derived from tdiv."""
        return self.tdiv * 10

    @tmax.setter
    def tmax(self, value):
        """Set tdiv based on desired total time span."""
        self.tdiv = value / 10


class Channel:
    """
    Oscilloscope channel interface.

    Provides property-based access to channel parameters. All changes
    are queued in the parent Scope's queue.

    Properties are automatically generated from CHANNEL_PARAMS table.
    """

    def __init__(self, scope: Scope, ch_num: int):
        """
        Initialize channel.

        Args:
            scope: Parent Scope instance
            ch_num: 0-based channel index (0-3 for 4-channel scope)
        """
        self._scope = scope
        self._ch_num = ch_num

    # Special property with custom logic
    @property
    def vmax(self):
        """Maximum voltage (full scale = 4 divisions). Derived from vdiv."""
        return self.vdiv * 4

    @vmax.setter
    def vmax(self, value):
        """Set vdiv based on desired maximum voltage."""
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

        # Read raw block and parse manually, potentially repeat until we get a usable result
        while True:
            self._scope._write(f":WAVeform:DATA?")
            data_bytes = _parse_tmc_block(self._scope.inst.read_raw())
            if len(data_bytes) != 0:
                break

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

    def __init__(self, scope: Scope):
        """
        Initialize AFG.

        Args:
            scope: Parent Scope instance
        """
        self._scope = scope


class Trigger:
    """
    Trigger interface.

    Provides property-based access to trigger parameters. All changes
    are queued in the parent Scope's queue.

    Properties are automatically generated from TRIGGER_PARAMS table.
    """

    def __init__(self, scope: Scope):
        """
        Initialize Trigger.

        Args:
            scope: Parent Scope instance
        """
        self._scope = scope

    @property
    def source(self):
        """Get trigger source (e.g., 'CHAN1', 'EXT')."""
        scpi_cmd = 'TRIGger:EDGE:SOURce'

        # Check cache first (committed values)
        if scpi_cmd in self._scope._cache:
            return self._scope._cache[scpi_cmd][0]

        # Not in cache: commit pending changes, query device, and cache the result
        self._scope.commit()
        result = self._scope._query(f':{scpi_cmd}?')
        value = result.strip()
        self._scope._cache[scpi_cmd] = (value, Type.STRING)
        return value

    @source.setter
    def source(self, value):
        """
        Set trigger source.

        Args:
            value: Can be a Channel object (e.g., scope.channels[0]) or a string (e.g., 'CHAN1', 'EXT')
        """
        # Import here to avoid circular dependency
        if isinstance(value, Channel):
            # Convert Channel object to SCPI string
            value = f'CHAN{value._ch_num + 1}'

        scpi_cmd = 'TRIGger:EDGE:SOURce'

        # Clear cache entry (will be re-populated on next read)
        if scpi_cmd in self._scope._cache:
            del self._scope._cache[scpi_cmd]

        self._scope._queue[scpi_cmd] = (value, Type.STRING)


# Unified property generator - eliminates duplication
def _generate_properties(cls, params, scpi_cmd_fn):
    """
    Generate and attach properties to a class from a parameter table.

    Args:
        cls: Class to attach properties to
        params: Parameter table (list of tuples)
        scpi_cmd_fn: Function to generate SCPI command from template and instance
    """
    for name, ptype, scpi_template, valid_values in params:
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
