"""
FreqProbe: helper for Bode-plot style measurements with a Rigol DHO924S.

Usage in your own code:

    probe = FreqProbe(
        resource="TCPIP0::192.168.5.2::INSTR",
        channel_a=1,
        channel_b=2,
        desired_cycles=1000,      # only affects screen timebase
        mem_depth="10K",
        max_voltage=12.0,         # channel voltage range (includes headroom)
        probe_factor=10,
        afg_amplitude_v=10.0,     # AFG output signal voltage
        headroom=1.2,             # headroom factor (1.2 = 20% headroom)
    )

    # Basic measurement
    ch_a, ch_b, dt = probe.measure(freq_hz=10e3)

    # Measurement with automatic dynamic range adjustment
    ch_a, ch_b, dt = probe.measure_dynamic(freq_hz=10e3)

    # ch_a, ch_b: NumPy arrays in volts
    # dt: sample interval in seconds
"""

import time
from typing import Tuple, Optional

import numpy as np
import pyvisa


class FreqProbe:
    def __init__(
        self,
        resource: str = "TCPIP0::192.168.5.2::INSTR",
        channel_a: int = 1,
        channel_b: int = 2,
        desired_cycles: int = 10,        # used only to choose a timebase
        mem_depth: str = "10K",          # "10K", "100K", "1M", "25M", ...
        max_voltage: float = 10.0,  # expected |V| at probe tip
        probe_factor: int = 10,          # 10× probe (must be integer)
        afg_amplitude_v: float = 10.0,
        headroom: float = 1.2,           # headroom factor for channel ranges
        visa_backend: Optional[str] = None,  # e.g. "@py" for pyvisa-py
        debug: bool = False,
        quiet: bool = False,             # suppress informational messages
    ):
        """
        Initialize FreqProbe for Bode plot measurements.

        After construction, call:
            ch_a, ch_b, dt = probe.measure(freq_hz=...)
            # or for automatic dynamic range adjustment:
            ch_a, ch_b, dt = probe.measure_dynamic(freq_hz=...)

        Parameters
        ----------
        max_voltage : float
            Channel voltage range (includes headroom above AFG signal).
            For example, if afg_amplitude_v=10.0 and headroom=1.2,
            then max_voltage should be 12.0.
            Converted to V/div ≈ max_voltage / 4 (±4 divisions on 8-div screen).
        headroom : float
            Headroom factor for channel ranges (must be >= 1.0).
            Used by measure_dynamic() to determine when to adjust scales.
            Example: 1.2 = 20% headroom above the signal.
        afg_amplitude_v : float
            AFG output signal amplitude in volts.
        """
        self.resource = resource
        self.channel_a = channel_a
        self.channel_b = channel_b
        self.desired_cycles = desired_cycles
        self.mem_depth_setting = mem_depth
        self.probe_factor = int(probe_factor)
        self.afg_amplitude_v = afg_amplitude_v
        self.headroom = headroom
        self.debug = debug
        self.quiet = quiet
        self.vdiv = max_voltage / 4.0
        self.max_voltage = max_voltage

        self.sample_interval_s: Optional[float] = None
        self.sample_rate_hz: Optional[float] = None

        # Track output channel scale across measurements
        self.current_scale = max_voltage

        # Open VISA connection
        if visa_backend is None:
            rm = pyvisa.ResourceManager()
        else:
            rm = pyvisa.ResourceManager(visa_backend)
        inst = rm.open_resource(resource)

        inst.timeout = 120_000

        self.rm = rm
        self.inst = inst

        idn = inst.query("*IDN?").strip()
        if not quiet:
            print("Connected to:", idn)

        # Validate and round memory depth to nearest supported value
        self.mem_depth_setting = self._validate_mem_depth(mem_depth)
        if self.mem_depth_setting != mem_depth.upper() and not quiet:
            print(f"Memory depth {mem_depth} rounded to {self.mem_depth_setting}")

        # Initial static configuration
        self._configure()

    # ------------------- low-level helpers -------------------

    def _write(self, cmd: str) -> None:
        """Execute a SCPI write command and check for errors."""
        if self.debug:
            print(f">> {cmd}")
        self.inst.write(cmd)

    def _query(self, cmd: str) -> str:
        """Execute a SCPI query command and check for errors."""
        if self.debug:
            print(f">> {cmd}")
        result = self.inst.query(cmd)
        if self.debug:
            print(f"<< {result.strip()}")
        return result

    @staticmethod
    def _parse_points(text: str) -> int:
        s = text.strip().upper()
        try:
            return int(float(s))
        except ValueError:
            pass
        suffix_mult = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}
        if not s:
            raise ValueError("Empty memory depth string from scope")
        last = s[-1]
        if last in suffix_mult:
            mantissa = float(s[:-1])
            return int(mantissa * suffix_mult[last])
        raise ValueError(f"Unrecognized memory depth format: {text!r}")

    @staticmethod
    def _validate_mem_depth(depth_str: str) -> str:
        """
        Validate and round memory depth to nearest supported value.

        Valid depths: 1K, 10K, 100K, 1M, 5M, 10M, 25M, 50M
        """
        # Parse the requested depth
        s = depth_str.strip().upper()
        try:
            requested = int(float(s))
        except ValueError:
            suffix_mult = {"K": 1_000, "M": 1_000_000}
            if s and s[-1] in suffix_mult:
                requested = int(float(s[:-1]) * suffix_mult[s[-1]])
            else:
                raise ValueError(f"Invalid memory depth format: {depth_str}")

        # Valid memory depths in points
        valid_depths = [1_000, 10_000, 100_000, 1_000_000, 5_000_000,
                       10_000_000, 25_000_000, 50_000_000]

        # Find nearest valid depth
        nearest = min(valid_depths, key=lambda x: abs(x - requested))

        # Convert to string format
        if nearest >= 1_000_000:
            if nearest % 1_000_000 == 0:
                return f"{nearest // 1_000_000}M"
            else:
                return f"{nearest / 1_000_000:.0f}M"
        else:
            return f"{nearest // 1_000}K"

    # ------------------- vertical scale -------------------

    def set_voltage_scale(self, channel: int, value: float) -> None:
        """
        Set the vertical scale of one channel.

        Parameters
        ----------
        channel : int
            Channel number (1-4)
        value : float
            Maximum voltage range for the channel (±value at probe tip)
            Converted to V/div ≈ value / 4 (8 vertical divisions total)
        """
        self._write(f":CHANnel{channel}:SCALe {value/4}")

    # ------------------- configuration -------------------

    def _configure(self) -> None:
        self._write(f":SOURce:FUNCtion SINusoid")
        self._write(f":SOURce:VOLTage:AMPLitude {self.afg_amplitude_v}")
        self._write(f":SOURce:VOLTage:OFFSet 0")
        self._write(f":SOURce:OUTPut:STATe ON")

        for ch in range(1, 5):
            if ch in (self.channel_a, self.channel_b):
                self._write(f":CHANnel{ch}:DISPlay ON")
                self._write(f":CHANnel{ch}:BWLimit 20M")
                self._write(f":CHANnel{ch}:COUPling DC")
                self._write(f":CHANnel{ch}:PROBe {self.probe_factor}")
                self._write(f":CHANnel{ch}:SCALe {self.vdiv}")
                self._write(f":CHANnel{ch}:OFFSet 0")
            else:
                self._write(f":CHANnel{ch}:DISPlay OFF")


        self._write(f":TIMebase:MODE MAIN")
        self._write(f":TIMebase:HREFerence:MODE CENTer")

        self._write(":ACQuire:TYPE NORMal")
        self._write(":ACQuire:AVERages 1")
        self._write(f":ACQuire:MDEPth {self.mem_depth_setting}")
        self.mem_depth_points = self._parse_points(self.mem_depth_setting)

        self._write(f":TRIGger:MODE EDGE")
        self._write(f":TRIGger:EDGE:SOURce CHANnel{self.channel_a}")
        self._write(f":TRIGger:EDGE:LEVel 0.0")
        self._write(f":TRIGger:EDGE:SLOPe POSitive")
        self._write(f":TRIGger:SWEep NORMal")

        self._write(":WAVeform:MODE RAW")
        self._write(":WAVeform:FORMat WORD")
        self._write(f":WAVeform:POINts {self.mem_depth_points}")

    def _wait_for_trigger_and_stop(self) -> None:
        """
        Wait for single-shot acquisition to complete and stop.

        Assumes :SINGle was already called to arm the trigger.
        Polls :TRIGger:STATus? until "TD" or "STOP", then issues :STOP
        to freeze the waveform buffer for reading.
        """
        # Wait for trigger to be detected
        # In single-shot mode, status goes: WAIT -> TD -> STOP
        # We check for TD to know when waveform is being captured
        t0 = time.time()
        while True:
            status = self._query(":TRIGger:STATus?").strip().upper()
            if status == "TD" or status == "STOP":  # Trigger detected or already stopped
                break
            if time.time() - t0 > 5.0:
                raise TimeoutError(
                    f"Acquisition did not trigger within 5 s; last status={status!r}"
                )
            time.sleep(0.01)

        # Explicitly stop to freeze waveform buffer for reading
        # This is needed even in single-shot mode to ensure RAW data is accessible
        self._write(":STOP")

    # ------------------- waveform I/O -------------------

    @staticmethod
    def _parse_tmc_block(raw: bytes) -> bytes:
        if not raw or raw[0:1] != b"#":
            raise ValueError(f"Expected SCPI block starting with '#', got: {raw[:10]!r}")
        n_digits = int(raw[1:2].decode("ascii"))
        length_str = raw[2:2 + n_digits].decode("ascii")
        length = int(length_str)
        start = 2 + n_digits
        end = start + length
        return raw[start:end]

    def _read_waveform(self, channel: int) -> Tuple[np.ndarray, dict]:
        """
        Read WORD-format waveform and scaling for one channel.

        Returns (v, meta), where v is in volts and meta contains xincr, etc.
        We do NOT transfer a time array; dt = xincr is all you need for
        regular sampling.

        Parameters
        ----------
        channel : int
            Channel number to read
        """
        # Batch source selection with preamble query
        # Use :WAVeform:PREamble? instead of 6 separate queries
        # Preamble format: format,type,points,count,xincr,xorig,xref,yincr,yorig,yref
        preamble = self._query(f":WAVeform:SOURce CHANnel{channel};:WAVeform:PREamble?").strip()
        parts = preamble.split(',')

        if len(parts) < 10:
            raise ValueError(f"Unexpected preamble format: {preamble}")

        # Parse preamble fields
        # format: 0=BYTE, 1=WORD, 2=ASCii
        # type: 0=NORMal, 1=PEAK, 2=AVERage
        xincr = float(parts[4])
        xorig = float(parts[5])
        xref = float(parts[6])
        yincr = float(parts[7])
        yorig = float(parts[8])
        yref = float(parts[9])

        # Batch source selection with data request
        self._write(f":WAVeform:SOURce CHANnel{channel};:WAVeform:DATA?")
        raw_block = self.inst.read_raw()
        data_bytes = self._parse_tmc_block(raw_block)

        codes = np.frombuffer(data_bytes, dtype="<u2")

        # Check that we got data
        if codes.size == 0:
            raise ValueError(
                f"No waveform data received for channel {channel}. "
                f"Raw block size: {len(raw_block)} bytes, "
                f"Data bytes: {len(data_bytes)} bytes. "
                f"Make sure acquisition completed before reading waveform."
            )

        volts = (codes.astype(np.float64) - (yorig + yref)) * yincr

        meta = {
            "channel": channel,
            "npoints": int(codes.size),
            "xincr": xincr,
            "xorig": xorig,
            "xref": xref,
            "yincr": yincr,
            "yorig": yorig,
            "yref": yref,
            "codes_min": int(codes.min()),
            "codes_max": int(codes.max()),
        }
        return volts, meta

    # ------------------- public API -------------------

    def measure(self, freq_hz: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Configure the scope and AFG for freq_hz, acquire one *fresh* record, and
        return (channel_a, channel_b, dt) where dt is the sample interval
        in seconds.

        channel_a and channel_b are NumPy arrays in volts, from the same
        acquisition (time-aligned). Their lengths should match.
        """
        if freq_hz <= 0:
            raise ValueError("freq_hz must be > 0")

        # Batch configure and trigger acquisition
        target_span = self.desired_cycles / freq_hz
        safety_factor = 1.2
        visible_span = target_span * safety_factor
        tscale = visible_span / 10.0

        # Batch: timebase + frequency + trigger setup + arm
        self._write(f":TIMebase:MAIN:SCALe {tscale};:SOURce:FREQuency {freq_hz};:TRIGger:SWEep SINGle;:SINGle")

        # Wait for trigger and stop
        self._wait_for_trigger_and_stop()

        # Read both channels from the same acquisition
        v_a, meta_a = self._read_waveform(self.channel_a)
        v_b, meta_b = self._read_waveform(self.channel_b)

        # Sanity: lengths
        assert len(v_a) == len(v_b), "CH1/CH2 length mismatch"

        # Sample interval (seconds), same for both channels
        dt_a = meta_a["xincr"]
        dt_b = meta_b["xincr"]
        assert np.isclose(dt_a, dt_b, rtol=1e-6, atol=0), "CH1/CH2 dt mismatch"
        dt = dt_a

        self.sample_interval_s = dt
        self.sample_rate_hz = 1.0 / dt if dt > 0 else None

        return v_a, v_b, dt

    def measure_dynamic(self, freq_hz: float, max_scale_adjustments: int = 6) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Measure with dynamic range adjustment for the output channel (channel_b).

        This method automatically adjusts the output channel voltage scale to optimize
        signal quality by analyzing the actual voltage samples:
        - Zooms out if >1% of samples exceed the headroom threshold (prevents clipping)
        - Zooms in if peak voltage < 25% of scale (improves SNR)

        Hysteresis prevents oscillation: with 2× scaling, 25% threshold ensures a stable
        zone exists between zoom-in and zoom-out thresholds.

        The voltage scale is maintained across calls, so successive measurements at
        different frequencies can benefit from the previously optimized scale.

        Parameters
        ----------
        freq_hz : float
            Frequency to measure at
        max_scale_adjustments : int
            Maximum number of scale adjustment iterations (default: 6)

        Returns
        -------
        v_a : np.ndarray
            Channel A (input) voltage samples
        v_b : np.ndarray
            Channel B (output) voltage samples
        dt : float
            Sample interval in seconds
        """
        # Set output channel to current scale (persistent across measurements)
        self.set_voltage_scale(channel=self.channel_b, value=self.current_scale)

        # Try to adapt the output scale if signal doesn't fit well
        for _ in range(max_scale_adjustments):
            v_a, v_b, dt = self.measure(freq_hz=freq_hz)

            # Voltage range is ±current_scale (8 divisions × scale/4 V/div = ±4 × scale/4 = ±scale)
            # E.g., scale=10V means we send SCALe 2.5V/div, giving range ±10V
            max_allowed = self.current_scale / self.headroom
            peak_voltage = np.max(np.abs(v_b))

            # Zoom out if >1% of samples exceed headroom
            pct_exceeding = 100.0 * np.sum(np.abs(v_b) > max_allowed) / len(v_b)
            if pct_exceeding > 1.0 and self.current_scale < self.max_voltage:
                new_scale = min(self.current_scale * 2.0, self.max_voltage)
                if not self.quiet:
                    print(f'  CH{self.channel_b}: {pct_exceeding:.1f}% samples exceed headroom, zooming out to {new_scale:.3f}V')
                self.current_scale = new_scale
                self.set_voltage_scale(channel=self.channel_b, value=self.current_scale)

            # Zoom in if peak < 25% of range
            elif peak_voltage < 0.25 * self.current_scale and self.current_scale > 0.001:
                new_scale = self.current_scale / 2.0
                if not self.quiet:
                    print(f'  CH{self.channel_b}: Peak {peak_voltage:.3f}V < 25% of scale, zooming in to {new_scale:.3f}V')
                self.current_scale = new_scale
                self.set_voltage_scale(channel=self.channel_b, value=self.current_scale)

            else:
                # Scale is good
                break

        return v_a, v_b, dt

    def resume_acquisition(self) -> None:
        """Return the oscilloscope to normal RUN mode."""
        self._write(f":SOURce:OUTPut:STATe OFF")
        self._write(":RUN")

    def close(self) -> None:
        """Return scope to RUN mode and close the VISA session."""
        try:
            self.resume_acquisition()
        finally:
            try:
                self.inst.close()
            finally:
                self.rm.close()
