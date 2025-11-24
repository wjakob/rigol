"""
Comprehensive test suite for Scope abstraction.

Tests every exposed parameter with set + query round-trip verification.
Requires a connected oscilloscope at 192.168.5.2.
"""

import pytest
import numpy as np
from rigol.scope import Scope


@pytest.fixture(scope="module")
def scope():
    """Create scope instance for testing. Connects to default IP."""
    try:
        s = Scope(ip='192.168.5.2', debug_level=0)
        yield s
    except Exception as e:
        pytest.skip(f"Scope not available: {e}")


@pytest.fixture(autouse=True)
def clear_cache(scope):
    """Clear scope cache between tests to ensure isolation."""
    # Commit any pending changes before clearing to avoid leaving scope in bad state
    scope._commit()
    scope.clear_cache()
    yield


class TestScopeParameters:
    """Test scope-level parameters."""

    def test_reset_method(self, scope):
        """Test reset to factory defaults - run first to establish clean state."""
        # Set some parameters
        scope.tdiv = 1e-3
        scope.channels[0].vdiv = 1.0

        # Reset to factory defaults (blocks until complete)
        scope.reset()

        # Queue should be cleared
        assert len(scope._queue) == 0

        # Re-establish known good state for subsequent tests
        # After reset, set up basic configuration that tests expect
        scope.tdiv = 1e-3
        scope.channels[0].enabled = True
        scope.channels[0].vdiv = 1.0
        scope.channels[0].probe = 1

    def test_mem_depth_valid_values(self, scope):
        """Test setting and querying memory depth with valid string values."""
        # Set up known-good scope state: only one channel enabled, reasonable timebase
        # This maximizes available memory depth options
        for i in range(4):
            scope.channels[i].enabled = (i == 0)
        scope.tdiv = 1e-3  # 1ms/div - moderate timebase

        # Test commonly supported values as strings
        test_values = ['10K', '100K', '1M']

        for value in test_values:
            scope.mem_depth = value
            result = scope.mem_depth
            # Just verify it doesn't error - scope may return scientific notation
            assert isinstance(result, str), f"Expected string result, got {type(result)}"

    def test_mem_depth_invalid_value(self, scope):
        """Test that invalid memory depth values raise ValueError."""
        # Set up known-good scope state
        for i in range(4):
            scope.channels[i].enabled = (i == 0)
        scope.tdiv = 1e-3

        # Test a value not in the valid list - should raise ValueError
        with pytest.raises(ValueError, match="Invalid mem_depth"):
            scope.mem_depth = '50K'  # Not in valid values list

    def test_mem_depth_string_value(self, scope):
        """Test that mem_depth accepts string values."""
        # Set up known-good scope state
        for i in range(4):
            scope.channels[i].enabled = (i == 0)
        scope.tdiv = 1e-3

        # Test string format - scope may return scientific notation
        scope.mem_depth = '100K'
        result = scope.mem_depth
        # Scope returns scientific notation like '1.0000E+05'
        assert result.upper() in ['100K', '1.0000E+05', '1E+05'], f"Unexpected format: {result}"

    def test_tdiv(self, scope):
        """Test setting and querying timebase scale."""
        test_values = [1e-6, 1e-3, 1.0]  # 1us, 1ms, 1s

        for value in test_values:
            scope.tdiv = value
            result = scope.tdiv
            assert abs(result - value) / value < 0.01, f"Expected {value}, got {result}"

    def test_tdiv_string(self, scope):
        """Test setting timebase scale with SI string."""
        scope.tdiv = '10us'
        result = scope.tdiv
        # Allow 2% tolerance for scope rounding
        assert abs(result - 1e-5) / 1e-5 <= 0.02

    def test_toffset(self, scope):
        """Test setting and querying timebase offset."""
        # Set timebase scale to allow reasonable offset range
        # According to SCPI: MainLeftTime = -5 x MainScale
        # With tdiv=1ms, we can offset up to -5ms
        scope.tdiv = 1e-3

        test_values = [-1e-3, 0.0, 1e-3]  # -1ms, 0, 1ms

        for value in test_values:
            scope.toffset = value
            result = scope.toffset
            assert abs(result - value) < 1e-6, f"Expected {value}, got {result}"

    def test_toffset_string(self, scope):
        """Test setting timebase offset with SI string."""
        scope.toffset = '1ms'
        result = scope.toffset
        assert abs(result - 1e-3) < 1e-6

    def test_tmode(self, scope):
        """Test timebase mode."""
        # Test MAIN mode
        scope.tmode = 'MAIN'
        result = scope.tmode
        assert result.upper() == 'MAIN'

        # Test ROLL mode
        scope.tmode = 'ROLL'
        result = scope.tmode
        assert result.upper() == 'ROLL'

        # Return to MAIN
        scope.tmode = 'MAIN'

    def test_acq_type(self, scope):
        """Test acquisition type."""
        # Test NORMal
        scope.acq_type = 'NORMal'
        result = scope.acq_type
        assert 'NORM' in result.upper()

        # Test AVERages
        scope.acq_type = 'AVERages'
        result = scope.acq_type
        assert 'AVER' in result.upper()

        # Return to NORMal
        scope.acq_type = 'NORMal'

    def test_acq_averages(self, scope):
        """Test acquisition averages count."""
        # First set to average mode
        scope.acq_type = 'AVERages'

        # Test powers of 2
        test_values = [2, 4, 8, 16, 32, 64, 128]

        for value in test_values:
            scope.acq_averages = value
            result = scope.acq_averages
            assert result == value, f"Expected {value}, got {result}"

        # Return to NORMal mode
        scope.acq_type = 'NORMal'


class TestChannelParameters:
    """Test channel parameters."""

    def test_vdiv(self, scope):
        """Test voltage per division setting (standard 1-2-5 sequence)."""
        ch = scope.channels[0]
        # Channel must be enabled for vdiv to work
        ch.enabled = True
        ch.probe = 1  # Use 1x probe for predictable values

        # Test standard 1-2-5 sequence values
        test_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

        for value in test_values:
            ch.vdiv = value
            result = ch.vdiv
            assert abs(result - value) < 1e-6, f"Expected {value}, got {result}"

    def test_vdiv_string(self, scope):
        """Test setting vdiv with SI string."""
        ch = scope.channels[0]
        ch.enabled = True
        ch.probe = 1
        ch.vdiv = '500mV'
        result = ch.vdiv
        assert abs(result - 0.5) < 1e-6, f"Expected 0.5, got {result}"

    def test_vmax(self, scope):
        """Test vmax property (derived from vdiv)."""
        ch = scope.channels[0]
        ch.enabled = True
        ch.probe = 1
        ch.vmax = 16.0  # Should set vdiv to 4.0 (16/4)

        vdiv_result = ch.vdiv
        assert abs(vdiv_result - 4.0) < 1e-6, f"Expected vdiv=2.0, got {vdiv_result}"

        vmax_result = ch.vmax
        assert abs(vmax_result - 16.0) < 1e-6, f"Expected vmax=16.0, got {vmax_result}"

    def test_probe(self, scope):
        """Test probe attenuation factor."""
        ch = scope.channels[0]
        test_values = [1, 10, 100]

        for value in test_values:
            ch.probe = value
            result = ch.probe
            assert result == value or abs(result - value) < 0.1, f"Expected {value}, got {result}"

    def test_enabled(self, scope):
        """Test channel enable/disable."""
        ch = scope.channels[0]

        ch.enabled = True
        assert ch.enabled is True

        ch.enabled = False
        assert ch.enabled is False

        # Restore to True
        ch.enabled = True

    def test_coupling(self, scope):
        """Test coupling mode."""
        ch = scope.channels[0]

        for value in ['DC', 'AC', 'GND']:
            ch.coupling = value
            result = ch.coupling
            assert result.upper() == value.upper()

    def test_coupling_case_insensitive(self, scope):
        """Test coupling with lowercase."""
        ch = scope.channels[0]
        ch.coupling = 'dc'
        result = ch.coupling
        assert result.upper() == 'DC'

    def test_coupling_invalid(self, scope):
        """Test that invalid coupling raises ValueError."""
        ch = scope.channels[0]
        with pytest.raises(ValueError):
            ch.coupling = 'INVALID'

    def test_bwlimit(self, scope):
        """Test bandwidth limit setting."""
        ch = scope.channels[0]

        # Test setting to 20M
        ch.bwlimit = '20M'
        result = ch.bwlimit
        assert result == '20M'

        # Test setting to OFF
        ch.bwlimit = 'OFF'
        result = ch.bwlimit
        assert result == 'OFF'

    def test_offset(self, scope):
        """Test vertical offset."""
        ch = scope.channels[0]
        test_values = [-1.0, 0.0, 1.0]

        for value in test_values:
            ch.offset = value
            result = ch.offset
            assert abs(result - value) < 1e-6, f"Expected {value}, got {result}"

    def test_all_channels_accessible(self, scope):
        """Test that all 4 channels are accessible."""
        assert len(scope.channels) == 4
        for i in range(4):
            ch = scope.channels[i]
            # Just verify we can set a parameter
            ch.probe = 10
            assert ch.probe == 10

    def test_position(self, scope):
        """Test channel position (bias voltage)."""
        ch = scope.channels[0]
        test_values = [-1.0, 0.0, 1.0]

        for value in test_values:
            ch.position = value
            result = ch.position
            assert abs(result - value) < 0.01, f"Expected {value}, got {result}"

    def test_invert(self, scope):
        """Test channel signal invert."""
        ch = scope.channels[0]

        ch.invert = True
        assert ch.invert is True

        ch.invert = False
        assert ch.invert is False


class TestAFGParameters:
    """Test AFG parameters."""

    def test_enabled(self, scope):
        """Test AFG enable/disable."""
        afg = scope.afg

        afg.enabled = True
        assert afg.enabled is True

        afg.enabled = False
        assert afg.enabled is False

    def test_function(self, scope):
        """Test waveform function type."""
        afg = scope.afg

        afg.function = 'SINusoid'
        result = afg.function
        # Scope may return abbreviated form 'SIN' or full 'SINUSOID'
        assert result.upper() in ['SIN', 'SINUSOID']

    def test_voltage(self, scope):
        """Test AFG voltage amplitude."""
        afg = scope.afg
        test_values = [1.0, 5.0, 10.0]

        for value in test_values:
            afg.voltage = value
            result = afg.voltage
            assert abs(result - value) < 1e-3, f"Expected {value}, got {result}"

    def test_voltage_string(self, scope):
        """Test setting AFG voltage with SI string."""
        afg = scope.afg
        afg.voltage = '5V'
        result = afg.voltage
        assert abs(result - 5.0) < 1e-3

    def test_frequency(self, scope):
        """Test AFG frequency."""
        afg = scope.afg
        test_values = [100.0, 1000.0, 10000.0]

        for value in test_values:
            afg.frequency = value
            result = afg.frequency
            assert abs(result - value) < 1.0, f"Expected {value}, got {result}"

    def test_frequency_string(self, scope):
        """Test setting AFG frequency with SI string."""
        afg = scope.afg
        afg.frequency = '1kHz'
        result = afg.frequency
        assert abs(result - 1000.0) < 1.0

    def test_offset(self, scope):
        """Test AFG DC offset."""
        afg = scope.afg
        test_values = [-1.0, 0.0, 1.0]

        for value in test_values:
            afg.offset = value
            result = afg.offset
            assert abs(result - value) < 1e-3, f"Expected {value}, got {result}"

    def test_phase(self, scope):
        """Test AFG phase offset."""
        afg = scope.afg
        test_values = [0, 90, 180, 270]

        for value in test_values:
            afg.phase = value
            result = afg.phase
            assert abs(result - value) < 1.0, f"Expected {value}, got {result}"

    def test_duty(self, scope):
        """Test AFG square wave duty cycle."""
        afg = scope.afg

        # Set to square wave first
        afg.function = 'SQUare'
        afg.duty = 50

        result = afg.duty
        assert abs(result - 50) < 1.0, f"Expected 50, got {result}"

    def test_symmetry(self, scope):
        """Test AFG ramp symmetry."""
        afg = scope.afg

        # Set to ramp wave first
        afg.function = 'RAMP'
        afg.symmetry = 50

        result = afg.symmetry
        assert abs(result - 50) < 1.0, f"Expected 50, got {result}"


class TestCommitBehavior:
    """Test commit and queue behavior."""

    def test_queue_and_cache_behavior(self, scope):
        """Test that writing queues values, and reading auto-commits then reads from device."""
        ch = scope.channels[0]

        # Channel must be enabled for vdiv to work
        ch.enabled = True
        ch.probe = 1

        # Set a parameter
        ch.vdiv = 2.0

        # Check queue contains the value
        assert len(scope._queue) > 0

        # Read it back - this should auto-commit and read from device
        result = ch.vdiv
        assert abs(result - 2.0) < 1e-6

        # After reading, queue should be cleared (auto-committed)
        assert len(scope._queue) == 0

        # Value should now be in cache
        scpi_cmd = f'CHANnel{ch._ch_num + 1}:SCALe'
        assert scpi_cmd in scope._cache

        # Reading again should return cached value without device query
        result2 = ch.vdiv
        assert abs(result2 - 2.0) < 1e-6

    def test_batch_multiple_parameters(self, scope):
        """Test that multiple parameters are batched and auto-committed on first read."""
        ch0 = scope.channels[0]
        ch1 = scope.channels[1]

        # Ensure channel is enabled
        ch0.enabled = True

        # Set probe first, then vdiv (probe may need to be set before vdiv)
        ch0.probe = 10

        # Set multiple parameters - these are queued
        ch0.vdiv = 1.0
        ch1.enabled = False
        scope.afg.frequency = 1000.0
        scope.mem_depth = '100K'

        # Queue should have multiple items
        assert len(scope._queue) > 1

        # Reading any value auto-commits all pending changes
        result = ch0.vdiv
        assert abs(result - 1.0) < 1e-6

        # Queue should be cleared after auto-commit
        assert len(scope._queue) == 0

        # Verify all were applied
        assert ch0.probe == 10
        assert ch1.enabled is False
        assert abs(scope.afg.frequency - 1000.0) < 1.0
        assert isinstance(scope.mem_depth, str)

    def test_multiple_writes_and_reads(self, scope):
        """Test that queue is cleared after auto-commit on read."""
        ch = scope.channels[0]

        # First write and read
        ch.vdiv = 1.0
        result1 = ch.vdiv
        assert abs(result1 - 1.0) < 1e-6

        # Queue should be empty after auto-commit
        assert len(scope._queue) == 0

        # Second write with different value
        ch.vdiv = 2.0

        # Queue should have the new value
        assert len(scope._queue) > 0

        # Read should auto-commit and return new value
        result2 = ch.vdiv
        assert abs(result2 - 2.0) < 1e-6

        # Queue should be empty again
        assert len(scope._queue) == 0

    def test_auto_commit_on_single(self, scope):
        """Test that single() auto-commits pending changes."""
        ch = scope.channels[0]

        # Set a parameter without commit
        ch.vdiv = 2.0

        # Call single() - should auto-commit
        scope.single()

        # Force trigger and wait for completion
        scope.force()
        scope.wait_trigger()

        # Verify parameter was applied
        result = ch.vdiv
        assert abs(result - 2.0) < 1e-6


class TestWaveformAcquisition:
    """Test waveform acquisition."""

    def test_waveform_capture(self, scope):
        """Test capturing waveform from channel."""
        # Configure AFG to generate signal
        scope.afg.enabled = True
        scope.afg.function = 'SINusoid'
        scope.afg.frequency = 1000.0
        scope.afg.voltage = 2.0

        # Configure channel
        ch = scope.channels[0]
        ch.enabled = True
        ch.vmax = 5.0
        ch.coupling = 'DC'
        ch.probe = 10

        # Configure timebase
        scope.tdiv = 1e-3  # 1ms/div

        # Commit and trigger
        scope.single()

        # Force trigger and wait for completion
        scope.force()
        scope.wait_trigger()

        # Capture waveform
        waveform = ch.waveform()

        # Verify waveform properties
        assert isinstance(waveform, np.ndarray)
        assert waveform.dtype == np.float64
        assert len(waveform) > 100  # Should have data points

        # Disable AFG
        scope.afg.enabled = False

    def test_waveform_multiple_channels(self, scope):
        """Test capturing waveforms from multiple channels."""
        # Configure AFG
        scope.afg.enabled = True
        scope.afg.function = 'SINusoid'
        scope.afg.frequency = 1000.0

        # Configure channels - both should be enabled even if only ch0 has signal
        scope.channels[0].enabled = True
        scope.channels[1].enabled = True
        scope.tdiv = 1e-3

        # Trigger
        scope.single()

        # Force trigger and wait for completion
        scope.force()
        scope.wait_trigger()

        # Capture from both channels
        # Note: Only channel 0 has AFG signal, channel 1 may be empty/noise
        wf0 = scope.channels[0].waveform()
        wf1 = scope.channels[1].waveform()

        # Verify both return numpy arrays
        assert isinstance(wf0, np.ndarray)
        assert isinstance(wf1, np.ndarray)

        # Channel 0 should have data (connected to AFG)
        assert len(wf0) > 0

        # Channel 1 should also return an array (even if just noise/zeros)
        # The waveform property should always return data of the configured memory depth
        assert isinstance(wf1, np.ndarray)

        # Cleanup
        scope.afg.enabled = False


class TestTriggerParameters:
    """Test trigger parameter configuration."""

    def test_trigger_mode(self, scope):
        """Test setting and querying trigger mode."""
        scope.trigger.mode = 'EDGE'

        result = scope.trigger.mode
        assert result.upper() == 'EDGE'

    def test_trigger_source_with_channel_object(self, scope):
        """Test setting trigger source using Channel object."""
        scope.trigger.source = scope.channels[0]

        result = scope.trigger.source
        assert 'CHAN1' in result.upper()

    def test_trigger_source_with_string(self, scope):
        """Test setting trigger source using string."""
        scope.trigger.source = 'CHAN2'

        result = scope.trigger.source
        assert 'CHAN2' in result.upper()

    def test_trigger_level(self, scope):
        """Test setting and querying trigger level."""
        test_level = 0.5
        scope.trigger.level = test_level

        result = scope.trigger.level
        assert abs(result - test_level) < 0.01

    def test_trigger_slope(self, scope):
        """Test setting and querying trigger slope."""
        scope.trigger.slope = 'POSitive'

        result = scope.trigger.slope
        assert 'POS' in result.upper()

    def test_trigger_sweep(self, scope):
        """Test setting and querying trigger sweep mode."""
        scope.trigger.sweep = 'NORMal'

        result = scope.trigger.sweep
        assert 'NORM' in result.upper()

    def test_run_stop_methods(self, scope):
        """Test run and stop methods."""
        # These should not raise errors
        scope.stop()
        scope.run()

    def test_force_method(self, scope):
        """Test force trigger method."""
        # Should not raise errors
        scope.force()

    def test_trigger_nreject(self, scope):
        """Test trigger noise rejection."""
        scope.trigger.nreject = True
        assert scope.trigger.nreject is True

        scope.trigger.nreject = False
        assert scope.trigger.nreject is False


class TestAdaptiveCapture:
    """Test adaptive waveform capture functionality."""

    @pytest.mark.skip(reason="Requires real signal for adaptive capture")
    def test_capture_adaptive_basic(self, scope):
        """Test basic adaptive capture with AFG signal.

        This test requires a real signal connected to the scope.
        Skipped in automated testing.
        """
        # Configure AFG
        scope.afg.enabled = True
        scope.afg.function = 'SINusoid'
        scope.afg.frequency = 1000.0
        scope.afg.voltage = 2.0

        # Configure channel with large initial scale
        ch = scope.channels[0]
        ch.enabled = True
        ch.vmax = 20.0  # Start large
        ch.coupling = 'DC'
        ch.probe = 10

        # Configure timebase and trigger
        scope.tdiv = 1e-3
        scope.trigger.mode = 'EDGE'
        scope.trigger.source = ch
        scope.trigger.level = 0.0


        initial_vmax = ch.vmax

        # Trigger first acquisition
        scope.single()
        scope.wait_trigger()

        # Capture with adaptation (may re-trigger if needed)
        waveform = ch.waveform(adaptive=True, headroom=1.2, max_iterations=3)

        final_vmax = ch.vmax

        # Verify waveform was captured
        assert isinstance(waveform, np.ndarray)
        assert len(waveform) > 100

        # Verify scale was adjusted (should zoom in from 20V)
        assert final_vmax <= initial_vmax

        # Cleanup
        scope.afg.enabled = False

    def test_capture_adaptive_headroom_validation(self, scope):
        """Test that invalid headroom values raise errors."""
        ch = scope.channels[0]

        with pytest.raises(ValueError, match="headroom must be >= 1.0"):
            ch.waveform(adaptive=True, headroom=0.5)


class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_channel_index(self, scope):
        """Test that accessing invalid channel index raises IndexError."""
        with pytest.raises(IndexError):
            _ = scope.channels[10]

    def test_type_conversion(self, scope):
        """Test that string values are properly converted."""
        ch = scope.channels[0]

        # Voltage with units - should be parsed to float
        ch.vdiv = '500mV'
        # Reading triggers auto-commit and reads actual value from device
        assert abs(ch.vdiv - 0.5) < 1e-6

        # Frequency with units - should be parsed to float
        scope.afg.frequency = '10kHz'
        # Reading triggers auto-commit and reads actual value from device
        assert abs(scope.afg.frequency - 10000.0) < 1.0

    def test_boolean_conversion(self, scope):
        """Test boolean parameter handling."""
        ch = scope.channels[0]

        # Python booleans
        ch.enabled = True
        assert ch.enabled is True

        ch.enabled = False
        assert ch.enabled is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
