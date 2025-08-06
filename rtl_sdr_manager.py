#!/usr/bin/env python3
"""
RTL-SDR Manager for Audio Signal Detection and Demodulation

This module provides a comprehensive interface for RTL-SDR operations,
focusing on efficient frequency sweeps and audio signal capture.
"""

import numpy as np
import scipy.signal
from scipy.io import wavfile
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, NamedTuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
from pathlib import Path

try:
    from rtlsdr import RtlSdr
    RTL_SDR_AVAILABLE = True
except ImportError:
    RTL_SDR_AVAILABLE = False
    logging.warning("RTL-SDR library not available. Mock mode only.")


class ModulationType(Enum):
    """Supported modulation types for audio demodulation."""
    AM = "AM"
    FM = "FM"
    WFM = "WFM"  # Wideband FM (broadcast)
    NFM = "NFM"  # Narrowband FM (comms)


@dataclass
class FrequencyBand:
    """Definition of a frequency band with its characteristics."""
    name: str
    start_freq: float  # Hz
    end_freq: float    # Hz
    modulation: ModulationType
    channel_spacing: float  # Hz
    audio_bandwidth: float  # Hz
    priority: int = 1  # Higher numbers = higher priority


@dataclass
class SignalDetection:
    """Represents a detected signal with its properties."""
    frequency: float
    power_db: float
    bandwidth: float
    snr_db: float
    band_info: FrequencyBand
    timestamp: float
    confidence: float = 0.0


@dataclass
class AudioCapture:
    """Contains demodulated audio data and metadata."""
    signal_info: SignalDetection
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    file_path: Optional[str] = None


class RTLSDRManager:
    """
    Comprehensive RTL-SDR manager for audio signal detection and capture.
    
    Supports efficient frequency sweeps, signal detection, and audio demodulation
    for various radio services.
    """
    
    # RTL-SDR hardware specifications
    MIN_FREQ = 500e3      # 500 kHz
    MAX_FREQ = 1.7e9      # 1.7 GHz  
    BANDWIDTH = 2.4e6     # 2.4 MHz
    
    # Default sample rates
    SWEEP_SAMPLE_RATE = 1.0e6    # Reduced to 1MHz to avoid overflow
    AUDIO_SAMPLE_RATE = 256e3    # Sufficient for audio demodulation
    
    # Audio frequency bands of interest
    AUDIO_BANDS = [
        FrequencyBand("AM Broadcast", 530e3, 1700e3, ModulationType.AM, 10e3, 5e3, priority=5),
        FrequencyBand("FM Broadcast", 88e6, 108e6, ModulationType.WFM, 200e3, 15e3, priority=5),
        FrequencyBand("Aviation", 108e6, 137e6, ModulationType.AM, 25e3, 3e3, priority=4),
        FrequencyBand("Marine VHF", 156e6, 174e6, ModulationType.NFM, 25e3, 3e3, priority=3),
        FrequencyBand("FRS/GMRS", 462e6, 467e6, ModulationType.NFM, 12.5e3, 3e3, priority=3),
        FrequencyBand("Ham 2m", 144e6, 148e6, ModulationType.NFM, 12.5e3, 3e3, priority=2),
        FrequencyBand("Ham 70cm", 420e6, 450e6, ModulationType.NFM, 12.5e3, 3e3, priority=2),
        FrequencyBand("Public Safety", 450e6, 470e6, ModulationType.NFM, 12.5e3, 3e3, priority=4),
        FrequencyBand("Business", 151e6, 159e6, ModulationType.NFM, 12.5e3, 3e3, priority=2),
        FrequencyBand("MURS", 151.82e6, 154.6e6, ModulationType.NFM, 12.5e3, 3e3, priority=2),
        # Emergency and distress frequencies
        FrequencyBand("Aviation Emergency", 121.5e6, 121.5e6, ModulationType.AM, 25e3, 3e3, priority=5),
        FrequencyBand("Marine Distress", 156.8e6, 156.8e6, ModulationType.NFM, 25e3, 3e3, priority=5),
        FrequencyBand("International Distress", 2.182e6, 2.182e6, ModulationType.AM, 3e3, 3e3, priority=5),
        # NOAA Weather Radio
        FrequencyBand("NOAA Weather", 162.4e6, 162.55e6, ModulationType.NFM, 25e3, 3e3, priority=5),
    ]
    
    def __init__(self, device_index: int = 0, gain: Union[str, float] = 'auto', 
                 mock_mode: bool = False):
        """
        Initialize RTL-SDR manager.
        
        Args:
            device_index: RTL-SDR device index (0 for first device)
            gain: Gain setting ('auto' or dB value)
            mock_mode: Use mock data instead of real hardware
        """
        self.device_index = device_index
        self.gain = gain
        self.mock_mode = mock_mode or not RTL_SDR_AVAILABLE
        self.sdr: Optional[RtlSdr] = None
        self.is_running = False
        
        # Thread-safe result storage
        self.detection_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.sweep_stats = {
            'total_sweeps': 0,
            'signals_detected': 0,
            'audio_captures': 0,
            'last_sweep_time': 0.0
        }
        
        if not self.mock_mode:
            self._initialize_sdr()
    
    def _initialize_sdr(self) -> None:
        """Initialize RTL-SDR hardware connection."""
        try:
            self.sdr = RtlSdr(device_index=self.device_index)
            self.sdr.gain = self.gain
            self.logger.info(f"RTL-SDR initialized: {self.sdr}")
        except Exception as e:
            self.logger.error(f"Failed to initialize RTL-SDR: {e}")
            self.mock_mode = True
            raise RuntimeError(f"RTL-SDR initialization failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
    
    def close(self) -> None:
        """Clean up RTL-SDR resources."""
        self.is_running = False
        if self.sdr:
            try:
                self.sdr.close()
            except Exception as e:
                self.logger.warning(f"Error closing RTL-SDR: {e}")
    
    def get_frequency_band(self, frequency: float) -> Optional[FrequencyBand]:
        """
        Determine which audio band a frequency belongs to.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            FrequencyBand if frequency is in an audio band, None otherwise
        """
        for band in self.AUDIO_BANDS:
            if band.start_freq <= frequency <= band.end_freq:
                return band
        return None
    
    def _generate_sweep_frequencies(self, start_freq: float, end_freq: float,
                                  step_size: Optional[float] = None) -> List[float]:
        """
        Generate frequency list for sweeping with optimal overlap.
        
        Args:
            start_freq: Start frequency in Hz
            end_freq: End frequency in Hz
            step_size: Frequency step size (defaults to 80% of bandwidth)
            
        Returns:
            List of center frequencies for sweep
        """
        if step_size is None:
            step_size = self.BANDWIDTH * 0.8  # 20% overlap for better coverage
        
        frequencies = []
        current = start_freq + self.BANDWIDTH / 2  # Start at center of first band
        
        while current + self.BANDWIDTH / 2 <= end_freq:
            frequencies.append(current)
            current += step_size
        
        return frequencies
    
    def _capture_samples(self, center_freq: float, sample_rate: float, 
                        duration: float) -> np.ndarray:
        """
        Capture IQ samples from RTL-SDR.
        
        Args:
            center_freq: Center frequency in Hz
            sample_rate: Sample rate in Hz
            duration: Capture duration in seconds
            
        Returns:
            Complex IQ samples
        """
        if self.mock_mode:
            return self._generate_mock_samples(center_freq, sample_rate, duration)
        
        try:
            # Set frequency and sample rate
            self.sdr.center_freq = center_freq
            self.sdr.sample_rate = sample_rate
            
            # Small delay to let hardware settle
            time.sleep(0.01)
            
            # Read in smaller chunks to avoid overflow
            chunk_size = 131072  # 128K samples per chunk (more conservative)
            num_samples = int(sample_rate * duration)
            
            if num_samples <= chunk_size:
                # Small enough to read at once
                samples = self.sdr.read_samples(num_samples)
            else:
                # Read in chunks
                samples = np.array([], dtype=complex)
                remaining = num_samples
                
                while remaining > 0:
                    chunk_samples = min(chunk_size, remaining)
                    chunk = self.sdr.read_samples(chunk_samples)
                    samples = np.concatenate([samples, chunk])
                    remaining -= chunk_samples
            
            return samples
        except Exception as e:
            self.logger.error(f"Failed to capture samples at {center_freq/1e6:.3f} MHz: {e}")
            return np.array([], dtype=complex)
    
    def _generate_mock_samples(self, center_freq: float, sample_rate: float, 
                              duration: float) -> np.ndarray:
        """Generate mock IQ samples for testing."""
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Create mock signal with noise
        noise = (np.random.normal(0, 0.1, num_samples) + 
                1j * np.random.normal(0, 0.1, num_samples))
        
        # Add a strong signal if in an audio band
        band = self.get_frequency_band(center_freq)
        if band and np.random.random() > 0.7:  # 30% chance of signal
            signal_freq = np.random.uniform(-sample_rate/4, sample_rate/4)
            signal_power = np.random.uniform(0.3, 0.8)
            
            if band.modulation == ModulationType.AM:
                # AM signal
                carrier = np.exp(1j * 2 * np.pi * signal_freq * t)
                modulation = 1 + 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1kHz tone
                signal = signal_power * modulation * carrier
            else:
                # FM signal
                audio_freq = 1000  # 1kHz tone
                fm_deviation = 5000  # 5kHz deviation
                phase = 2 * np.pi * signal_freq * t + (fm_deviation/audio_freq) * np.sin(2 * np.pi * audio_freq * t)
                signal = signal_power * np.exp(1j * phase)
            
            noise += signal
        
        return noise
    
    def _detect_signals(self, samples: np.ndarray, center_freq: float, 
                       sample_rate: float) -> List[SignalDetection]:
        """
        Detect signals in captured samples using power spectral density.
        
        Args:
            samples: Complex IQ samples
            center_freq: Center frequency of capture
            sample_rate: Sample rate used
            
        Returns:
            List of detected signals
        """
        if len(samples) == 0:
            return []
        
        # Calculate power spectral density
        freqs, psd = scipy.signal.welch(samples, fs=sample_rate, nperseg=1024)
        psd_db = 10 * np.log10(psd + 1e-12)  # Avoid log(0)
        
        # Estimate noise floor (median of lower 25% of PSD)
        noise_floor = np.median(np.sort(psd_db)[:len(psd_db)//4])
        
        # Find peaks above noise floor
        threshold = noise_floor + 10  # 10 dB above noise floor
        peaks, properties = scipy.signal.find_peaks(
            psd_db, 
            height=threshold,
            distance=int(len(psd_db) * 0.01),  # Minimum 1% of spectrum apart
            prominence=5  # Minimum 5 dB prominence
        )
        
        detections = []
        for i, peak_idx in enumerate(peaks):
            freq = center_freq + freqs[peak_idx]
            power = psd_db[peak_idx]
            snr = power - noise_floor
            
            # Estimate bandwidth (width at -3dB from peak)
            try:
                width_samples = properties['widths'][i]
                bandwidth = width_samples * sample_rate / len(psd_db)
            except (KeyError, IndexError):
                bandwidth = sample_rate / len(psd_db)  # Single bin width
            
            # Check if in audio band
            band = self.get_frequency_band(freq)
            if band:
                # Calculate confidence based on SNR and band priority
                confidence = min(1.0, (snr / 20.0) * (band.priority / 5.0))
                
                detection = SignalDetection(
                    frequency=freq,
                    power_db=power,
                    bandwidth=bandwidth,
                    snr_db=snr,
                    band_info=band,
                    timestamp=time.time(),
                    confidence=confidence
                )
                detections.append(detection)
        
        return sorted(detections, key=lambda x: x.confidence, reverse=True)
    
    def _demodulate_am(self, samples: np.ndarray, sample_rate: float) -> np.ndarray:
        """Demodulate AM signal."""
        # Calculate envelope
        envelope = np.abs(samples)
        
        # Remove DC component
        envelope = envelope - np.mean(envelope)
        
        # Low-pass filter to audio bandwidth
        nyquist = sample_rate / 2
        cutoff = min(5000, nyquist * 0.4)  # 5kHz or 40% of Nyquist
        sos = scipy.signal.butter(6, cutoff / nyquist, btype='low', output='sos')
        audio = scipy.signal.sosfilt(sos, envelope)
        
        return audio.astype(np.float32)
    
    def _demodulate_fm(self, samples: np.ndarray, sample_rate: float, 
                      wideband: bool = False) -> np.ndarray:
        """Demodulate FM signal."""
        # Calculate instantaneous phase
        phase = np.unwrap(np.angle(samples))
        
        # Differentiate to get frequency
        audio = np.diff(phase) * sample_rate / (2 * np.pi)
        
        # Low-pass filter
        nyquist = sample_rate / 2
        if wideband:
            cutoff = min(15000, nyquist * 0.4)  # 15kHz for WFM
        else:
            cutoff = min(3000, nyquist * 0.4)   # 3kHz for NFM
        
        sos = scipy.signal.butter(6, cutoff / nyquist, btype='low', output='sos')
        audio = scipy.signal.sosfilt(sos, audio)
        
        return audio.astype(np.float32)
    
    def capture_audio(self, detection: SignalDetection, duration: float = 3.0,
                     save_to_file: bool = False, output_dir: str = "audio_captures") -> AudioCapture:
        """
        Capture and demodulate audio from a detected signal.
        
        Args:
            detection: Signal detection information
            duration: Capture duration in seconds
            save_to_file: Whether to save audio to WAV file
            output_dir: Directory for saved audio files
            
        Returns:
            AudioCapture with demodulated audio data
        """
        # Capture samples at audio sample rate
        samples = self._capture_samples(
            detection.frequency, 
            self.AUDIO_SAMPLE_RATE, 
            duration
        )
        
        if len(samples) == 0:
            raise RuntimeError(f"Failed to capture audio at {detection.frequency/1e6:.3f} MHz")
        
        # Demodulate based on modulation type
        if detection.band_info.modulation == ModulationType.AM:
            audio = self._demodulate_am(samples, self.AUDIO_SAMPLE_RATE)
        elif detection.band_info.modulation == ModulationType.WFM:
            audio = self._demodulate_fm(samples, self.AUDIO_SAMPLE_RATE, wideband=True)
        else:  # NFM
            audio = self._demodulate_fm(samples, self.AUDIO_SAMPLE_RATE, wideband=False)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Resample to standard audio rate (44.1kHz)
        audio_sample_rate = 44100
        if len(audio) > 1:
            num_samples = int(len(audio) * audio_sample_rate / self.AUDIO_SAMPLE_RATE)
            audio = scipy.signal.resample(audio, num_samples)
        
        file_path = None
        if save_to_file:
            # Create output directory
            Path(output_dir).mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            freq_mhz = detection.frequency / 1e6
            filename = f"{timestamp}_{freq_mhz:.3f}MHz_{detection.band_info.name.replace(' ', '_')}.wav"
            file_path = Path(output_dir) / filename
            
            # Save as 16-bit WAV
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(str(file_path), audio_sample_rate, audio_int16)
        
        return AudioCapture(
            signal_info=detection,
            audio_data=audio,
            sample_rate=audio_sample_rate,
            duration=duration,
            file_path=str(file_path) if file_path else None
        )
    
    def panoramic_sweep(self, start_freq: float = None, end_freq: float = None,
                       step_size: float = None, min_confidence: float = 0.3) -> List[SignalDetection]:
        """
        Perform panoramic frequency sweep to detect signals.
        
        Args:
            start_freq: Start frequency (defaults to MIN_FREQ)
            end_freq: End frequency (defaults to MAX_FREQ)
            step_size: Frequency step size (defaults to optimal overlap)
            min_confidence: Minimum confidence threshold for detections
            
        Returns:
            List of detected signals sorted by confidence
        """
        start_freq = start_freq or self.MIN_FREQ
        end_freq = end_freq or self.MAX_FREQ
        
        self.logger.info(f"Starting panoramic sweep: {start_freq/1e6:.1f} - {end_freq/1e6:.1f} MHz")
        
        sweep_start = time.time()
        all_detections = []
        
        # Generate sweep frequencies
        frequencies = self._generate_sweep_frequencies(start_freq, end_freq, step_size)
        
        for i, freq in enumerate(frequencies):
            if not self.is_running:
                break
            
            try:
                # Capture samples for signal detection
                samples = self._capture_samples(freq, self.SWEEP_SAMPLE_RATE, 0.1)  # 100ms
                
                # Detect signals
                detections = self._detect_signals(samples, freq, self.SWEEP_SAMPLE_RATE)
                
                # Filter by confidence
                filtered_detections = [d for d in detections if d.confidence >= min_confidence]
                all_detections.extend(filtered_detections)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    progress = (i + 1) / len(frequencies) * 100
                    self.logger.info(f"Sweep progress: {progress:.1f}% ({freq/1e6:.1f} MHz)")
                
            except Exception as e:
                self.logger.warning(f"Error sweeping {freq/1e6:.3f} MHz: {e}")
        
        # Update statistics
        sweep_time = time.time() - sweep_start
        self.sweep_stats['total_sweeps'] += 1
        self.sweep_stats['signals_detected'] += len(all_detections)
        self.sweep_stats['last_sweep_time'] = sweep_time
        
        self.logger.info(f"Sweep completed in {sweep_time:.1f}s, found {len(all_detections)} signals")
        
        # Remove duplicates and sort by confidence
        unique_detections = self._remove_duplicate_detections(all_detections)
        return sorted(unique_detections, key=lambda x: x.confidence, reverse=True)
    
    def _remove_duplicate_detections(self, detections: List[SignalDetection], 
                                   freq_tolerance: float = 50e3) -> List[SignalDetection]:
        """Remove duplicate detections within frequency tolerance."""
        if not detections:
            return []
        
        unique = []
        sorted_detections = sorted(detections, key=lambda x: x.frequency)
        
        for detection in sorted_detections:
            is_duplicate = False
            for existing in unique:
                if abs(detection.frequency - existing.frequency) < freq_tolerance:
                    # Keep the one with higher confidence
                    if detection.confidence > existing.confidence:
                        unique.remove(existing)
                        unique.append(detection)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(detection)
        
        return unique
    
    def targeted_audio_sweep(self, bands: List[str] = None, duration: float = 3.0,
                           save_audio: bool = False, max_captures: int = 10) -> List[AudioCapture]:
        """
        Perform targeted sweep of audio bands with automatic capture.
        
        Args:
            bands: List of band names to sweep (None for all)
            duration: Audio capture duration per signal
            save_audio: Whether to save audio files
            max_captures: Maximum number of audio captures
            
        Returns:
            List of audio captures
        """
        self.is_running = True
        captures = []
        
        # Select bands to sweep
        target_bands = self.AUDIO_BANDS
        if bands:
            target_bands = [band for band in self.AUDIO_BANDS if band.name in bands]
        
        # Sort by priority
        target_bands = sorted(target_bands, key=lambda x: x.priority, reverse=True)
        
        self.logger.info(f"Starting targeted audio sweep of {len(target_bands)} bands")
        
        for band in target_bands:
            if len(captures) >= max_captures or not self.is_running:
                break
            
            self.logger.info(f"Sweeping {band.name} ({band.start_freq/1e6:.1f}-{band.end_freq/1e6:.1f} MHz)")
            
            # Sweep this band
            detections = self.panoramic_sweep(
                start_freq=band.start_freq,
                end_freq=band.end_freq,
                min_confidence=0.5  # Higher threshold for audio capture
            )
            
            # Capture audio from top detections
            for detection in detections[:3]:  # Top 3 per band
                if len(captures) >= max_captures:
                    break
                
                try:
                    self.logger.info(f"Capturing audio: {detection.frequency/1e6:.3f} MHz "
                                   f"({detection.band_info.name}, confidence: {detection.confidence:.2f})")
                    
                    capture = self.capture_audio(
                        detection, 
                        duration=duration, 
                        save_to_file=save_audio
                    )
                    captures.append(capture)
                    self.sweep_stats['audio_captures'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to capture audio from {detection.frequency/1e6:.3f} MHz: {e}")
        
        self.logger.info(f"Targeted sweep completed: {len(captures)} audio captures")
        return captures
    
    def export_mobile_json(self, detections: List[SignalDetection] = None,
                          captures: List[AudioCapture] = None) -> Dict:
        """
        Export results in mobile-optimized JSON format.
        
        Args:
            detections: Signal detections to export
            captures: Audio captures to export
            
        Returns:
            Mobile-optimized JSON data structure
        """
        timestamp = time.time()
        
        # Convert detections to mobile format
        signals = []
        if detections:
            for detection in detections:
                signal_data = {
                    'frequency': detection.frequency,
                    'frequency_mhz': round(detection.frequency / 1e6, 6),
                    'power_db': round(detection.power_db, 1),
                    'snr_db': round(detection.snr_db, 1),
                    'bandwidth': detection.bandwidth,
                    'band': detection.band_info.name,
                    'modulation': detection.band_info.modulation.value,
                    'confidence': round(detection.confidence, 3),
                    'timestamp': detection.timestamp
                }
                signals.append(signal_data)
        
        # Convert audio captures to mobile format
        audio_data = []
        if captures:
            for capture in captures:
                # Calculate audio statistics
                audio_stats = {
                    'duration': round(capture.duration, 2),
                    'sample_rate': capture.sample_rate,
                    'peak_amplitude': float(np.max(np.abs(capture.audio_data))),
                    'rms_level': float(np.sqrt(np.mean(capture.audio_data**2))),
                    'file_path': capture.file_path
                }
                
                # Convert signal info to dict manually to handle Enum
                signal_dict = {
                    'frequency': capture.signal_info.frequency,
                    'power_db': capture.signal_info.power_db,
                    'bandwidth': capture.signal_info.bandwidth,
                    'snr_db': capture.signal_info.snr_db,
                    'timestamp': capture.signal_info.timestamp,
                    'confidence': capture.signal_info.confidence,
                    'band_info': {
                        'name': capture.signal_info.band_info.name,
                        'start_freq': capture.signal_info.band_info.start_freq,
                        'end_freq': capture.signal_info.band_info.end_freq,
                        'modulation': capture.signal_info.band_info.modulation.value,
                        'channel_spacing': capture.signal_info.band_info.channel_spacing,
                        'audio_bandwidth': capture.signal_info.band_info.audio_bandwidth,
                        'priority': capture.signal_info.band_info.priority
                    }
                }
                
                capture_data = {
                    'signal': signal_dict,
                    'audio_stats': audio_stats,
                    'timestamp': capture.signal_info.timestamp
                }
                audio_data.append(capture_data)
        
        # Main data structure
        result = {
            'metadata': {
                'timestamp': timestamp,
                'format_version': '1.0',
                'sdr_info': {
                    'device': 'RTL-SDR',
                    'mock_mode': self.mock_mode,
                    'frequency_range': {
                        'min_hz': self.MIN_FREQ,
                        'max_hz': self.MAX_FREQ,
                        'bandwidth_hz': self.BANDWIDTH
                    }
                },
                'stats': self.sweep_stats.copy()
            },
            'signals': signals,
            'audio_captures': audio_data,
            'band_definitions': [
                {
                    'name': band.name,
                    'start_freq': band.start_freq,
                    'end_freq': band.end_freq,
                    'modulation': band.modulation.value,
                    'priority': band.priority
                }
                for band in self.AUDIO_BANDS
            ]
        }
        
        return result
    
    def get_status(self) -> Dict:
        """Get current status and statistics."""
        status = {
            'connected': not self.mock_mode and self.sdr is not None,
            'mock_mode': self.mock_mode,
            'running': self.is_running,
            'stats': self.sweep_stats.copy(),
            'queue_sizes': {
                'detections': self.detection_queue.qsize(),
                'audio': self.audio_queue.qsize()
            }
        }
        
        if not self.mock_mode and self.sdr:
            try:
                status['sdr_info'] = {
                    'center_freq': self.sdr.center_freq,
                    'sample_rate': self.sdr.sample_rate,
                    'gain': self.sdr.gain
                }
            except Exception:
                pass
        
        return status


def main():
    """Example usage and testing."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize manager
    with RTLSDRManager(mock_mode=True) as sdr_manager:
        print("RTL-SDR Manager initialized")
        print(f"Status: {sdr_manager.get_status()}")
        
        # Test panoramic sweep
        print("\n=== Panoramic Sweep Test ===")
        detections = sdr_manager.panoramic_sweep(
            start_freq=88e6, 
            end_freq=108e6,  # FM band only for quick test
            min_confidence=0.3
        )
        
        print(f"Found {len(detections)} signals")
        for det in detections[:5]:  # Show top 5
            print(f"  {det.frequency/1e6:.3f} MHz - {det.band_info.name} "
                  f"(SNR: {det.snr_db:.1f} dB, conf: {det.confidence:.2f})")
        
        # Test audio capture
        if detections:
            print(f"\n=== Audio Capture Test ===")
            capture = sdr_manager.capture_audio(detections[0], duration=1.0, save_to_file=True)
            print(f"Captured {capture.duration:.1f}s of audio from {detections[0].frequency/1e6:.3f} MHz")
            if capture.file_path:
                print(f"Saved to: {capture.file_path}")
        
        # Export to JSON
        print(f"\n=== JSON Export Test ===")
        json_data = sdr_manager.export_mobile_json(detections, [capture] if detections else None)
        print(f"JSON export: {len(json_data['signals'])} signals, "
              f"{len(json_data['audio_captures'])} audio captures")
        
        # Print sample JSON (truncated)
        print("\nSample JSON structure:")
        sample_json = json.dumps(json_data, indent=2)[:500] + "..."
        print(sample_json)


if __name__ == "__main__":
    main()