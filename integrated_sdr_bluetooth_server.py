#!/usr/bin/env python3
"""
Integrated SDR-Bluetooth Server for SIGMA Project

Combines RTL-SDR signal detection and audio capture with Bluetooth communication
to the Android SIGMA app. Provides real SDR data instead of mock data.

Usage:
    sudo python3 integrated_sdr_bluetooth_server.py [--test-mode] [--sdr-mode] [--sweep-interval SECONDS]
"""

import bluetooth
import json
import time
import argparse
import logging
import threading
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path
import signal
import sys
import uuid
import base64
import os

from rtl_sdr_manager import RTLSDRManager, SignalDetection, AudioCapture

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiagnosticDisplay:
    """Clean diagnostic display with progress indicators."""
    
    def __init__(self):
        self.last_update = 0
        self.min_update_interval = 0.1  # Update at most 10 times per second
        
    def clear_screen(self):
        """Clear terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def progress_bar(self, current: float, total: float, width: int = 40, label: str = "") -> str:
        """Create a text progress bar."""
        if total == 0:
            percent = 0
        else:
            percent = min(current / total, 1.0)
        
        filled = int(width * percent)
        bar = "█" * filled + "░" * (width - filled)
        
        return f"{label:20} [{bar}] {percent*100:5.1f}%"
    
    def format_frequency(self, freq_hz: float) -> str:
        """Format frequency for display."""
        if freq_hz >= 1e9:
            return f"{freq_hz/1e9:.3f} GHz"
        elif freq_hz >= 1e6:
            return f"{freq_hz/1e6:.3f} MHz"
        elif freq_hz >= 1e3:
            return f"{freq_hz/1e3:.3f} kHz"
        else:
            return f"{freq_hz:.0f} Hz"
    
    def display_status(self, server_status: Dict, force: bool = False):
        """Display server status dashboard."""
        current_time = time.time()
        
        # Rate limit updates unless forced
        if not force and current_time - self.last_update < self.min_update_interval:
            return
        
        self.last_update = current_time
        self.clear_screen()
        
        # Header
        print("╔" + "═" * 78 + "╗")
        print("║" + " SIGMA SDR-BLUETOOTH SERVER ".center(78) + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        
        # Connection Status
        bt_status = "🟢 Connected" if server_status.get('bluetooth_connected') else "🔴 Waiting"
        sdr_status = "🟢 Active" if server_status.get('sdr_status', {}).get('connected') else "🟡 Mock"
        
        print(f"Bluetooth: {bt_status:<20} SDR: {sdr_status:<20}")
        print(f"Mode: {'Test' if server_status.get('test_mode') else 'SDR':<20} "
              f"Uptime: {server_status.get('uptime_formatted', '0h 0m')}")
        print()
        
        # Progress Section
        stats = server_status.get('statistics', {})
        latest = server_status.get('latest_data', {})
        
        # Show current operation
        if server_status.get('current_operation'):
            op = server_status['current_operation']
            if op['type'] == 'sweep':
                print(self.progress_bar(
                    op.get('progress', 0),
                    op.get('total', 100),
                    label="SDR Sweep"
                ))
            elif op['type'] == 'transmission':
                print(self.progress_bar(
                    op.get('messages_sent', 0),
                    op.get('total_messages', 1),
                    label="Data Transmission"
                ))
            print()
        
        # Statistics
        print("─" * 80)
        print(f"Sweeps: {stats.get('sdr_sweeps_completed', 0):<10} "
              f"Signals: {stats.get('signals_detected', 0):<10} "
              f"Audio: {stats.get('audio_captures', 0):<10} "
              f"Messages: {stats.get('messages_sent', 0)}")
        
        # Latest Signals
        if 'latest_signals' in server_status:
            print("\n" + "─" * 80)
            print("Latest Signals Detected:")
            print(f"{'Frequency':>12} {'Power':>8} {'SNR':>6} {'Band':>20} {'Type':>8}")
            print("─" * 80)
            
            for sig in server_status['latest_signals'][:5]:
                print(f"{self.format_frequency(sig['frequency']):>12} "
                      f"{sig['power_db']:>7.1f}dB "
                      f"{sig['snr_db']:>5.1f}dB "
                      f"{sig['band']:>20} "
                      f"{sig['modulation']:>8}")
        
        # Show any errors
        if server_status.get('statistics', {}).get('last_error'):
            print("\n" + "─" * 80)
            print(f"⚠️  ERROR: {server_status['statistics']['last_error']}")
        
        # Footer with next action
        if server_status.get('next_action'):
            print("\n" + "─" * 80)
            print(f"Next: {server_status['next_action']}")
        
        print("\n[Ctrl+C to stop]", end='', flush=True)


class IntegratedSigmaServer:
    """
    Integrated SIGMA server combining RTL-SDR operations with Bluetooth communication.
    
    Features:
    - Real-time SDR frequency sweeps
    - Audio signal detection and capture
    - Bluetooth communication with Android app
    - Continuous operation with configurable sweep intervals
    """
    
    def __init__(self, test_mode: bool = False, sdr_mode: bool = False, 
                 sweep_interval: float = 60.0, max_iterations: Optional[int] = None,
                 quiet_mode: bool = False, best_per_band: bool = True):
        """
        Initialize integrated SIGMA server.
        
        Args:
            test_mode: Use mock Bluetooth data (for testing without Android app)
            sdr_mode: Use real SDR hardware instead of mock SDR data
            sweep_interval: Time between SDR sweeps in seconds
            max_iterations: Maximum number of sweep reports to send before disconnecting (None = unlimited)
            quiet_mode: Use diagnostic display instead of streaming logs
            best_per_band: Only report strongest signal per band (default: True)
        """
        self.test_mode = test_mode
        self.sdr_mode = sdr_mode
        self.sweep_interval = sweep_interval
        self.max_iterations = max_iterations
        self.quiet_mode = quiet_mode
        self.best_per_band = best_per_band
        
        # Bluetooth setup
        self.server_sock = None
        self.client_sock = None
        self.running = False
        self.uuid = "00001101-0000-1000-8000-00805F9B34FB"
        self.service_name = "SIGMA Sensor"
        
        # SDR setup
        self.sdr_manager: Optional[RTLSDRManager] = None
        self.latest_detections: List[SignalDetection] = []
        self.latest_captures: List[AudioCapture] = []
        self.sdr_thread: Optional[threading.Thread] = None
        self.sdr_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'bluetooth_connections': 0,
            'messages_sent': 0,
            'sdr_sweeps_completed': 0,
            'signals_detected': 0,
            'audio_captures': 0,
            'last_sweep_time': 0,
            'server_start_time': time.time(),
            'last_error': None
        }
        
        # Message sequencing
        self.message_sequence = 0
        self.current_report_id = None
        
        # Diagnostic display
        self.display = DiagnosticDisplay() if quiet_mode else None
        self.current_operation = None
        
        # Configure logging for quiet mode
        if quiet_mode:
            logging.getLogger().setLevel(logging.WARNING)
        
        # Debug mode flag
        self.debug_mode = False
        
        # Audio storage
        self.audio_dir = Path("sigma_audio_captures")
        self.audio_dir.mkdir(exist_ok=True)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_server()
        sys.exit(0)
    
    def initialize_sdr(self) -> bool:
        """
        Initialize RTL-SDR manager.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.sdr_manager = RTLSDRManager(mock_mode=not self.sdr_mode)
            logger.info(f"SDR Manager initialized (mock_mode={not self.sdr_mode})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SDR: {e}")
            return False
    
    def start_sdr_monitoring(self):
        """Start continuous SDR monitoring in a separate thread."""
        if not self.sdr_manager:
            logger.error("SDR Manager not initialized")
            return
        
        # Set running flag BEFORE starting thread
        self.running = True
        
        self.sdr_thread = threading.Thread(target=self._sdr_monitoring_loop, daemon=True)
        self.sdr_thread.start()
        logger.info(f"SDR monitoring started (sweep interval: {self.sweep_interval}s)")
    
    def _sdr_monitoring_loop(self):
        """Continuous SDR monitoring loop."""
        logger.info("SDR monitoring loop started")
        
        # Wait for first client connection before starting sweeps
        while self.running and not self.client_sock:
            time.sleep(1)
            logger.debug("Waiting for client connection before starting SDR sweeps...")
        
        while self.running:
            try:
                sweep_start = time.time()
                if not self.quiet_mode:
                    logger.info("Starting SDR sweep...")
                
                # Update operation status
                self.current_operation = {
                    'type': 'sweep',
                    'progress': 0,
                    'total': 100
                }
                
                # Perform frequency marching through bands of interest
                captures = []
                detections = []
                
                try:
                    # Define scan windows for each band
                    # Window size is based on SDR bandwidth (1 MHz with some overlap)
                    scan_plan = [
                        # Skip AM Broadcast - RTL-SDR V4 struggles with frequencies below 24 MHz
                        # FM Broadcast - 2.4 MHz windows for better quality
                        {"band": "FM Broadcast", "windows": [
                            (88e6, 93e6, 2.048e6),      # 88-93 MHz
                            (93e6, 98e6, 2.048e6),      # 93-98 MHz
                            (98e6, 103e6, 2.048e6),     # 98-103 MHz
                            (103e6, 108e6, 2.048e6),    # 103-108 MHz
                        ]},
                        # Aviation band - 2.048 MHz for better stability  
                        {"band": "Aviation", "windows": [
                            (118e6, 123e6, 2.048e6),   # 118-123 MHz
                            (123e6, 128e6, 2.048e6),   # 123-128 MHz
                            (128e6, 133e6, 2.048e6),   # 128-133 MHz
                            (133e6, 137e6, 2.048e6),   # 133-137 MHz
                        ]},
                        # Marine VHF - 2.048 MHz windows
                        {"band": "Marine VHF", "windows": [
                            (155e6, 159e6, 2.048e6),   # Marine channels around 156.8 MHz
                            (159e6, 163e6, 2.048e6),   # NOAA weather channels too
                        ]},
                        # Ham 2m band
                        {"band": "Ham 2m", "windows": [
                            (144e6, 146e6, 2.048e6),   # 144-146 MHz
                            (146e6, 148e6, 2.048e6),   # 146-148 MHz
                        ]},
                        # Ham 70cm band
                        {"band": "Ham 70cm", "windows": [
                            (420e6, 430e6, 2.048e6),   # 420-430 MHz
                            (430e6, 440e6, 2.048e6),   # 430-440 MHz  
                            (440e6, 450e6, 2.048e6),   # 440-450 MHz
                        ]},
                        # GMRS/FRS
                        {"band": "GMRS/FRS", "windows": [
                            (462e6, 463e6, 2.048e6),   # FRS/GMRS channels 1-7
                            (467e6, 468e6, 2.048e6),   # FRS/GMRS channels 15-22
                        ]},
                        # Public Safety
                        {"band": "Public Safety", "windows": [
                            (450e6, 455e6, 2.048e6),   # 450-455 MHz
                            (455e6, 460e6, 2.048e6),   # 455-460 MHz
                            (460e6, 470e6, 2.048e6),   # 460-470 MHz
                        ]},
                    ]
                    
                    total_windows = sum(len(band["windows"]) for band in scan_plan)
                    window_count = 0
                    
                    logger.info(f"Starting frequency march through {total_windows} windows")
                    
                    for band_info in scan_plan:
                        band_name = band_info["band"]
                        
                        for start_freq, end_freq, sample_rate in band_info["windows"]:
                            window_count += 1
                            center_freq = (start_freq + end_freq) / 2
                            
                            # Update progress
                            self.current_operation = {
                                'type': 'sweep',
                                'progress': int((window_count / total_windows) * 100),
                                'total': 100
                            }
                            
                            try:
                                # Longer settling time for frequency changes
                                time.sleep(0.1)  # 100ms settling time
                                
                                # Skip frequencies below 24 MHz (RTL-SDR V4 limitation)
                                if center_freq < 24e6:
                                    logger.debug(f"Skipping {center_freq/1e6:.1f} MHz - below RTL-SDR V4 range")
                                    continue
                                
                                # Capture samples at this window
                                samples = self.sdr_manager._capture_samples(
                                    center_freq=center_freq,
                                    sample_rate=sample_rate,
                                    duration=0.02  # Even shorter 20ms capture to avoid overflow
                                )
                                
                                if len(samples) > 0:
                                    # Detect signals in this window
                                    window_detections = self.sdr_manager._detect_signals(
                                        samples, center_freq, sample_rate
                                    )
                                    
                                    # Add band info to detections
                                    for det in window_detections:
                                        # Only keep if actually in the band
                                        if start_freq <= det.frequency <= end_freq:
                                            detections.append(det)
                                    
                                    if not self.quiet_mode and window_detections:
                                        logger.info(f"{band_name} window {center_freq/1e6:.1f} MHz: "
                                                   f"found {len(window_detections)} signals")
                                
                            except Exception as e:
                                logger.warning(f"Failed to scan {band_name} window "
                                             f"{center_freq/1e6:.1f} MHz: {e}")
                                # Don't fail the whole sweep for one bad window
                                continue
                    
                    logger.info(f"Frequency march complete: {len(detections)} total signals detected")
                    
                except Exception as e:
                    logger.error(f"Frequency march failed: {e}")
                    self.stats['last_error'] = f"Scan error: {str(e)}"
                
                # Sort detections by signal strength and filter weak ones
                strong_detections = [d for d in detections if d.snr_db > 10]  # Only strong signals
                
                if self.best_per_band:
                    # Group by band and take strongest from each
                    detections_by_band = {}
                    for det in strong_detections:
                        band = det.band_info.name
                        if band not in detections_by_band or det.power_db > detections_by_band[band].power_db:
                            detections_by_band[band] = det
                    
                    # Get list of strongest per band, sorted by power
                    final_detections = list(detections_by_band.values())
                    final_detections.sort(key=lambda x: x.power_db, reverse=True)
                    
                    logger.info(f"Found signals in {len(detections_by_band)} bands: "
                               f"{', '.join(detections_by_band.keys())}")
                else:
                    # Just take strongest overall
                    strong_detections.sort(key=lambda x: x.power_db, reverse=True)
                    final_detections = strong_detections[:15]
                
                # Update shared data with thread safety
                with self.sdr_lock:
                    self.latest_detections = final_detections
                    self.latest_captures = captures
                    self.stats['sdr_sweeps_completed'] += 1
                    self.stats['signals_detected'] += len(strong_detections)
                    self.stats['audio_captures'] += len(captures)
                    self.stats['last_sweep_time'] = time.time()
                
                sweep_time = time.time() - sweep_start
                logger.info(f"SDR sweep completed in {sweep_time:.1f}s: "
                           f"{len(detections)} signals, {len(captures)} audio captures")
                
                # Wait for next sweep
                sleep_time = max(0, self.sweep_interval - sweep_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in SDR monitoring loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def create_signal_report(self) -> Dict:
        """
        Create signal report from latest SDR data.
        
        Returns:
            Signal report in SIGMA format
        """
        current_time = time.time()
        
        with self.sdr_lock:
            detections = self.latest_detections.copy()
            captures = self.latest_captures.copy()
        
        if not detections and not self.test_mode:
            # No real data available, create minimal report
            return {
                "timestamp": current_time,
                "status": "scanning",
                "message": "Scanning for signals...",
                "signals_detected": 0,
                "audio_captures": 0,
                "next_sweep": self.stats['last_sweep_time'] + self.sweep_interval
            }
        
        # Use SDR manager's mobile JSON export
        if self.sdr_manager:
            json_data = self.sdr_manager.export_mobile_json(detections, captures)
            
            # Add SIGMA-specific fields
            json_data['sigma_info'] = {
                'server_version': '2.0',
                'bluetooth_connections': self.stats['bluetooth_connections'],
                'uptime_seconds': current_time - self.stats['server_start_time'],
                'sweep_interval': self.sweep_interval,
                'next_sweep': self.stats['last_sweep_time'] + self.sweep_interval
            }
            
            return json_data
        
        # Fallback to test data if SDR manager not available
        return self._create_test_signal_report()
    
    def _create_test_detections(self) -> List[SignalDetection]:
        """Create test signal detections for testing mode."""
        from rtl_sdr_manager import SignalDetection, FrequencyBand, ModulationType
        import random
        
        # Create mock frequency bands
        fm_band = FrequencyBand(
            name="FM Broadcast",
            start_freq=88e6,
            end_freq=108e6,
            modulation=ModulationType.WFM,
            channel_spacing=200000,
            audio_bandwidth=150000,
            priority=5
        )
        
        aviation_band = FrequencyBand(
            name="Aviation Emergency",
            start_freq=121.4e6,
            end_freq=121.6e6,
            modulation=ModulationType.AM,
            channel_spacing=25000,
            audio_bandwidth=10000,
            priority=5
        )
        
        noaa_band = FrequencyBand(
            name="NOAA Weather",
            start_freq=162.4e6,
            end_freq=162.55e6,
            modulation=ModulationType.NFM,
            channel_spacing=25000,
            audio_bandwidth=15000,
            priority=5
        )
        
        test_signals = [
            SignalDetection(
                frequency=100.1e6,
                power_db=-45.2 + random.uniform(-5, 5),
                snr_db=18.5 + random.uniform(-2, 2),
                bandwidth=200000,
                band_info=fm_band,
                confidence=0.85,
                timestamp=time.time()
            ),
            SignalDetection(
                frequency=121.5e6,
                power_db=-52.1 + random.uniform(-5, 5),
                snr_db=15.2 + random.uniform(-2, 2), 
                bandwidth=25000,
                band_info=aviation_band,
                confidence=0.72,
                timestamp=time.time()
            ),
            SignalDetection(
                frequency=162.55e6,
                power_db=-48.3 + random.uniform(-5, 5),
                snr_db=22.1 + random.uniform(-2, 2),
                bandwidth=25000,
                band_info=noaa_band,
                confidence=0.91,
                timestamp=time.time()
            )
        ]
        
        return test_signals
    
    def start_bluetooth_server(self):
        """Start Bluetooth server and listen for connections."""
        try:
            # Create Bluetooth socket
            self.server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.server_sock.bind(("", bluetooth.PORT_ANY))
            self.server_sock.listen(1)
            
            port = self.server_sock.getsockname()[1]
            
            # Advertise service
            bluetooth.advertise_service(
                self.server_sock,
                self.service_name,
                service_id=self.uuid,
                service_classes=[self.uuid, bluetooth.SERIAL_PORT_CLASS],
                profiles=[bluetooth.SERIAL_PORT_PROFILE]
            )
            
            logger.info(f"Bluetooth server started")
            logger.info(f"Service: {self.service_name}")
            logger.info(f"UUID: {self.uuid}")
            logger.info(f"Port: {port}")
            logger.info("Waiting for Android app connection...")
            
            # Note: self.running is already set True in start_sdr_monitoring()
            
            while self.running:
                try:
                    # Accept connection
                    client_sock, client_info = self.server_sock.accept()
                    self.client_sock = client_sock
                    self.stats['bluetooth_connections'] += 1
                    
                    logger.info(f"Connection from {client_info}")
                    
                    # Handle client
                    self._handle_client()
                    
                except bluetooth.BluetoothError as e:
                    logger.error(f"Bluetooth error: {e}")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error accepting connection: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            logger.error(f"Failed to start Bluetooth server: {e}")
            raise
    
    def _handle_client(self):
        """Handle connected Android client following SIGMA spec."""
        try:
            logger.info("Client connected, sending connection acknowledgment...")
            
            # Send connection acknowledgment immediately
            self._send_connection_ack()
            
            # Main communication loop
            last_sweep_report_time = 0
            report_interval = 60.0  # SIGMA spec default: 60 seconds between reports
            iterations_completed = 0
            
            while self.running and self.client_sock:
                try:
                    current_time = time.time()
                    
                    # Send sweep report at intervals
                    if current_time - last_sweep_report_time >= report_interval:
                        # Wait for fresh data if we just connected
                        if last_sweep_report_time == 0:
                            logger.info("Waiting for initial SDR sweep to complete...")
                            time.sleep(5)  # Give SDR time to complete first sweep
                        
                        # Generate new report ID for this sweep cycle
                        self.current_report_id = str(uuid.uuid4())
                        
                        # Send sweep report
                        if self._send_sweep_report():
                            last_sweep_report_time = current_time
                            iterations_completed += 1
                            
                            # Send audio packets for captured signals
                            time.sleep(0.5)  # Small delay before audio
                            self._send_audio_packets()
                            
                            # Check if we've completed requested iterations
                            if self.max_iterations and iterations_completed >= self.max_iterations:
                                logger.info(f"Completed {iterations_completed} iterations, disconnecting client")
                                break
                    
                    # Sleep to prevent busy loop
                    time.sleep(1)
                    
                except bluetooth.BluetoothError as e:
                    logger.warning(f"Bluetooth communication error: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error in client handler: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            self._cleanup_client()
    
    def _send_message(self, message: Dict) -> bool:
        """
        Send a JSON message to the client with newline delimiter.
        
        Args:
            message: Dictionary to send
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Increment sequence counter
            self.message_sequence += 1
            message['sequence'] = self.message_sequence
            
            # Convert to JSON with newline delimiter
            json_data = json.dumps(message, separators=(',', ':')) + '\n'
            
            # Send to client
            self.client_sock.send(json_data.encode('utf-8'))
            self.stats['messages_sent'] += 1
            
            logger.debug(f"Sent message type: {message.get('message_type', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def _send_connection_ack(self):
        """Send connection acknowledgment per SIGMA spec."""
        message = {
            "message_type": "connection",
            "status": "connected", 
            "version": "1.0",
            "timestamp": int(time.time() * 1000)  # Milliseconds
        }
        
        if self._send_message(message):
            logger.info("Sent connection acknowledgment")
    
    def _send_sweep_report(self) -> bool:
        """Send sweep report per SIGMA spec."""
        try:
            # Update operation status
            self.current_operation = {
                'type': 'transmission',
                'messages_sent': 0,
                'total_messages': 1  # Will update when we know audio count
            }
            
            with self.sdr_lock:
                detections = self.latest_detections.copy()
                captures = self.latest_captures.copy()
            
            # Use test data if in test mode and no real data
            if self.test_mode and not detections:
                detections = self._create_test_detections()
                if not self.quiet_mode:
                    logger.info("Using test signal data")
            
            # Build signals array following spec format
            signals = []
            
            # Add detected signals
            for idx, detection in enumerate(detections[:15]):  # Limit to 15 signals
                signal_data = {
                    "signal_id": f"sig_{idx+1:03d}",
                    "frequency_mhz": round(detection.frequency / 1e6, 3),
                    "frequency_hz": int(detection.frequency),
                    "strength_dbm": round(detection.power_db, 1),
                    "snr_db": round(detection.snr_db, 1),
                    "bandwidth_khz": round(detection.bandwidth / 1000, 1),
                    "band": detection.band_info.name,
                    "modulation": detection.band_info.modulation.name,
                    "has_audio": False,
                    "audio_pending": False,
                    "confidence": round(detection.confidence, 2)
                }
                
                # Check if we have audio for this signal
                for capture in captures:
                    if abs(capture.signal_info.frequency - detection.frequency) < 1000:  # Within 1kHz
                        signal_data["has_audio"] = True
                        signal_data["audio_pending"] = True
                        break
                
                signals.append(signal_data)
            
            # Build sweep report message
            sweep_start_time = self.stats.get('last_sweep_time', time.time())
            
            message = {
                "message_type": "sweep_report",
                "report_id": self.current_report_id,
                "timestamp": int(time.time() * 1000),
                "sweep_info": {
                    "start_freq_mhz": 0.5,
                    "end_freq_mhz": 1700.0,
                    "duration_ms": int((time.time() - sweep_start_time) * 1000),
                    "device_info": "RTL-SDR" if self.sdr_mode else "Mock SDR"
                },
                "signals_detected": len(signals),
                "signals": signals
            }
            
            if self._send_message(message):
                logger.info(f"Sent sweep report: {len(signals)} signals")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to send sweep report: {e}")
            return False
    
    def _send_audio_packets(self):
        """Send audio packets for captured signals per SIGMA spec."""
        try:
            with self.sdr_lock:
                captures = self.latest_captures.copy()
            
            if not captures:
                logger.debug("No audio captures to send")
                return
            
            for idx, capture in enumerate(captures[:5]):  # Max 5 audio samples
                signal_id = f"sig_{idx+1:03d}"
                
                # Check if we have audio data
                if not hasattr(capture, 'audio_data') or capture.audio_data is None:
                    logger.warning(f"No audio data for signal {signal_id}")
                    continue
                
                # Convert audio to base64
                audio_bytes = capture.audio_data.tobytes()
                total_size = len(audio_bytes)
                
                # Calculate chunking info for raw bytes
                chunk_size = 32768  # 32KB chunks
                total_chunks = (total_size + chunk_size - 1) // chunk_size
                
                # Send chunks
                for chunk_idx in range(total_chunks):
                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, total_size)
                    
                    # Encode this chunk to base64
                    chunk_bytes = audio_bytes[start:end]
                    chunk_data = base64.b64encode(chunk_bytes).decode('utf-8')
                    
                    message = {
                        "message_type": "audio_packet",
                        "report_id": self.current_report_id,
                        "signal_id": signal_id,
                        "timestamp": int(time.time() * 1000),
                        "audio_info": {
                            "frequency_mhz": round(capture.signal_info.frequency / 1e6, 2),
                            "band": capture.signal_info.band_info.name,
                            "modulation": capture.signal_info.band_info.modulation.name,
                            "capture_time": int(capture.signal_info.timestamp * 1000),
                            "duration_ms": int(capture.duration * 1000),
                            "sample_rate": capture.sample_rate,
                            "channels": 1,
                            "bits_per_sample": 16,
                            "format": "pcm_s16le",
                            "compression": "none",
                            "total_size": total_size,
                            "chunk_info": {
                                "total_chunks": total_chunks,
                                "chunk_index": chunk_idx + 1,
                                "chunk_size": len(chunk_data),
                                "final_chunk_size": total_size % chunk_size if chunk_idx == total_chunks - 1 else chunk_size
                            }
                        },
                        "audio_data": chunk_data
                    }
                    
                    if not self._send_message(message):
                        logger.error(f"Failed to send audio chunk {chunk_idx+1}/{total_chunks}")
                        break
                    
                    time.sleep(0.1)  # Small delay between chunks
                
                logger.info(f"Sent audio for signal {signal_id}: {total_chunks} chunks")
                
        except Exception as e:
            logger.error(f"Failed to send audio packets: {e}")
    
    def _cleanup_client(self):
        """Clean up client connection."""
        if self.client_sock:
            try:
                self.client_sock.close()
            except:
                pass
            self.client_sock = None
            logger.info("Client disconnected")
    
    def stop_server(self):
        """Stop the server gracefully."""
        logger.info("Stopping SIGMA server...")
        self.running = False
        
        # Stop SDR monitoring
        if self.sdr_manager:
            self.sdr_manager.close()
        
        # Close Bluetooth connections
        self._cleanup_client()
        
        if self.server_sock:
            try:
                self.server_sock.close()
            except:
                pass
        
        logger.info("SIGMA server stopped")
    
    def get_status(self) -> Dict:
        """Get current server status."""
        current_time = time.time()
        uptime = current_time - self.stats['server_start_time']
        
        status = {
            'running': self.running,
            'test_mode': self.test_mode,
            'sdr_mode': self.sdr_mode,
            'uptime_seconds': uptime,
            'uptime_formatted': f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m",
            'bluetooth_connected': self.client_sock is not None,
            'statistics': self.stats.copy(),
            'current_operation': self.current_operation
        }
        
        # Add SDR status
        if self.sdr_manager:
            status['sdr_status'] = self.sdr_manager.get_status()
        
        # Add latest data summary
        with self.sdr_lock:
            status['latest_data'] = {
                'signals_count': len(self.latest_detections),
                'audio_captures_count': len(self.latest_captures),
                'last_update': self.stats['last_sweep_time']
            }
            
            # Add latest signals for display
            if self.latest_detections:
                status['latest_signals'] = [
                    {
                        'frequency': d.frequency,
                        'power_db': d.power_db,
                        'snr_db': d.snr_db,
                        'band': d.band_info.name,
                        'modulation': d.band_info.modulation.name
                    }
                    for d in self.latest_detections[:5]
                ]
        
        # Calculate next action
        if not self.running:
            status['next_action'] = "Server stopped"
        elif not self.client_sock:
            status['next_action'] = "Waiting for Android connection..."
        else:
            next_sweep = self.stats['last_sweep_time'] + self.sweep_interval
            time_to_sweep = max(0, next_sweep - current_time)
            status['next_action'] = f"Next sweep in {time_to_sweep:.0f}s"
        
        return status
    
    def print_status(self):
        """Print current status to console."""
        status = self.get_status()
        
        print("\n" + "="*60)
        print("SIGMA INTEGRATED SDR-BLUETOOTH SERVER STATUS")
        print("="*60)
        print(f"Running: {status['running']}")
        print(f"Test Mode: {status['test_mode']}")
        print(f"SDR Mode: {status['sdr_mode']}")
        print(f"Uptime: {status['uptime_formatted']}")
        print(f"Bluetooth Connected: {status['bluetooth_connected']}")
        print(f"Sweep Interval: {self.sweep_interval}s")
        
        print(f"\nStatistics:")
        stats = status['statistics']
        print(f"  Bluetooth Connections: {stats['bluetooth_connections']}")
        print(f"  Messages Sent: {stats['messages_sent']}")
        print(f"  SDR Sweeps: {stats['sdr_sweeps_completed']}")
        print(f"  Signals Detected: {stats['signals_detected']}")
        print(f"  Audio Captures: {stats['audio_captures']}")
        
        latest = status['latest_data']
        print(f"\nLatest Data:")
        print(f"  Active Signals: {latest['signals_count']}")
        print(f"  Audio Captures: {latest['audio_captures_count']}")
        
        if latest['last_update'] > 0:
            last_update = datetime.fromtimestamp(latest['last_update'])
            print(f"  Last Update: {last_update.strftime('%H:%M:%S')}")
        
        if 'sdr_status' in status:
            sdr = status['sdr_status']
            print(f"\nSDR Status:")
            print(f"  Connected: {sdr['connected']}")
            print(f"  Mock Mode: {sdr['mock_mode']}")
        
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Integrated SDR-Bluetooth Server for SIGMA Project"
    )
    parser.add_argument(
        '--test-mode', 
        action='store_true',
        help='Use test data instead of waiting for Android app'
    )
    parser.add_argument(
        '--sdr-mode',
        action='store_true', 
        help='Use real RTL-SDR hardware instead of mock data'
    )
    parser.add_argument(
        '--sweep-interval',
        type=float,
        default=60.0,
        help='Time between SDR sweeps in seconds (default: 60)'
    )
    parser.add_argument(
        '--status-interval',
        type=float,
        default=30.0,
        help='Time between status prints in seconds (default: 30)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='Number of sweep reports to send before disconnecting (default: unlimited)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Use diagnostic display instead of streaming logs'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show debug output even in quiet mode'
    )
    parser.add_argument(
        '--all-signals',
        action='store_true',
        help='Report all strong signals instead of just strongest per band'
    )
    
    args = parser.parse_args()
    
    # Create and start server
    server = IntegratedSigmaServer(
        test_mode=args.test_mode,
        sdr_mode=args.sdr_mode,
        sweep_interval=args.sweep_interval,
        max_iterations=args.iterations,
        quiet_mode=args.quiet,
        best_per_band=not args.all_signals  # Invert the flag
    )
    
    try:
        # Initialize SDR
        if not server.initialize_sdr():
            logger.error("Failed to initialize SDR, exiting")
            return 1
        
        # Start SDR monitoring
        server.start_sdr_monitoring()
        
        # Start status display
        if args.quiet:
            # Use diagnostic display
            def display_updater():
                while server.running:
                    server.display.display_status(server.get_status())
                    time.sleep(0.5)  # Update display twice per second
            
            display_thread = threading.Thread(target=display_updater, daemon=True)
            display_thread.start()
            
            # Show initial display
            server.display.display_status(server.get_status(), force=True)
        else:
            # Use regular status printer
            def status_printer():
                while server.running:
                    time.sleep(args.status_interval)
                    if server.running:
                        server.print_status()
            
            status_thread = threading.Thread(target=status_printer, daemon=True)
            status_thread.start()
            
            # Print initial status
            server.print_status()
        
        # Start Bluetooth server (blocking)
        server.start_bluetooth_server()
        
    except KeyboardInterrupt:
        logger.info("Received Ctrl+C, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1
    finally:
        server.stop_server()
    
    return 0


if __name__ == "__main__":
    exit(main())