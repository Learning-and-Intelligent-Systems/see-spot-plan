#!/usr/bin/env python3
"""
Synchronized Data Collection Server (runs on GPU machine)

This server:
1. Streams ZED frames at native frequency in background thread
2. Streams Kiwi frames at native frequency in background thread
3. Listens for joint data from Mac client at ~50 Hz on separate thread
4. Creates a master time grid at policy frequency (e.g., 20 Hz)
5. Intelligently resamples all data to align with master grid:
   - Joint data: linear interpolation (handles any downsampling/upsampling)
   - ZED images: nearest neighbor, automatically detects if downsampling or upsampling
   - Kiwi images: nearest neighbor, automatically detects if downsampling or upsampling
6. Saves synchronized data to HDF5 in ACT++ format

Note: Resampling is fully adaptive - source frequencies are detected automatically
from the collected data timestamps, no hardcoded assumptions about Hz values.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │   Synchronized Data Collection Server                   │
    │                                                          │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
    │  │  ZED Stream  │  │ Joint Server │  │ Kiwi Stream  │  │
    │  │   (~12 Hz)   │  │   (~50 Hz)   │  │   (~5 Hz)    │  │
    │  │ RGB + Depth  │  │              │  │     RGB      │  │
    │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
    │         │                 │                 │           │
    │         └─────────┬───────┴────────┬────────┘           │
    │                   │                │                    │
    │         ┌─────────▼────────┐       │                    │
    │         │ Circular Buffers │       │                    │
    │         │ (timestamped)    │       │                    │
    │         └─────────┬────────┘       │                    │
    │                   │                │                    │
    │         ┌─────────▼────────────────▼───────┐            │
    │         │  Master Time Grid (20 Hz)        │            │
    │         │  Policy frequency                │            │
    │         └─────────┬──────────────────────┘            │
    │                   │                                    │
    │         ┌─────────▼─────────┐                          │
    │         │ Resampling Engine │                          │
    │         │ - Interpolate qpos│                          │
    │         │ - Upsample images │                          │
    │         │   (nearest neighb)│                          │
    │         └─────────┬─────────┘                          │
    │                   │                                    │
    │         ┌─────────▼──────────────┐                     │
    │         │ HDF5 Writer (ACT++)    │                     │
    │         │ qpos, action           │                     │
    │         │ arm_camera, zed_camera │                     │
    │         └────────────────────────┘                     │
    └─────────────────────────────────────────────────────────┘

Usage:
    python synchronized_data_collection.py \
        --output-dir teleoperation_data \
        --policy-hz 20 \
        --joint-port 9999 \
        --kiwi-port 8888
"""

import argparse
import socket
import struct
import time
import h5py
import threading
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from zed import stream_zed_frames
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TimestampedData:
    """Container for timestamped sensor data"""
    timestamp: float  # System time in seconds
    data: np.ndarray  # The actual data


class CircularTimestampBuffer:
    """Thread-safe circular buffer that stores timestamped data"""

    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def append(self, timestamp: float, data: np.ndarray):
        """Add timestamped data to buffer"""
        with self.lock:
            self.buffer.append(TimestampedData(timestamp, data))

    def get_at_time(self, target_time: float) -> Optional[np.ndarray]:
        """
        Get data closest to target_time (nearest neighbor).
        """
        with self.lock:
            if not self.buffer:
                return None

            # Find closest timestamp
            closest = min(self.buffer, key=lambda x: abs(x.timestamp - target_time))
            return closest.data.copy()

    def interpolate_at_time(self, target_time: float) -> Optional[np.ndarray]:
        """
        Interpolate data at target_time using linear interpolation.
        Used for joint data which varies smoothly.
        """
        with self.lock:
            if len(self.buffer) < 2:
                return None if not self.buffer else self.buffer[0].data.copy()

            # Find surrounding points
            items = sorted(list(self.buffer), key=lambda x: x.timestamp)

            if target_time <= items[0].timestamp:
                return items[0].data.copy()
            if target_time >= items[-1].timestamp:
                return items[-1].data.copy()

            # Find two surrounding points
            for i in range(len(items) - 1):
                if items[i].timestamp <= target_time <= items[i + 1].timestamp:
                    t0, t1 = items[i].timestamp, items[i + 1].timestamp
                    d0, d1 = items[i].data, items[i + 1].data

                    # Linear interpolation
                    alpha = (target_time - t0) / (t1 - t0)
                    return d0 * (1 - alpha) + d1 * alpha

            return items[-1].data.copy()

    def get_all(self) -> List[TimestampedData]:
        """Get all buffered data"""
        with self.lock:
            return list(self.buffer)

    def get_sampling_frequency(self) -> float:
        """
        Estimate sampling frequency in Hz based on timestamp deltas.
        Uses median of time deltas to be robust to outliers.
        """
        with self.lock:
            if len(self.buffer) < 3:
                return 0.0

            items = sorted(list(self.buffer), key=lambda x: x.timestamp)
            time_deltas = [items[i+1].timestamp - items[i].timestamp
                          for i in range(len(items)-1)]

            # Use median to be robust to outliers
            median_delta = np.median(time_deltas)
            if median_delta > 0:
                return 1.0 / median_delta
            return 0.0


def recv_exact(sock: socket.socket, num_bytes: int) -> Optional[bytes]:
    """Receive exactly num_bytes from socket"""
    data = b''
    while len(data) < num_bytes:
        try:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                return None
            data += chunk
        except socket.timeout:
            return None
    return data


def wait_for_client_connection(
    server_sock: socket.socket,
    device_name: str,
    timeout_seconds: float = 60.0
) -> Optional[socket.socket]:
    """
    Wait for a client connection with timeout.

    Args:
        server_sock: Server socket in listening state
        device_name: Name of device (e.g., "Mac", "iPhone")
        timeout_seconds: Maximum time to wait for connection

    Returns:
        Connected socket or None if timeout
    """
    server_sock.settimeout(timeout_seconds)
    try:
        logger.info(f"Waiting for {device_name} client connection (timeout: {timeout_seconds}s)...")
        conn, addr = server_sock.accept()
        conn.settimeout(10.0)
        logger.info(f"{device_name} client connected from {addr[0]}:{addr[1]}")
        return conn
    except socket.timeout:
        logger.error(f"Timeout waiting for {device_name} client (no connection within {timeout_seconds}s)")
        return None
    except Exception as e:
        logger.error(f"Error accepting {device_name} connection: {e}")
        return None


def start_joint_server(port: int = 9999) -> socket.socket:
    """Start TCP server for joint data from Mac"""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('0.0.0.0', port))
    server_sock.listen(1)

    logger.info(f"Joint server listening on 0.0.0.0:{port}")

    return server_sock


def start_kiwi_server(port: int = 8888) -> socket.socket:
    """Start TCP server for Kiwi frames from iPhone"""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('0.0.0.0', port))
    server_sock.listen(1)

    logger.info(f"Kiwi server listening on 0.0.0.0:{port}")

    return server_sock


def stream_joint_data_worker(conn: socket.socket, buffer: CircularTimestampBuffer, disconnect_event: threading.Event):
    """
    Thread worker: Receive joint data from Mac and add to buffer.

    Packet format:
        [timestamp (8 bytes, double)] [num_values (4 bytes)] [value1 (8)] [value2 (8)] ...
    """
    try:
        while True:
            header = recv_exact(conn, 12)
            if header is None:
                logger.info("Joint client disconnected - stopping collection")
                disconnect_event.set()
                break

            timestamp = struct.unpack('>d', header[:8])[0]
            num_values = struct.unpack('>I', header[8:12])[0]

            values_data = recv_exact(conn, num_values * 8)
            if values_data is None:
                logger.info("Joint client disconnected - stopping collection")
                disconnect_event.set()
                break

            values = struct.unpack('>' + 'd' * num_values, values_data)

            # Add to buffer with system time as key
            system_time = time.time()
            buffer.append(system_time, np.array(values, dtype=np.float64))

    except socket.timeout:
        logger.warning("Joint server timeout - stopping collection")
        disconnect_event.set()
    except Exception as e:
        logger.error(f"Joint server error: {e}")
        disconnect_event.set()


def stream_zed_data_worker(buffer: CircularTimestampBuffer, stop_event: threading.Event):
    """
    Thread worker: Stream ZED frames at native ~12 Hz and add to buffer.
    """
    try:
        logger.info("Starting ZED camera streaming (~12 Hz)...")
        for rgb_bgr, depth in stream_zed_frames():
            if stop_event.is_set():
                break

            # Convert BGR to RGB
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

            # Add to buffer with system time
            system_time = time.time()
            buffer.append(system_time, rgb)

    except Exception as e:
        logger.error(f"ZED streaming error: {e}")


def stream_kiwi_data_worker(conn: socket.socket, buffer: CircularTimestampBuffer, stop_event: threading.Event, disconnect_event: threading.Event):
    """
    Thread worker: Receive Kiwi frames from iPhone at ~5 Hz and add to buffer.

    Expects Protobuf format: [length (4 bytes)] [Protobuf FrameBundle data]
    Extracts JPEG image data from the Protobuf message.
    """
    try:
        logger.info("Starting Kiwi frame reception (~5 Hz)...")
        rgb_field_num = None  # Cache RGB field number after first frame
        frame_count = 0

        while not stop_event.is_set():
            # Read length prefix (4 bytes, big-endian)
            length_data = recv_exact(conn, 4)
            if length_data is None:
                logger.info("Kiwi client disconnected - stopping collection")
                disconnect_event.set()
                break

            frame_length = struct.unpack('>I', length_data)[0]

            # Read Protobuf payload
            protobuf_data = recv_exact(conn, frame_length)
            if protobuf_data is None:
                logger.info("Kiwi client disconnected - stopping collection")
                disconnect_event.set()
                break

            # Extract RGB image from Protobuf wire format
            rgb_data_manual = None

            # Find RGB field in the Protobuf message
            if rgb_field_num is None:
                # First frame: scan to find RGB field
                i = 0
                while i < len(protobuf_data):
                    tag = protobuf_data[i]
                    field_num = tag >> 3
                    wire_type = tag & 0x7
                    i += 1

                    if wire_type == 0:  # varint
                        while i < len(protobuf_data) and (protobuf_data[i] & 0x80):
                            i += 1
                        i += 1
                    elif wire_type == 2:  # length-delimited
                        length_field = 0
                        shift = 0
                        start_pos = i
                        while i < len(protobuf_data):
                            byte = protobuf_data[i]
                            length_field |= (byte & 0x7F) << shift
                            i += 1
                            if not (byte & 0x80):
                                break
                            shift += 7

                        # Look for JPEG/PNG data (large length-delimited field with image magic bytes)
                        if length_field > 1000 and i + length_field <= len(protobuf_data):
                            data = protobuf_data[i:i+length_field]
                            if data[:2] == b'\xff\xd8' or data[:4] == b'\x89PNG':
                                rgb_data_manual = data
                                rgb_field_num = field_num
                                logger.debug(f"Found RGB image in field {field_num} ({length_field} bytes)")
                                break
                        i += length_field
                    elif wire_type == 5:  # 32-bit fixed
                        i += 4
                    elif wire_type == 1:  # 64-bit fixed
                        i += 8
                    else:
                        break
            else:
                # Subsequent frames: directly extract from known field
                i = 0
                while i < len(protobuf_data):
                    tag = protobuf_data[i]
                    field_num = tag >> 3
                    wire_type = tag & 0x7
                    i += 1

                    if field_num == rgb_field_num and wire_type == 2:  # length-delimited
                        length_field = 0
                        shift = 0
                        while i < len(protobuf_data):
                            byte = protobuf_data[i]
                            length_field |= (byte & 0x7F) << shift
                            i += 1
                            if not (byte & 0x80):
                                break
                            shift += 7

                        if i + length_field <= len(protobuf_data):
                            rgb_data_manual = protobuf_data[i:i+length_field]
                        break
                    elif wire_type == 0:  # varint
                        while i < len(protobuf_data) and (protobuf_data[i] & 0x80):
                            i += 1
                        i += 1
                    elif wire_type == 2:  # length-delimited (skip)
                        length_field = 0
                        shift = 0
                        while i < len(protobuf_data):
                            byte = protobuf_data[i]
                            length_field |= (byte & 0x7F) << shift
                            i += 1
                            if not (byte & 0x80):
                                break
                            shift += 7
                        i += length_field
                    elif wire_type == 5:  # 32-bit fixed
                        i += 4
                    elif wire_type == 1:  # 64-bit fixed
                        i += 8
                    else:
                        break

            # Decode image if found
            if not rgb_data_manual:
                continue

            try:
                # Decode JPEG/PNG using PIL
                pil_image = Image.open(BytesIO(rgb_data_manual))
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                elif pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                rgb = np.array(pil_image)

                # Add to buffer with system time
                system_time = time.time()
                buffer.append(system_time, rgb)

                frame_count += 1
                if frame_count % 10 == 0:
                    logger.debug(f"Received {frame_count} Kiwi frames")

            except Exception as e:
                logger.warning(f"Failed to decode Kiwi frame: {e}")
                continue

    except socket.timeout:
        logger.warning("Kiwi server timeout - stopping collection")
        disconnect_event.set()
    except Exception as e:
        logger.error(f"Kiwi streaming error: {e}")
        disconnect_event.set()


def create_master_time_grid(
    start_time: float,
    end_time: float,
    policy_hz: float
) -> np.ndarray:
    """
    Create master time grid at policy frequency.

    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        policy_hz: Policy frequency in Hz

    Returns:
        Array of timestamps at policy frequency
    """
    dt = 1.0 / policy_hz
    num_steps = int(np.ceil((end_time - start_time) / dt)) + 1
    return np.linspace(start_time, start_time + (num_steps - 1) * dt, num_steps)


def intelligently_resample_data(
    buffer: CircularTimestampBuffer,
    master_grid: np.ndarray,
    data_type: str = 'discrete'
) -> List[Optional[np.ndarray]]:
    """
    Intelligently resample data based on source vs target frequency.

    This function automatically detects whether the source frequency is higher
    or lower than the policy frequency and applies the appropriate resampling
    method:
    - Continuous data (joint positions): Uses linear interpolation
    - Discrete data (images): Uses nearest neighbor

    Args:
        buffer: CircularTimestampBuffer with timestamped data
        master_grid: Target timestamps at policy frequency
        data_type: Either 'continuous' (joint data) or 'discrete' (image data)

    Returns:
        List of resampled data at each master grid timestamp
    """
    # Estimate source frequency
    source_hz = buffer.get_sampling_frequency()
    policy_hz = 1.0 / (master_grid[1] - master_grid[0]) if len(master_grid) > 1 else 0

    logger.info(f"  Resampling {data_type} data: source={source_hz:.2f} Hz, target={policy_hz:.2f} Hz")

    resampled = []

    if data_type == 'continuous':
        # For continuous data (joint positions), use linear interpolation
        # This works well for both upsampling and downsampling
        for target_time in master_grid:
            data = buffer.interpolate_at_time(target_time)
            resampled.append(data)

    else:  # 'discrete' (images)
        # For discrete data (images), always use nearest neighbor
        # Log when we're downsampling for monitoring
        if source_hz > policy_hz * 1.1:
            logger.debug(f"    Downsampling images: {source_hz:.2f} Hz → {policy_hz:.2f} Hz")
        elif source_hz < policy_hz * 0.9:
            logger.debug(f"    Upsampling images: {source_hz:.2f} Hz → {policy_hz:.2f} Hz")

        for target_time in master_grid:
            data = buffer.get_at_time(target_time)
            resampled.append(data)

    return resampled


def collect_synchronized(
    output_dir: str,
    policy_hz: float = 20.0,
    joint_port: int = 9999,
    kiwi_port: int = 8888,
    duration_seconds: float = 60.0
):
    """
    Main synchronized data collection loop.

    Args:
        output_dir: Directory to save HDF5 files
        policy_hz: Policy frequency for master time grid (Hz)
        joint_port: Port for joint data server
        kiwi_port: Port for Kiwi data server
        duration_seconds: How long to collect data (approximate)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create circular buffers for each data stream
    joint_buffer = CircularTimestampBuffer(max_size=2000)  # ~40s at 50 Hz
    zed_buffer = CircularTimestampBuffer(max_size=500)    # ~40s at 12 Hz
    kiwi_buffer = CircularTimestampBuffer(max_size=500)   # ~100s at 5 Hz

    # Threading events
    stop_event = threading.Event()
    disconnect_event = threading.Event()  # Signals when any client disconnects

    logger.info("="*70)
    logger.info("Synchronized Data Collection Server (ACT++ Format)")
    logger.info("="*70)
    logger.info(f"Master clock: {policy_hz} Hz (dt = {1.0/policy_hz:.4f}s)")
    logger.info(f"Expected sources:")
    logger.info(f"  Joint data: ~50 Hz → {policy_hz} Hz (interpolate + downsample)")
    logger.info(f"  ZED images: ~12 Hz → {policy_hz} Hz (nearest neighbor upsample)")
    logger.info(f"  Kiwi images: ~5 Hz → {policy_hz} Hz (nearest neighbor upsample)")
    logger.info(f"Collection duration: ~{duration_seconds}s")
    logger.info("")

    # Start ZED streaming in background
    zed_thread = threading.Thread(
        target=stream_zed_data_worker,
        args=(zed_buffer, stop_event),
        daemon=True
    )
    zed_thread.start()
    time.sleep(1.0)  # Let ZED initialize

    # Start server sockets (non-blocking setup)
    joint_server = start_joint_server(port=joint_port)
    kiwi_server = start_kiwi_server(port=kiwi_port)

    # Wait for both clients to connect (with timeout)
    logger.info("\nWaiting for both Mac and iPhone clients to connect...")
    connection_timeout = 120.0  # Allow up to 2 minutes for both connections

    joint_conn = wait_for_client_connection(joint_server, "Mac", timeout_seconds=connection_timeout)
    if joint_conn is None:
        logger.error("Failed to establish Mac connection - aborting data collection")
        stop_event.set()
        return

    kiwi_conn = wait_for_client_connection(kiwi_server, "iPhone", timeout_seconds=connection_timeout)
    if kiwi_conn is None:
        logger.error("Failed to establish iPhone connection - aborting data collection")
        stop_event.set()
        joint_conn.close()
        return

    logger.info("✓ Both Mac and iPhone clients connected - starting data streams\n")

    # Start joint data thread
    joint_thread = threading.Thread(
        target=stream_joint_data_worker,
        args=(joint_conn, joint_buffer, disconnect_event),
        daemon=True
    )
    joint_thread.start()

    # Start Kiwi data thread
    kiwi_thread = threading.Thread(
        target=stream_kiwi_data_worker,
        args=(kiwi_conn, kiwi_buffer, stop_event, disconnect_event),
        daemon=True
    )
    kiwi_thread.start()

    logger.info("All streams started. Collecting data...")
    logger.info("Press Ctrl+C to stop.\n")

    start_time = time.time()

    try:
        while time.time() - start_time < duration_seconds:
            # Check if any client disconnected
            if disconnect_event.is_set():
                logger.info("\nClient disconnection detected - stopping collection")
                break

            # Check if we have data from all sources
            joint_count = len(joint_buffer.buffer)
            zed_count = len(zed_buffer.buffer)
            kiwi_count = len(kiwi_buffer.buffer)

            if int(time.time() - start_time) % 5 == 0 and (time.time() - start_time) % 5 < 0.5:
                logger.info(f"[{time.time() - start_time:.1f}s] Buffers: joint={joint_count}, zed={zed_count}, kiwi={kiwi_count}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("\nCollection stopped by user")

    # Stop all streams
    stop_event.set()
    joint_conn.close()
    kiwi_conn.close()
    zed_thread.join(timeout=2.0)
    joint_thread.join(timeout=2.0)
    kiwi_thread.join(timeout=2.0)

    logger.info("\nAll streams stopped")

    # Calculate actual FPS from collected data
    joint_data = joint_buffer.get_all()
    zed_data = zed_buffer.get_all()
    kiwi_data = kiwi_buffer.get_all()

    if len(joint_data) < 2:
        logger.error("Not enough joint data collected")
        return

    # Compute actual frame rates
    joint_times = [item.timestamp for item in joint_data]
    joint_duration = joint_times[-1] - joint_times[0]
    actual_joint_fps = len(joint_data) / joint_duration if joint_duration > 0 else 0

    actual_zed_fps = 0
    if len(zed_data) >= 2:
        zed_times = [item.timestamp for item in zed_data]
        zed_duration = zed_times[-1] - zed_times[0]
        actual_zed_fps = len(zed_data) / zed_duration if zed_duration > 0 else 0

    actual_kiwi_fps = 0
    if len(kiwi_data) >= 2:
        kiwi_times = [item.timestamp for item in kiwi_data]
        kiwi_duration = kiwi_times[-1] - kiwi_times[0]
        actual_kiwi_fps = len(kiwi_data) / kiwi_duration if kiwi_duration > 0 else 0

    logger.info(f"\nActual Frame Rates:")
    logger.info(f"  Joint data: {actual_joint_fps:.2f} Hz ({len(joint_data)} samples in {joint_duration:.2f}s)")
    logger.info(f"  ZED images: {actual_zed_fps:.2f} Hz ({len(zed_data)} frames in {zed_times[-1] - zed_times[0]:.2f}s)" if len(zed_data) >= 2 else f"  ZED images: No valid data")
    logger.info(f"  Kiwi images: {actual_kiwi_fps:.2f} Hz ({len(kiwi_data)} frames in {kiwi_times[-1] - kiwi_times[0]:.2f}s)" if len(kiwi_data) >= 2 else f"  Kiwi images: No valid data")
    logger.info("")

    logger.info(f"Resampling data ({len(joint_data)} joint samples)...")

    # Get time range from joint data
    grid_start = joint_times[0]
    grid_end = joint_times[-1]

    # Create master grid
    master_grid = create_master_time_grid(grid_start, grid_end, policy_hz)
    logger.info(f"Master grid: {len(master_grid)} timesteps ({grid_end - grid_start:.2f}s)")

    # Resample all data to master grid with intelligent frequency detection
    logger.info("  → Resampling joint data...")
    qpos_list = intelligently_resample_data(joint_buffer, master_grid, data_type='continuous')

    logger.info("  → Resampling ZED images...")
    zed_list = intelligently_resample_data(zed_buffer, master_grid, data_type='discrete')

    logger.info("  → Resampling Kiwi images...")
    kiwi_list = intelligently_resample_data(kiwi_buffer, master_grid, data_type='discrete')

    # Handle missing data
    valid_indices = [i for i, qpos in enumerate(qpos_list) if qpos is not None]
    if not valid_indices:
        logger.error("No valid qpos data after resampling")
        return

    logger.info(f"Valid timesteps: {len(valid_indices)} / {len(qpos_list)} ({100*len(valid_indices)/len(qpos_list):.1f}%)")

    # Check data availability
    valid_zed = [i for i, img in enumerate(zed_list) if img is not None]
    valid_kiwi = [i for i, img in enumerate(kiwi_list) if img is not None]
    logger.info(f"Valid ZED frames: {len(valid_zed)} / {len(zed_list)} ({100*len(valid_zed)/len(zed_list):.1f}%)")
    logger.info(f"Valid Kiwi frames: {len(valid_kiwi)} / {len(kiwi_list)} ({100*len(valid_kiwi)/len(kiwi_list):.1f}%)")

    if len(valid_kiwi) == 0:
        logger.warning("WARNING: No valid Kiwi frames received! Check iPhone client connection.")

    # Create contiguous arrays
    dt = 1.0 / policy_hz
    num_timesteps = len(master_grid)

    qpos_array = np.zeros((num_timesteps, 11), dtype=np.float64)
    zed_array = np.zeros((num_timesteps, 720, 1280, 3), dtype=np.uint8)
    kiwi_array = np.zeros((num_timesteps, 720, 1280, 3), dtype=np.uint8)

    # Fill arrays with valid data
    for i in range(num_timesteps):
        if qpos_list[i] is not None:
            qpos = qpos_list[i]
            if len(qpos) >= 11:
                qpos_array[i] = qpos[:11]
            else:
                qpos_array[i, :len(qpos)] = qpos

        if zed_list[i] is not None:
            img = zed_list[i]
            if img.shape != (720, 1280, 3):
                zed_array[i] = cv2.resize(img, (1280, 720))
            else:
                zed_array[i] = img

        if kiwi_list[i] is not None:
            img = kiwi_list[i]
            if img.shape != (720, 1280, 3):
                kiwi_array[i] = cv2.resize(img, (1280, 720))
            else:
                kiwi_array[i] = img

    # Action = qpos for teleoperation
    action_array = qpos_array.copy()

    # Save to HDF5 in ACT++ format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"episode_{timestamp}.hdf5"

    logger.info(f"\nWriting to {output_path}...")
    t0 = time.time()

    with h5py.File(output_path, 'w') as root:
        # Attributes
        root.attrs['sim'] = False
        root.attrs['compress'] = False

        # Observations group
        obs = root.create_group('observations')
        obs.create_dataset('qpos', data=qpos_array, dtype='float64')

        # Images group
        images = obs.create_group('images')
        images.create_dataset(
            'zed_camera',
            data=zed_array,
            dtype='uint8',
            chunks=(1, 720, 1280, 3),
            compression='gzip',
            compression_opts=4
        )
        images.create_dataset(
            'arm_camera',
            data=kiwi_array,
            dtype='uint8',
            chunks=(1, 720, 1280, 3),
            compression='gzip',
            compression_opts=4
        )

        # Action
        root.create_dataset('action', data=action_array, dtype='float64')

    elapsed = time.time() - t0
    logger.info(f"Written in {elapsed:.1f}s")

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("Dataset Summary")
    logger.info("="*70)
    logger.info(f"File: {output_path.name}")
    logger.info(f"Timesteps: {len(master_grid)}")
    logger.info(f"Duration: {(grid_end - grid_start):.2f}s")
    logger.info(f"Policy frequency: {policy_hz} Hz")
    logger.info(f"")
    logger.info(f"Data Shapes:")
    logger.info(f"  observations/qpos: {qpos_array.shape}")
    logger.info(f"  observations/images/zed_camera: {zed_array.shape}")
    logger.info(f"  observations/images/arm_camera: {kiwi_array.shape}")
    logger.info(f"  action: {action_array.shape}")
    logger.info(f"")
    file_size_gb = output_path.stat().st_size / (1024**3)
    logger.info(f"File size: {file_size_gb:.2f} GB")
    logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Synchronized Data Collection - ACT++ Format'
    )
    parser.add_argument('--output-dir', type=str, default='teleoperation_data',
                        help='Directory to save HDF5 files')
    parser.add_argument('--policy-hz', type=float, default=20.0,
                        help='Master clock frequency in Hz')
    parser.add_argument('--joint-port', type=int, default=9999,
                        help='Port for joint data server')
    parser.add_argument('--kiwi-port', type=int, default=8888,
                        help='Port for Kiwi data server')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Collection duration in seconds')
    args = parser.parse_args()

    collect_synchronized(
        output_dir=args.output_dir,
        policy_hz=args.policy_hz,
        joint_port=args.joint_port,
        kiwi_port=args.kiwi_port,
        duration_seconds=args.duration
    )


if __name__ == "__main__":
    main()
