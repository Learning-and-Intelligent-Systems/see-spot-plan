#!/usr/bin/env python3
"""
ZED Collection Server (runs on Nova GPU machine)

This server:
1. Listens for joint data from Mac client
2. Captures ZED frames locally in sync with incoming data
3. Saves both joint data + ZED images to HDF5 file

Usage:
    python zed_collection_server.py --output-dir teleoperation_data --port 9999
"""

import argparse
import socket
import struct
import time
import h5py
import json
import threading
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from zed import stream_zed_frames


def recv_exact(sock, num_bytes):
    """Receive exactly num_bytes from socket (TCP requires this)"""
    data = b''
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            return None  # Connection closed
        data += chunk
    return data


def start_server(host: str = '0.0.0.0', port: int = 9999):
    """
    Start TCP server and wait for Mac client connection.

    Args:
        host: Host to listen on
        port: Port to listen on

    Returns:
        Tuple[socket.socket, tuple]: Connected socket and client address
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)

    print("=" * 60)
    print("ZED Collection Server (Joint Data Receiver)")
    print("=" * 60)
    print(f"Listening on {host}:{port}")
    print("Waiting for Mac client connection...\n")

    conn, addr = server_sock.accept()
    print(f"Connected from {addr[0]}:{addr[1]}\n")

    return conn, addr


def recv_joint_data(conn: socket.socket):
    """
    Receive joint data packets from Mac client.

    Packet format:
        [timestamp (8 bytes, double)] [num_values (4 bytes)] [value1 (8)] [value2 (8)] ...

    Args:
        conn: Connected socket

    Yields:
        Tuple[float, list]: (timestamp, joint_values)
    """
    try:
        while True:
            # Read header: timestamp (8) + num_values (4)
            header = recv_exact(conn, 12)
            if header is None:
                print("[OK] Client disconnected")
                break

            timestamp = struct.unpack('>d', header[:8])[0]
            num_values = struct.unpack('>I', header[8:12])[0]

            # Read joint values
            values_data = recv_exact(conn, num_values * 8)
            if values_data is None:
                print("[OK] Client disconnected")
                break

            values = struct.unpack('>' + 'd' * num_values, values_data)
            yield timestamp, list(values)

    except KeyboardInterrupt:
        print("[OK] Server stopped")
    except Exception as e:
        print(f"[ERROR] {e}")


def collect_with_zed(output_dir: str, port: int = 9999):
    """
    Main collection loop: receive joint data from Mac and capture ZED frames locally.

    Args:
        output_dir: Directory to save HDF5 file
        port: Port to listen on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create HDF5 file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = output_dir / f"body_arm_{timestamp}.hdf5"

    print(f"Output file: {dataset_path}\n")

    # Start ZED streaming in background thread
    latest_zed_frame = {"rgb": None, "depth": None}
    zed_lock = threading.Lock()
    stop_zed_streaming = threading.Event()

    def zed_stream_worker():
        """Background thread that streams ZED frames."""
        try:
            for rgb_bgr, depth in stream_zed_frames():
                if stop_zed_streaming.is_set():
                    break
                with zed_lock:
                    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                    latest_zed_frame["rgb"] = rgb.copy()
                    latest_zed_frame["depth"] = depth.copy()
        except Exception as e:
            print(f"ZED streaming error: {e}")

    zed_thread = threading.Thread(target=zed_stream_worker, daemon=True)
    zed_thread.start()
    print("Started ZED camera streaming...\n")
    time.sleep(1.0)  # Give ZED time to start

    # Start server and receive joint data
    conn, addr = start_server(port=port)

    qpos_data = []
    body_pose_data = []
    body_vel_data = []
    action_data = []
    timestamps = []
    zed_rgb_data = []
    zed_depth_data = []

    frame_count = 0

    try:
        for timestamp, values in recv_joint_data(conn):
            # values format: [6 arm joints + 1 gripper + 6 body pose + 3 body vel]
            # = [joint0, joint1, joint2, joint3, joint4, joint5, gripper, body_x, body_y, body_z, body_yaw, body_pitch, body_roll, v_x, v_y, v_rot]

            if len(values) != 16:
                print(f"[WARN] Expected 16 values, got {len(values)}, skipping")
                continue

            # Extract components
            joints = values[:6]
            gripper = values[6]
            body_pose = values[7:13]
            body_vel = values[13:16]

            qpos = values
            qpos_data.append(qpos)
            body_pose_data.append(body_pose)
            body_vel_data.append(body_vel)
            action_data.append(qpos)
            timestamps.append(timestamp)

            # Capture latest ZED frame
            with zed_lock:
                if latest_zed_frame["rgb"] is not None:
                    zed_rgb_data.append(latest_zed_frame["rgb"].copy())
                    zed_depth_data.append(latest_zed_frame["depth"].copy())
                else:
                    # Placeholder if ZED not ready yet
                    zed_rgb_data.append(np.zeros((720, 1280, 3), dtype=np.uint8))
                    zed_depth_data.append(np.zeros((720, 1280), dtype=np.float32))

            frame_count += 1
            if frame_count % 50 == 0:
                print(f"[Received {frame_count} frames]")

    except KeyboardInterrupt:
        print(f"\nStopped. Collected {len(qpos_data)} timesteps.")

    finally:
        # Stop ZED streaming
        stop_zed_streaming.set()
        zed_thread.join(timeout=2.0)
        print("Stopped ZED camera streaming")

        conn.close()
        print("Closed connection\n")

        # Save to HDF5
        print(f"Saving {len(qpos_data)} timesteps to {dataset_path}...")
        t0 = time.time()

        qpos_array = np.array(qpos_data, dtype=np.float64)
        body_pose_array = np.array(body_pose_data, dtype=np.float64)
        body_vel_array = np.array(body_vel_data, dtype=np.float64)
        action_array = np.array(action_data, dtype=np.float64)
        timestamps_array = np.array(timestamps, dtype=np.float64)
        zed_rgb_array = np.array(zed_rgb_data, dtype=np.uint8)
        zed_depth_array = np.array(zed_depth_data, dtype=np.float32)

        with h5py.File(dataset_path, 'w') as root:
            root.attrs['sim'] = False
            root.attrs['rate_hz'] = 100.0  # Nominal rate
            root.attrs['arm_joint_names'] = ["arm0.sh0", "arm0.sh1", "arm0.el0", "arm0.el1", "arm0.wr0", "arm0.wr1"]
            root.attrs['timestamp'] = timestamp

            obs = root.create_group('observations')
            obs.create_dataset('qpos', data=qpos_array, dtype='float64')
            obs.create_dataset('body_pose', data=body_pose_array, dtype='float64')
            obs.create_dataset('body_vel', data=body_vel_array, dtype='float64')
            obs.create_dataset('timestamps', data=timestamps_array, dtype='float64')

            # Save ZED images
            images = obs.create_group('images')
            images.create_dataset('zed_rgb', data=zed_rgb_array, dtype='uint8',
                                  chunks=(1, 720, 1280, 3), compression='gzip', compression_opts=4)
            images.create_dataset('zed_depth', data=zed_depth_array, dtype='float32',
                                  chunks=(1, 720, 1280), compression='gzip', compression_opts=4)

            root.create_dataset('action', data=action_array, dtype='float64')

        print(f'Saving completed in {time.time() - t0:.1f} seconds')
        print(f'HDF5 file created: {dataset_path}')
        print(f'Dataset structure:')
        print(f'  observations/qpos: {qpos_array.shape}')
        print(f'  observations/body_pose: {body_pose_array.shape}')
        print(f'  observations/body_vel: {body_vel_array.shape}')
        print(f'  observations/images/zed_rgb: {zed_rgb_array.shape}')
        print(f'  observations/images/zed_depth: {zed_depth_array.shape}')
        print(f'  action: {action_array.shape}')


def main():
    parser = argparse.ArgumentParser(
        description='ZED Collection Server - receives joint data from Mac and captures ZED images'
    )
    parser.add_argument('--output-dir', type=str, default='teleoperation_data',
                        help='Directory to save HDF5 files')
    parser.add_argument('--port', type=int, default=9999,
                        help='Port to listen on')
    args = parser.parse_args()

    collect_with_zed(args.output_dir, args.port)


if __name__ == "__main__":
    main()
