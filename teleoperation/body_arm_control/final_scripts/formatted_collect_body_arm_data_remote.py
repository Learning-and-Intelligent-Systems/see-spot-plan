#!/usr/bin/env python3
"""
Spot data collection script for Mac (connects to robot, streams data to Nova GPU machine)

This script:
1. Runs on Mac connected to Spot robot
2. Collects joint, gripper, and body pose/velocity data at ~50 Hz
3. Streams data to Nova GPU machine (synchronized_data_collection.py)
4. Nova synchronizes with ZED (~12 Hz) and Kiwi (~5 Hz) streams on a master grid

Packet format sent to server:
    [timestamp (8 bytes, double)] [num_values (4 bytes, int)] [value1 (8 bytes)] ... [valueN (8 bytes)]
    Values: [6 arm joints] [1 gripper] [6 body pose] [3 body velocity] = 16 total

Usage:
    python formatted_collect_body_arm_data_remote.py --hostname <spot-ip> --nova-host <nova-ip> --nova-port 9999
"""

import argparse
import socket
import struct
import time
import sys
import os
from pathlib import Path

from bosdyn.client import create_standard_sdk
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.util import authenticate
import numpy as np

from spot_utils.utils import get_robot_state, verify_estop


def send_data(sock, timestamp, joint_values):
    """
    Send joint data packet to synchronized_data_collection.py server.

    Packet format (big-endian):
        [timestamp (8 bytes, double)]
        [num_values (4 bytes, uint32)]
        [value1 (8 bytes, double)] ... [valueN (8 bytes, double)]

    Example with 16 values:
        timestamp | num_values | j0 | j1 | j2 | j3 | j4 | j5 | gripper | x | y | z | yaw | pitch | roll | vx | vy | v_rot

    Args:
        sock: Connected socket to server
        timestamp: Unix timestamp in seconds (float)
        joint_values: List of 16 floats:
            [0-5]: arm joint positions (radians)
            [6]: gripper state (0-1 normalized)
            [7-9]: body position (x, y, z in meters)
            [10-12]: body orientation (yaw, pitch, roll in radians)
            [13-15]: body velocity (vx, vy, v_angular_z)
    """
    timestamp_bytes = struct.pack('>d', timestamp)
    num_values = len(joint_values)
    num_values_bytes = struct.pack('>I', num_values)
    values_bytes = struct.pack('>' + 'd' * num_values, *joint_values)

    packet = timestamp_bytes + num_values_bytes + values_bytes
    sock.sendall(packet)


def connect_to_nova(host: str, port: int):
    """
    Connect to Nova collection server.

    Args:
        host: Nova hostname/IP
        port: Nova port

    Returns:
        socket.socket: Connected socket
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print(f"Connected to Nova at {host}:{port}\n")
    return sock


def collect_body_arm_data_remote(robot, nova_sock, rate_hz=100.0):
    """
    Collect body/arm data and stream to Nova.

    Data collected per frame:
    - 6 arm joint positions (radians)
    - 1 gripper state (0-1 normalized, 0=closed, 1=open)
    - 6 body pose values (x, y, z, yaw, pitch, roll)
    - 3 body velocities (vx, vy, v_angular_z)
    Total: 16 values per frame

    Nova server (synchronized_data_collection.py) will:
    - Downsample this ~50 Hz stream to 20 Hz via interpolation
    - Synchronize with ZED (~12 Hz) and Kiwi (~5 Hz) streams
    - Upsample images to match 20 Hz policy frequency

    Args:
        robot: Spot robot object
        nova_sock: Connected socket to Nova server
        rate_hz: Collection rate in Hz (default 100 Hz, will be downsampled to policy Hz)
    """
    arm_joint_names = ["arm0.sh0", "arm0.sh1", "arm0.el0", "arm0.el1", "arm0.wr0", "arm0.wr1"]
    dt = 1.0 / rate_hz

    print(f"Collecting at {rate_hz} Hz")
    print(f"Streaming to Nova...\n")

    timestep = 0
    prev_body_x = None
    prev_body_y = None
    prev_body_yaw = None
    prev_timestamp = None

    try:
        while True:
            loop_start_time = time.time()
            current_timestamp = time.time()

            robot_state = get_robot_state(robot)
            joint_states = robot_state.kinematic_state.joint_states
            joint_dict = {js.name: js for js in joint_states}

            # Collect arm joint positions
            positions = []
            for joint_name in arm_joint_names:
                if joint_name in joint_dict:
                    position = joint_dict[joint_name].position.value
                    positions.append(position)
                else:
                    positions.append(0.0)

            # Collect gripper state
            gripper_raw = robot_state.manipulator_state.gripper_open_percentage
            gripper_normalized = gripper_raw / 100.0
            gripper_normalized = max(0.0, min(1.0, gripper_normalized))
            if gripper_normalized < 0.02:
                gripper_normalized = 0.0

            # Collect body pose
            odom_tform_body = get_odom_tform_body(robot_state.kinematic_state.transforms_snapshot)
            body_pos = odom_tform_body.position
            body_rot = odom_tform_body.rotation

            body_x = body_pos.x
            body_y = body_pos.y
            body_z = body_pos.z

            # Extract Euler angles from quaternion
            w, x, y, z = body_rot.w, body_rot.x, body_rot.y, body_rot.z
            body_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            body_pitch = np.arcsin(2*(w*y - z*x))
            body_roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))

            # Collect body velocity
            v_x = 0.0
            v_y = 0.0
            v_rot = 0.0

            try:
                if hasattr(robot_state.kinematic_state, 'velocity_of_body_in_odom'):
                    vel = robot_state.kinematic_state.velocity_of_body_in_odom
                    if vel is not None:
                        if hasattr(vel, 'linear') and vel.linear is not None:
                            v_x = vel.linear.x if hasattr(vel.linear, 'x') else 0.0
                            v_y = vel.linear.y if hasattr(vel.linear, 'y') else 0.0
                        if hasattr(vel, 'angular') and vel.angular is not None:
                            v_rot = vel.angular.z if hasattr(vel.angular, 'z') else 0.0
            except (AttributeError, KeyError, TypeError):
                pass

            # Transform velocity from odom to body frame
            if v_x != 0.0 or v_y != 0.0:
                cos_yaw = np.cos(body_yaw)
                sin_yaw = np.sin(body_yaw)
                v_x_body = v_x * cos_yaw + v_y * sin_yaw
                v_y_body = -v_x * sin_yaw + v_y * cos_yaw
            else:
                v_x_body = 0.0
                v_y_body = 0.0

            # Fallback: compute velocity from position differences if not available
            velocity_available = abs(v_x_body) > 1e-6 or abs(v_y_body) > 1e-6 or abs(v_rot) > 1e-6
            if not velocity_available and prev_body_x is not None and prev_timestamp is not None:
                dt_vel = current_timestamp - prev_timestamp
                if dt_vel > 1e-6:
                    v_x_odom = (body_x - prev_body_x) / dt_vel
                    v_y_odom = (body_y - prev_body_y) / dt_vel
                    cos_yaw = np.cos(body_yaw)
                    sin_yaw = np.sin(body_yaw)
                    v_x_body = v_x_odom * cos_yaw + v_y_odom * sin_yaw
                    v_y_body = -v_x_odom * sin_yaw + v_y_odom * cos_yaw
                    yaw_diff = body_yaw - prev_body_yaw
                    while yaw_diff > np.pi:
                        yaw_diff -= 2 * np.pi
                    while yaw_diff < -np.pi:
                        yaw_diff += 2 * np.pi
                    v_rot = yaw_diff / dt_vel

            # Build qpos: [6 joints + 1 gripper + 6 body pose + 3 body vel]
            qpos = positions + [gripper_normalized] + [body_x, body_y, body_z, body_yaw, body_pitch, body_roll] + [v_x_body, v_y_body, v_rot]

            # Send to Nova
            try:
                send_data(nova_sock, current_timestamp, qpos)
            except (BrokenPipeError, ConnectionResetError):
                print("[ERROR] Lost connection to Nova")
                break

            # Update previous values
            prev_body_x = body_x
            prev_body_y = body_y
            prev_body_yaw = body_yaw
            prev_timestamp = current_timestamp

            # Print progress
            print(f"[Timestep {timestep}]")
            for idx, joint_name in enumerate(arm_joint_names):
                print(f"  {joint_name}: {positions[idx]:.6f} rad")
            gripper_status = "OPEN" if gripper_normalized > 0.8 else ("CLOSING" if gripper_normalized > 0.2 else "CLOSED")
            print(f"  gripper: {gripper_normalized:.6f} [{gripper_status}]")
            print(f"  body: x={body_x:.6f} m, y={body_y:.6f} m, z={body_z:.6f} m")
            velocity_magnitude = np.sqrt(v_x_body**2 + v_y_body**2)
            print(f"  velocity: {velocity_magnitude:.6f} m/s")
            print()

            timestep += 1

            elapsed = time.time() - loop_start_time
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\nStopped. Sent {timestep} timesteps.\n")


def main():
    parser = argparse.ArgumentParser(
        description='Spot data collection client - streams joint data to Nova GPU machine'
    )
    parser.add_argument("--hostname", type=str, required=True,
                        help="Spot robot hostname/IP")
    parser.add_argument("--nova-host", type=str, required=True,
                        help="Nova GPU machine hostname/IP")
    parser.add_argument("--nova-port", type=int, default=9999,
                        help="Nova collection server port")
    parser.add_argument("--rate-hz", type=float, default=100.0,
                        help="Collection rate in Hz")
    args = parser.parse_args()

    sdk = create_standard_sdk("SpotBodyArmCollectRemote")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()

    nova_sock = connect_to_nova(args.nova_host, args.nova_port)

    try:
        collect_body_arm_data_remote(robot, nova_sock, args.rate_hz)
    finally:
        nova_sock.close()
        print("Closed Nova connection")


if __name__ == "__main__":
    main()
