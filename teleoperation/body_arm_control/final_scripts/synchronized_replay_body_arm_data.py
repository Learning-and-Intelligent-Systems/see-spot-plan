#!/usr/bin/env python3
"""
Synchronized Data Replay Script

Replays trajectories collected from synchronized_data_collection.py.
Reads HDF5 files in ACT++ format and replays arm, gripper, and body movements.

Data structure in HDF5:
- observations/qpos: [timesteps, 11] (6 arm joints + 1 gripper + 4 body pose params)
- observations/images/zed_camera: [timesteps, 720, 1280, 3] (ZED RGB images)
- observations/images/arm_camera: [timesteps, 720, 1280, 3] (Kiwi/arm RGB images)
- action: [timesteps, 11] (same as qpos)

Usage:
    python synchronized_replay_body_arm_data.py \
        --hostname 192.168.1.100 \
        --file teleoperation_data/episode_20251125_165947.hdf5
"""

import argparse
import h5py
import time
import numpy as np
from pathlib import Path

from bosdyn.api import arm_command_pb2, geometry_pb2, mobility_command_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, BODY_FRAME_NAME, get_odom_tform_body
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.util import authenticate
from bosdyn.geometry import EulerZXY
from google.protobuf import wrappers_pb2

from spot_utils.utils import get_robot_state, verify_estop


def synchronized_replay_body_arm_data(robot, hdf5_filename, rate_hz=None):
    """
    Replay data collected from synchronized_data_collection.py (HDF5 format).

    Args:
        robot: Spot robot instance
        hdf5_filename: Path to HDF5 file from synchronized_data_collection.py
        rate_hz: Replay frequency in Hz (if None, uses collection frequency)
    """
    print(f"\nReading synchronized HDF5 data from: {hdf5_filename}")

    # Load HDF5 file
    with h5py.File(hdf5_filename, 'r') as f:
        # Get metadata
        if rate_hz is None:
            rate_hz = f.attrs.get('rate_hz', 20.0)  # Default 20 Hz from synchronized collection

        # Load observation data
        qpos_data = f['observations/qpos'][:]  # [timesteps, 11]
        action_data = f['action'][:]  # [timesteps, 11] - usually same as qpos

    num_timesteps = len(qpos_data)
    print(f"Loaded {num_timesteps} timesteps at {rate_hz} Hz")
    print(f"Data shape - qpos: {qpos_data.shape}, action: {action_data.shape}")

    # Extract initial state
    # qpos format: [6 arm joints, 1 gripper, body_x, body_y, body_z, body_yaw]
    # Note: synchronized_data_collection.py stores 11-dim data
    start_qpos = qpos_data[0]
    start_arm_joints = start_qpos[:6]
    start_gripper = start_qpos[6]
    start_body_x = start_qpos[7]
    start_body_y = start_qpos[8]
    start_body_z = start_qpos[9]
    start_body_yaw = start_qpos[10] if len(start_qpos) > 10 else 0.0

    print(f"\nInitial state:")
    print(f"  Arm joints: {start_arm_joints}")
    print(f"  Gripper: {start_gripper}")
    print(f"  Body pose: x={start_body_x:.3f}, y={start_body_y:.3f}, z={start_body_z:.3f}, yaw={start_body_yaw:.3f}")

    # Get lease and command client
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()

    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    # Move arm to start position
    print("\nMoving to start position...")
    start_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
        start_arm_joints[0], start_arm_joints[1], start_arm_joints[2],
        start_arm_joints[3], start_arm_joints[4], start_arm_joints[5],
        time_since_reference_secs=2.0
    )
    start_traj = arm_command_pb2.ArmJointTrajectory(points=[start_point])
    start_move = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=start_traj)
    start_arm_cmd = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=start_move)

    start_gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(start_gripper)
    start_sync = synchronized_command_pb2.SynchronizedCommand.Request(
        arm_command=start_arm_cmd,
        gripper_command=start_gripper_cmd.synchronized_command.gripper_command
    )

    start_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=start_sync)
    command_client.robot_command(start_robot_cmd)
    time.sleep(2.2)

    # Get nominal standing height
    nominal_height = -5.850433805135551  # From calibration (same as in formatted_replay_body_arm_data.py)

    # Calculate height offset
    initial_height_offset = start_body_z - nominal_height

    # Stand command with correct height and orientation
    stand_cmd = RobotCommandBuilder.synchro_stand_command(
        body_height=initial_height_offset,
        footprint_R_body=EulerZXY(yaw=0.0, roll=0.0, pitch=0.0)
    )
    command_client.robot_command(stand_cmd)
    time.sleep(1.5)

    initial_body_z = start_body_z
    print(f"Body positioned. Nominal height: {nominal_height:.3f} m, Start height: {start_body_z:.3f} m, Offset applied: {initial_height_offset:.3f} m")

    print("Starting replay...\n")

    dt = 1.0 / rate_hz
    smoothed_height_offset = 0.0
    last_commanded_height_offset = None
    last_arm_positions = None

    try:
        for i in range(num_timesteps):
            loop_start_time = time.time()

            # Get current state
            qpos = qpos_data[i]
            arm_joints = qpos[:6]
            gripper = qpos[6]
            body_x = qpos[7]
            body_y = qpos[8]
            body_z = qpos[9]
            body_yaw = qpos[10] if len(qpos) > 10 else 0.0

            # Check if arm position changed
            arm_position_changed = (
                last_arm_positions is None or
                not np.allclose(arm_joints, last_arm_positions, atol=1e-4)
            )

            arm_command = None

            # Send arm command if position changed
            if arm_position_changed:
                point = RobotCommandBuilder.create_arm_joint_trajectory_point(
                    arm_joints[0], arm_joints[1], arm_joints[2],
                    arm_joints[3], arm_joints[4], arm_joints[5],
                    time_since_reference_secs=dt,
                )

                max_vel = wrappers_pb2.DoubleValue(value=15.0)
                max_acc = wrappers_pb2.DoubleValue(value=30.0)

                arm_joint_traj = arm_command_pb2.ArmJointTrajectory(
                    points=[point],
                    maximum_velocity=max_vel,
                    maximum_acceleration=max_acc,
                )

                joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(
                    trajectory=arm_joint_traj
                )
                arm_command = arm_command_pb2.ArmCommand.Request(
                    arm_joint_move_command=joint_move_command
                )
                last_arm_positions = arm_joints.copy()

            # Gripper command
            gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(gripper)
            gripper_command = gripper_cmd.synchronized_command.gripper_command

            # Height correction (when stationary)
            mobility_command = None
            height_offset = body_z - initial_body_z
            height_offset = max(-0.1, min(0.1, height_offset))

            height_needs_correction = False
            if height_offset is not None:
                height_alpha = 0.15
                smoothed_height_offset = (
                    height_alpha * height_offset + (1 - height_alpha) * smoothed_height_offset
                    if i > 0
                    else height_offset
                )

                height_change_threshold = 0.001  # 1mm
                if (last_commanded_height_offset is None or
                    abs(smoothed_height_offset - last_commanded_height_offset) > height_change_threshold):
                    height_needs_correction = True

            # Send height adjustment when needed
            if height_needs_correction:
                final_height_offset = max(-0.1, min(0.1, smoothed_height_offset))
                stand_cmd = RobotCommandBuilder.synchro_stand_command(
                    body_height=final_height_offset,
                    footprint_R_body=EulerZXY(yaw=0.0, roll=0.0, pitch=0.0)
                )
                mobility_command = stand_cmd.synchronized_command.mobility_command
                last_commanded_height_offset = final_height_offset

                if i % 20 == 0:
                    print(f"[{i:4d}] Height adjustment: {final_height_offset:.4f} m")

            # Build synchronized command
            sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                arm_command=arm_command,
                gripper_command=gripper_command,
                mobility_command=mobility_command
            )

            robot_command = robot_command_pb2.RobotCommand(synchronized_command=sync_command)

            # Send command
            command_client.robot_command(robot_command)

            if i % 50 == 0 and i > 0:
                print(f"[{i:4d}] Arm: {arm_joints}, Gripper: {gripper:.3f}")

            # Timing
            elapsed = time.time() - loop_start_time
            sleep_time = dt - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nReplay stopped by user.")

    print(f"\nReplay complete! Executed {num_timesteps} timesteps ({num_timesteps / rate_hz:.2f}s).")


def main():
    # Configuration
    ROBOT_HOSTNAME = "192.168.80.3"  # Set your robot's IP/hostname here
    HDF5_FILE = "teleoperation/body_arm_control/final_scripts/episode_20251125_165947 (2).hdf5"  # Path to HDF5 file to replay
    REPLAY_RATE_HZ = 20.0  # None = use collection rate, or set to specific Hz (e.g., 20.0)

    parser = argparse.ArgumentParser(
        description='Replay synchronized teleoperation data collected from synchronized_data_collection.py'
    )
    parser.add_argument("--hostname", type=str, default=ROBOT_HOSTNAME,
                        help="Robot hostname or IP address")
    parser.add_argument("--file", type=str, default=HDF5_FILE,
                        help="Path to HDF5 file from synchronized_data_collection.py")
    parser.add_argument("--rate-hz", type=float, default=REPLAY_RATE_HZ,
                        help="Replay frequency in Hz (default: from collection)")
    args = parser.parse_args()

    sdk = create_standard_sdk("SpotSynchronizedReplay")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()

    synchronized_replay_body_arm_data(robot, args.file, rate_hz=args.rate_hz)


if __name__ == "__main__":
    main()
