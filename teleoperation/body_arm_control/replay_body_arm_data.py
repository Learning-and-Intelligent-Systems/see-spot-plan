import argparse
import os
import time
from datetime import datetime

from bosdyn.api import arm_command_pb2, geometry_pb2, mobility_command_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, BODY_FRAME_NAME, get_odom_tform_body
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.util import authenticate
from bosdyn.geometry import EulerZXY
from google.protobuf import wrappers_pb2
import numpy as np

from spot_utils.utils import get_robot_state, verify_estop


def replay_body_arm_data(robot, filename, rate_hz=50.0, window_size=3):
    print(f"\nReading data from: {filename}")

    positions_data = []
    arm_joint_names = ["arm0.sh0", "arm0.sh1", "arm0.el0", "arm0.el1", "arm0.wr0", "arm0.wr1"]

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split(',')
            timestep = int(parts[0])

            timestamp_utc = float(parts[1].strip())
            joint_start_idx = 2

            joint_positions = [float(p.strip()) for p in parts[joint_start_idx:joint_start_idx+6]]
            gripper_value = float(parts[joint_start_idx+6].strip()) if len(parts) > joint_start_idx+6 else None

            body_x = float(parts[joint_start_idx+7].strip())
            body_y = float(parts[joint_start_idx+8].strip())
            body_z = float(parts[joint_start_idx+9].strip())
            body_yaw = float(parts[joint_start_idx+10].strip())

            body_roll = None
            body_pitch = float(parts[joint_start_idx+11].strip())
            velocity_start_idx = joint_start_idx+12

            v_x_body = float(parts[velocity_start_idx].strip())
            v_y_body = float(parts[velocity_start_idx+1].strip())

            positions_data.append((timestep, timestamp_utc, joint_positions, gripper_value, body_x, body_y, body_z, body_yaw, body_roll, body_pitch, v_x_body, v_y_body))

    start_body = positions_data[0]
    start_positions = start_body[2]
    start_gripper = start_body[3]
    start_body_z = start_body[6]
    start_body_yaw = start_body[7]
    start_body_roll = start_body[8]
    start_body_pitch = start_body[9]

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()

    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    print("Moving to start position...")
    start_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
        start_positions[0], start_positions[1], start_positions[2],
        start_positions[3], start_positions[4], start_positions[5],
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

    # stand_cmd = RobotCommandBuilder.synchro_stand_command(
    #     body_height=0.0,
    #     footprint_R_body=footprint_R_body
    # )
    # command_client.robot_command(stand_cmd)
    # time.sleep(1.5)

    # # Detect nominal standing height dynamically
    # robot_state = get_robot_state(robot)
    # odom_tform_body = get_odom_tform_body(robot_state.kinematic_state.transforms_snapshot)
    # nominal_height = odom_tform_body.position.z
    # print("nominal height: ", nominal_height)

    nominal_height = -5.850433805135551 # calculated this with the above code
    
    # Calculate offset needed to reach the collected start height
    initial_height_offset = start_body_z - nominal_height

    # Re-issue stand command with correct offset to match collected start position
    stand_cmd = RobotCommandBuilder.synchro_stand_command(
        body_height=initial_height_offset,
        footprint_R_body= EulerZXY(yaw=0.0, roll=0.0, pitch=start_body_pitch)
    )
    command_client.robot_command(stand_cmd)
    time.sleep(1.5)

    initial_body_z = start_body_z
    print(f"Body positioned. Nominal height: {nominal_height:.3f} m, Start height: {start_body_z:.3f} m, Offset applied: {initial_height_offset:.3f} m")

    print("Starting replay...\n")

    actual_window_size = 1
    dt = 1.0 / rate_hz

    smoothed_v_x_body = 0.0
    smoothed_v_y_body = 0.0

    smoothed_height_offset = 0.0
    last_commanded_height_offset = None
    last_arm_positions = None  # Track last sent arm positions to detect stationarity

    try:
        i = 0
        while i < len(positions_data):
            loop_start_time = time.time()

            if i > 0:
                prev_timestamp = positions_data[i - 1][1]
                curr_timestamp = positions_data[i][1]
                if prev_timestamp is not None and curr_timestamp is not None:
                    dt = curr_timestamp - prev_timestamp
                    dt = max(0.001, min(0.2, dt))
            else:
                dt = 1.0 / rate_hz

            # Check if arm position has changed since last command
            current_positions = positions_data[i][2]
            arm_position_changed = (
                last_arm_positions is None or
                not np.allclose(current_positions, last_arm_positions, atol=1e-4)
            )

            trajectory_points = []
            arm_command = None  # Only set if arm needs to move

            if arm_position_changed:
                trajectory_time = max(dt, 0.03)

                for j in range(min(actual_window_size, len(positions_data) - i)):
                    data = positions_data[i + j]
                    timestep = data[0]
                    timestamp_utc = data[1]
                    positions = data[2]
                    gripper = data[3]
                    body_x = data[4]
                    body_y = data[5]
                    body_z = data[6]
                    body_yaw = data[7]
                    body_roll = data[8]
                    body_pitch = data[9]

                    point = RobotCommandBuilder.create_arm_joint_trajectory_point(
                        positions[0],
                        positions[1],
                        positions[2],
                        positions[3],
                        positions[4],
                        positions[5],
                        time_since_reference_secs=trajectory_time,
                    )
                    trajectory_points.append(point)

                # Remove velocity/acceleration limits for maximum accuracy
                max_vel = wrappers_pb2.DoubleValue(value=15.0)
                max_acc = wrappers_pb2.DoubleValue(value=30.0)

                arm_joint_traj = arm_command_pb2.ArmJointTrajectory(
                    points=trajectory_points,
                    maximum_velocity=max_vel,
                    maximum_acceleration=max_acc,
                )

                joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(
                    trajectory=arm_joint_traj
                )
                arm_command = arm_command_pb2.ArmCommand.Request(
                    arm_joint_move_command=joint_move_command
                )
                last_arm_positions = current_positions

            data = positions_data[i]
            timestep = data[0]
            timestamp_utc = data[1]
            positions = data[2]
            current_gripper = data[3]
            body_x = data[4]
            body_y = data[5]
            body_z = data[6]
            body_yaw = data[7]
            body_roll = data[8]
            body_pitch = data[9]
            v_x_body = data[10]
            v_y_body = data[11]
            gripper_command = None
            mobility_command = None

            # SEND GRIPPER COMMAND
            gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
                current_gripper
            )
            gripper_command = gripper_cmd.synchronized_command.gripper_command
            last_gripper_value = current_gripper

            # Determine mobility command independently:
            # 1. Check for walking (velocity data available and magnitude > threshold)
            # 2. Check for height adjustment (body pose data available and height offset != 0)
            mobility_command = None
            is_velocity_mobility = False  # Track whether mobility_command is a velocity command
            case_type = None  # Track what type of movement this is

            # Calculate height offset for stand command
            height_offset = None
            if body_z is not None and initial_body_z is not None:
                height_offset = body_z - initial_body_z
                height_offset = max(-0.1, min(0.1, height_offset))

            # Check if we should send velocity commands (walking)
            velocity_magnitude = 0.0
            sending_velocity = False
            if  (v_x_body is not None or v_y_body is not None):
                v_x_body_raw = v_x_body
                v_y_body_raw = v_y_body

                alpha = 0.3
                smoothed_v_x_body = alpha * v_x_body_raw + (1 - alpha) * smoothed_v_x_body if i > 0 else v_x_body_raw
                smoothed_v_y_body = alpha * v_y_body_raw + (1 - alpha) * smoothed_v_y_body if i > 0 else v_y_body_raw

                v_x_body_final = smoothed_v_x_body
                v_y_body_final = smoothed_v_y_body

                velocity_magnitude = np.sqrt(v_x_body_final**2 + v_y_body_final**2)
                velocity_threshold = 0.03  # 5 cm/s - below this, treat as stationary (was too sensitive at 1 cm/s)

                if velocity_magnitude > velocity_threshold:
                    sending_velocity = True
                    max_velocity = 1.5
                    final_v_x = max(-max_velocity, min(max_velocity, v_x_body_final))
                    final_v_y = max(-max_velocity, min(max_velocity, v_y_body_final))
                    final_v_rot = 0.0

                    velocity_cmd = RobotCommandBuilder.synchro_velocity_command(
                        v_x=final_v_x,  # forward/back
                        v_y=final_v_y,  # left/right
                        v_rot=final_v_rot
                    )
                    mobility_command = velocity_cmd.synchronized_command.mobility_command
                    is_velocity_mobility = True

                    if i % 10 == 0:
                        print(f"[WALKING] Timestep {timestep}: Walking with arm movement")
                        print(f"  Velocity: v_x={final_v_x:.4f} m/s (forward/back), v_y={final_v_y:.4f} m/s (left/right)")

            # Height commands should NEVER be sent when walking - maintain height only when stationary
            # Only check and send height commands when NOT sending velocity commands
            if sending_velocity:
                # When walking, do NOT send any height commands - maintain current height
                case_type = "WALKING"
            else:
                # When stationary, check and correct height to maintain accuracy
                height_needs_correction = False
                if height_offset is not None:
                    # Smooth height changes using exponential moving average for smooth transitions
                    height_alpha = 0.15
                    smoothed_height_offset = height_alpha * height_offset + (1 - height_alpha) * smoothed_height_offset if i > 0 else height_offset

                    # Check if smoothed height has changed significantly from last commanded height
                    height_change_threshold = 0.001  # 1mm - smooth transitions
                    if (last_commanded_height_offset is None or
                        abs(smoothed_height_offset - last_commanded_height_offset) > height_change_threshold):
                        height_needs_correction = True

                # Send height adjustment command when needed (using smoothed value)
                if height_needs_correction and height_offset is not None:
                    # Use smoothed height offset for smooth transitions
                    final_height_offset = max(-0.1, min(0.1, smoothed_height_offset))

                    # Only pass footprint_R_body if we have orientation data, otherwise omit it
                    if body_pitch is not None:
                        footprint_R_body = EulerZXY(yaw=0.0, roll=0.0, pitch=body_pitch)
                        stand_cmd = RobotCommandBuilder.synchro_stand_command(
                            body_height=final_height_offset,
                            footprint_R_body=footprint_R_body
                        )
                    else:
                        stand_cmd = RobotCommandBuilder.synchro_stand_command(
                            body_height=final_height_offset
                        )

                    mobility_command = stand_cmd.synchronized_command.mobility_command
                    last_commanded_height_offset = final_height_offset
                    case_type = "ARM_WITH_HEIGHT"
                    if i % 10 == 0:
                        print(f"[ARM_WITH_HEIGHT] Timestep {timestep}: Moving arm (standing), height offset={final_height_offset:.4f} m (smoothed from {height_offset:.4f} m)")
                else:
                    case_type = "ARM_ONLY"
                    if i % 10 == 0:
                        print(f"[ARM_ONLY] Timestep {timestep}: Moving arm (no walking, no height adjustment)")

            if mobility_command is not None:
                if gripper_command is not None:
                    if arm_command is not None:
                        sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                            arm_command=arm_command,
                            gripper_command=gripper_command,
                            mobility_command=mobility_command
                        )
                    else:
                        sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                            gripper_command=gripper_command,
                            mobility_command=mobility_command
                        )
                else:
                    if arm_command is not None:
                        sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                            arm_command=arm_command,
                            mobility_command=mobility_command
                        )
                    else:
                        sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                            mobility_command=mobility_command
                        )
            elif gripper_command is not None:
                if arm_command is not None:
                    sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                        arm_command=arm_command,
                        gripper_command=gripper_command
                    )
                else:
                    sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                        gripper_command=gripper_command
                    )
            else:
                if arm_command is not None:
                    sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                        arm_command=arm_command
                    )
                else:
                    # No commands to send - skip this iteration
                    i += 1
                    continue

            robot_command = robot_command_pb2.RobotCommand(synchronized_command=sync_command)

            # Velocity commands need an expiration time (end_time_secs)
            # The robot will either walk (velocity) OR move arm, not both simultaneously
            # Send commands continuously matching the original data collection timing
            if is_velocity_mobility:
                loop_period = dt
                expiration_duration = loop_period * 2.0 + 0.05  # Cover next 2 timesteps + 50ms buffer
                expiration_duration = max(0.1, min(expiration_duration, 1.0))  # Clamp for safety

                end_time_secs = time.time() + expiration_duration
                cmd_id = command_client.robot_command(robot_command, end_time_secs=end_time_secs)

                if i % 50 == 0:
                    print(f"  Velocity command: v_x={v_x_body_final:.4f}, v_y={v_y_body_final:.4f}, dt={loop_period:.3f}s, expiration={expiration_duration:.3f}s")
            else:
                cmd_id = command_client.robot_command(robot_command)
                timeout = max(trajectory_time * 2.5, 0.2)
                start_wait = time.time()
                while time.time() - start_wait < timeout:
                    try:
                        feedback = command_client.robot_command_feedback(cmd_id)
                        arm_feedback = feedback.feedback.synchronized_feedback.arm_command_feedback
                        if hasattr(arm_feedback, 'arm_joint_move_feedback'):
                            if arm_feedback.arm_joint_move_feedback.status == 2:  # STATUS_COMPLETE
                                break
                    except:
                        pass
                    time.sleep(0.01)

            i += 1

            elapsed = time.time() - loop_start_time
            sleep_time = dt - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nReplay stopped.")

    print(f"\nReplay complete! Executed {len(positions_data)} timesteps.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    args = parser.parse_args()

    sdk = create_standard_sdk("SpotBodyArmReplay")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()

    replay_body_arm_data(robot, "teleoperation_data/body_arm_20251123_164133.txt")


if __name__ == "__main__":
    main()
