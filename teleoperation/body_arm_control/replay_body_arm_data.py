import argparse
import os
import time
from datetime import datetime

from bosdyn.api import arm_command_pb2, geometry_pb2, mobility_command_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import create_standard_sdk
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, BODY_FRAME_NAME, get_odom_tform_body
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.util import authenticate
from google.protobuf import wrappers_pb2

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
            
            # Check if file has timestamps (new format) or not (old format)
            if len(parts) >= 9:
                try:
                    timestamp_utc = float(parts[1].strip())
                    joint_start_idx = 2
                except (ValueError, IndexError):
                    timestamp_utc = None
                    joint_start_idx = 1
            else:
                timestamp_utc = None
                joint_start_idx = 1
            
            joint_positions = [float(p.strip()) for p in parts[joint_start_idx:joint_start_idx+6]]
            gripper_value = float(parts[joint_start_idx+6].strip()) if len(parts) > joint_start_idx+6 else None
            body_x = float(parts[joint_start_idx+7].strip()) if len(parts) > joint_start_idx+7 else None
            body_y = float(parts[joint_start_idx+8].strip()) if len(parts) > joint_start_idx+8 else None
            body_z = float(parts[joint_start_idx+9].strip()) if len(parts) > joint_start_idx+9 else None
            body_theta = float(parts[joint_start_idx+10].strip()) if len(parts) > joint_start_idx+10 else None
            
            positions_data.append((timestep, timestamp_utc, joint_positions, gripper_value, body_x, body_y, body_z, body_theta))
    
    has_timestamps = positions_data[0][1] is not None
    has_gripper_data = positions_data[0][3] is not None
    has_body_pose = positions_data[0][4] is not None
    print(f"Loaded {len(positions_data)} timesteps")
    print(f"Timestamps: {'Yes' if has_timestamps else 'No (using fixed rate)'}")
    print(f"Gripper data: {'Yes' if has_gripper_data else 'No'}")
    print(f"Body pose data: {'Yes' if has_body_pose else 'No'}")
    if has_body_pose:
        start_body = positions_data[0]
        print(f"  Start body pose: x={start_body[4]:.3f}, y={start_body[5]:.3f}, z={start_body[6]:.3f} m, theta={start_body[7]:.3f} rad")
    if has_timestamps:
        print(f"Using original collection timestamps for timing (ignoring --rate parameter)")
    else:
        print(f"Replaying at {rate_hz} Hz with {window_size}-point trajectory window")
    print("Press Ctrl+C to stop.\n")
    
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    
    print("Moving to start position...")
    start_positions = positions_data[0][2]
    start_gripper = positions_data[0][3] if has_gripper_data else None
    start_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
        start_positions[0], start_positions[1], start_positions[2],
        start_positions[3], start_positions[4], start_positions[5],
        time_since_reference_secs=2.0
    )
    start_traj = arm_command_pb2.ArmJointTrajectory(points=[start_point])
    start_move = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=start_traj)
    start_arm_cmd = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=start_move)
    
    if has_gripper_data and start_gripper is not None:
        start_gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(start_gripper)
        start_sync = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=start_arm_cmd,
            gripper_command=start_gripper_cmd.synchronized_command.gripper_command
        )
    else:
        start_sync = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=start_arm_cmd)
    
    start_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=start_sync)
    command_client.robot_command(start_robot_cmd)
    time.sleep(2.2)
    
    # Get initial body height for computing height offsets
    initial_body_z = None
    if has_body_pose:
        robot_state = get_robot_state(robot)
        odom_tform_body = get_odom_tform_body(robot_state.kinematic_state.transforms_snapshot)
        initial_body_z = odom_tform_body.position.z
        print(f"Initial body height: {initial_body_z:.3f} m")
    
    print("Starting replay...\n")
    
    # Use smaller window or single point for more accurate replay
    if window_size == 3 and has_timestamps:
        # With timestamps, single-point commands are more accurate
        actual_window_size = 1
    else:
        actual_window_size = window_size
    
    dt = 1.0 / rate_hz
    last_gripper_value = start_gripper if start_gripper is not None else None
    last_commanded_body_z = initial_body_z if initial_body_z is not None else None
    last_commanded_body_x = None
    last_commanded_body_y = None
    last_commanded_body_theta = None
    
    try:
        i = 0
        while i < len(positions_data):
            loop_start_time = time.time()
            
            # Calculate actual dt from timestamps if available
            if has_timestamps and i > 0:
                prev_timestamp = positions_data[i - 1][1]
                curr_timestamp = positions_data[i][1]
                if prev_timestamp is not None and curr_timestamp is not None:
                    dt = curr_timestamp - prev_timestamp
                    dt = max(0.005, min(0.1, dt))
            
            trajectory_points = []
            for j in range(min(actual_window_size, len(positions_data) - i)):
                timestep, timestamp_utc, positions, gripper, body_x, body_y, body_z, body_theta = positions_data[i + j]
                
                point = RobotCommandBuilder.create_arm_joint_trajectory_point(
                    positions[0],
                    positions[1],
                    positions[2],
                    positions[3],
                    positions[4],
                    positions[5],
                    time_since_reference_secs=(j + 1) * dt,
                )
                trajectory_points.append(point)
            
            # Higher velocity/acceleration limits for more accurate replay
            # (closer to natural robot movements during collection)
            max_vel = wrappers_pb2.DoubleValue(value=8.0)
            max_acc = wrappers_pb2.DoubleValue(value=15.0)
            
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
            
            timestep, timestamp_utc, positions, current_gripper, body_x, body_y, body_z, body_theta = positions_data[i]
            gripper_command = None
            mobility_command = None
            
            # Always send gripper command for accuracy (remove threshold filtering)
            if has_gripper_data and current_gripper is not None:
                gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
                    current_gripper
                )
                gripper_command = gripper_cmd.synchronized_command.gripper_command
                last_gripper_value = current_gripper
            
            # Reduced thresholds for more accurate replay (5mm instead of 1cm)
            body_position_changed = False
            if has_body_pose and body_x is not None and body_y is not None and body_theta is not None:
                if (last_commanded_body_x is None or 
                    abs(body_x - last_commanded_body_x) > 0.005 or
                    abs(body_y - last_commanded_body_y) > 0.005 or
                    abs(body_theta - last_commanded_body_theta) > 0.005):
                    body_position_changed = True
            
            # Reduced threshold for height changes (5mm instead of 1cm)
            height_command_needed = False
            height_offset = None
            if has_body_pose and body_z is not None and initial_body_z is not None:
                height_offset = body_z - initial_body_z
                height_offset = max(-0.2, min(0.2, height_offset))
                
                if last_commanded_body_z is None or abs(body_z - last_commanded_body_z) > 0.005:
                    height_command_needed = True
            
            # Strategy: 
            # - If body is moving (x, y, theta changed), use SE2 commands (which will override height)
            #   In this case, the robot should automatically adjust height based on arm pose
            # - If body is NOT moving but height needs to change, use stand command with arm command
            #   This allows explicit height control without movement conflicts
            
            mobility_command = None
            
            if body_position_changed:
                # Body is moving - use SE2 commands
                # Note: SE2 commands override stand commands, so we can't explicitly control height
                # However, Spot's mobility stack should automatically adjust body height based on
                # the arm's pose to maintain balance. This is the expected automatic behavior.
                # The robot will automatically lower/raise its body based on where the arm is positioned.
                body_pose_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                    goal_x=body_x,
                    goal_y=body_y,
                    goal_heading=body_theta,
                    frame_name=ODOM_FRAME_NAME
                )
                mobility_command = body_pose_cmd.synchronized_command.mobility_command
                last_commanded_body_x = body_x
                last_commanded_body_y = body_y
                last_commanded_body_theta = body_theta
                # Note: We don't update last_commanded_body_z here because we're relying on
                # automatic adjustment, not explicit height commands
                if i % 10 == 0:
                    print(f"  SE2 movement: x={body_x:.3f}, y={body_y:.3f}, theta={body_theta:.3f} (height={body_z:.3f}, auto adjustment expected)")
            
            elif height_command_needed and height_offset is not None:
                # Body is stationary but height needs to change - use stand command
                # This works when combined with arm commands without SE2 movement
                stand_cmd = RobotCommandBuilder.synchro_stand_command(body_height=height_offset)
                mobility_command = stand_cmd.synchronized_command.mobility_command
                last_commanded_body_z = body_z
                if i % 10 == 0:
                    print(f"  Stand height command: offset={height_offset:.3f} m (z={body_z:.3f} m)")
            
            if mobility_command is not None:
                if gripper_command is not None:
                    sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                        arm_command=arm_command,
                        gripper_command=gripper_command,
                        mobility_command=mobility_command
                    )
                else:
                    sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                        arm_command=arm_command,
                        mobility_command=mobility_command
                    )
            elif gripper_command is not None:
                sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                    arm_command=arm_command,
                    gripper_command=gripper_command
                )
            else:
                sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                    arm_command=arm_command
                )
            
            robot_command = robot_command_pb2.RobotCommand(synchronized_command=sync_command)
            command_client.robot_command(robot_command)
            
            if i % 10 == 0:
                timestep, timestamp_utc, positions, gripper_val, body_x, body_y, body_z, body_theta = positions_data[i]
                gripper_str = f", gripper={gripper_val:.4f}" if gripper_val is not None else ""
                dt_str = f", dt={dt*1000:.1f}ms" if has_timestamps else ""
                print(f"Timestep {timestep}: sh0={positions[0]:.4f}, sh1={positions[1]:.4f}, el0={positions[2]:.4f}{gripper_str}{dt_str}")
                if has_body_pose:
                    print(f"  Body: x={body_x:.3f}, y={body_y:.3f}, z={body_z:.3f} m, theta={body_theta:.3f} rad")
            
            i += 1
            
            elapsed = time.time() - loop_start_time
            sleep_time = max(0, dt - elapsed)
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
    
    # CHANGE THIS FILE TO REPLAY THE MOTION!
    # YOUR FILE SHOULD BE IN THE 'teleoperation_data' FOLDER!
    replay_body_arm_data(robot, "teleoperation_data/body_arm_20251117_232047.txt")


if __name__ == "__main__":
    main()

