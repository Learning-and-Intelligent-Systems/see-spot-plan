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
            
            # Format: body_x, body_y, body_z, body_yaw, body_pitch (roll not stored, always 0.0)
            # Backward compatibility: check if old format has roll column
            body_x = float(parts[joint_start_idx+7].strip()) if len(parts) > joint_start_idx+7 else None
            body_y = float(parts[joint_start_idx+8].strip()) if len(parts) > joint_start_idx+8 else None
            body_z = float(parts[joint_start_idx+9].strip()) if len(parts) > joint_start_idx+9 else None
            body_yaw = float(parts[joint_start_idx+10].strip()) if len(parts) > joint_start_idx+10 else None
            # Check if file has roll column (old format) or not (new format)
            # New format: timestep, timestamp, 6 joints, gripper, body_x, body_y, body_z, body_yaw, body_pitch, v_x, v_y, v_rot = 17 columns
            # Old format: same but with body_roll between body_yaw and body_pitch = 18 columns
            if len(parts) >= 18:
                # Old format: has roll column (18+ columns)
                body_roll = float(parts[joint_start_idx+11].strip()) if parts[joint_start_idx+11].strip() else None
                body_pitch = float(parts[joint_start_idx+12].strip()) if len(parts) > joint_start_idx+12 else None
                velocity_start_idx = joint_start_idx+13
            else:
                # New format: no roll column (17 columns)
                body_roll = None
                body_pitch = float(parts[joint_start_idx+11].strip()) if len(parts) > joint_start_idx+11 else None
                velocity_start_idx = joint_start_idx+12
            
            # Read velocity data (v_x_body, v_y_body in body frame - no rotation for now)
            v_x_body = float(parts[velocity_start_idx].strip()) if len(parts) > velocity_start_idx else None
            v_y_body = float(parts[velocity_start_idx+1].strip()) if len(parts) > velocity_start_idx+1 else None

            positions_data.append((timestep, timestamp_utc, joint_positions, gripper_value, body_x, body_y, body_z, body_yaw, body_roll, body_pitch, v_x_body, v_y_body))
    
    has_timestamps = positions_data[0][1] is not None
    has_gripper_data = positions_data[0][3] is not None
    has_body_pose = positions_data[0][4] is not None
    has_velocity_data = positions_data[0][10] is not None if len(positions_data[0]) > 10 else False
    print(f"Loaded {len(positions_data)} timesteps")
    print(f"Timestamps: {'Yes' if has_timestamps else 'No (using fixed rate)'}")
    print(f"Gripper data: {'Yes' if has_gripper_data else 'No'}")
    print(f"Body pose data: {'Yes' if has_body_pose else 'No'}")
    print(f"Velocity data: {'Yes' if has_velocity_data else 'No'}")
    if has_body_pose:
        start_body = positions_data[0]
        start_body_yaw = start_body[7]
        start_body_roll = start_body[8]
        start_body_pitch = start_body[9]
        print(f"  Start body pose: x={start_body[4]:.3f}, y={start_body[5]:.3f}, z={start_body[6]:.3f} m, yaw={start_body_yaw:.3f} rad")
        if start_body_pitch is not None:
            print(f"    roll=0.0 rad, pitch={start_body_pitch:.3f} rad")
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
    
    # Move body to start position to match recorded data
    if has_body_pose:
        start_body = positions_data[0]
        start_body_x = start_body[4]
        start_body_y = start_body[5]
        start_body_z = start_body[6]
        start_body_yaw = start_body[7]
        start_body_roll = start_body[8]
        start_body_pitch = start_body[9]
        
        print(f"Moving body to start position: x={start_body_x:.3f}, y={start_body_y:.3f}, z={start_body_z:.3f} m, yaw={start_body_yaw:.3f} rad")
        
        # Move body to start position (x, y, yaw) first
        if start_body_x is not None and start_body_y is not None and start_body_yaw is not None:
            body_pose_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                goal_x=start_body_x,
                goal_y=start_body_y,
                goal_heading=start_body_yaw,
                frame_name=ODOM_FRAME_NAME
            )
            cmd_id = command_client.robot_command(body_pose_cmd)
            
            # Wait for SE2 command to complete
            timeout = 5.0
            start_wait = time.time()
            while time.time() - start_wait < timeout:
                try:
                    feedback = command_client.robot_command_feedback(cmd_id)
                    mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
                    if hasattr(mobility_feedback, 'se2_trajectory_feedback'):
                        if mobility_feedback.se2_trajectory_feedback.status == 2:  # STATUS_AT_GOAL
                            break
                except:
                    pass
                time.sleep(0.1)
            time.sleep(0.5)
        
        # Get current body pose AFTER SE2 command has completed
        robot_state = get_robot_state(robot)
        odom_tform_body = get_odom_tform_body(robot_state.kinematic_state.transforms_snapshot)
        current_body_z = odom_tform_body.position.z
        initial_body_z = current_body_z
        
        # Set body height and orientation to match start position
        # footprint_R_body controls body orientation relative to footprint (mainly roll/pitch)
        # Yaw is already handled by SE2 command, so we only use roll and pitch here
        if start_body_z is not None:
            height_offset = start_body_z - current_body_z
            height_offset = max(-0.2, min(0.2, height_offset))
            
            # Only use roll and pitch in footprint_R_body, not yaw (yaw handled by SE2)
            # footprint_R_body yaw should be 0 (body aligned with footprint)
            footprint_R_body = None
            if start_body_pitch is not None:
                footprint_R_body = EulerZXY(yaw=0.0, roll=0.0, pitch=start_body_pitch)
            
            if abs(height_offset) > 0.001 or footprint_R_body is not None:
                print(f"Adjusting body height: offset={height_offset:.3f} m")
                if footprint_R_body is not None:
                    print(f"  Setting body orientation: roll=0.0 rad, pitch={start_body_pitch:.3f} rad")
                stand_cmd = RobotCommandBuilder.synchro_stand_command(
                    body_height=height_offset,
                    footprint_R_body=footprint_R_body
                )
                command_client.robot_command(stand_cmd)
                time.sleep(1.5)
        
        # Update initial_body_z to the target start position for computing future offsets
        initial_body_z = start_body_z
        print(f"Body positioned. Initial body height for offsets: {initial_body_z:.3f} m")
        
        # Initialize last commanded body pose to start position to prevent immediate movement
        last_commanded_body_x = start_body_x
        last_commanded_body_y = start_body_y
        last_commanded_body_yaw = start_body_yaw
        last_commanded_body_z = start_body_z
    else:
        # No body pose data - just get current height for offsets
        robot_state = get_robot_state(robot)
        odom_tform_body = get_odom_tform_body(robot_state.kinematic_state.transforms_snapshot)
        initial_body_z = odom_tform_body.position.z
        print(f"Initial body height: {initial_body_z:.3f} m")
        last_commanded_body_x = None
        last_commanded_body_y = None
        last_commanded_body_yaw = None
        last_commanded_body_z = initial_body_z
    
    print("Starting replay...\n")
    
    # For maximum accuracy: use single-point trajectories (no interpolation)
    # This ensures exact position replay, critical for precise tasks like grasping
    actual_window_size = 1
    
    dt = 1.0 / rate_hz
    last_gripper_value = start_gripper if start_gripper is not None else None
    
    # Initialize smoothed velocity variables for exponential moving average
    smoothed_v_x_body = 0.0
    smoothed_v_y_body = 0.0
    smoothed_v_x_odom = 0.0
    smoothed_v_y_odom = 0.0
    
    # Initialize smoothed height offset for smooth height transitions
    smoothed_height_offset = 0.0
    last_commanded_height_offset = None
    
    try:
        i = 0
        while i < len(positions_data):
            loop_start_time = time.time()
            
            # Calculate actual dt from timestamps if available
            # This preserves the original collection timing
            if has_timestamps and i > 0:
                prev_timestamp = positions_data[i - 1][1]
                curr_timestamp = positions_data[i][1]
                if prev_timestamp is not None and curr_timestamp is not None:
                    dt = curr_timestamp - prev_timestamp
                    # Don't clamp dt too aggressively - preserve original timing
                    dt = max(0.001, min(0.2, dt))
            else:
                # No timestamps - use default rate
                dt = 1.0 / rate_hz
            
            trajectory_points = []
            
            # For single-point trajectories, use exact dt from timestamps
            # Ensure minimum time for command processing - slightly higher for perfect execution
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
            # High limits allow robot to reach exact positions without artificial constraints
            # This is critical when arm is extended - small errors become large at the end-effector
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
            v_x_body = data[10] if has_velocity_data and len(data) > 10 else None
            v_y_body = data[11] if has_velocity_data and len(data) > 11 else None
            gripper_command = None
            mobility_command = None
            
            # Always send gripper command for PERFECT accuracy - no threshold
            # This ensures perfect synchronization even for tiny changes
            if has_gripper_data and current_gripper is not None:
                gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
                    current_gripper
                )
                gripper_command = gripper_cmd.synchronized_command.gripper_command
                last_gripper_value = current_gripper
            
            body_position_changed = False
            if has_body_pose and body_x is not None and body_y is not None and body_yaw is not None:
                # Threshold: 5mm for position, 0.01 rad (~0.5 degree) for rotation
                # Smaller thresholds for better accuracy while still filtering minor noise
                if (last_commanded_body_x is None or
                    abs(body_x - last_commanded_body_x) > 0.005 or
                    abs(body_y - last_commanded_body_y) > 0.005 or
                    abs(body_yaw - last_commanded_body_yaw) > 0.01):
                    body_position_changed = True

            # Determine mobility command independently:
            # 1. Check for walking (velocity data available and magnitude > threshold)
            # 2. Check for height adjustment (body pose data available and height offset != 0)
            mobility_command = None
            is_velocity_mobility = False  # Track whether mobility_command is a velocity command
            case_type = None  # Track what type of movement this is

            # Calculate height offset for stand command
            height_offset = None
            desired_height_z = None
            if has_body_pose and body_z is not None and initial_body_z is not None:
                desired_height_z = body_z
                height_offset = body_z - initial_body_z
                height_offset = max(-0.1, min(0.1, height_offset))

            # Check if we should send velocity commands (walking)
            velocity_magnitude = 0.0
            sending_velocity = False
            if has_velocity_data and (v_x_body is not None or v_y_body is not None):
                # v_x_body and v_y_body are in body frame (forward/back and left/right)
                v_x_body_raw = v_x_body if v_x_body is not None else 0.0
                v_y_body_raw = v_y_body if v_y_body is not None else 0.0

                # Smooth body frame velocities
                alpha = 0.3
                smoothed_v_x_body = alpha * v_x_body_raw + (1 - alpha) * smoothed_v_x_body if i > 0 else v_x_body_raw
                smoothed_v_y_body = alpha * v_y_body_raw + (1 - alpha) * smoothed_v_y_body if i > 0 else v_y_body_raw

                # No transformation needed - already in body frame!
                v_x_body_final = smoothed_v_x_body
                v_y_body_final = smoothed_v_y_body

                velocity_magnitude = np.sqrt(v_x_body_final**2 + v_y_body_final**2)
                velocity_threshold = 0.01  # 1 cm/s - below this, treat as stationary

                # Send velocity commands if magnitude is meaningful
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

            # ALWAYS check and correct height when not walking - maintain height accuracy
            # Check height adjustment when stationary (not sending velocity commands)
            height_needs_correction = False
            if not sending_velocity:
                if height_offset is not None:
                    # Smooth height changes using exponential moving average for smooth transitions
                    # Lower alpha (0.15) for smoother, slower height changes
                    height_alpha = 0.15
                    smoothed_height_offset = height_alpha * height_offset + (1 - height_alpha) * smoothed_height_offset if i > 0 else height_offset
                    
                    # Check if smoothed height has changed significantly from last commanded height
                    # Use a slightly higher threshold (1mm) to avoid too frequent corrections
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
            elif sending_velocity:
                case_type = "WALKING"
            else:
                case_type = "ARM_ONLY"
                if i % 10 == 0:
                    print(f"[ARM_ONLY] Timestep {timestep}: Moving arm (no walking, no height adjustment)")
            
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
            
            # Velocity commands need an expiration time (end_time_secs)
            # The robot will either walk (velocity) OR move arm, not both simultaneously
            # Send commands continuously matching the original data collection timing
            if is_velocity_mobility:
                loop_period = dt if has_timestamps else (1.0 / rate_hz)
                
                # Set expiration to ensure command persists until next command is sent
                # Use the actual dt from timestamps to match original collection rate
                # Add small buffer to ensure no gaps between commands
                expiration_duration = loop_period * 2.0 + 0.05  # Cover next 2 timesteps + 50ms buffer
                expiration_duration = max(0.1, min(expiration_duration, 1.0))  # Clamp for safety
                
                end_time_secs = time.time() + expiration_duration
                cmd_id = command_client.robot_command(robot_command, end_time_secs=end_time_secs)
                command_completed = False
                
                if i % 50 == 0:
                    # Use v_x_body_final and v_y_body_final which are always defined
                    print(f"  Velocity command: v_x={v_x_body_final:.4f}, v_y={v_y_body_final:.4f}, dt={loop_period:.3f}s, expiration={expiration_duration:.3f}s")
            else:
                cmd_id = command_client.robot_command(robot_command)
                command_completed = False
                timeout = max(trajectory_time * 2.5, 0.2)
                start_wait = time.time()
                while time.time() - start_wait < timeout:
                    try:
                        feedback = command_client.robot_command_feedback(cmd_id)
                        arm_feedback = feedback.feedback.synchronized_feedback.arm_command_feedback
                        if hasattr(arm_feedback, 'arm_joint_move_feedback'):
                            if arm_feedback.arm_joint_move_feedback.status == 2:  # STATUS_COMPLETE
                                command_completed = True
                                break
                    except:
                        pass
                    time.sleep(0.01)
                    
            try:
                robot_state = get_robot_state(robot)
                joint_states = robot_state.kinematic_state.joint_states
                joint_dict = {js.name: js for js in joint_states}
                
                # Verify arm positions
                max_error = 0.0
                for idx, joint_name in enumerate(arm_joint_names):
                    if joint_name in joint_dict:
                        actual_pos = joint_dict[joint_name].position.value
                        commanded_pos = positions[idx]
                        error = abs(actual_pos - commanded_pos)
                        max_error = max(max_error, error)
                
                if max_error > 0.03:  # Warn if error > 0.03 rad (~1.7 degrees)
                    print(f"  WARNING: Arm position error at timestep {timestep}, max_error={max_error:.4f} rad (cmd_completed={command_completed})")
            except:
                pass
            
            if i % 10 == 0:
                data = positions_data[i]
                timestep = data[0]
                timestamp_utc = data[1]
                positions = data[2]
                gripper_val = data[3]
                body_x = data[4]
                body_y = data[5]
                body_z = data[6]
                body_yaw = data[7]
                body_roll = data[8]
                body_pitch = data[9]
                
                gripper_str = f", gripper={gripper_val:.4f}" if gripper_val is not None else ""
                dt_str = f", dt={dt*1000:.1f}ms" if has_timestamps else ""
                cmd_status = "✓" if command_completed else "⏳"
                print(f"Timestep {timestep} {cmd_status}: sh0={positions[0]:.4f}, sh1={positions[1]:.4f}, el0={positions[2]:.4f}{gripper_str}{dt_str}")
                if has_body_pose:
                    print(f"  Body: x={body_x:.3f}, y={body_y:.3f}, z={body_z:.3f} m, yaw={body_yaw:.3f} rad")
            
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
    
    replay_body_arm_data(robot, "teleoperation_data/body_arm_20251120_171339.txt")


if __name__ == "__main__":
    main()

