import argparse
from calendar import c
from math import e
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
            if len(parts) > joint_start_idx+12:
                # Old format: has roll column
                body_roll = float(parts[joint_start_idx+11].strip()) if parts[joint_start_idx+11].strip() else None
                body_pitch = float(parts[joint_start_idx+12].strip()) if len(parts) > joint_start_idx+12 else None
                velocity_start_idx = joint_start_idx+13
            else:
                # New format: no roll column
                body_roll = None
                body_pitch = float(parts[joint_start_idx+11].strip()) if len(parts) > joint_start_idx+11 else None
                velocity_start_idx = joint_start_idx+12
            
            v_x = float(parts[velocity_start_idx].strip()) if len(parts) > velocity_start_idx else None
            v_y = float(parts[velocity_start_idx+1].strip()) if len(parts) > velocity_start_idx+1 else None
            v_rot = float(parts[velocity_start_idx+2].strip()) if len(parts) > velocity_start_idx+2 else None
            
            positions_data.append((timestep, timestamp_utc, joint_positions, gripper_value, body_x, body_y, body_z, body_yaw, body_roll, body_pitch, v_x, v_y, v_rot))
    
    has_timestamps = positions_data[0][1] is not None
    has_gripper_data = positions_data[0][3] is not None
    has_body_pose = positions_data[0][4] is not None
    has_velocity_data = positions_data[0][10] is not None

    print(f"Loaded {len(positions_data)} timesteps")
    print(f"Timestamps: {'Yes' if has_timestamps else 'No (using fixed rate)'}")
    print(f"Gripper data: {'Yes' if has_gripper_data else 'No'}")
    print(f"Body pose data: {'Yes' if has_body_pose else 'No'}")
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
    smoothed_v_rot_body = 0.0
    
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
            v_x_recorded = data[10]
            v_y_recorded = data[11]
            v_rot_recorded = data[12]
            gripper_command = None
            mobility_command = None

            v_x_odom = 0.0
            v_y_odom = 0.0
            v_rot_odom = 0.0
            if has_velocity_data and v_x_recorded is not None and v_y_recorded is not None and v_rot_recorded is not None:
                v_x_odom = v_x_recorded
                v_y_odom = v_y_recorded
                v_rot_odom = v_rot_recorded
            
            # Always send gripper command for PERFECT accuracy - no threshold
            # This ensures perfect synchronization even for tiny changes
            if has_gripper_data and current_gripper is not None:
                gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
                    current_gripper
                )
                gripper_command = gripper_cmd.synchronized_command.gripper_command
                last_gripper_value = current_gripper
            
            # Body position change detection with hysteresis for noise filtering
            # Use meaningful thresholds to filter out small variations/noise in recorded data
            # Only send SE2 commands when there's a significant position change
            # Reduced thresholds for better accuracy while still filtering noise
            body_position_changed = False
            if has_body_pose and body_x is not None and body_y is not None and body_yaw is not None:
                # Threshold: 5mm for position, 0.01 rad (~0.5 degree) for rotation
                # Smaller thresholds for better accuracy while still filtering minor noise
                if (last_commanded_body_x is None or 
                    abs(body_x - last_commanded_body_x) > 0.005 or
                    abs(body_y - last_commanded_body_y) > 0.005 or
                    abs(body_yaw - last_commanded_body_yaw) > 0.01):
                    body_position_changed = True
            
            # Minimal threshold for height (0.5mm) - PERFECT accuracy
            # Reduced threshold for better height matching
            # Also check actual body height and correct if off
            height_command_needed = False
            height_offset = None
            if has_body_pose and body_z is not None and initial_body_z is not None:
                # Check if we need to adjust based on recorded height
                height_change_needed = (last_commanded_body_z is None or abs(body_z - last_commanded_body_z) > 0.0005)
                
                # Also check actual body height if available (from previous verification)
                actual_height_check_needed = False
                try:
                    robot_state_check = get_robot_state(robot)
                    odom_tform_body_check = get_odom_tform_body(robot_state_check.kinematic_state.transforms_snapshot)
                    actual_body_z_check = odom_tform_body_check.position.z
                    # If actual height is off by more than 2mm, force correction
                    if abs(actual_body_z_check - body_z) > 0.002:
                        actual_height_check_needed = True
                except:
                    pass
                
                if height_change_needed or actual_height_check_needed:
                    height_offset = body_z - initial_body_z
                    height_offset = max(-0.2, min(0.2, height_offset))
                    height_command_needed = True
            

            mobility_command = None

            if body_position_changed:
                # Velocity is already in odom frame from collection, use directly
                # Apply smoothing to reduce noise
                alpha = 0.3
                smoothed_v_x_body = alpha * v_x_odom + (1 - alpha) * smoothed_v_x_body
                smoothed_v_y_body = alpha * v_y_odom + (1 - alpha) * smoothed_v_y_body
                smoothed_v_rot_body = alpha * v_rot_odom + (1 - alpha) * smoothed_v_rot_body

                velocity_cmd = RobotCommandBuilder.synchro_velocity_command(
                    v_x=smoothed_v_x_body,
                    v_y=smoothed_v_y_body,
                    r_rot=smoothed_v_rot_body
                )
                mobility_command = velocity_cmd.synchronized_command.mobility_command

                # update last commanded position to prevent position change detection from triggering SE2
                if has_body_pose and body_x is not None and body_y is not None and body_yaw is not None:
                    last_commanded_body_x = body_x
                    last_commanded_body_y = body_y
                    last_commanded_body_yaw = body_yaw
                    last_commanded_body_z = body_z

            elif has_velocity_data and (v_x_odom != 0.0 or v_y_odom != 0.0 or v_rot_odom != 0.0):
                # Even if body position doesn't change, send velocity commands if robot is moving
                alpha = 0.3
                smoothed_v_x_body = alpha * v_x_odom + (1 - alpha) * smoothed_v_x_body
                smoothed_v_y_body = alpha * v_y_odom + (1 - alpha) * smoothed_v_y_body
                smoothed_v_rot_body = alpha * v_rot_odom + (1 - alpha) * smoothed_v_rot_body

                velocity_cmd = RobotCommandBuilder.synchro_velocity_command(
                    v_x=smoothed_v_x_body,
                    v_y=smoothed_v_y_body,
                    r_rot=smoothed_v_rot_body
                )
                mobility_command = velocity_cmd.synchronized_command.mobility_command

            elif height_command_needed and height_offset is not None:
                # Body is stationary but height needs to change - use stand command
                # This works when combined with arm commands without SE2 movement
                
                # Only use roll and pitch in footprint_R_body, not yaw (yaw handled by SE2 if body moves)
                # footprint_R_body yaw should be 0 (body aligned with footprint)
                footprint_R_body = None
                if body_pitch is not None:
                    footprint_R_body = EulerZXY(yaw=0.0, roll=0.0, pitch=body_pitch)
                
                stand_cmd = RobotCommandBuilder.synchro_stand_command(
                    body_height=height_offset,
                    footprint_R_body=footprint_R_body
                )
                mobility_command = stand_cmd.synchronized_command.mobility_command
                last_commanded_body_z = body_z
                if i % 10 == 0:
                    orient_str = " (with orientation)" if footprint_R_body is not None else ""
                    print(f"  Stand height command: offset={height_offset:.3f} m (z={body_z:.3f} m){orient_str}")
            
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
            cmd_id = command_client.robot_command(robot_command)
            
            # For PERFECT accuracy: wait for ALL commands to complete
            # This ensures no command queue buildup and perfect synchronization
            # Waiting on every command guarantees exact position matching
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
            
            # If body moved (SE2 command), verify and correct height if needed
            # SE2 commands can override height, so we check and adjust after movement
            if body_position_changed and command_completed and has_body_pose and body_z is not None and initial_body_z is not None:
                try:
                    robot_state_check = get_robot_state(robot)
                    odom_tform_body_check = get_odom_tform_body(robot_state_check.kinematic_state.transforms_snapshot)
                    actual_body_z_check = odom_tform_body_check.position.z
                    # If height is off by more than 1mm after SE2 movement, correct it
                    if abs(actual_body_z_check - body_z) > 0.001:
                        height_offset_followup = body_z - initial_body_z
                        height_offset_followup = max(-0.2, min(0.2, height_offset_followup))
                        
                        # Only use roll and pitch in footprint_R_body, not yaw (yaw handled by SE2)
                        # footprint_R_body yaw should be 0 (body aligned with footprint)
                        footprint_R_body_followup = None
                        if body_pitch is not None:
                            footprint_R_body_followup = EulerZXY(yaw=0.0, roll=0.0, pitch=body_pitch)
                        
                        stand_cmd_followup = RobotCommandBuilder.synchro_stand_command(
                            body_height=height_offset_followup,
                            footprint_R_body=footprint_R_body_followup
                        )
                        followup_sync = synchronized_command_pb2.SynchronizedCommand.Request(
                            arm_command=arm_command,
                            mobility_command=stand_cmd_followup.synchronized_command.mobility_command
                        )
                        followup_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=followup_sync)
                        command_client.robot_command(followup_robot_cmd)
                        last_commanded_body_z = body_z
                        if i % 10 == 0:
                            print(f"  Height correction after SE2: offset={height_offset_followup:.3f} m (error was {abs(actual_body_z_check - body_z)*1000:.2f} mm)")
                except:
                    pass
            
            # Verify position accuracy for PERFECT replay
            # This reads actual robot state and compares to commanded
            # More frequent checking for better accuracy validation, especially body height
            if i % 10 == 0:  # Check more frequently for better accuracy
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
                    
                    # Verify body height - critical for leg matching
                    body_height_error = None
                    if has_body_pose and body_z is not None:
                        odom_tform_body = get_odom_tform_body(robot_state.kinematic_state.transforms_snapshot)
                        actual_body_z = odom_tform_body.position.z
                        body_height_error = abs(actual_body_z - body_z)
                    
                    if max_error > 0.03:  # Warn if error > 0.03 rad (~1.7 degrees)
                        print(f"  WARNING: Arm position error at timestep {timestep}, max_error={max_error:.4f} rad (cmd_completed={command_completed})")
                    
                    if body_height_error is not None and body_height_error > 0.003:  # Warn if height error > 3mm
                        print(f"  WARNING: Body height error at timestep {timestep}, error={body_height_error*1000:.2f} mm (target={body_z:.3f}, actual={actual_body_z:.3f})")
                        
                        # If height error is significant, adjust it in next command
                        if body_height_error > 0.005:  # If error > 5mm, force correction
                            # Will be corrected in next iteration by height command logic
                            if i % 20 == 0:  # Print every 20 timesteps when correcting
                                print(f"  Will correct body height in next command...")
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
            
            # Adjust timing: account for command execution time
            # If command completed, we've already waited, so minimal additional sleep
            elapsed = time.time() - loop_start_time
            if command_completed:
                # Command already completed - just maintain minimum timing
                sleep_time = max(0, dt - elapsed)
            else:
                # Command didn't complete in time - give it more time next iteration
                sleep_time = max(0, dt - elapsed - 0.05)
            
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
    
    # CHANGE THIS FILE TO REPLAY THE MOTION!
    # YOUR FILE SHOULD BE IN THE 'teleoperation_data' FOLDER!
    replay_body_arm_data(robot, "teleoperation_data/body_arm_20251118_194055.txt")


if __name__ == "__main__":
    main()

