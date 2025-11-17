import argparse
import os
import time
from datetime import datetime

from bosdyn.api import arm_command_pb2, geometry_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.util import authenticate
from google.protobuf import wrappers_pb2

from spot_utils.utils import get_robot_state, verify_estop

def replay_all_joint_data(robot, filename, rate_hz=50.0, window_size=3):
    print(f"\nReading data from: {filename}")
    
    positions_data = []
    all_joint_names = None
    arm_joint_names = ["arm0.sh0", "arm0.sh1", "arm0.el0", "arm0.el1", "arm0.wr0", "arm0.wr1"]
    leg_joint_prefixes = ["fl.", "fr.", "hl.", "hr."]
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                if line.startswith('# Format:') and all_joint_names is None:
                    parts = line.strip().split(': ')[1].split(', ')
                    all_joint_names = parts[1:-1]
                    has_body_pose = len(parts) > len(all_joint_names) + 2
                continue
            
            parts = line.strip().split(',')
            timestep = int(parts[0])
            num_joints = len(all_joint_names) if all_joint_names is not None else 0
            if num_joints == 0:
                all_joint_positions = [float(p) for p in parts[1:-1]]
                num_joints = len(all_joint_positions)
            
            all_joint_positions = [float(p) for p in parts[1:1+num_joints]]
            gripper_value = float(parts[1+num_joints]) if len(parts) > 1+num_joints else None
            
            body_x = float(parts[2+num_joints]) if len(parts) > 2+num_joints else None
            body_y = float(parts[3+num_joints]) if len(parts) > 3+num_joints else None
            body_z = float(parts[4+num_joints]) if len(parts) > 4+num_joints else None
            body_qw = float(parts[5+num_joints]) if len(parts) > 5+num_joints else None
            body_qx = float(parts[6+num_joints]) if len(parts) > 6+num_joints else None
            body_qy = float(parts[7+num_joints]) if len(parts) > 7+num_joints else None
            body_qz = float(parts[8+num_joints]) if len(parts) > 8+num_joints else None
            
            if all_joint_names is None:
                all_joint_names = [f"joint_{i}" for i in range(len(all_joint_positions))]
            
            joint_dict = {all_joint_names[i]: all_joint_positions[i] for i in range(len(all_joint_names))}
            arm_positions = [joint_dict[joint_name] for joint_name in arm_joint_names if joint_name in joint_dict]
            leg_positions = {name: joint_dict[name] for name in all_joint_names if any(name.startswith(prefix) for prefix in leg_joint_prefixes)}
            
            body_pose = None
            if body_x is not None and body_z is not None and body_qw is not None:
                body_pose = math_helpers.SE3Pose(
                    x=body_x, y=body_y, z=body_z,
                    rot=math_helpers.Quat(w=body_qw, x=body_qx, y=body_qy, z=body_qz)
                )
            
            if len(arm_positions) == len(arm_joint_names):
                positions_data.append((timestep, arm_positions, gripper_value, leg_positions, body_pose))
    
    has_gripper_data = positions_data[0][2] is not None
    has_leg_data = len(positions_data[0][3]) > 0 if len(positions_data) > 0 else False
    has_body_pose = positions_data[0][4] is not None if len(positions_data) > 0 else False
    print(f"Loaded {len(positions_data)} timesteps")
    print(f"Gripper data: {'Yes' if has_gripper_data else 'No'}")
    print(f"Leg joint data: {'Yes' if has_leg_data else 'No'}")
    print(f"Body pose data: {'Yes' if has_body_pose else 'No'}")
    if has_leg_data:
        leg_joint_names = sorted(positions_data[0][3].keys())
        print(f"  Leg joints found: {', '.join(leg_joint_names)}")
    if has_body_pose:
        start_body = positions_data[0][4]
        print(f"  Body pose (height control): x={start_body.x:.3f}, y={start_body.y:.3f}, z={start_body.z:.3f} m")
    print(f"Replaying at {rate_hz} Hz with {window_size}-point trajectory window")
    print("Press Ctrl+C to stop.\n")
    
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    
    print("Moving to start position...")
    start_positions = positions_data[0][1]
    start_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
        start_positions[0], start_positions[1], start_positions[2],
        start_positions[3], start_positions[4], start_positions[5],
        time_since_reference_secs=2.0
    )
    start_traj = arm_command_pb2.ArmJointTrajectory(points=[start_point])
    start_move = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=start_traj)
    start_arm_cmd = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=start_move)
    start_sync = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=start_arm_cmd)
    start_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=start_sync)
    command_client.robot_command(start_robot_cmd)
    time.sleep(2.2)
    print("Starting replay...\n")
    
    dt = 1.0 / rate_hz
    last_gripper_value = None
    
    try:
        i = 0
        while i < len(positions_data):
            start_time = time.time()
            
            trajectory_points = []
            for j in range(min(window_size, len(positions_data) - i)):
                timestep, positions, gripper, leg_positions, body_pose = positions_data[i + j]
                
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
            
            max_vel = wrappers_pb2.DoubleValue(value=5.0)
            max_acc = wrappers_pb2.DoubleValue(value=10.0)
            
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
            
            timestep, positions, current_gripper, leg_positions, body_pose = positions_data[i]
            gripper_command = None
            mobility_command = None
            
            if has_gripper_data and current_gripper is not None:
                if last_gripper_value is None or abs(current_gripper - last_gripper_value) > 0.05:
                    gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
                        current_gripper
                    )
                    gripper_command = gripper_cmd.synchronized_command.gripper_command
                    last_gripper_value = current_gripper
            
            if has_body_pose and body_pose is not None:
                body_pose_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                    goal_x=body_pose.x,
                    goal_y=body_pose.y,
                    goal_heading=body_pose.rot.to_yaw(),
                    frame_name=ODOM_FRAME_NAME
                )
                mobility_command = body_pose_cmd.synchronized_command.mobility_command
            
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
                timestep, positions, gripper_val, leg_positions, body_pose = positions_data[i]
                gripper_str = f", gripper={gripper_val:.4f}" if gripper_val is not None else ""
                print(f"Timestep {timestep}: sh0={positions[0]:.4f}, sh1={positions[1]:.4f}, el0={positions[2]:.4f}{gripper_str}")
                if has_body_pose and body_pose is not None:
                    print(f"  Body height (z): {body_pose.z:.3f} m")
                if has_leg_data and leg_positions:
                    leg_str = ", ".join([f"{name}={leg_positions[name]:.3f}" for name in sorted(leg_positions.keys())[:4]])
                    print(f"  Leg joints: {leg_str}...")
            
            i += 1
            
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nReplay stopped.")
    
    print(f"\nReplay complete! Executed {len(positions_data)} timesteps.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    args = parser.parse_args()
    
    sdk = create_standard_sdk("SpotAllJointReplay")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()
    

    # CHANGE THIS FILE TO REPLAY THE MOTION!
    # YOUR FILE SHOULD BE IN THE 'teleoperation_data' FOLDER!
    replay_all_joint_data(robot, "teleoperation_data/all_joints_20251117_175629.txt")


if __name__ == "__main__":
    main()

