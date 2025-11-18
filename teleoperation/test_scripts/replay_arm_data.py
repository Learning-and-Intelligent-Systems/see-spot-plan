import argparse
import os
import time
from datetime import datetime

from bosdyn.api import arm_command_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.client import create_standard_sdk
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.util import authenticate
from google.protobuf import wrappers_pb2

from spot_utils.utils import get_robot_state, verify_estop

def replay_arm_data(robot, filename, rate_hz=50.0, window_size=3):
    print(f"\nReading data from: {filename}")
    
    positions_data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split(',')
            timestep = int(parts[0])
            joint_positions = [float(p.strip()) for p in parts[1:7]]
            gripper_value = float(parts[7].strip()) if len(parts) > 7 else None
            positions_data.append((timestep, joint_positions, gripper_value))
    
    has_gripper_data = positions_data[0][2] is not None
    print(f"Loaded {len(positions_data)} timesteps")
    print(f"Gripper data: {'Yes' if has_gripper_data else 'No'}")
    print(f"Replaying at {rate_hz} Hz with {window_size}-point trajectory window")
    print("Press Ctrl+C to stop.\n")
    
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    
    print("Moving to start position...")
    start_positions = positions_data[0][1]
    start_gripper = positions_data[0][2] if has_gripper_data else None
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
    print("Starting replay...\n")
    
    dt = 1.0 / rate_hz
    last_gripper_value = start_gripper if start_gripper is not None else None
    
    try:
        i = 0
        while i < len(positions_data):
            start_time = time.time()
            
            trajectory_points = []
            for j in range(min(window_size, len(positions_data) - i)):
                timestep, positions, gripper = positions_data[i + j]
                
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
            
            current_gripper = positions_data[i][2]
            
            if has_gripper_data and current_gripper is not None:
                gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
                    current_gripper
                )
                gripper_command = gripper_cmd.synchronized_command.gripper_command
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
                positions = positions_data[i][1]
                gripper_val = positions_data[i][2]
                gripper_str = f", gripper={gripper_val:.4f}" if gripper_val is not None else ""
                print(f"Timestep {positions_data[i][0]}: sh0={positions[0]:.4f}, sh1={positions[1]:.4f}, el0={positions[2]:.4f}{gripper_str}")
            
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
    
    sdk = create_standard_sdk("SpotArmJointReplay")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()
    

    # CHANGE THIS FILE TO REPLAY THE MOTION!
    # YOUR FILE SHOULD BE IN THE 'teleoperation_data' FOLDER!
    replay_arm_data(robot, "teleoperation_data/arm_joints_20251117_165048.txt")


if __name__ == "__main__":
    main()
