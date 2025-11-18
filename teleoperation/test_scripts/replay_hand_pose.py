import argparse
import time
from datetime import datetime

from bosdyn.api import arm_command_pb2, geometry_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.client import create_standard_sdk
from bosdyn.client.frame_helpers import BODY_FRAME_NAME
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.util import authenticate

from spot_utils.utils import verify_estop


def replay_hand_pose_data(robot, filename, rate_hz=50.0):
    print(f"\nReading data from: {filename}")
    
    poses_data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split(',')
            utc = float(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            gripper = float(parts[8]) if len(parts) > 8 else None
            
            poses_data.append((utc, x, y, z, qw, qx, qy, qz, gripper))
    
    has_gripper_data = poses_data[0][8] is not None
    print(f"Loaded {len(poses_data)} poses")
    print(f"Gripper data: {'Yes' if has_gripper_data else 'No'}")
    print(f"Replaying at {rate_hz} Hz")
    print("Press Ctrl+C to stop.\n")
    
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    
    print("Moving to start position...")
    start_pose = poses_data[0]
    start_pos = geometry_pb2.Vec3(x=start_pose[1], y=start_pose[2], z=start_pose[3])
    start_rot = geometry_pb2.Quaternion(w=start_pose[4], x=start_pose[5], y=start_pose[6], z=start_pose[7])
    start_hand_pose = geometry_pb2.SE3Pose(position=start_pos, rotation=start_rot)
    start_gripper = start_pose[8] if has_gripper_data and start_pose[8] is not None else None
    
    start_arm_cmd = RobotCommandBuilder.arm_pose_command(
        start_hand_pose.position.x,
        start_hand_pose.position.y,
        start_hand_pose.position.z,
        start_hand_pose.rotation.w,
        start_hand_pose.rotation.x,
        start_hand_pose.rotation.y,
        start_hand_pose.rotation.z,
        BODY_FRAME_NAME,
        seconds=2.0
    )
    
    if start_gripper is not None:
        start_gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(start_gripper)
        sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=start_arm_cmd.synchronized_command.arm_command,
            gripper_command=start_gripper_cmd.synchronized_command.gripper_command
        )
        start_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_command)
    else:
        start_cmd = start_arm_cmd
    
    command_client.robot_command(start_cmd)
    time.sleep(2.2)
    print("Starting replay with frozen body...\n")
    print(f"Using original collection timestamps for timing (ignoring --rate parameter)")
    
    last_gripper_value = start_gripper if start_gripper is not None else None
    
    try:
        for i, pose_data in enumerate(poses_data):
            loop_start_time = time.time()
            
            utc, x, y, z, qw, qx, qy, qz, gripper = pose_data
            
            if i > 0:
                prev_utc = poses_data[i - 1][0]
                dt = utc - prev_utc
            else:
                dt = 0.02
            
            dt = max(0.005, min(0.1, dt))
            
            arm_cmd = RobotCommandBuilder.arm_pose_command(
                x, y, z, qw, qx, qy, qz,
                BODY_FRAME_NAME,
                seconds=dt * 1.5
            )
            
            arm_command = arm_cmd.synchronized_command.arm_command
            
            gripper_command = None
            if has_gripper_data and gripper is not None:
                if last_gripper_value is None or abs(gripper - last_gripper_value) > 0.01:
                    gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(gripper)
                    gripper_command = gripper_cmd.synchronized_command.gripper_command
                    last_gripper_value = gripper
            
            if gripper_command is not None:
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
            
            if i % 50 == 0:
                gripper_str = f", gripper={gripper:.4f}" if gripper is not None else ""
                print(f"Step {i}: x={x:.3f}, y={y:.3f}, z={z:.3f}{gripper_str} (dt={dt*1000:.1f}ms)")
            
            elapsed = time.time() - loop_start_time
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nReplay stopped.")
    
    print(f"\nReplay complete! Executed {len(poses_data)} poses.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--rate", type=float, default=50.0)
    args = parser.parse_args()
    
    sdk = create_standard_sdk("SpotHandPoseReplay")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()
    
    replay_hand_pose_data(robot, args.file, rate_hz=args.rate)


if __name__ == "__main__":
    main()

