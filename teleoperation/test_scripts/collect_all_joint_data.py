import argparse
import os
import time
from datetime import datetime

from bosdyn.api import arm_command_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.client import create_standard_sdk
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, BODY_FRAME_NAME, get_odom_tform_body
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.util import authenticate
from google.protobuf import wrappers_pb2

from spot_utils.utils import get_robot_state, verify_estop


def get_all_joint_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    args = parser.parse_args()
    
    sdk = create_standard_sdk("SpotAllJointMonitor")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()
    
    arm_joint_names = ["arm0.sh0", "arm0.sh1", "arm0.el0", "arm0.el1", "arm0.wr0", "arm0.wr1"]
    leg_joint_prefixes = ["fl.", "fr.", "hl.", "hr."]
    
    rate_hz = 50.0
    dt = 1.0 / rate_hz
    
    os.makedirs("teleoperation_data", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"teleoperation_data/all_joints_{timestamp}.txt"
    
    print(f"Reading all joint positions (arm + legs) at {rate_hz} Hz. Press Ctrl+C to stop.")
    print(f"Saving data to: {filename}\n")
    
    all_joint_names = None
    
    with open(filename, "w") as f:
        try:
            timestep = 0
            while True:
                start_time = time.time()
                
                robot_state = get_robot_state(robot)
                joint_states = robot_state.kinematic_state.joint_states
                joint_dict = {js.name: js for js in joint_states}
                
                odom_tform_body = get_odom_tform_body(robot_state.kinematic_state.transforms_snapshot)
                body_pos = odom_tform_body.position
                body_rot = odom_tform_body.rotation
                
                if all_joint_names is None:
                    all_joint_names = sorted(joint_dict.keys())
                    f.write(f"# All joint positions collected at {rate_hz} Hz\n")
                    f.write(f"# Timestamp: {timestamp}\n")
                    f.write(f"# Format: timestep, {', '.join(all_joint_names)}, gripper_open_percentage, body_x, body_y, body_z, body_qw, body_qx, body_qy, body_qz\n")
                    f.write(f"# Body pose is in odom frame (position x,y,z and quaternion qw,qx,qy,qz)\n\n")
                    print(f"Collecting {len(all_joint_names)} joints (arm + legs + base) + body pose")
                    arm_joints = [j for j in all_joint_names if j.startswith("arm0.")]
                    leg_joints = [j for j in all_joint_names if any(j.startswith(prefix) for prefix in leg_joint_prefixes)]
                    print(f"  Arm joints ({len(arm_joints)}): {', '.join(arm_joints)}")
                    print(f"  Leg joints ({len(leg_joints)}): {', '.join(leg_joints)}")
                    print()
                
                positions = []
                print(f"[Timestep {timestep}]")
                for joint_name in all_joint_names:
                    if joint_name in joint_dict:
                        position = joint_dict[joint_name].position.value
                        positions.append(position)
                        if joint_name in arm_joint_names:
                            print(f"  {joint_name}: {position:.4f} rad")
                        elif any(joint_name.startswith(prefix) for prefix in leg_joint_prefixes):
                            print(f"  {joint_name}: {position:.4f} rad")
                
                gripper_raw = robot_state.manipulator_state.gripper_open_percentage
                gripper_normalized = gripper_raw / 100.0
                gripper_normalized = max(0.0, min(1.0, gripper_normalized))
                if gripper_normalized < 0.02:
                    gripper_normalized = 0.0
                
                gripper_status = "OPEN" if gripper_normalized > 0.8 else ("CLOSING" if gripper_normalized > 0.2 else "CLOSED")
                print(f"  gripper: {gripper_normalized:.4f} [{gripper_status}] (raw={gripper_raw:.1f})")
                print(f"  body height (z): {body_pos.z:.4f} m")
                print()
                
                f.write(f"{timestep}, {', '.join(f'{p}' for p in positions)}, {gripper_normalized}, {body_pos.x}, {body_pos.y}, {body_pos.z}, {body_rot.w}, {body_rot.x}, {body_rot.y}, {body_rot.z}\n")
                f.flush()
                
                timestep += 1
                
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print(f"\nStopped. Data saved to {filename}")

def main():
    get_all_joint_data()

if __name__ == "__main__":
    main()

