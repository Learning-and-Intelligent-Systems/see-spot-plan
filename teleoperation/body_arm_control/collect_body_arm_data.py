import argparse
import os
import time
from datetime import datetime

from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, BODY_FRAME_NAME, get_odom_tform_body
from bosdyn.client.util import authenticate

from spot_utils.utils import get_robot_state, verify_estop


def collect_body_arm_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    args = parser.parse_args()
    
    sdk = create_standard_sdk("SpotBodyArmCollect")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()
    
    arm_joint_names = ["arm0.sh0", "arm0.sh1", "arm0.el0", "arm0.el1", "arm0.wr0", "arm0.wr1"]
    
    rate_hz = 50.0
    dt = 1.0 / rate_hz
    
    os.makedirs("teleoperation_data", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"teleoperation_data/body_arm_{timestamp}.txt"
    
    print(f"Collecting arm joints + body pose (x, y, z, theta) at {rate_hz} Hz. Press Ctrl+C to stop.")
    print(f"Saving data to: {filename}\n")
    
    with open(filename, "w") as f:
        f.write(f"# Arm joint positions + body pose collected at {rate_hz} Hz\n")
        f.write(f"# Timestamp: {timestamp}\n")
        f.write(f"# Format: timestep, timestamp_utc, {', '.join(arm_joint_names)}, gripper_open_percentage, body_x, body_y, body_z, body_theta\n")
        f.write(f"# Body pose is in odom frame (x, y, z in meters, theta in radians)\n\n")
        
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
                body_theta = body_rot.to_yaw()
                
                positions = []
                print(f"[Timestep {timestep}]")
                for joint_name in arm_joint_names:
                    if joint_name in joint_dict:
                        position = joint_dict[joint_name].position.value
                        positions.append(position)
                        print(f"  {joint_name}: {position:.4f} rad")
                
                gripper_raw = robot_state.manipulator_state.gripper_open_percentage
                gripper_normalized = gripper_raw / 100.0
                gripper_normalized = max(0.0, min(1.0, gripper_normalized))
                if gripper_normalized < 0.02:
                    gripper_normalized = 0.0
                
                gripper_status = "OPEN" if gripper_normalized > 0.8 else ("CLOSING" if gripper_normalized > 0.2 else "CLOSED")
                print(f"  gripper: {gripper_normalized:.4f} [{gripper_status}] (raw={gripper_raw:.1f})")
                print(f"  body: x={body_pos.x:.4f} m, y={body_pos.y:.4f} m, z={body_pos.z:.4f} m, theta={body_theta:.4f} rad")
                print()
                
                timestamp_utc = time.time()
                f.write(f"{timestep}, {timestamp_utc}, {', '.join(f'{p}' for p in positions)}, {gripper_normalized}, {body_pos.x}, {body_pos.y}, {body_pos.z}, {body_theta}\n")
                f.flush()
                
                timestep += 1
                
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print(f"\nStopped. Data saved to {filename}")


def main():
    collect_body_arm_data()


if __name__ == "__main__":
    main()

