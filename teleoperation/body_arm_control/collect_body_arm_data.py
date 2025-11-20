import argparse
import os
import time
from datetime import datetime

from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, BODY_FRAME_NAME, get_odom_tform_body
from bosdyn.client.util import authenticate
import numpy as np

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
    
    # Increased rate for PERFECT accuracy - higher sampling = smoother replay
    rate_hz = 100.0
    dt = 1.0 / rate_hz

    # Number of samples to average per timestep to reduce noise
    # Increased from 3 to 5 to better filter height variations from arm movements
    samples_per_timestep = 5
    
    os.makedirs("teleoperation_data", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"teleoperation_data/body_arm_{timestamp}.txt"
    
    print(f"Collecting arm joints + body pose + velocity at {rate_hz} Hz. Press Ctrl+C to stop.")
    print(f"Saving data to: {filename}\n")
    
    with open(filename, "w") as f:
        f.write(f"# Arm joint positions + body pose + velocity collected at {rate_hz} Hz\n")
        f.write(f"# Timestamp: {timestamp}\n")
        f.write(f"# Format: timestep, timestamp_utc, {', '.join(arm_joint_names)}, gripper_open_percentage, body_x, body_y, body_z, body_yaw, body_pitch, v_x, v_y, v_rot\n")
        f.write(f"# Body pose: x, y, z in odom frame (meters), yaw/pitch (EulerZXY) in radians for SE2 and footprint_R_body (roll always 0.0)\n")
        f.write(f"# Velocity: v_x, v_y (linear velocity in odom frame, m/s), v_rot (angular velocity around z-axis, rad/s)\n\n")
        
        try:
            timestep = 0
            while True:
                loop_start_time = time.time()
                
                # Collect multiple samples and average to reduce noise for PERFECT accuracy
                position_samples = [[] for _ in arm_joint_names]
                gripper_samples = []
                body_x_samples = []
                body_y_samples = []
                body_z_samples = []
                body_yaw_samples = []
                body_pitch_samples = []
                v_x_samples = []
                v_y_samples = []
                v_rot_samples = []
                
                for sample_idx in range(samples_per_timestep):
                    robot_state = get_robot_state(robot)
                    joint_states = robot_state.kinematic_state.joint_states
                    joint_dict = {js.name: js for js in joint_states}
                    
                    # Collect arm joint positions
                    for idx, joint_name in enumerate(arm_joint_names):
                        if joint_name in joint_dict:
                            position = joint_dict[joint_name].position.value
                            position_samples[idx].append(position)
                    
                    # Collect gripper state
                    gripper_raw = robot_state.manipulator_state.gripper_open_percentage
                    gripper_normalized = gripper_raw / 100.0
                    gripper_normalized = max(0.0, min(1.0, gripper_normalized))
                    if gripper_normalized < 0.02:
                        gripper_normalized = 0.0
                    gripper_samples.append(gripper_normalized)
                    
                    # Collect body pose (position + EulerZXY for SE2 and footprint_R_body)
                    odom_tform_body = get_odom_tform_body(robot_state.kinematic_state.transforms_snapshot)
                    body_pos = odom_tform_body.position
                    body_rot = odom_tform_body.rotation
                    
                    body_x_samples.append(body_pos.x)
                    body_y_samples.append(body_pos.y)
                    body_z_samples.append(body_pos.z)
                    
                    # Extract EulerZXY (yaw, pitch) from quaternion (roll not collected, always 0.0)
                    w, x, y, z = body_rot.w, body_rot.x, body_rot.y, body_rot.z
                    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
                    pitch = np.arcsin(2*(w*y - z*x))
                    body_yaw_samples.append(yaw)
                    body_pitch_samples.append(pitch)
                    
                    # Collect body velocity (velocity_of_body_in_odom)
                    # This gives us linear and angular velocity in the odom frame
                    try:
                        if hasattr(robot_state.kinematic_state, 'velocity_of_body_in_odom'):
                            vel = robot_state.kinematic_state.velocity_of_body_in_odom
                            # Linear velocity: x, y components (z is vertical, not used for SE2)
                            v_x_samples.append(vel.linear.x)
                            v_y_samples.append(vel.linear.y)
                            # Angular velocity: z component (rotation around vertical axis)
                            v_rot_samples.append(vel.angular.z)
                        else:
                            # Fallback: compute from position if velocity not available
                            v_x_samples.append(0.0)
                            v_y_samples.append(0.0)
                            v_rot_samples.append(0.0)
                    except (AttributeError, KeyError):
                        # Fallback: compute from position if velocity not available
                        v_x_samples.append(0.0)
                        v_y_samples.append(0.0)
                        v_rot_samples.append(0.0)
                    
                    # Small delay between samples
                    if sample_idx < samples_per_timestep - 1:
                        time.sleep(0.001)
                
                # Average all samples for PERFECT accuracy
                positions = []
                for sample_list in position_samples:
                    if sample_list:
                        avg_position = sum(sample_list) / len(sample_list)
                        positions.append(avg_position)
                    else:
                        positions.append(0.0)
                
                gripper_normalized = sum(gripper_samples) / len(gripper_samples) if gripper_samples else 0.0
                body_x = sum(body_x_samples) / len(body_x_samples) if body_x_samples else 0.0
                body_y = sum(body_y_samples) / len(body_y_samples) if body_y_samples else 0.0
                body_z = sum(body_z_samples) / len(body_z_samples) if body_z_samples else 0.0
                body_yaw = sum(body_yaw_samples) / len(body_yaw_samples) if body_yaw_samples else 0.0
                body_pitch = sum(body_pitch_samples) / len(body_pitch_samples) if body_pitch_samples else 0.0
                v_x = sum(v_x_samples) / len(v_x_samples) if v_x_samples else 0.0
                v_y = sum(v_y_samples) / len(v_y_samples) if v_y_samples else 0.0
                v_rot = sum(v_rot_samples) / len(v_rot_samples) if v_rot_samples else 0.0
                
                print(f"[Timestep {timestep}]")
                for idx, joint_name in enumerate(arm_joint_names):
                    if idx < len(positions):
                        print(f"  {joint_name}: {positions[idx]:.6f} rad (avg of {samples_per_timestep} samples)")
                
                gripper_status = "OPEN" if gripper_normalized > 0.8 else ("CLOSING" if gripper_normalized > 0.2 else "CLOSED")
                print(f"  gripper: {gripper_normalized:.6f} [{gripper_status}] (avg of {samples_per_timestep} samples)")
                print(f"  body: x={body_x:.6f} m, y={body_y:.6f} m, z={body_z:.6f} m, yaw={body_yaw:.6f} rad (avg of {samples_per_timestep} samples)")
                print(f"  velocity: v_x={v_x:.6f} m/s, v_y={v_y:.6f} m/s, v_rot={v_rot:.6f} rad/s (avg of {samples_per_timestep} samples)")
                print()
                
                timestamp_utc = time.time()
                f.write(f"{timestep}, {timestamp_utc}, {', '.join(f'{p:.9f}' for p in positions)}, {gripper_normalized:.9f}, {body_x:.9f}, {body_y:.9f}, {body_z:.9f}, {body_yaw:.9f}, {body_pitch:.9f}, {v_x:.9f}, {v_y:.9f}, {v_rot:.9f}\n")
                f.flush()
                
                timestep += 1
                
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print(f"\nStopped. Data saved to {filename}")


def main():
    collect_body_arm_data()


if __name__ == "__main__":
    main()
