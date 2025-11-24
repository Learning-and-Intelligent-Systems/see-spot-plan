'''
Spot data collection script that is compatible with the ACT architecture

- 6 joint angles + 1 gripper open percentage + 3 body (body_x, body_y, body_z) + 3 body orientation (body_yaw, body_pitch, body_roll) + 3 body velocity (v_x, v_y, v_rot)
- 2 cameras with UTC timestamps (video will get streamed)

'''

import argparse
import os
import time
import h5py
from datetime import datetime

from bosdyn.client import create_standard_sdk
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.util import authenticate
import numpy as np

from spot_utils.utils import get_robot_state, verify_estop


def collect_body_arm_data_formatted():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    args = parser.parse_args()

    sdk = create_standard_sdk("SpotBodyArmCollect")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()

    arm_joint_names = ["arm0.sh0", "arm0.sh1", "arm0.el0", "arm0.el1", "arm0.wr0", "arm0.wr1"]
    num_arm_joints = len(arm_joint_names)

    # Collection parameters - match original collect_body_arm_data.py
    rate_hz = 100.0
    dt = 1.0 / rate_hz

    os.makedirs("teleoperation_data", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"teleoperation_data/body_arm_{timestamp}"
    dataset_path = f"{dataset_name}.hdf5"

    print(f"Collecting arm joints + body pose + velocity at {rate_hz} Hz")
    print(f"Saving data to: {dataset_path}\n")

    # Prepare HDF5 file with the following structure:
    # observations:
    #   - qpos: [timesteps, 16] (6 arm joints + 1 gripper + 6 body pose + 3 body vel)
    #   - body_pose: [timesteps, 6] (x, y, z, yaw, pitch, roll)
    #   - body_vel: [timesteps, 3] (v_x, v_y, v_rot)
    # action: [timesteps, 16] (6 arm joints + 1 gripper + 6 body pose + 3 body vel)

    # Initialize data storage
    qpos_data = []  # arm joints + gripper
    body_pose_data = []  # x, y, z, yaw, pitch, roll
    body_vel_data = []  # v_x, v_y, v_rot in body frame
    action_data = []  # same as qpos for now (observations become actions in ACT)
    timestamps = []  # actual UTC timestamps for accurate replay timing

    try:
        timestep = 0
        prev_body_x = None
        prev_body_y = None
        prev_body_yaw = None
        prev_timestamp = None

        while True:
            loop_start_time = time.time()
            current_timestamp = time.time()

            robot_state = get_robot_state(robot)
            joint_states = robot_state.kinematic_state.joint_states
            joint_dict = {js.name: js for js in joint_states}

            # Collect arm joint positions
            positions = []
            for joint_name in arm_joint_names:
                if joint_name in joint_dict:
                    position = joint_dict[joint_name].position.value
                    positions.append(position)
                else:
                    positions.append(0.0)

            # Collect gripper state
            gripper_raw = robot_state.manipulator_state.gripper_open_percentage
            gripper_normalized = gripper_raw / 100.0
            gripper_normalized = max(0.0, min(1.0, gripper_normalized))
            if gripper_normalized < 0.02:
                gripper_normalized = 0.0

            # Collect body pose
            odom_tform_body = get_odom_tform_body(robot_state.kinematic_state.transforms_snapshot)
            body_pos = odom_tform_body.position
            body_rot = odom_tform_body.rotation

            body_x = body_pos.x
            body_y = body_pos.y
            body_z = body_pos.z

            # Extract Euler angles from quaternion
            w, x, y, z = body_rot.w, body_rot.x, body_rot.y, body_rot.z
            body_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            body_pitch = np.arcsin(2*(w*y - z*x))
            body_roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))

            # Collect body velocity
            v_x = 0.0
            v_y = 0.0
            v_rot = 0.0

            try:
                if hasattr(robot_state.kinematic_state, 'velocity_of_body_in_odom'):
                    vel = robot_state.kinematic_state.velocity_of_body_in_odom
                    if vel is not None:
                        if hasattr(vel, 'linear') and vel.linear is not None:
                            v_x = vel.linear.x if hasattr(vel.linear, 'x') else 0.0
                            v_y = vel.linear.y if hasattr(vel.linear, 'y') else 0.0
                        if hasattr(vel, 'angular') and vel.angular is not None:
                            v_rot = vel.angular.z if hasattr(vel.angular, 'z') else 0.0
            except (AttributeError, KeyError, TypeError):
                pass

            # Transform velocity from odom to body frame
            if v_x != 0.0 or v_y != 0.0:
                cos_yaw = np.cos(body_yaw)
                sin_yaw = np.sin(body_yaw)
                v_x_body = v_x * cos_yaw + v_y * sin_yaw
                v_y_body = -v_x * sin_yaw + v_y * cos_yaw
            else:
                v_x_body = 0.0
                v_y_body = 0.0

            # Fallback: compute velocity from position differences if not available
            velocity_available = abs(v_x_body) > 1e-6 or abs(v_y_body) > 1e-6 or abs(v_rot) > 1e-6
            if not velocity_available and prev_body_x is not None and prev_timestamp is not None:
                dt_vel = current_timestamp - prev_timestamp
                if dt_vel > 1e-6:
                    v_x_odom = (body_x - prev_body_x) / dt_vel
                    v_y_odom = (body_y - prev_body_y) / dt_vel
                    cos_yaw = np.cos(body_yaw)
                    sin_yaw = np.sin(body_yaw)
                    v_x_body = v_x_odom * cos_yaw + v_y_odom * sin_yaw
                    v_y_body = -v_x_odom * sin_yaw + v_y_odom * cos_yaw
                    yaw_diff = body_yaw - prev_body_yaw
                    while yaw_diff > np.pi:
                        yaw_diff -= 2 * np.pi
                    while yaw_diff < -np.pi:
                        yaw_diff += 2 * np.pi
                    v_rot = yaw_diff / dt_vel

            # Store data: qpos includes arm joints + gripper + body pose + body velocity
            qpos = positions + [gripper_normalized] + [body_x, body_y, body_z, body_yaw, body_pitch, body_roll] + [v_x_body, v_y_body, v_rot]
            qpos_data.append(qpos)
            body_pose_data.append([body_x, body_y, body_z, body_yaw, body_pitch, body_roll])
            body_vel_data.append([v_x_body, v_y_body, v_rot])
            action_data.append(qpos)  # Action includes all DOF: arm joints + gripper + body pose + body velocity
            timestamps.append(current_timestamp)  # Store actual UTC timestamp

            # Update previous values
            prev_body_x = body_x
            prev_body_y = body_y
            prev_body_yaw = body_yaw
            prev_timestamp = current_timestamp

            # Print progress
            print(f"[Timestep {timestep}]")
            for idx, joint_name in enumerate(arm_joint_names):
                print(f"  {joint_name}: {positions[idx]:.6f} rad")
            gripper_status = "OPEN" if gripper_normalized > 0.8 else ("CLOSING" if gripper_normalized > 0.2 else "CLOSED")
            print(f"  gripper: {gripper_normalized:.6f} [{gripper_status}]")
            print(f"  body: x={body_x:.6f} m, y={body_y:.6f} m, z={body_z:.6f} m, yaw={body_yaw:.6f} rad")
            velocity_magnitude = np.sqrt(v_x_body**2 + v_y_body**2)
            print(f"  velocity: v_x={v_x_body:.6f} m/s, v_y={v_y_body:.6f} m/s, v_rot={v_rot:.6f} rad/s (mag={velocity_magnitude:.6f} m/s)")
            print()

            timestep += 1

            elapsed = time.time() - loop_start_time
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\nStopped. Collected {len(qpos_data)} timesteps.")

        # Convert to numpy arrays
        qpos_array = np.array(qpos_data, dtype=np.float64)
        body_pose_array = np.array(body_pose_data, dtype=np.float64)
        body_vel_array = np.array(body_vel_data, dtype=np.float64)
        action_array = np.array(action_data, dtype=np.float64)
        timestamps_array = np.array(timestamps, dtype=np.float64)

        # Write to HDF5 file
        print(f"\nSaving {len(qpos_data)} timesteps to {dataset_path}...")
        t0 = time.time()
        with h5py.File(dataset_path, 'w') as root:
            root.attrs['sim'] = False
            root.attrs['rate_hz'] = rate_hz
            root.attrs['arm_joint_names'] = arm_joint_names
            root.attrs['timestamp'] = timestamp

            obs = root.create_group('observations')
            obs.create_dataset('qpos', data=qpos_array, dtype='float64')
            obs.create_dataset('body_pose', data=body_pose_array, dtype='float64')
            obs.create_dataset('body_vel', data=body_vel_array, dtype='float64')
            obs.create_dataset('timestamps', data=timestamps_array, dtype='float64')

            root.create_dataset('action', data=action_array, dtype='float64')

        print(f'Saving completed in {time.time() - t0:.1f} seconds')
        print(f'HDF5 file created: {dataset_path}')
        print(f'Dataset structure:')
        print(f'  observations/qpos: {qpos_array.shape}  (arm_joints[6] + gripper[1] + body_pose[6] + body_vel[3])')
        print(f'  observations/body_pose: {body_pose_array.shape}  (x, y, z, yaw, pitch, roll)')
        print(f'  observations/body_vel: {body_vel_array.shape}  (v_x, v_y, v_rot)')
        print(f'  action: {action_array.shape}  (arm_joints[6] + gripper[1] + body_pose[6] + body_vel[3])')


def main():
    collect_body_arm_data_formatted()


if __name__ == "__main__":
    main()
