import argparse
import h5py
import tempfile
import csv
import os

from bosdyn.client import create_standard_sdk
from bosdyn.client.util import authenticate

from spot_utils.utils import verify_estop
import sys
import os
# Add teleoperation directory to path to import replay_body_arm_data
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from testing.replay_body_arm_data import replay_body_arm_data


def formatted_replay_body_arm_data(robot, hdf5_filename, rate_hz=None):
    """
    Replay data collected from formatted_collect_body_arm_data.py (HDF5 format).

    Converts HDF5 data to the CSV format expected by replay_body_arm_data(),
    then calls that function to perform the actual replay.

    Data structure in HDF5:
    - observations/qpos: [timesteps, 16] (6 arm joints + 1 gripper + 6 body pose + 3 body vel)
    - observations/body_pose: [timesteps, 6] (x, y, z, yaw, pitch, roll)
    - observations/body_vel: [timesteps, 3] (v_x, v_y, v_rot)
    - action: [timesteps, 16] (same as qpos)
    """
    print(f"\nReading formatted HDF5 data from: {hdf5_filename}")

    # Load HDF5 file
    with h5py.File(hdf5_filename, 'r') as f:
        # Get metadata
        if rate_hz is None:
            rate_hz = f.attrs.get('rate_hz', 50.0)
        arm_joint_names = f.attrs.get('arm_joint_names',
            [b"arm0.sh0", b"arm0.sh1", b"arm0.el0", b"arm0.el1", b"arm0.wr0", b"arm0.wr1"])

        # Decode arm_joint_names if they're bytes
        # Convert to list if it's a numpy array, then check and decode
        if arm_joint_names is not None and len(arm_joint_names) > 0:
            arm_joint_names = list(arm_joint_names)  # Convert to list to avoid numpy array issues
            if isinstance(arm_joint_names[0], bytes):
                arm_joint_names = [name.decode('utf-8') for name in arm_joint_names]

        # Load observation data
        qpos_data = f['observations/qpos'][:]
        body_vel_data = f['observations/body_vel'][:]
        
        # Load actual timestamps if available (for accurate replay timing)
        # Fall back to calculated timestamps for backward compatibility
        if 'observations/timestamps' in f:
            timestamps_data = f['observations/timestamps'][:]
            use_actual_timestamps = True
            print(f"Using actual timestamps from collection for accurate replay timing")
        else:
            timestamps_data = None
            use_actual_timestamps = False
            print(f"Using calculated timestamps (timestep / rate_hz) - consider re-collecting with updated script for accurate timing")

    num_timesteps = len(qpos_data)
    print(f"Loaded {num_timesteps} timesteps at {rate_hz} Hz")
    print(f"Arm joints: {arm_joint_names}")

    # Convert to CSV format expected by replay_body_arm_data
    # Format: timestep, timestamp_utc, joint1, ..., joint6, gripper, body_x, body_y, body_z, body_yaw, body_pitch, v_x_body, v_y_body
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        csv_filename = f.name
        writer = csv.writer(f)

        for timestep in range(num_timesteps):
            qpos = qpos_data[timestep]
            body_vel = body_vel_data[timestep]

            # Extract components from qpos
            # qpos = [6 arm joints + 1 gripper + 6 body pose + 3 body vel]
            joint_positions = qpos[:6]
            gripper = qpos[6]
            body_pose = qpos[7:13]  # x, y, z, yaw, pitch, roll

            body_x = body_pose[0]
            body_y = body_pose[1]
            body_z = body_pose[2]
            body_yaw = body_pose[3]
            body_pitch = body_pose[4]  # Fixed: pitch is at index 4, not 5

            v_x_body = body_vel[0]
            v_y_body = body_vel[1]
            v_rot = body_vel[2]  # Include v_rot to match original format

            # Use actual timestamp if available, otherwise calculate from timestep
            if use_actual_timestamps:
                timestamp_utc = timestamps_data[timestep]
            else:
                timestamp_utc = timestep / rate_hz

            # Write CSV row: timestep, timestamp_utc, joint1-6, gripper, body_x, body_y, body_z, body_yaw, body_pitch, v_x, v_y, v_rot
            # Format matches original collect_body_arm_data.py
            row = [
                timestep,
                timestamp_utc,
                *joint_positions,
                gripper,
                body_x,
                body_y,
                body_z,
                body_yaw,
                body_pitch,
                v_x_body,
                v_y_body,
                v_rot
            ]
            writer.writerow(row)

    print(f"Converted to CSV format: {csv_filename}")

    try:
        # Call the existing replay function with the converted data
        replay_body_arm_data(robot, csv_filename, rate_hz=rate_hz)
    finally:
        # Clean up temporary file
        if os.path.exists(csv_filename):
            os.remove(csv_filename)
            print(f"Cleaned up temporary CSV file")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    parser.add_argument("--file", type=str, required=True,
                        help="Path to HDF5 file from formatted_collect_body_arm_data.py")
    args = parser.parse_args()

    sdk = create_standard_sdk("SpotFormattedBodyArmReplay")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()

    formatted_replay_body_arm_data(robot, args.file)


if __name__ == "__main__":
    main()
