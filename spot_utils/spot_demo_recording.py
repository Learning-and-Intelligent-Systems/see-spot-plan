"""Script to record demonstration data from the Spot. Run this script and then use the tablet to
teleop the robot for data collection.

Example usage:
python spot_utils/spot_demo_recording.py --hostname 192.168.80.3 --demo_folder_name test_data_recording0
"""

import argparse
import os
import time

import dill as pkl
from bosdyn.client.image import ImageClient
from bosdyn.client.time_sync import TimeSyncClient

from spot_utils.utils import get_robot_state, verify_estop

DATA_COLLECTION_INTERVAL = 1.0  # seconds


def main():
    """Record demonstration data from the Spot robot."""
    # pylint: disable=ungrouped-imports
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.util import authenticate

    # Argparse setup to get robot hostname
    parser = argparse.ArgumentParser(description="Parse the robot's hostname.")
    parser.add_argument(
        "--hostname",
        type=str,
        required=True,
        help="The robot's hostname/ip-address (e.g. 192.168.80.3)",
    )
    parser.add_argument(
        "--demo_folder_name",
        type=str,
        required=True,
        help="The name of the folder to save the demonstration data",
    )
    args = parser.parse_args()

    # Get constants.
    hostname = args.hostname
    sdk = create_standard_sdk("SpotDataLogger")
    robot = sdk.create_robot(hostname)
    authenticate(robot)
    verify_estop(robot)
    
    # Ensure time sync client is created
    robot.time_sync = robot.ensure_client(TimeSyncClient.default_service_name)
    robot.time_sync.wait_for_sync()
    
    image_client = robot.ensure_client(ImageClient.default_service_name)
    camera_sources = [
        "frontleft_fisheye_image",
        "frontright_fisheye_image",
        "left_fisheye_image",
        "right_fisheye_image",
        "back_fisheye_image",
        "hand_color_image",
    ]

    # Create a directory to save the demonstration data.
    demo_folder_name = args.demo_folder_name
    os.makedirs("demonstrations/" + demo_folder_name, exist_ok=False)

    print("Starting data collection! Press Ctrl+C to stop.")
    try:
        timestep = 0
        start_time = time.time()  # Record the start time
        
        while True:
            # Record the current timestamp relative to start
            current_time = time.time()
            relative_timestamp = current_time - start_time
            
            # Make a folder corresponding to the current timestep.
            os.makedirs(
                f"demonstrations/{demo_folder_name}/{timestep}",
                exist_ok=False,
            )
            # Capture images from all selected cameras
            image_responses = image_client.get_image_from_sources(camera_sources)
            for image_response in image_responses:
                if image_response.shot.image.data:
                    source_name = image_response.source.name
                    with open(
                        f"demonstrations/{demo_folder_name}/{timestep}/{source_name}.jpg",
                        "wb",
                    ) as img_file:
                        img_file.write(image_response.shot.image.data)

            # Get robot state (includes arm joint angles)
            robot_state = get_robot_state(robot)
            arm_joint_state = robot_state.kinematic_state.joint_states

            # Convert arm_joint_state to a list of dictionaries
            arm_joint_state_list = [
                {
                    "name": joint_state.name,
                    "position": joint_state.position.value,
                    "velocity": joint_state.velocity.value,
                }
                for joint_state in arm_joint_state
            ]

            # Get end-effector pose
            end_effector_pose = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map[
                "hand"
            ].parent_tform_child

            # Get gripper open/close value
            gripper_state = robot_state.manipulator_state.gripper_open_percentage

            # Add end-effector pose and gripper state to the dictionary
            robot_data = {
                "arm_joint_state": arm_joint_state_list,
                "end_effector_pose": {
                    "position": {
                        "x": end_effector_pose.position.x,
                        "y": end_effector_pose.position.y,
                        "z": end_effector_pose.position.z,
                    },
                    "rotation": {
                        "x": end_effector_pose.rotation.x,
                        "y": end_effector_pose.rotation.y,
                        "z": end_effector_pose.rotation.z,
                        "w": end_effector_pose.rotation.w,
                    },
                },
                "gripper_open_percentage": gripper_state,
                "timestamp": relative_timestamp,  # Add the relative timestamp
            }

            # Save the robot state to a pickle file
            with open(
                f"demonstrations/{demo_folder_name}/{timestep}/robot_state.pkl", "wb"
            ) as state_file:
                pkl.dump(robot_data, state_file)

            print(f"Saving data for timestep {timestep}: {robot_data}")

            timestep += 1
            time.sleep(DATA_COLLECTION_INTERVAL)

    except KeyboardInterrupt:
        print("Stopping data collection.")


if __name__ == "__main__":
    main()  # type: ignore
