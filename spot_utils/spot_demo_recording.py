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
    robot.time_sync.wait_for_sync()
    image_client = robot.ensure_client(ImageClient.default_service_name)
    camera_sources = [
        "frontleft_fisheye_image",
        "frontright_fisheye_image",
        "left_fisheye_image",
        "right_fisheye_image",
        "back_fisheye_image",
    ]

    # Create a directory to save the demonstration data.
    demo_folder_name = args.demo_folder_name
    os.makedirs("demonstrations/" + demo_folder_name, exist_ok=False)

    print("Starting data collection! Press Ctrl+C to stop.")
    try:
        timestep = 0
        while True:
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
            # Save the robot state to a pickle file
            with open(
                f"demonstrations/{demo_folder_name}/{timestep}/robot_state.pkl", "wb"
            ) as state_file:
                pkl.dump(arm_joint_state_list, state_file)

            print(f"Saving data for timestep {timestep}: {arm_joint_state_list}")

            timestep += 1
            time.sleep(DATA_COLLECTION_INTERVAL)

    except KeyboardInterrupt:
        print("Stopping data collection.")


if __name__ == "__main__":
    main()  # type: ignore
