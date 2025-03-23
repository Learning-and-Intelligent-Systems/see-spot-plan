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
from rich import print
from spot_utils.utils import get_robot_state, verify_estop

DATA_COLLECTION_INTERVAL = 1.0 / 4.0  # 10 Hz


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
            # Ensure we're capturing both position and velocity data
            arm_joint_state_list = []
            for joint_state in arm_joint_state:
                joint_data = {
                    "name": joint_state.name,
                    "position": joint_state.position.value,
                }

                # Make sure velocity exists before accessing it
                if joint_state.velocity is not None and hasattr(
                    joint_state.velocity, "value"
                ):
                    joint_data["velocity"] = joint_state.velocity.value
                else:
                    joint_data["velocity"] = 0.0  # Default to zero if no velocity data

                arm_joint_state_list.append(joint_data)

            # Get end-effector pose
            end_effector_pose = robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map[
                "hand"
            ].parent_tform_child

            # Get gripper open/close value and force information
            gripper_state = robot_state.manipulator_state.gripper_open_percentage

            # Get gripper force information - look for various potential sources of force data
            gripper_force = None
            gripper_holding = False

            # Check if we're currently holding something (gripper is closed and applying force)
            if gripper_state < 0.2:  # Less than 20% open means mostly closed
                gripper_holding = True

            # Try to get estimated end effector force if available
            if hasattr(
                robot_state.manipulator_state, "estimated_end_effector_force_in_hand"
            ):
                force_in_hand = (
                    robot_state.manipulator_state.estimated_end_effector_force_in_hand
                )
                gripper_force = {
                    "x": force_in_hand.x,
                    "y": force_in_hand.y,
                    "z": force_in_hand.z,
                    "magnitude": (
                        force_in_hand.x**2 + force_in_hand.y**2 + force_in_hand.z**2
                    )
                    ** 0.5,
                }
                print(f"Recorded gripper force: {gripper_force['magnitude']:.2f} N")

            # Record whether the gripper might be holding an object
            gripper_data = {
                "percentage": gripper_state,
                "force": gripper_force,
                "holding": gripper_holding,
            }

            # Add gripper information to the robot data
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
                "gripper_data": gripper_data,
                "gripper_open_percentage": gripper_state,
                "gripper_force": gripper_force,
                "gripper_holding": gripper_holding,
                "timestamp": relative_timestamp,
            }

            # Save the robot state to a pickle file
            with open(
                f"demonstrations/{demo_folder_name}/{timestep}/robot_state.pkl", "wb"
            ) as state_file:
                pkl.dump(robot_data, state_file)

            # Print information about the capture, including velocity data
            arm_joints = {joint["name"]: joint for joint in arm_joint_state_list}
            velocity_info = ""
            for joint_name in [
                "arm0.sh0",
                "arm0.sh1",
                "arm0.el0",
                "arm0.el1",
                "arm0.wr0",
                "arm0.wr1",
            ]:
                if joint_name in arm_joints:
                    velocity_info += (
                        f"{joint_name}: {arm_joints[joint_name]['velocity']:.3f} "
                    )

            print(f"Saving data for timestep {timestep} at {relative_timestamp:.2f}s")
            print(f"Velocities: {velocity_info}")

            timestep += 1
            time.sleep(DATA_COLLECTION_INTERVAL)

    except KeyboardInterrupt:
        print("Stopping data collection.")


if __name__ == "__main__":
    main()  # type: ignore
