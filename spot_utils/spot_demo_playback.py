"""Script to playback demonstration data on the Spot robot.

Example usage:
python spot_utils/spot_demo_playback.py --hostname 192.168.80.3 --demo_folder_name test_data_recording0
"""

import argparse
import time
import os

import dill as pkl
from bosdyn.api import arm_command_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.client import create_standard_sdk
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.util import authenticate


ARM_JOINT_NAMES = [
    "arm0.sh0",
    "arm0.sh1",
    "arm0.el0",
    "arm0.el1",
    "arm0.wr0",
    "arm0.wr1",
]


def make_robot_command(arm_joint_traj):
    """Helper function to create a RobotCommand from an ArmJointTrajectory.
    The returned command will be a SynchronizedCommand with an ArmJointMoveCommand
    filled out to follow the passed in trajectory.
    """
    joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(
        trajectory=arm_joint_traj
    )
    arm_command = arm_command_pb2.ArmCommand.Request(
        arm_joint_move_command=joint_move_command
    )
    sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(
        arm_command=arm_command
    )
    arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
    return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)


def main():
    """Playback demonstration data on the Spot robot."""
    # Argparse setup to get robot hostname and demo folder name
    parser = argparse.ArgumentParser(
        description="Parse the robot's hostname and demo folder name."
    )
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
        help="The name of the folder containing the demonstration data",
    )
    args = parser.parse_args()

    # Get constants.
    hostname = args.hostname
    demo_folder_name = args.demo_folder_name

    # Create SDK and robot objects
    sdk = create_standard_sdk("SpotDemoPlayback")
    robot = sdk.create_robot(hostname)
    authenticate(robot)
    
    # Ensure time sync client is created
    robot.time_sync.wait_for_sync()
    
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    manipulation_api_client = robot.ensure_client(
        ManipulationApiClient.default_service_name
    )
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    lease_keepalive = LeaseKeepAlive(
        lease_client, must_acquire=True, return_at_exit=True
    )

    # Verify the robot is estopped
    from spot_utils.utils import verify_estop

    verify_estop(robot)

    # Get the number of timesteps in the demo folder
    demo_path = f"demonstrations/{demo_folder_name}"
    timesteps = sorted([int(ts) for ts in os.listdir(demo_path) if ts.isdigit()])
    
    if not timesteps:
        print(f"No timesteps found in {demo_path}")
        return
    
    print(f"Found {len(timesteps)} timesteps in {demo_path}")
    
    # Set up for timestamp-based playback
    playback_start_time = time.time()
    
    # Playback the demonstration data
    timestep_index = 0
    try:
        while timestep_index < len(timesteps):
            timestep = timesteps[timestep_index]
            
            # Load the robot state from the pickle file
            with open(
                f"{demo_path}/{timestep}/robot_state.pkl", "rb"
            ) as state_file:
                robot_data = pkl.load(state_file)

            # Get the timestamp of this data point
            data_timestamp = robot_data.get("timestamp", timestep_index)  # Default to index if no timestamp
            
            # Calculate how much time has passed in our playback
            current_playback_time = time.time() - playback_start_time
            
            # If we're ahead of schedule, wait until it's time to execute this step
            if current_playback_time < data_timestamp:
                wait_time = data_timestamp - current_playback_time
                print(f"Waiting {wait_time:.2f}s for timestep {timestep}")
                time.sleep(wait_time)
            
            # Extract arm joint states
            arm_joint_state_list = robot_data["arm_joint_state"]
            # Extract gripper state
            gripper_open_percentage = robot_data["gripper_open_percentage"]
            # Now we need to extract the position value of each of the robot's joints.
            joint_name_to_position = {}
            for joint in arm_joint_state_list:
                if joint["name"] in ARM_JOINT_NAMES:
                    joint_name_to_position[joint["name"]] = joint["position"]
            assert len(joint_name_to_position) == len(ARM_JOINT_NAMES), (
                "Missing joint positions in the data."
            )
            
            # Create and send the arm joint command
            joint_trajectory_point = (
                RobotCommandBuilder.create_arm_joint_trajectory_point(
                    joint_name_to_position["arm0.sh0"],
                    joint_name_to_position["arm0.sh1"],
                    joint_name_to_position["arm0.el0"],
                    joint_name_to_position["arm0.el1"],
                    joint_name_to_position["arm0.wr0"],
                    joint_name_to_position["arm0.wr1"],
                    # Use the timestamp for trajectory timing
                    # TODO verify if this is correct
                    time_since_reference_secs=data_timestamp
                )
            )
            
            # Create ArmJointTrajectory with points
            arm_joint_traj = arm_command_pb2.ArmJointTrajectory(
                points=[joint_trajectory_point]
            )
            
            # Make a RobotCommand
            command = make_robot_command(arm_joint_traj)
            
            # Send the request
            cmd_id = command_client.robot_command(command)
            print(f"Executed command for timestep {timestep} at timestamp {data_timestamp:.2f}s")

            timestep_index += 1

    except (KeyboardInterrupt, FileNotFoundError):
        print("Stopping playback.")


if __name__ == "__main__":
    main()  # type: ignore
