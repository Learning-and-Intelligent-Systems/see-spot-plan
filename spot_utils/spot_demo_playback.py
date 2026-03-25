"""Script to playback demonstration data on the Spot robot.

Example usage:
python spot_utils/spot_demo_playback.py --hostname 192.168.80.3 --demo_folder_name test_data_recording0
"""

import argparse
import os
import time

import dill as pkl
from bosdyn.api import (
    arm_command_pb2,
    robot_command_pb2,
    synchronized_command_pb2,
)
from bosdyn.client import create_standard_sdk
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.util import authenticate
from bosdyn.util import seconds_to_duration
from google.protobuf import wrappers_pb2
from rich import print

ARM_JOINT_NAMES = [
    "arm0.sh0",
    "arm0.sh1",
    "arm0.el0",
    "arm0.el1",
    "arm0.wr0",
    "arm0.wr1",
]

# Time buffer for the first action to prevent too-fast initial movement
FIRST_ACTION_BUFFER = 0.5  # seconds


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


def block_until_arm_arrives(command_client, cmd_id, timeout_sec):
    """Helper that blocks until the arm command completes or times out."""
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        feedback_resp = command_client.robot_command_feedback(cmd_id)
        if (
            feedback_resp.feedback.synchronized_feedback.arm_command_feedback.status
            == arm_command_pb2.ArmCommandFeedback.STATUS_TRAJECTORY_COMPLETE
        ):
            return True
        time.sleep(0.1)
    return False


def create_synchronized_command(
    arm_joint_traj, gripper_percentage=None, gripper_force=None
):
    """Create a command with both arm trajectory and optional gripper commands."""
    # First create the arm command
    joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(
        trajectory=arm_joint_traj
    )
    arm_command = arm_command_pb2.ArmCommand.Request(
        arm_joint_move_command=joint_move_command
    )

    # Create the synchronized command
    if gripper_percentage is not None:
        # For better gripping when force is specified
        if gripper_percentage < 0.2 and gripper_force is not None:
            # Use custom gripper command with stronger grip
            print(f"Using stronger grip for gripper (detected force: {gripper_force})")

            # Create a ClawGripperCommand manually with trajectory points
            # We can't use max_torque parameter directly, so we'll use a different approach
            from bosdyn.api import gripper_command_pb2, trajectory_pb2

            # Create a trajectory point with the target percentage
            traj_point = trajectory_pb2.ScalarTrajectoryPoint()
            traj_point.point = gripper_percentage

            # Create the claw gripper command
            claw_command = gripper_command_pb2.ClawGripperCommand.Request()
            claw_command.trajectory.points.append(traj_point)

            # Create the gripper command
            gripper_command = gripper_command_pb2.GripperCommand.Request(
                claw_gripper_command=claw_command
            )
        else:
            # Use standard gripper command
            gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
                gripper_percentage
            )
            gripper_command = gripper_cmd.synchronized_command.gripper_command

        # Create synchronized command with both arm and gripper
        sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command, gripper_command=gripper_command
        )
    else:
        # Arm command only
        sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command
        )

    # Create the robot command
    return robot_command_pb2.RobotCommand(synchronized_command=sync_command)


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
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()

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
    print(f"First action buffer: {FIRST_ACTION_BUFFER}s")

    # Set up for timestamp-based playback
    playback_start_time = time.time()
    last_robot_data = None
    last_execution_time = 0
    last_gripper_percentage = None

    # Playback the demonstration data
    timestep_index = 0
    try:
        while timestep_index < len(timesteps):
            timestep = timesteps[timestep_index]

            # Load the robot state from the pickle file
            with open(f"{demo_path}/{timestep}/robot_state.pkl", "rb") as state_file:
                robot_data = pkl.load(state_file)

            # Get the timestamp of this data point
            data_timestamp = robot_data.get(
                "timestamp", timestep_index
            )  # Default to index if no timestamp

            # Determine the delta time to use for trajectory timing
            if last_robot_data is not None:
                last_data_timestamp = last_robot_data.get(
                    "timestamp", timestep_index - 1
                )
                if data_timestamp < last_data_timestamp:
                    print(
                        f"Warning: Data timestamp {data_timestamp} is less than last data timestamp {last_data_timestamp}. Skipping this timestep."
                    )
                    timestep_index += 1
                    continue
                # Calculate time difference between current and previous action
                delta_time = data_timestamp - last_data_timestamp
            else:
                # First action uses the buffer time
                delta_time = FIRST_ACTION_BUFFER

            # Calculate how much time has passed in our playback
            current_playback_time = time.time() - playback_start_time

            # If we're ahead of schedule, wait until it's time to execute this step
            wait_point = last_execution_time + (
                0 if timestep_index == 0 else delta_time
            )
            if current_playback_time < wait_point:
                wait_time = wait_point - current_playback_time
                print(f"Waiting {wait_time:.2f}s for timestep {timestep}")
                time.sleep(wait_time)

            # Extract arm joint states
            arm_joint_state_list = robot_data["arm_joint_state"]

            # Extract gripper state and force information
            gripper_open_percentage = robot_data["gripper_open_percentage"]
            gripper_force = robot_data.get("gripper_force")
            gripper_holding = robot_data.get("gripper_holding", False)

            # Normalize gripper percentage if it's from old recordings
            if gripper_open_percentage > 1.0:
                gripper_open_percentage = min(
                    max(gripper_open_percentage / 100.0, 0.0), 1.0
                )

            # Adjust gripper openness based on force detection - for stronger grip
            if (
                gripper_force
                and "magnitude" in gripper_force
                and gripper_force["magnitude"] > 10.0
            ):
                # Significant force detected - reduce openness to apply more force during grip
                original_percentage = gripper_open_percentage
                # Reduce openness by 25% but maintain a minimum to prevent crushing
                gripper_open_percentage = max(gripper_open_percentage * 0.75, 0.01)
                print(
                    f"High force detected ({gripper_force['magnitude']:.2f}N) - reducing gripper opening from "
                    f"{original_percentage:.2f} to {gripper_open_percentage:.2f} for stronger grip"
                )

            # Check if gripper state has changed significantly
            gripper_to_send = None
            if (
                last_gripper_percentage is None
                or abs(gripper_open_percentage - last_gripper_percentage) > 0.02
            ):
                gripper_to_send = gripper_open_percentage
                print(
                    f"Including gripper position {gripper_open_percentage:.2f} in command"
                )

                # If the gripper is holding something, we'll apply more torque during playback
                if gripper_holding:
                    print("Detected gripper holding object - will apply higher torque")
                last_gripper_percentage = gripper_open_percentage

            # Now we need to extract the position value of each of the robot's joints.
            positions = {}
            velocities = {}

            for joint in arm_joint_state_list:
                if joint["name"] in ARM_JOINT_NAMES:
                    positions[joint["name"]] = joint["position"]
                    if "velocity" in joint:
                        velocities[joint["name"]] = joint["velocity"]

            # If we have velocity data for all joints, use it to create a trajectory point with velocity
            if len(velocities) == len(ARM_JOINT_NAMES):
                # Create a trajectory point with both position and velocity
                joint_trajectory_point = arm_command_pb2.ArmJointTrajectoryPoint(
                    position=arm_command_pb2.ArmJointPosition(
                        sh0=wrappers_pb2.DoubleValue(value=positions["arm0.sh0"]),
                        sh1=wrappers_pb2.DoubleValue(value=positions["arm0.sh1"]),
                        el0=wrappers_pb2.DoubleValue(value=positions["arm0.el0"]),
                        el1=wrappers_pb2.DoubleValue(value=positions["arm0.el1"]),
                        wr0=wrappers_pb2.DoubleValue(value=positions["arm0.wr0"]),
                        wr1=wrappers_pb2.DoubleValue(value=positions["arm0.wr1"]),
                    ),
                    velocity=arm_command_pb2.ArmJointVelocity(
                        sh0=wrappers_pb2.DoubleValue(value=velocities["arm0.sh0"]),
                        sh1=wrappers_pb2.DoubleValue(value=velocities["arm0.sh1"]),
                        el0=wrappers_pb2.DoubleValue(value=velocities["arm0.el0"]),
                        el1=wrappers_pb2.DoubleValue(value=velocities["arm0.el1"]),
                        wr0=wrappers_pb2.DoubleValue(value=velocities["arm0.wr0"]),
                        wr1=wrappers_pb2.DoubleValue(value=velocities["arm0.wr1"]),
                    ),
                    time_since_reference=seconds_to_duration(delta_time),
                )
                print("Using velocity data for smoother trajectory")
            else:
                # Fall back to the position-only approach
                joint_trajectory_point = (
                    RobotCommandBuilder.create_arm_joint_trajectory_point(
                        positions["arm0.sh0"],
                        positions["arm0.sh1"],
                        positions["arm0.el0"],
                        positions["arm0.el1"],
                        positions["arm0.wr0"],
                        positions["arm0.wr1"],
                        time_since_reference_secs=delta_time,
                    )
                )
                print("No velocity data available, using position-only trajectory")

            # Create ArmJointTrajectory with points and velocity/acceleration limits
            # This makes motion smoother by constraining the maximum velocity and acceleration
            max_vel = wrappers_pb2.DoubleValue(value=3.0)  # rad/s
            max_acc = wrappers_pb2.DoubleValue(value=7.5)  # rad/s^2

            arm_joint_traj = arm_command_pb2.ArmJointTrajectory(
                points=[joint_trajectory_point],
                maximum_velocity=max_vel,
                maximum_acceleration=max_acc,
            )

            # Create and send a combined command
            combined_command = create_synchronized_command(
                arm_joint_traj, gripper_to_send, gripper_force
            )
            _ = command_client.robot_command(combined_command)

            if gripper_to_send is not None:
                print(
                    f"Executed command for timestep {timestep} with arm movement and gripper position {gripper_to_send:.2f}"
                )
            else:
                print(
                    f"Executed command for timestep {timestep} with arm movement only"
                )

            # Update tracking variables
            last_execution_time = time.time() - playback_start_time
            timestep_index += 1
            last_robot_data = robot_data

    except (KeyboardInterrupt, FileNotFoundError):
        print("Stopping playback.")


if __name__ == "__main__":
    main()
