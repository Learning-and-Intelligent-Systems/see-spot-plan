"""Interface for Spot navigation."""

import argparse
import logging
import time
from typing import Tuple

from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, \
    get_se2_a_tform_b
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.sdk import Robot

from spot_utils.spot_localization import SpotLocalizer
from spot_utils.utils import get_graph_nav_dir, get_robot_state, verify_estop


def navigate_to_relative_pose(robot: Robot,
                              body_tform_goal: math_helpers.SE2Pose,
                              max_xytheta_vel: Tuple[float, float,
                                                     float] = (2.0, 2.0, 1.0),
                              min_xytheta_vel: Tuple[float, float,
                                                     float] = (-2.0, -2.0,
                                                               -1.0),
                              timeout: float = 20.0) -> None:
    """Execute a relative move.

    The pose is dx, dy, dyaw relative to the robot's body.
    """
    # Get the robot's current state.
    robot_state = get_robot_state(robot)
    transforms = robot_state.kinematic_state.transforms_snapshot
    assert str(transforms) != ""

    # We do not want to command this goal in body frame because the body will
    # move, thus shifting our goal. Instead, we transform this offset to get
    # the goal position in the output frame (odometry).
    out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME,
                                       BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified
    # frame. The command will stop at the new position.
    # Constrain the robot not to turn, forcing it to strafe laterally.
    speed_limit = SE2VelocityLimit(
        max_vel=SE2Velocity(linear=Vec2(x=max_xytheta_vel[0],
                                        y=max_xytheta_vel[1]),
                            angular=max_xytheta_vel[2]),
        min_vel=SE2Velocity(linear=Vec2(x=min_xytheta_vel[0],
                                        y=min_xytheta_vel[1]),
                            angular=min_xytheta_vel[2]))
    mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit)

    robot_command_client = robot.ensure_client(
        RobotCommandClient.default_service_name)
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x,
        goal_y=out_tform_goal.y,
        goal_heading=out_tform_goal.angle,
        frame_name=ODOM_FRAME_NAME,
        params=mobility_params)
    cmd_id = robot_command_client.robot_command(lease=None,
                                                command=robot_cmd,
                                                end_time_secs=time.time() +
                                                timeout)
    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) <= timeout:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.\
            synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:  # pylint: disable=no-member,line-too-long
            logging.warning("Failed to reach the goal")
            return
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL
                and traj_feedback.body_movement_status
                == traj_feedback.BODY_STATUS_SETTLED):
            return
    if (time.perf_counter() - start_time) > timeout:
        logging.warning("Timed out waiting for movement to execute!")



def navigate_to_absolute_pose(robot: Robot,
                              localizer: SpotLocalizer,
                              target_pose: math_helpers.SE2Pose,
                              max_xytheta_vel: Tuple[float, float,
                                                     float] = (2.0, 2.0, 1.0),
                              min_xytheta_vel: Tuple[float, float,
                                                     float] = (-2.0, -2.0,
                                                               -1.0),
                              timeout: float = 20.0) -> None:
    """Move to the absolute SE2 pose."""
    localizer.localize()
    robot_pose = localizer.get_last_robot_pose()
    robot_se2 = robot_pose.get_closest_se2_transform()
    rel_pose = robot_se2.inverse() * target_pose
    return navigate_to_relative_pose(robot, rel_pose, max_xytheta_vel,
                                     min_xytheta_vel, timeout)



if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # pylint: disable=ungrouped-imports
    import numpy as np
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    def _run_manual_test() -> None:
        # Get constants.
        parser = argparse.ArgumentParser(description="Parse the robot's hostname.")
        parser.add_argument('--hostname', type=str, required=True, help="The robot's hostname/ip-address (e.g. 192.168.80.3)")
        parser.add_argument('--map_name', type=str, required=True, help="The name of the map folder to load (sub-folder under graph_nav_maps)")
        args = parser.parse_args()

        hostname = args.hostname

        sdk = create_standard_sdk('NavigationSkillTestClient')
        robot = sdk.create_robot(hostname)
        authenticate(robot)
        verify_estop(robot)
        path = get_graph_nav_dir(args.map_name)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        lease_keepalive = LeaseKeepAlive(lease_client,
                                         must_acquire=True,
                                         return_at_exit=True)
        localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
        robot.time_sync.wait_for_sync()
        localizer.localize()
        robot_pose_SE3 = localizer.get_last_robot_pose()
        robot_pose_SE2 = robot_pose_SE3.get_closest_se2_transform()
        offset = math_helpers.SE2Pose(x=0, y=10.5, angle=0)
        desired_pose_SE2 = robot_pose_SE2.mult(offset)
        navigate_to_absolute_pose(robot, localizer, desired_pose_SE2)
        """relative_poses = [ (math_helpers.SE2Pose(x=0, y=0, angle=0),
        "Standing still"), (math_helpers.SE2Pose(x=0, y=0.5, angle=0), "Moving
        dy"), (math_helpers.SE2Pose(x=0, y=-0.5, angle=0), "Moving -dy"),
        (math_helpers.SE2Pose(x=0.5, y=0, angle=0), "Moving dx"),
        (math_helpers.SE2Pose(x=-0.5, y=0, angle=0), "Moving -dx"),
        (math_helpers.SE2Pose(x=0, y=0, angle=np.pi / 2), "Moving yaw"),
        (math_helpers.SE2Pose(x=0, y=0, angle=-np.pi / 2), "Moving -yaw"), ]
        for relative_pose, msg in relative_poses:

        print(msg) navigate_to_relative_pose(robot, relative_pose)
        """
    _run_manual_test()
