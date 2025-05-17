"""Parsing script for running high-level commands on the spot.

This code makes it possible to write down a sequence of robot commands
in a text file and have them run on the real robot. The logic for
managing, e.g., the lease for the spot and the relevant datastructures
for specifiying SE(2) poses can be abstracted away from the high-level
plan provided as input.
"""

import argparse
from typing import Dict, Optional

import numpy as np
import yaml
from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.util import authenticate
from numpy.typing import NDArray

from skills.grasp import grasp_at_pixel
from skills.spot_hand_move import (
    close_gripper,
    move_hand_to_relative_pose,
    open_gripper,
)
from skills.spot_navigation import (
    navigate_to_absolute_pose_precise,
)
from spot_utils.perception.spot_cameras import capture_images
from spot_utils.spot_localization import SpotLocalizer
from spot_utils.utils import (
    get_graph_nav_dir,
    get_pixel_from_grounded_sam,
    get_pixel_from_user,
    verify_estop,
)

DEFAULT_HAND_LOOK_FLOOR_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 3)
)

DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2)
)

direction_to_pose = {
    "DOWN": DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE,
    "AHEAD": DEFAULT_HAND_LOOK_FLOOR_POSE,
}

grasp_offset = math_helpers.SE3Pose(0, 0, 0, math_helpers.Quat.from_pitch(np.pi / 2))

LOCALIZER = None
ROBOT = None
SAM_ENDPOINT = None
SPOT_ROOM_POSE: Dict[str, float] = dict()


def init(hostname: str, map_name: str, endpoint_url: Optional[str]) -> None:
    """Initialize the robot and the localizer."""
    global LOCALIZER
    global ROBOT
    global SAM_ENDPOINT
    sdk = create_standard_sdk("NavigationSkillTestClient")
    ROBOT = sdk.create_robot(hostname)
    authenticate(ROBOT)
    verify_estop(ROBOT)
    path = get_graph_nav_dir(map_name)
    lease_client = ROBOT.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    lease_keepalive = LeaseKeepAlive(
        lease_client, must_acquire=True, return_at_exit=True
    )
    LOCALIZER = SpotLocalizer(ROBOT, path, lease_client, lease_keepalive)
    ROBOT.time_sync.wait_for_sync()
    LOCALIZER.localize()
    SAM_ENDPOINT = endpoint_url


def map_to_spot(pose: math_helpers.SE2Pose) -> math_helpers.SE2Pose:
    """Convert from coordinates in the "room" frame, to spot coordinates."""
    tf = math_helpers.SE2Pose(
        SPOT_ROOM_POSE["x"], SPOT_ROOM_POSE["y"], SPOT_ROOM_POSE["angle"]
    )
    return tf.mult(pose)


def move_to(x_abs: float, y_abs: float, yaw_abs: float) -> None:
    """Move the robot to the specified absolute pose."""
    desired_pose = math_helpers.SE2Pose(x=x_abs, y=y_abs, angle=yaw_abs)
    desired_pose_spot = map_to_spot(desired_pose)
    if ROBOT is not None and LOCALIZER is not None:
        navigate_to_absolute_pose_precise(
            ROBOT, LOCALIZER, desired_pose_spot, tolerance=0.015
        )


def gaze(direction: str) -> None:
    """Move the hand to look in a certain direction."""
    look_pose = direction_to_pose[direction]
    move_hand_to_relative_pose(ROBOT, look_pose)
    open_gripper(ROBOT)


def grasp(text_prompt: Optional[str]) -> None:
    """Grasp an object at a specified pixel."""
    # Capture an image.
    camera = "hand_color_image"
    if ROBOT is not None and LOCALIZER is not None:
        rgbd = capture_images(ROBOT, LOCALIZER, [camera])[camera]

        if text_prompt and SAM_ENDPOINT:
            # Select a pixel by querying GroundedSAM.
            pixel = get_pixel_from_grounded_sam(rgbd.rgb, text_prompt, SAM_ENDPOINT)
        else:
            # Select a pixel by querying the user.
            pixel = get_pixel_from_user(rgbd.rgb)

        if pixel is not None:
            # Grasp at the pixel with a top-down grasp.
            top_down_rot = math_helpers.Quat.from_pitch(np.pi / 2)
            grasp_at_pixel(ROBOT, rgbd, pixel, grasp_rot=top_down_rot)


def grasp_at_pose(X_RobEE: NDArray) -> None:
    """Grasp an object at a specified pose relative to the robot."""
    open_gripper(ROBOT)
    pose = math_helpers.SE3Pose(
        x=X_RobEE[0],
        y=X_RobEE[1],
        z=X_RobEE[2],
        rot=math_helpers.Quat(X_RobEE[6], X_RobEE[3], X_RobEE[4], X_RobEE[5]),
    )
    move_hand_to_relative_pose(ROBOT, pose.mult(grasp_offset))
    close_gripper(ROBOT)
    move_hand_to_relative_pose(ROBOT, DEFAULT_HAND_LOOK_FLOOR_POSE)


def place_at_pose(X_RobEE: NDArray) -> None:
    """Place an object at a specified pose relative to the robot."""
    pose = math_helpers.SE3Pose(
        x=X_RobEE[0],
        y=X_RobEE[1],
        z=X_RobEE[2],
        rot=math_helpers.Quat(X_RobEE[6], X_RobEE[3], X_RobEE[4], X_RobEE[5]),
    )
    move_hand_to_relative_pose(ROBOT, pose.mult(grasp_offset))
    open_gripper(ROBOT)
    move_hand_to_relative_pose(ROBOT, DEFAULT_HAND_LOOK_FLOOR_POSE)


if __name__ == "__main__":
    # running this script standalone initializes a bosdyn robot and localizer.
    # It then executes the list of commands provided in the plan file
    parser = argparse.ArgumentParser(description="Parse the robot's hostname.")
    parser.add_argument(
        "--hostname",
        type=str,
        required=True,
        help="The robot's hostname/ip-address (e.g. 192.168.80.3)",
    )
    parser.add_argument(
        "--map_name",
        type=str,
        required=True,
        help="The name of the map folder to load (sub-folder under graph_nav_maps)",
    )
    parser.add_argument(
        "--plan", type=str, required=True, help="Path of the Plan to run"
    )
    parser.add_argument(
        "--sam_endpoint",
        type=str,
        required=False,
        help="Address of endpoint hosting GroundedSAM",
    )
    args = parser.parse_args()
    init(args.hostname, args.map_name, args.sam_endpoint)
    with open(get_graph_nav_dir(args.map_name) / "metadata.yaml", "rb") as f:
        metadata = yaml.safe_load(f)
        if "spot-room-pose" in metadata.keys():
            SPOT_ROOM_POSE = metadata["spot-room-pose"]
        else:
            print("spot-room-pose not found in metadata.yaml, using default val")
            SPOT_ROOM_POSE = {"x": 0.0, "y": 0.0, "z": 0.0, "angle": 0.0}
    with open(args.plan, "r") as plan_file:
        exec(plan_file.read())
    print("done")
