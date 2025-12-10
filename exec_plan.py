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

from skills.grasp_vlm import grasp_with_vlm
from skills.spot_hand_move import (
    close_gripper,
    move_hand_to_relative_pose,
    open_gripper,
    stow_arm,
)
from skills.wipe import wipe_multiple_strokes
from skills.wipe_online import wipe_online as run_wipe_online
from skills.push_button import push_button as run_push_button
from skills.open_drawer import open_drawer as run_open_drawer
from skills.spot_navigation import navigate_to_absolute_pose
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


def np_pose_to_SE3(X_RobEE: NDArray) -> math_helpers.SE3Pose:
    return math_helpers.SE3Pose(
        X_RobEE[0],
        X_RobEE[1],
        X_RobEE[2],
        rot=math_helpers.Quat(X_RobEE[6], X_RobEE[3], X_RobEE[4], X_RobEE[5]),
    )


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
        navigate_to_absolute_pose(ROBOT, LOCALIZER, desired_pose_spot)


def gaze(direction: str) -> None:
    """Move the hand to look in a certain direction."""
    look_pose = direction_to_pose[direction]
    move_hand_to_relative_pose(ROBOT, look_pose)
    open_gripper(ROBOT)

def gaze_without_open(direction: str) -> None:
    """ Move the hand to look in a certain direction without opening the gripper."""
    look_pose = direction_to_pose[direction]
    move_hand_to_relative_pose(ROBOT, look_pose)


def grasp(text_prompt: Optional[str]) -> None:
    """Grasp an object at a specified pixel."""
    assert ROBOT is not None, "Sahit why!"
    assert LOCALIZER is not None, "SAHIT WHY!!!!"
    grasp_with_vlm(ROBOT, LOCALIZER, text_prompt)


def grasp_at_pose(X_RobEE: NDArray) -> None:
    """Grasp an object at a specified pose relative to the robot."""
    open_gripper(ROBOT)
    pose = np_pose_to_SE3(X_RobEE)
    move_hand_to_relative_pose(ROBOT, pose.mult(grasp_offset))
    close_gripper(ROBOT)
    move_hand_to_relative_pose(ROBOT, DEFAULT_HAND_LOOK_FLOOR_POSE)


def place_at_pose(X_RobEE: NDArray) -> None:
    """Place an object at a specified xyz position with a top-down approach.

    The first three entries of X_RobEE are interpreted as (x, y, z) in the
    robot body frame. Any additional entries (e.g. quaternion components) are
    ignored for placement. A fixed positional offset of (0, 0, 0.05) meters in
    the body frame is applied so the hand stops slightly above the nominal
    target position. The orientation is set to a fixed top-down pose so that
    the arm motion is simple and predictable, independent of any orientation
    passed in.
    """
    assert ROBOT is not None
    # Interpret the input as an xyz position in the body frame.
    x, y, z = X_RobEE[0], X_RobEE[1], X_RobEE[2]
    # Fixed positional offset (dx, dy, dz) expressed in the body frame.
    dx, dy, dz = 0.0, 0.0, 0.05
    # Use the same "straight down" orientation used elsewhere for looking down.
    top_down_rot = DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE.rot
    # Apply the offset directly in the body frame so that positive dz moves
    # the hand upward relative to the robot body.
    place_pose = math_helpers.SE3Pose(
        x=x + dx,
        y=y + dy,
        z=z + dz,
        rot=top_down_rot,
    )
    # Move to the placement pose, open the gripper to release, then the plan
    # can decide when to stow the arm.
    move_hand_to_relative_pose(ROBOT, place_pose)
    open_gripper(ROBOT)


def vertical_wipe(
    X_RobEE_start: NDArray, stroke_dx: float, y_delta: float, num_strokes: int
) -> None:
    """Wipes a surface at a given pose with known height and width.."""
    start_pose = np_pose_to_SE3(X_RobEE_start)
    wipe_multiple_strokes(
        ROBOT,
        start_pose,
        start_pose,
        stroke_dx=stroke_dx,
        stroke_dy=0,
        delta_x_y_between_strokes=(0, y_delta),
        num_strokes=num_strokes,
        duration_per_stroke=3.0,
        num_attempts_per_stroke=1,
    )


def wipe_at(*args, **kwargs) -> None:
    """Run the online wipe skill using defaults from the skill module."""
    run_wipe_online(
        ROBOT,
        None,
        None,
        LOCALIZER,
    )


def press_button(text_prompt: Optional[str]) -> None:
    """Identify a button and push it using the hand camera.

    If text_prompt is provided, it will be used as the label (e.g., "button").
    """
    label = text_prompt if text_prompt else "button"
    if ROBOT is not None and LOCALIZER is not None:
        run_push_button(
            ROBOT,
            LOCALIZER,
            label=label,
        )


def open_cabinet_drawer(
    standoff_dist: float = 0.8,
    body_height_offset: float = 0.0,
    retreat_offset: float = 0.1,
    checkpoint: int = 7,
) -> None:
    """Open a drawer using the high-level open_drawer skill.

    This delegates to ``skills.open_drawer.open_drawer``, passing the
    initialized global ``ROBOT`` and ``LOCALIZER``.
    """
    assert ROBOT is not None, "Robot is not initialized; call init(...) first."
    assert LOCALIZER is not None, "Localizer is not initialized; call init(...) first."

    run_open_drawer(
        ROBOT,
        LOCALIZER,
        standoff_dist=standoff_dist,
        body_height_offset=body_height_offset,
        retreat_offset=retreat_offset,
        checkpoint=checkpoint,
    )


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
            SPOT_ROOM_POSE = {"x": 0.0, "y": 0.0, "angle": 0.0}
            
    with open(args.plan, "r") as plan_file:
        exec(plan_file.read())
    print("done")
