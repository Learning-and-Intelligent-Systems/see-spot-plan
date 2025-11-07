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
    stow_arm,
)
from skills.spot_navigation import (
    navigate_to_absolute_pose_precise,
)
from spot_utils.gemini_utils import get_pixel_from_gemini
from spot_utils.perception.spot_cameras import capture_images
from spot_utils.spot_localization import SpotLocalizer
from spot_utils.utils import (
    get_graph_nav_dir,
    get_pixel_from_grounded_sam,
    get_pixel_from_user,
    verify_estop,
)
import rerun as rr
from PIL import Image
import cv2

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
    return ROBOT, LOCALIZER, SAM_ENDPOINT


def map_to_spot(pose: math_helpers.SE2Pose) -> math_helpers.SE2Pose:
    """Convert from coordinates in the "room" frame, to spot coordinates."""
    tf = math_helpers.SE2Pose(
        SPOT_ROOM_POSE["x"], SPOT_ROOM_POSE["y"], SPOT_ROOM_POSE["angle"]
    )
    return tf.mult(pose)


def move_to(x_abs: float, y_abs: float, yaw_abs: float) -> None:
    """Move the robot to the specified absolute pose."""
    print(f"move_to(x_abs={x_abs}, y_abs={y_abs}, yaw_abs={yaw_abs}")
    desired_pose_spot = math_helpers.SE2Pose(x=x_abs, y=y_abs, angle=yaw_abs)
    # desired_pose_spot = map_to_spot(desired_pose)
    if ROBOT is not None and LOCALIZER is not None:
        navigate_to_absolute_pose_precise(
            ROBOT, LOCALIZER, desired_pose_spot, max_xytheta_vel=[1,1,1], min_xytheta_vel=[-1,-1,-1], tolerance=0.05
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
    assert ROBOT is not None, "Sahit why!"
    assert LOCALIZER is not None, "SAHIT WHY!!!!"

    images = capture_images(ROBOT, LOCALIZER, [camera])
    rgbd = images[camera]
    rgb_np = rgbd.rgb
    rr.log("rgb_raw", rr.Image(rgb_np))

    # FIXME: Don't do this!!! Hiding implementation
    # if text_prompt and SAM_ENDPOINT:
    #     # Select a pixel by querying GroundedSAM.
    #     pixel = get_pixel_from_grounded_sam(rgbd.rgb, text_prompt, SAM_ENDPOINT)
    # else:
    #     # Select a pixel by querying the user.
    #     pixel = get_pixel_from_user(rgbd.rgb)

    # Call Gemini to point
    vlm_query_template = f"""
    Point to the {text_prompt}. If you cannot see the {text_prompt} fully, point to the best guess.
    The answer should follow the json format: [{{"point": , "label": }}, ...]. The points are in [y, x] format normalized to 0-1000.
    """
    image_pil = Image.fromarray(rgb_np)
    pixel = get_pixel_from_gemini(vlm_query_template, image_pil)

    # Draw pixel on the image
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    cv2.circle(bgr, pixel, 5, (0, 0, 255), -1)
    rgb_annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rr.log("pointing", rr.Image(rgb_annotated))

    if pixel is not None:
        # Grasp at the pixel with a top-down grasp.
        top_down_rot = math_helpers.Quat.from_pitch(np.pi / 2)
        grasp_at_pixel(ROBOT, rgbd, pixel, grasp_rot=top_down_rot)
        return
    else:
        raise RuntimeError("WTF. Grasp failed!")


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
    rr.init("exec_plan", spawn=True)
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
    # Stow before running plan

    # grasp("teddy bear")
    stow_arm(ROBOT)
    # open_gripper(ROBOT)
    # close_gripper(ROBOT)
    # gaze("DOWN")
    # move_to(x_abs=2.454116588554405, y_abs=-2.759339339399913, yaw_abs=-2.401609411896162)
    # move_to(x_abs=2.542117893278808, y_abs=-0.752019696769485, yaw_abs=-1.933767708350784)
    #
    # move_to(x_abs=2.723552922357717, y_abs=-3.172621984462674, yaw_abs=-2.753087971468099)
    # gaze("AHEAD")
    # grasp("caterpillar")
    #
    # move_to(x_abs=4.471199645580120, y_abs=-3.217432928943325, yaw_abs=-0.023820034339159)
    # place_at_pose([0.9594754639330341, 0.0, 0.09019530213554317, 0.0, 0.0, -0.18379480399141906, 0.9829646331510385])

    # move_to(x_abs=4.483301381932474, y_abs=-3.486604871046227, yaw_abs=0.130814244458979)
    # place_at_pose([0.8673816028445435, 0.0, -0.011952195088877238, 0.0, 0.0, 0.1621424933557064, 0.9867673544703406])
    # print()
    with open(args.plan, "r") as plan_file:
        exec(plan_file.read())
    print("done")
