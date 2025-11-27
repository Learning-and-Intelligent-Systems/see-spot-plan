"""Parsing script for running high-level commands on the spot.

This code makes it possible to write down a sequence of robot commands
in a text file and have them run on the real robot. The logic for
managing, e.g., the lease for the spot and the relevant datastructures
for specifiying SE(2) poses can be abstracted away from the high-level
plan provided as input.
"""

import argparse
import json
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
from skills.wipe import wipe_multiple_strokes
from skills.wipe_online import wipe_online as run_wipe_online
from skills.push_button import push_button as run_push_button
from skills.spot_navigation import navigate_to_absolute_pose
from spot_utils.pretrained_model_interface import GoogleGeminiVLM
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


def _get_pixel_from_gemini(vlm_query_str: str, pil_image: Image.Image) -> tuple[int, int]:
    """Query Gemini VLM to get a single pixel [y, x] normalized to 0-1000, then
    denormalize to image pixel coordinates.

    This mirrors the usage pattern in the wipe tool: construct a
    GoogleGeminiVLM instance and call sample_completions directly.
    """

    vlm = GoogleGeminiVLM("gemini-2.5-pro")

    def _strip_markdown_fence(json_output_str: str) -> str:
        """Remove ```json fences if present and return the inner JSON string."""
        lines = json_output_str.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                json_output_str = "\n".join(lines[i + 1 :])
                json_output_str = json_output_str.split("```")[0]
                break
        return json_output_str.strip()

    # 1) Query the VLM
    vlm_output_list = vlm.sample_completions(
        prompt=vlm_query_str,
        imgs=[pil_image],
        temperature=0.0,
        seed=42,
        num_completions=1,
    )
    vlm_output_str = vlm_output_list[0]

    # 2) Parse JSON output
    json_string_to_parse = _strip_markdown_fence(vlm_output_str)
    parsed_data = json.loads(json_string_to_parse)

    if not isinstance(parsed_data, list) or not parsed_data:
        raise ValueError("Parsed JSON is not a non-empty list.")

    first_point_obj = parsed_data[0]
    if (
        "point" not in first_point_obj
        or not isinstance(first_point_obj["point"], list)
        or len(first_point_obj["point"]) != 2
    ):
        raise ValueError(
            "First element in JSON does not contain a valid 'point' list [y, x]."
        )

    y_norm, x_norm = first_point_obj["point"]
    if not isinstance(y_norm, (int, float)) or not isinstance(x_norm, (int, float)):
        raise ValueError("Normalized coordinates are not numbers.")

    # 3) Denormalize from 0–1000 range to image pixel coordinates
    img_height = pil_image.height
    img_width = pil_image.width
    y = int(y_norm * img_height / 1000.0)
    x = int(x_norm * img_width / 1000.0)

    # Clamp to image bounds
    y = max(0, min(y, img_height - 1))
    x = max(0, min(x, img_width - 1))

    # Return as (x, y) pixel coordinate
    return (x, y)


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
    pixel = _get_pixel_from_gemini(vlm_query_template, image_pil)

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
    pose = np_pose_to_SE3(X_RobEE)
    move_hand_to_relative_pose(ROBOT, pose.mult(grasp_offset))
    close_gripper(ROBOT)
    move_hand_to_relative_pose(ROBOT, DEFAULT_HAND_LOOK_FLOOR_POSE)


def place_at_pose(X_RobEE: NDArray) -> None:
    """Place an object at a specified pose relative to the robot."""
    pose = np_pose_to_SE3(X_RobEE)
    move_hand_to_relative_pose(ROBOT, pose.mult(grasp_offset))
    open_gripper(ROBOT)
    move_hand_to_relative_pose(ROBOT, DEFAULT_HAND_LOOK_FLOOR_POSE)


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


def press(text_prompt: Optional[str]) -> None:
    """Identify a button and push it using the hand camera.

    If text_prompt is provided, it will be used as the label (e.g., "button").
    """
    label = text_prompt if text_prompt else "button"
    if ROBOT is not None and LOCALIZER is not None:
        run_push_button(
            ROBOT,
            LOCALIZER,
            label=label,
            sam_endpoint=SAM_ENDPOINT,
            use_vlm=True,
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
