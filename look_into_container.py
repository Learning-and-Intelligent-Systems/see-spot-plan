"""Skill for looking into containers (buckets, boxes) with Spot."""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image  # type: ignore[import]
from bosdyn.client import math_helpers
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.sdk import Robot, create_standard_sdk
from bosdyn.client.util import authenticate

from spot_utils.perception.spot_cameras import capture_images
from spot_utils.spot_localization import SpotLocalizer
from spot_utils.utils import (
    DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE,
    get_graph_nav_dir,
    verify_estop,
)
from skills.spot_hand_move import move_hand_to_relative_pose, stow_arm, open_gripper


def _build_look_pose(
    forward_offset: float,
    vertical_offset: float,
    pitch_deg: float,
) -> math_helpers.SE3Pose:
    pitch_rad = np.deg2rad(pitch_deg)
    return math_helpers.SE3Pose(
        x=forward_offset,
        y=0.0,
        z=vertical_offset,
        rot=math_helpers.Quat.from_pitch(pitch_rad),
    )


def look_into_container(
    robot: Robot,
    localizer: SpotLocalizer,
    forward_offset: float = 1.0,
    vertical_offset: float = 0.25,
    pitch_deg: float = 45.0,
    settle_seconds: float = 1.0,
    image_basename: str = "container_inspection",
    stow_after: bool = True,
    open_before_capture: bool = True,
) -> Path:
    """Tilt the arm down and capture a hand-camera image of a container.

    Args:
        robot: Spot robot instance (lease already acquired).
        localizer: Existing SpotLocalizer for keeping localization fresh.
        forward_offset: Forward reach of the hand relative to the body (m).
        vertical_offset: Vertical placement of the hand relative to body origin (m).
        pitch_deg: Downward pitch angle for the wrist (degrees).
        settle_seconds: Wait time after moving the arm before capturing.
        image_basename: Base filename for the saved RGB image.
        stow_after: Whether to stow the arm after the capture.
        open_before_capture: Whether to open the gripper before capturing.

    Returns:
        Path to the saved RGB image.
    """

    look_pose = _build_look_pose(forward_offset, vertical_offset, pitch_deg)
    move_hand_to_relative_pose(robot, look_pose)
    time.sleep(settle_seconds)

    if open_before_capture:
        open_gripper(robot)
        time.sleep(0.3)

    rgbds = capture_images(robot, localizer, camera_names=["hand_color_image"])
    rgbd = rgbds["hand_color_image"]

    pil_image = Image.fromarray(rgbd.rgb)
    image_path = Path(f"{image_basename}.jpg")
    pil_image.save(image_path)

    if stow_after:
        stow_arm(robot)

    return image_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Look into a container with Spot's hand camera")
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--map_name", required=True, help="GraphNav map name under spot_utils/graph_nav_maps")
    parser.add_argument("--forward_offset", type=float, default=0.8)
    parser.add_argument("--vertical_offset", type=float, default=0.2)
    parser.add_argument("--pitch_deg", type=float, default=70.0, help="Downward pitch angle in degrees")
    parser.add_argument("--settle_seconds", type=float, default=1.0)
    parser.add_argument("--image_basename", default="container_inspection")
    parser.add_argument("--no_stow", action="store_true", help="Keep the arm deployed after capturing")
    parser.add_argument("--no_open_gripper", action="store_true", help="Skip opening the gripper before imaging")
    args = parser.parse_args()

    sdk = create_standard_sdk("SpotLookIntoContainer")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    lease_keepalive = LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)

    path = get_graph_nav_dir(args.map_name)
    localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
    robot.time_sync.wait_for_sync()
    localizer.localize()

    try:
        image_path = look_into_container(
            robot,
            localizer,
            forward_offset=args.forward_offset,
            vertical_offset=args.vertical_offset,
            pitch_deg=args.pitch_deg,
            settle_seconds=args.settle_seconds,
            image_basename=args.image_basename,
            stow_after=not args.no_stow,
            open_before_capture=not args.no_open_gripper,
        )
        print(f"Saved container inspection image to {image_path}")
    finally:
        pass


if __name__ == "__main__":
    main()
