"""Skill for looking into containers (buckets, boxes) with Spot."""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw  # type: ignore[import]
from bosdyn.client import math_helpers
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.sdk import Robot, create_standard_sdk
from bosdyn.client.util import authenticate
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, VISION_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_state import RobotStateClient

from spot_utils.perception.spot_cameras import capture_images
from spot_utils.spot_localization import SpotLocalizer
from spot_utils.utils import (
    DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE,
    get_graph_nav_dir,
    verify_estop,
)
from skills.spot_hand_move import move_hand_to_relative_pose, stow_arm, open_gripper
from spot_utils.gemini_utils import get_pixel_from_gemini
from spot_utils.pretrained_model_interface import GoogleGeminiVLM
from open_drawer import pixels_to_vision_points
import json

prompt_get_container_center = """
You are looking down at a container (bucket or box) from above.
Find the center point of the container's opening (the center of the top rim/circle/rectangle).
Rules:
- The container is directly in front of the robot.
- Ignore the robot's gripper, floor, walls, and any objects outside the container.
- The point must be at the visual center of the container opening.
- For circular containers, return the center of the circle.
- For rectangular containers, return the center of the rectangle.
Output: [{"point": [y, x], "label": "container_center"}] with coordinates normalized to 0-1000.
"""

prompt_check_visibility = """
You are looking down at a container (bucket or box) from above.
Determine if you can clearly see the ENTIRE interior contents of the container and identify which parts are cut off.

CRITICAL REQUIREMENTS for "fully_visible" to be true:
1. You can see the BOTTOM of the container clearly - the bottom surface/base is visible, not cut off
2. You can see the TOP rim/opening of the container clearly - not cut off
3. You can see the LEFT side of the container opening clearly - not cut off
4. You can see the RIGHT side of the container opening clearly - not cut off
5. You can see ALL objects/items inside the container completely

Check each edge of the container opening:
- Is the BOTTOM of the container (the bottom surface/base inside) visible or cut off?
- Is the TOP of the container (the top rim/opening) visible or cut off?
- Is the LEFT side of the container opening visible or cut off?
- Is the RIGHT side of the container opening visible or cut off?

Output JSON with:
- "fully_visible": true ONLY if the entire container interior is visible including bottom, top, left, and right edges, false otherwise
- "bottom_cut_off": true if the bottom of the container is cut off or not visible, false otherwise
- "top_cut_off": true if the top rim/opening is cut off at the top edge of the image, false otherwise
- "left_cut_off": true if the left side of the container is cut off at the left edge of the image, false otherwise
- "right_cut_off": true if the right side of the container is cut off at the right edge of the image, false otherwise
- "reason": brief explanation of what's cut off (e.g., "bottom is cut off", "right side is cut off", "top and bottom are cut off", etc.)
- "needs_adjustment": true if any part is cut off or not fully visible

Output format: {"fully_visible": true/false, "bottom_cut_off": true/false, "top_cut_off": true/false, "left_cut_off": true/false, "right_cut_off": true/false, "reason": "string", "needs_adjustment": true/false}
"""

prompt_get_adjustment = """
You are looking down at a container from above. You CANNOT clearly see inside the container to view its contents.

Your goal: Adjust the camera position AND angle to get a better view INSIDE the container so you can see the bottom and all contents.

CRITICAL: If the bottom of the container is cut off or not visible:
1. INCREASE pitch to 90 degrees (straight down) - this allows you to look directly into the container
2. MOVE BACK (negative shift_y) - moving the camera away from the container helps see deeper inside
3. These two actions together maximize your ability to see the bottom and contents

Analyze the current view:
- Is the bottom of the container cut off or not visible? → Increase pitch to 90 degrees AND move back
- Are container walls blocking the view of bottom? → Increase pitch significantly (aim for 90 degrees) AND move back slightly
- Is the container opening centered in the frame? If not, shift to center it.
- Can you see down into the container? If not:
  * INCREASE pitch to look more straight down (up to 90 degrees)
  * MOVE BACK (negative shift_y) to increase viewing distance into container
- Are contents partially hidden? Shift to reveal hidden areas.
- Is the container cut off at image edges? Shift away from that edge.

Movement rules:
- "shift_x": positive = move camera RIGHT (to see more of LEFT side/contents), negative = move LEFT (to see more of RIGHT side)
- "shift_y": positive = move camera FORWARD/away from robot (to see more of BACK of container), negative = move BACK/toward robot (to see more of FRONT and deeper into container)
- "pitch_adjustment": adjustment to pitch angle in degrees
  * Positive = increase pitch (look MORE straight down, deeper into container)
  * Negative = decrease pitch (look less down, more horizontal)
  * Range: -10 to +30 degrees (can go up to 90 degrees total)
  * If bottom is cut off or not visible: use +15 to +30 degrees (aim for 90 degrees total)
  * If walls are blocking view: use +10 to +20 degrees
  * If already at 90 degrees, focus on shift_x/shift_y and shift_y (moving back)

Values: -1.0 to 1.0 for shift_x/shift_y
  - 0.1-0.3 = small adjustment (slightly off-center but mostly visible)
  - 0.4-0.6 = medium adjustment (container partially obscured or significantly off-center)
  - 0.7-1.0 = large adjustment (container cut off, major obstruction, or can't see inside at all)
  - For bottom cut off: use shift_y = -0.3 to -0.6 (move back to see deeper)

Priority (in order):
1. If bottom is cut off or not visible → pitch_adjustment = +15 to +30 degrees AND shift_y = -0.3 to -0.6 (move back)
2. If walls are blocking view of bottom → pitch_adjustment = +10 to +20 degrees AND shift_y = -0.2 to -0.4 (move back)
3. If container is off-center → shift to center (shift_x)
4. If can't see contents → combine pitch increase with moving back and centering

Output format: {"shift_x": float, "shift_y": float, "pitch_adjustment": float, "explanation": "detailed reason for this adjustment"}
"""


def _find_valid_depth_pixel(
    pixel: tuple[int, int],
    depth_image: np.ndarray,
    max_radius: int = 10,
) -> Optional[tuple[int, int]]:
    """Find a pixel with valid depth near the given pixel."""
    x, y = pixel
    height, width = depth_image.shape
    for radius in range(max_radius + 1):
        for dy in range(-radius, radius + 1):
            ny = y + dy
            if ny < 0 or ny >= height:
                continue
            for dx in range(-radius, radius + 1):
                nx = x + dx
                if nx < 0 or nx >= width:
                    continue
                depth_val = float(depth_image[ny, nx])
                if depth_image.dtype == np.uint16:
                    depth_val = depth_val / 1000.0
                if depth_val > 0.01 and depth_val < 3.0:
                    return (nx, ny)
    return None

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
    forward_offset: float = 0.8,
    vertical_offset: float = 0.6,
    pitch_deg: float = 85.0,
    settle_seconds: float = 1.5,
    image_basename: str = "container_inspection",
    stow_after: bool = True,
    open_before_capture: bool = True,
    center_on_container: bool = True,
) -> Image.Image:
    
    if center_on_container:
        print("→ Capturing initial image to find container center...")
        initial_pose = _build_look_pose(forward_offset, vertical_offset, pitch_deg)
        move_hand_to_relative_pose(robot, initial_pose)
        time.sleep(settle_seconds)
        
        if open_before_capture:
            open_gripper(robot)
            time.sleep(0.3)
        
        rgbds = capture_images(robot, localizer, camera_names=["hand_color_image"])
        rgbd = rgbds["hand_color_image"]
        pil_image = Image.fromarray(rgbd.rgb)
        
        print("→ Finding container center...")
        try:
            center_points = get_pixel_from_gemini(
                pil_image,
                prompt_get_container_center,
                num_points=1,
            )
            if center_points:
                center_point = center_points[0]
                pixel_y, pixel_x = center_point["point"]
                
                print(f"→ Found container center at pixel ({pixel_x}, {pixel_y})")
                
                height, width = rgbd.depth.shape
                pixel_x_norm = pixel_x / 1000.0 * width
                pixel_y_norm = pixel_y / 1000.0 * height
                pixel = (int(pixel_x_norm), int(pixel_y_norm))
                
                valid_pixel = _find_valid_depth_pixel(pixel, rgbd.depth)
                if valid_pixel:
                    pixel = valid_pixel
                
                vision_points = pixels_to_vision_points(
                    rgbd, [(pixel[1], pixel[0])]
                )
                if vision_points:
                    vision_point = vision_points[0]
                    body_T_vision = get_a_tform_b(
                        robot.ensure_client(RobotStateClient.default_service_name)
                        .get_robot_state()
                        .kinematic_state.transforms_snapshot,
                        BODY_FRAME_NAME,
                        VISION_FRAME_NAME,
                    )
                    body_point = body_T_vision * vision_point
                    
                    print(f"→ Container center in body frame: x={body_point.x:.3f}, y={body_point.y:.3f}, z={body_point.z:.3f}")
                    forward_offset = body_point.x
                    y_offset = body_point.y
                else:
                    print("→ Could not get 3D point, using default forward_offset")
                    y_offset = 0.0
            else:
                print("→ Could not find container center, using default position")
                y_offset = 0.0
        except Exception as exc:
            print(f"→ Warning: Could not find container center: {exc}")
            print("→ Using default position")
            y_offset = 0.0
    else:
        y_offset = 0.0
    
    print(f"→ Positioning arm high above container (z={vertical_offset:.2f}m, pitch={pitch_deg:.1f}deg)...")
    look_pose = _build_look_pose(forward_offset, vertical_offset, pitch_deg)
    look_pose = math_helpers.SE3Pose(
        x=look_pose.x,
        y=y_offset,
        z=look_pose.z,
        rot=look_pose.rot,
    )
    move_hand_to_relative_pose(robot, look_pose)
    time.sleep(settle_seconds)

    if open_before_capture:
        open_gripper(robot)
        time.sleep(0.3)

    print("→ Capturing final image...")
    rgbds_final = capture_images(robot, localizer, camera_names=["hand_color_image"])
    rgbd_final = rgbds_final["hand_color_image"]
    pil_final = Image.fromarray(rgbd_final.rgb)
    image_path = Path(f"{image_basename}.jpg")
    pil_final.save(image_path)
    print(f"✓ Saved container inspection image to {image_path}")

    if stow_after:
        stow_arm(robot)

    return pil_final


def main() -> None:
    parser = argparse.ArgumentParser(description="Look into a container with Spot's hand camera")
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--map_name", required=True, help="GraphNav map name under spot_utils/graph_nav_maps")
    parser.add_argument("--forward_offset", type=float, default=0.8)
    parser.add_argument("--vertical_offset", type=float, default=0.6)
    parser.add_argument("--pitch_deg", type=float, default=85.0, help="Downward pitch angle in degrees")
    parser.add_argument("--settle_seconds", type=float, default=1.5)
    parser.add_argument("--image_basename", default="container_inspection")
    parser.add_argument("--no_stow", action="store_true", help="Keep the arm deployed after capturing")
    parser.add_argument("--no_open_gripper", action="store_true", help="Skip opening the gripper before imaging")
    parser.add_argument("--no_center", action="store_true", help="Disable automatic container centering")
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
        image = look_into_container(
            robot,
            localizer,
            forward_offset=args.forward_offset,
            vertical_offset=args.vertical_offset,
            pitch_deg=args.pitch_deg,
            settle_seconds=args.settle_seconds,
            image_basename=args.image_basename,
            stow_after=not args.no_stow,
            open_before_capture=not args.no_open_gripper,
            center_on_container=not args.no_center,
        )
        image_path = Path(f"{args.image_basename}.jpg")
        print(f"Container inspection image captured and saved to {image_path}")
        print(f"Image size: {image.size[0]}x{image.size[1]} pixels")
    finally:
        pass


if __name__ == "__main__":
    main()
