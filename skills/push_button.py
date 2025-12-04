"""Skill to identify a button and push it.

This module reuses existing perception and manipulation helpers. It supports
three selection modes for the button location:
1) VLM-based bounding box (Gemini) → center pixel
2) GroundedSAM endpoint → center pixel from mask
3) Manual click from user
"""

from typing import Optional, Literal, Tuple, List

import json
import numpy as np
from PIL import Image
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot
import cv2
import os
from datetime import datetime
from spot_utils.spot_localization import SpotLocalizer
from spot_utils.perception.spot_cameras import capture_images
from skills.spot_hand_move import (
    move_hand_to_relative_pose,
    move_hand_to_relative_pose_with_velocity,
    stow_arm,
    open_gripper,
    close_gripper,
)
from spot_utils.pretrained_model_interface import GoogleGeminiVLM
from spot_utils.utils import verify_estop, get_graph_nav_dir
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.util import authenticate
from bosdyn.client import create_standard_sdk
import rerun as rr
import argparse

def init_robot(hostname: str, map_name: str) -> tuple[Robot, LeaseClient, LeaseKeepAlive, SpotLocalizer]:
    sdk = create_standard_sdk("WipeOnlineClient")
    robot = sdk.create_robot(hostname)
    authenticate(robot)
    verify_estop(robot)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    lease_keepalive = LeaseKeepAlive(
        lease_client, must_acquire=True, return_at_exit=True
    )
    robot.time_sync.wait_for_sync()
    
    # Initialize localizer
    path = get_graph_nav_dir(map_name)
    localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
    localizer.localize()
    print("[INFO] Localization successful.")
    
    return robot, lease_client, lease_keepalive, localizer

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

def gaze(robot, direction: str) -> None:
    """Move the hand to look in a certain direction."""
    look_pose = direction_to_pose[direction]
    move_hand_to_relative_pose(robot, look_pose)
    open_gripper(robot)


def get_points_from_pixels(rgb_image_path, depth_image_path, intrinsics):
    rgb = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    if rgb is None:
        raise FileNotFoundError(f"Could not read RGB image at: {rgb_image_path}")
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image at: {depth_image_path}")

    # Ensure single-channel depth
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    # Convert depth to meters if given as uint16 millimeters
    if depth.dtype == np.uint16:
        depth_m = depth.astype(np.float32) / 1000.0
    else:
        depth_m = depth.astype(np.float32)

    h, w = depth_m.shape
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)

    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

    # Create pixel grid
    u_coords, v_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    z = depth_m
    valid = (z > 0) & (z <= 1.5)

    x = (u_coords - cx) / fx * z
    y = (v_coords - cy) / fy * z

    # Stack and mask
    points = np.stack((x, y, z), axis=-1)[valid]
    if points.shape[0] == 0:
        print("No points passed the depth filter! Check depth image units and max distance.")

    # Colors: convert BGR (cv2) to RGB and normalize to [0,1]
    rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    colors = (rgb_rgb.reshape(-1, 3)[valid.ravel()] / 255.0).astype(np.float32)
    print("positions:", points.shape, points.dtype)
    print("colors:", colors.shape, colors.dtype)
    return points, colors


def get_multiple_pixels_from_gemini(
    vlm_query_str: str, pil_image: Image.Image, num_pixels: int = 15
) -> List[Tuple[int, int]]:
    """Query Gemini VLM to return multiple pixels on the target object.

    Expects the model to respond with a JSON array like:
    [ {"point": [y, x]}, ... ] with coordinates normalized to [0, 1000].
    """
    vlm = GoogleGeminiVLM("gemini-2.5-pro")

    def parse_json_output(json_output_str: str) -> str:
        lines = json_output_str.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                json_output_str = "\n".join(lines[i + 1 :])
                json_output_str = json_output_str.split("```")[0]
                break
        json_output_str = json_output_str.strip()
        return json_output_str

    vlm_output_list = vlm.sample_completions(
        prompt=vlm_query_str,
        imgs=[pil_image],
        temperature=0.0,
        seed=42,
        num_completions=1,
    )
    vlm_output_str = vlm_output_list[0]
    json_string_to_parse = parse_json_output(vlm_output_str)
    parsed_data = json.loads(json_string_to_parse)

    if not isinstance(parsed_data, list) or not parsed_data:
        raise ValueError("Parsed JSON is not a non-empty list.")
    # if len(parsed_data) < num_pixels:
    #     raise ValueError(f"Parsed JSON has less than {num_pixels} points.")

    pixels: List[Tuple[int, int]] = []
    for point_obj in parsed_data[:num_pixels]:
        if (
            "point" not in point_obj
            or not isinstance(point_obj["point"], list)
            or len(point_obj["point"]) != 2
        ):
            raise ValueError(
                "Some element in JSON does not contain a valid 'point' list [y, x]."
            )
        y_norm, x_norm = point_obj["point"]
        if not isinstance(y_norm, (int, float)) or not isinstance(x_norm, (int, float)):
            raise ValueError("Normalized coordinates are not numbers.")
        img_height = pil_image.height
        img_width = pil_image.width
        y = int(y_norm * img_height / 1000.0)
        x = int(x_norm * img_width / 1000.0)
        y = max(0, min(y, img_height - 1))
        x = max(0, min(x, img_width - 1))
        pixels.append((x, y))

    return pixels


def _pixel_to_body_xyz(u: int, v: int, rgbd, intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
    """Back-project pixel (u,v) to BODY frame using supplied intrinsics."""
    from bosdyn.client.frame_helpers import (
        BODY_FRAME_NAME,
        VISION_FRAME_NAME,
        get_a_tform_b,
    )

    depth = rgbd.depth
    depth_m = (
        depth.astype(np.float32) / 1000.0 if depth.dtype == np.uint16 else depth.astype(np.float32)
    )
    if v < 0 or v >= depth_m.shape[0] or u < 0 or u >= depth_m.shape[1]:
        raise ValueError("Pixel out of bounds")
    z = float(depth_m[v, u])
    if not np.isfinite(z) or z <= 0:
        win = 3
        v0, v1 = max(0, v - win), min(depth_m.shape[0], v + win + 1)
        u0, u1 = max(0, u - win), min(depth_m.shape[1], u + win + 1)
        patch = depth_m[v0:v1, u0:u1]
        vals = patch[np.isfinite(patch) & (patch > 0)]
        if vals.size == 0:
            raise RuntimeError("No valid depth near pixel")
        z = float(np.median(vals))

    fx, fy, cx, cy = intrinsics
    x_cam = (u - cx) / fx * z
    y_cam = (v - cy) / fy * z
    p_cam_h = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)

    T_vision_cam = get_a_tform_b(
        rgbd.transforms_snapshot, VISION_FRAME_NAME, rgbd.frame_name_image_sensor
    ).to_matrix()
    T_body_vision = get_a_tform_b(
        rgbd.transforms_snapshot, BODY_FRAME_NAME, VISION_FRAME_NAME
    ).to_matrix()
    return (T_body_vision @ (T_vision_cam @ p_cam_h))[:3]


def push_button(
    robot: Robot,
    localizer: SpotLocalizer,
    label: str = "button",
    surface: Literal["vertical", "horizontal"] = "horizontal",
    z_clearance: float = 0.03,
    press_depth: float = 0.025,
    press_duration: float = 0.7,
) -> None:
    """Identify a button and push on it.

    Args:
        robot: Spot robot handle
        localizer: Localizer for camera capture
        label: Target label to find (default: "button")
        surface: "vertical" to push forward, "horizontal" to push downward
        z_clearance: Approach standoff distance in meters
        press_depth: Linear press distance (m)
        press_duration: Duration for press motion (s)
    """
    # 1) Prepare and capture
    stow_arm(robot)

    gaze(robot, "DOWN") ## this looks straight down at the button 
    ## this behavior changes if the button is on the wall

    rgbd = capture_images(robot, localizer, camera_names=["hand_color_image"])  # type: ignore
    rgbd = rgbd["hand_color_image"]
    rgb = rgbd.rgb
    depth = rgbd.depth
    pil = Image.fromarray(rgb)

    rr.log("image/rgb", rr.Image(rgb))
    rr.log("image/depth", rr.Image(depth))

    # get the 3D points from the rgb and depth images, and the camera intrinsics
    cam_model = rgbd.camera_model  

    fx = cam_model.intrinsics.focal_length.x
    fy = cam_model.intrinsics.focal_length.y
    cx = cam_model.intrinsics.principal_point.x
    cy = cam_model.intrinsics.principal_point.y

    intrinsics = [fx, fy, cx, cy]

    save_folderpath = "push_button_images"
    os.makedirs(save_folderpath, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ## save the rgb and the depth image to the disk
    rgb_pil = Image.fromarray(rgb)
    depth_pil = Image.fromarray(depth)
    rgb_pil.save(os.path.join(save_folderpath, f"rgb_{timestamp}.png"))
    depth_pil.save(os.path.join(save_folderpath, f"depth_{timestamp}.png"))

    points, colors = get_points_from_pixels(os.path.join(save_folderpath, f"rgb_{timestamp}.png"), os.path.join(save_folderpath, f"depth_{timestamp}.png"), intrinsics)
    rr.log("pcd", rr.Points3D(positions=points, colors=colors, radii=0.01))
    num_points = 10

    vlm_query = f"""
    Point up to {num_points} points on the {label} in the image. Return a JSON list like [{{"point": [y, x]}}, ...] with coordinates normalized to 0-1000.
    """
    # 2) Select pixel(s) via VLM only
    pixels = get_multiple_pixels_from_gemini(vlm_query, pil, num_pixels=num_points)

    # rr.log("pixels", rr.Points2D(pixels=pixels))

    # 3) Back-project all pixels to BODY frame (filter invalid)
    pts_body: List[np.ndarray] = []
    for (u_px, v_px) in pixels:
        try:
            p = _pixel_to_body_xyz(int(u_px), int(v_px), rgbd, (fx, fy, cx, cy))
            if np.all(np.isfinite(p)):
                pts_body.append(np.asarray(p, dtype=np.float64))
        except Exception:
            continue

    # If insufficient valid 3D points, fall back to mean pixel back-projection
    if len(pts_body) < 3:
        u_mean = int(round(np.mean([u for (u, _) in pixels])))
        v_mean = int(round(np.mean([v for (_, v) in pixels])))
        p = _pixel_to_body_xyz(u_mean, v_mean, rgbd, (fx, fy, cx, cy))
        pts_body = [np.asarray(p, dtype=np.float64)]

    P = np.vstack(pts_body)

    def fit_plane_to_points(points_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert points_body.shape[1] == 3
        centroid = np.mean(points_body, axis=0)
        Q = points_body - centroid
        # SVD for plane normal (smallest singular vector)
        _, _, vh = np.linalg.svd(Q, full_matrices=False)
        normal = vh[-1, :]
        normal_norm = np.linalg.norm(normal)
        normal = normal / normal_norm if normal_norm > 1e-9 else np.array([1.0, 0.0, 0.0])
        # Ensure forward-facing (+x in BODY)
        if normal[0] < 0:
            normal = -normal
        return centroid, normal

    if P.shape[0] >= 3:
        centroid_body, normal_body = fit_plane_to_points(P)
    else:
        centroid_body = P[0]
        normal_body = np.array([1.0, 0.0, 0.0])

    # 4) Build approach/press poses.
    # For horizontal surfaces, use a "tip-down" orientation (hand looking down)
    # and press along the body Z axis so the tip of the hand makes contact.
    if surface == "horizontal":
        tip_down_rot = DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE.rot
        approach = math_helpers.SE3Pose(
            x=float(centroid_body[0]),
            y=float(centroid_body[1]),
            z=float(centroid_body[2] + z_clearance),
            rot=tip_down_rot,
        )
        press = math_helpers.SE3Pose(
            x=approach.x,
            y=approach.y,
            z=approach.z - press_depth,
            rot=tip_down_rot,
        )
    else:
        # Preserve the previous normal-based behavior for vertical surfaces:
        # align the hand's forward axis with the plane normal and press along it.
        approach = math_helpers.SE3Pose(
            x=float(centroid_body[0] - z_clearance * normal_body[0]),
            y=float(centroid_body[1] - z_clearance * normal_body[1]),
            z=float(centroid_body[2] - z_clearance * normal_body[2]),
            rot=math_helpers.Quat(),  # temporary; set below
        )

        press = math_helpers.SE3Pose(
            x=float(approach.x + press_depth * normal_body[0]),
            y=float(approach.y + press_depth * normal_body[1]),
            z=float(approach.z + press_depth * normal_body[2]),
            rot=math_helpers.Quat(),
        )

        # Orient hand so its forward axis aligns with normal (yaw+pitch approximation)
        nx, ny, nz = normal_body
        yaw = float(np.arctan2(ny, nx))
        hyp = float(np.sqrt(nx * nx + ny * ny))
        pitch = float(-np.arctan2(nz, max(hyp, 1e-9)))
        rot = math_helpers.Quat.from_yaw(yaw) * math_helpers.Quat.from_pitch(pitch)
        approach = math_helpers.SE3Pose(x=approach.x, y=approach.y, z=approach.z, rot=rot)
        press = math_helpers.SE3Pose(x=press.x, y=press.y, z=press.z, rot=rot)

    ## close the gripper 
    close_gripper(robot)

    # 5) Execute approach, press, retreat, and stow
    move_hand_to_relative_pose(robot, approach)
    move_hand_to_relative_pose_with_velocity(robot, approach, press, press_duration)
    move_hand_to_relative_pose_with_velocity(robot, press, approach, press_duration)
    stow_arm(robot)

def main():
    parser = argparse.ArgumentParser(description="Online wiping controller.")
    parser.add_argument(
        "--hostname",
        type=str,
        required=True,
        help="Spot hostname/IP (e.g., 192.168.80.3)",
    )
    parser.add_argument(
        "--map_name",
        type=str,
        required=True,
        help="The name of the map folder to load (sub-folder under graph_nav_maps)",
    )
    args = parser.parse_args()
    robot, lease_client, lease_keepalive, localizer = init_robot(args.hostname, args.map_name)
    rr.init("push_button", spawn=True)
    push_button(robot, localizer)

if __name__ == "__main__":
    main()