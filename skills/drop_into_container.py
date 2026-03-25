"""Skill for dropping a grasped object into a container using Spot's arm."""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rerun as rr
from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.sdk import Robot
from bosdyn.client.util import authenticate
from PIL import Image

from calibrate_iphone import rgbd_to_point_cloud
from iphone_streaming import get_latest_frame
from skills.spot_hand_move import (
    move_hand_to_relative_pose,
    open_gripper,
    stow_arm,
)
from spot_utils.perception.spot_cameras import _image_response_to_image
from spot_utils.pretrained_model_interface import GoogleGeminiVLM
from spot_utils.utils import verify_estop

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

DEFAULT_SPOT_HAND_CAMERA_NAME = "hand_color_image"
DEFAULT_IPHONE_EXTRINSICS_PATH = str((Path(__file__).resolve().parents[1] / "iphone_extrinsics.json"))


def gaze_without_open(robot, direction: str) -> None:
    """Move the hand to look in a certain direction without opening the gripper."""
    look_pose = direction_to_pose[direction]
    move_hand_to_relative_pose(robot, look_pose)


def init_robot(hostname: str, map_name: str) -> tuple[Robot, LeaseClient, LeaseKeepAlive]:
    """Initialize and authenticate the robot, returning the robot and lease handles."""
    sdk = create_standard_sdk("WipeOnlineClient")
    robot = sdk.create_robot(hostname)
    authenticate(robot)
    verify_estop(robot)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    lease_keepalive = LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)
    robot.time_sync.wait_for_sync()

    print("[INFO] Robot connection and time sync successful.")
    return robot, lease_client, lease_keepalive


def _load_T_hand_iphone(extrinsics_path: str) -> np.ndarray:
    """Load T_hand_iphone (hand-camera<-iphone) from JSON."""
    with open(extrinsics_path, "r") as f:
        extr = json.load(f)
    if "T_hand_iphone" not in extr:
        raise KeyError(f"Extrinsics JSON must contain key 'T_hand_iphone': {extrinsics_path}")
    T_hand_iphone = np.array(extr["T_hand_iphone"], dtype=np.float64)
    if T_hand_iphone.shape != (4, 4):
        raise ValueError(f"T_hand_iphone must be 4x4, got shape {T_hand_iphone.shape} in {extrinsics_path}")
    return T_hand_iphone


def _get_T_body_hand_camera(robot: Robot, hand_camera_name: str = DEFAULT_SPOT_HAND_CAMERA_NAME) -> np.ndarray:
    """Fetch the current BODY->hand camera transform using the snapshot attached to a hand-camera image."""
    image_client = robot.ensure_client(ImageClient.default_service_name)
    rgb_req = build_image_request(hand_camera_name, quality_percent=100, pixel_format=None)
    responses = image_client.get_image([rgb_req])
    if not responses:
        raise RuntimeError(f"No image responses returned for camera '{hand_camera_name}'.")
    resp = responses[0]
    _ = _image_response_to_image(resp)

    T_body_hand = get_a_tform_b(
        resp.shot.transforms_snapshot,
        BODY_FRAME_NAME,
        resp.shot.frame_name_image_sensor,
    )
    if T_body_hand is None:
        raise RuntimeError(
            f"Could not compute BODY->hand camera transform from image snapshot. "
            f"camera_name={hand_camera_name}, sensor_frame={resp.shot.frame_name_image_sensor}"
        )
    return np.asarray(T_body_hand.to_matrix(), dtype=np.float64)


def _iphone_pixel_to_body_xyz(
    u: int,
    v: int,
    depth_m: np.ndarray,
    K_iphone: np.ndarray,
    T_body_iphone: np.ndarray,
) -> np.ndarray:
    """Back-project an iPhone pixel (u, v) to BODY frame using depth, intrinsics, and T_body_iphone."""
    H, W = depth_m.shape
    if v < 0 or v >= H or u < 0 or u >= W:
        raise ValueError("Pixel out of bounds")

    z = float(depth_m[v, u])
    if not np.isfinite(z) or z <= 0:
        win = 3
        v0, v1 = max(0, v - win), min(H, v + win + 1)
        u0, u1 = max(0, u - win), min(W, u + win + 1)
        patch = depth_m[v0:v1, u0:u1]
        vals = patch[np.isfinite(patch) & (patch > 0)]
        if vals.size == 0:
            raise RuntimeError("No valid depth near pixel")
        z = float(np.median(vals))

    fx, fy = float(K_iphone[0, 0]), float(K_iphone[1, 1])
    cx, cy = float(K_iphone[0, 2]), float(K_iphone[1, 2])

    x_cam = (u - cx) / fx * z
    y_cam = (v - cy) / fy * z
    p_cam_h = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)

    p_body = (T_body_iphone @ p_cam_h)[:3]
    return p_body


DEFAULT_PLACE_VLM_QUERY_TEMPLATE = (
    "You are given an image of a surface. Return a point on the surface that's away from the walls of the container where an object can be placed."
    "OUTPUT FORMAT (return EXACTLY one JSON object in the FORMAT below and NOTHING ELSE):\n"
    '{"point": [y, x], "label": "open_container_region"}. '
    "Coordinates MUST be normalized to 0-1000.\n"
)


def _parse_point_json(raw: str) -> Optional[Tuple[float, float]]:
    """Parse {"point": [y,x] | null} from Gemini output.
    Robust to:
      - ```json fences
      - stray text
      - malformed JSON (fallback regex)
    Returns (y_norm, x_norm) or None.
    """
    print(raw)
    s = raw.strip()

    # Strip fenced code blocks
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 2:
            block = parts[1]
            if block.startswith("json\n"):
                block = "\n".join(block.splitlines()[1:])
            s = block.strip()

    # Extract first JSON-like object
    left, right = s.find("{"), s.rfind("}")
    s_obj = s[left:right + 1] if (left != -1 and right != -1 and right > left) else s

    # Try strict JSON
    try:
        obj = json.loads(s_obj)
        if not isinstance(obj, dict):
            raise ValueError("Expected JSON object.")

        if "point" in obj:
            p = obj["point"]
            if p is None:
                return None
            if not (isinstance(p, list) and len(p) == 2):
                raise ValueError("'point' must be [y, x] or null.")
            return (float(p[0]), float(p[1]))

        # Backwards compatibility if model returns {"points":[[y,x],...]}
        if "points" in obj:
            pts = obj["points"]
            if pts is None:
                return None
            if isinstance(pts, list) and len(pts) > 0:
                # Flatten [[[y,x],...]] -> [[y,x],...]
                if isinstance(pts[0], list) and len(pts) == 1 and len(pts[0]) > 0 and isinstance(pts[0][0], list):
                    pts = pts[0]
                for p in pts:
                    if isinstance(p, list) and len(p) == 2:
                        return (float(p[0]), float(p[1]))
            raise ValueError("'points' present but did not contain a valid [y,x] pair.")

        raise ValueError("Expected key 'point' (preferred) or 'points' (fallback).")

    except Exception:
        # Fallback: regex any [y,x] pair (first match)
        m = re.search(r"\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]", s_obj)
        if not m:
            raise ValueError(f"Could not parse JSON or find a [y,x] pair in:\n{s_obj}")
        return (float(m.group(1)), float(m.group(2)))


def get_open_table_point_from_gemini(
    vlm_query_str: str,
    pil_image: Image.Image,
    model_name: str = "gemini-2.5-pro",
) -> Optional[Tuple[int, int]]:
    """Returns ONE point in RGB pixel coords: (v, u) where v=row(y), u=col(x).
    Gemini returns normalized (0-1000) [y,x].
    """
    vlm = GoogleGeminiVLM(model_name)
    outputs = vlm.sample_completions(
        prompt=vlm_query_str,
        imgs=[pil_image],
        temperature=0.0,
        seed=42,
        num_completions=1,
    )
    s = outputs[0]
    p_norm = _parse_point_json(s)
    if p_norm is None:
        return None

    y_n, x_n = p_norm
    H, W = pil_image.height, pil_image.width
    v = int(round(y_n * H / 1000.0))
    u = int(round(x_n * W / 1000.0))
    v = max(0, min(v, H - 1))
    u = max(0, min(u, W - 1))
    return (v, u)


def compute_place_pose_from_iphone_pixel(
    u: int,
    v: int,
    depth_m: np.ndarray,
    K_iphone: np.ndarray,
    T_body_iphone: np.ndarray,
    z_clearance_m: float = 0.10,
    pitch_down: float = np.pi / 2,
) -> math_helpers.SE3Pose:
    """Backproject (u,v) from iPhone depth into BODY, then produce a pose above it."""
    p_body = _iphone_pixel_to_body_xyz(u, v, depth_m, K_iphone, T_body_iphone)
    return math_helpers.SE3Pose(
        x=float(p_body[0]),
        y=float(p_body[1]),
        z=float(p_body[2] + z_clearance_m),
        rot=math_helpers.Quat.from_pitch(pitch_down),
    )


def _rr_log_points2d(name: str, pts_vu: List[Tuple[int, int]], radii: float = 2.0) -> None:
    """pts_vu is [(v,u), ...]; Rerun Points2D expects (x,y)=(u,v)."""
    if not pts_vu:
        return
    xy = np.array([[u, v] for (v, u) in pts_vu], dtype=np.float32)
    rr.log(name, rr.Points2D(xy, radii=radii))


def _rr_log_pose_point3d(name: str, pose: math_helpers.SE3Pose, radii: float = 0.02) -> None:
    rr.log(
        name,
        rr.Points3D(
            positions=np.array([[pose.x, pose.y, pose.z]], dtype=np.float32),
            radii=radii,
        ),
    )


def drop_into_container(
    robot,
    iphone_extrinsics_path: str = DEFAULT_IPHONE_EXTRINSICS_PATH,
    vlm_query_template: str = DEFAULT_PLACE_VLM_QUERY_TEMPLATE,
    z_above_surface_m: float = 0.3,
    save_debug_images: bool = True,
) -> None:
    """Drop an object into a container.

    1) Move arm to look pose
    2) Capture iPhone RGBD
    3) Ask Gemini for ONE open-surface point
    4) Backproject to BODY
    5) Move above that point
    6) Open gripper
    """
    rr.init("drop_into_container_skill", spawn=True)

    # 1) Move arm so iPhone can see the container clearly
    gaze_without_open(robot, "DOWN")

    # 2) Receive RGBD from iPhone using the shared streaming process
    # Get the latest frame from the streaming process (must be started before running this skill)
    time.sleep(0.5)
    frame = get_latest_frame()
    if frame is None:
        raise RuntimeError("No iPhone frame received yet. Ensure iPhone is streaming.")

    rgb_img = frame.rgb
    depth_img = frame.depth
    if depth_img is None:
        raise RuntimeError("iPhone depth image is missing; cannot place safely.")
    K_full = np.asarray(frame.intrinsics, dtype=np.float32)  # intrinsics at RGB resolution

    H_rgb, W_rgb = rgb_img.shape[:2]
    H_d, W_d = depth_img.shape[:2]
    scale_x = W_d / float(W_rgb)
    scale_y = H_d / float(H_rgb)

    # Intrinsics scaled to DEPTH resolution
    K_iphone = K_full.copy()
    K_iphone[0, 0] *= scale_x
    K_iphone[1, 1] *= scale_y
    K_iphone[0, 2] *= scale_x
    K_iphone[1, 2] *= scale_y

    save_folderpath = "place_at_images_iphone"
    os.makedirs(save_folderpath, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rgb_pil = Image.fromarray(rgb_img)
    rgb_image_path = os.path.join(save_folderpath, f"rgb_{timestamp}.png")
    depth_image_path = os.path.join(save_folderpath, f"depth_{timestamp}.npy")
    intrinsics_path = os.path.join(save_folderpath, f"intrinsics_{timestamp}.json")

    rgb_pil.save(rgb_image_path)
    np.save(depth_image_path, depth_img.astype(np.float32))
    with open(intrinsics_path, "w") as f:
        json.dump(
            {
                "K_rgb": K_full.tolist(),
                "K_depth": K_iphone.tolist(),
                "width": int(W_rgb),
                "height": int(H_rgb),
            },
            f,
            indent=2,
        )

    # Log images to Rerun
    rr.log("place_at/iphone/rgb", rr.Image(rgb_img))
    rr.log("place_at/iphone/depth", rr.Image(depth_img.astype(np.float32)))

    # 3) Compose T_body_iphone (BODY <- iPhone camera)
    T_hand_iphone = _load_T_hand_iphone(iphone_extrinsics_path)
    T_body_hand = _get_T_body_hand_camera(robot, DEFAULT_SPOT_HAND_CAMERA_NAME)
    T_body_iphone = (T_body_hand @ T_hand_iphone).astype(np.float64)

    # Build point cloud (camera frame) + transform to BODY for viz
    points_cam, colors = rgbd_to_point_cloud(rgb_img, depth_img, K_iphone)
    points_cam = points_cam.astype(np.float32)
    num_pts = points_cam.shape[0]
    if num_pts > 0:
        points_cam_h = np.concatenate([points_cam, np.ones((num_pts, 1), dtype=np.float32)], axis=1)
        points_body_h = (T_body_iphone @ points_cam_h.T).T
        points_body = points_body_h[:, :3].astype(np.float32)
    else:
        points_body = points_cam

    voxel_size = 0.005
    rr.log(
        "scene/points3d_body",
        rr.Points3D(positions=points_body, colors=colors, radii=voxel_size / 2),
    )

    # 4) Ask Gemini for ONE open-surface point (in RGB pixel coords)
    p_rgb = get_open_table_point_from_gemini(vlm_query_template, rgb_pil)
    if p_rgb is None:
        stow_arm(robot)
        raise RuntimeError("Gemini returned no placement point; aborting place.")

    v_r, u_r = p_rgb

    # Log the chosen 2D point (separate entity for visibility)
    _rr_log_points2d("place_at/iphone/rgb/placement_point", [(v_r, u_r)], radii=10.0)

    # 5) Convert RGB pixel -> depth pixel
    v_d = int(round(v_r * scale_y))
    u_d = int(round(u_r * scale_x))
    v_d = max(0, min(v_d, H_d - 1))
    u_d = max(0, min(u_d, W_d - 1))

    _rr_log_points2d("place_at/iphone/depth/placement_point", [(v_d, u_d)], radii=10.0)

    depth_m = depth_img.astype(np.float32)

    # Backproject that point to BODY and log it
    p_place_body = _iphone_pixel_to_body_xyz(u_d, v_d, depth_m, K_iphone, T_body_iphone)
    rr.log(
        "place_at/target/place_point_body",
        rr.Points3D(np.array([p_place_body], dtype=np.float32), radii=0.03),
    )

    # 6) Compute target pose above that point (BODY frame)
    above_pose = compute_place_pose_from_iphone_pixel(
        u=u_d,
        v=v_d,
        depth_m=depth_m,
        K_iphone=K_iphone,
        T_body_iphone=T_body_iphone,
        z_clearance_m=z_above_surface_m,
        pitch_down=np.pi / 2,
    )
    _rr_log_pose_point3d("place_at/target/above_pose_body", above_pose, radii=0.02)

    # 7) Move above, open gripper, stow war
    move_hand_to_relative_pose(robot, above_pose)
    open_gripper(robot)
    stow_arm(robot)

def main() -> None:
    """Parse arguments and execute the drop-into-container skill."""
    parser = argparse.ArgumentParser(description="Drop into controller.")
    parser.add_argument(
        "--hostname",
        type=str,
        required=True,
        help="Spot hostname/IP (e.g., 192.168.80.3)",
    )
    args = parser.parse_args()
    robot, _, _ = init_robot(args.hostname, "")

    drop_into_container(robot, z_above_surface_m=0.5)


if __name__ == "__main__":
    main()
