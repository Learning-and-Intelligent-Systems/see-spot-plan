import argparse
from typing import Optional
import json
import os
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
from PIL import Image
from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, VISION_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.sdk import Robot
from bosdyn.client.util import authenticate

from spot_utils.utils import verify_estop, get_graph_nav_dir
from skills.spot_navigation import navigate_to_relative_pose
from spot_utils.pretrained_model_interface import GoogleGeminiVLM
from spot_utils.perception.spot_cameras import _image_response_to_image
from skills.spot_hand_move import (
    move_hand_to_relative_pose,
    move_hand_to_relative_pose_with_velocity,
    open_gripper,
    close_gripper,
    stow_arm,
)
from calibrate_iphone import rgbd_to_point_cloud
from iphone_streaming import get_latest_frame

rr.init("erase_whiteboard", spawn=True)

def init_robot(hostname: str, map_name: str) -> tuple[Robot, LeaseClient, LeaseKeepAlive]:
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
    
    print("[INFO] Robot connection and time sync successful.")

    return robot, lease_client, lease_keepalive

DEFAULT_HAND_LOOK_FLOOR_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 3)
)

DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2)
)

DEFAULT_HAND_LOOK_AT_WALL_POSE = math_helpers.SE3Pose(
    x=0.55, y=0.0, z=0.5, rot=math_helpers.Quat.from_pitch(0)
)

direction_to_pose = {
    "DOWN": DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE,
    "AHEAD": DEFAULT_HAND_LOOK_FLOOR_POSE,
    "WALL": DEFAULT_HAND_LOOK_AT_WALL_POSE
}

DEFAULT_WIPE_ONLINE_Z_OFFSET = 0.02


DEFAULT_WIPE_VLM_QUERY_TEMPLATE = (
    "You are given an image of a wall-mounted whiteboard. Identify the entire region that contains whiteboard marker writing (ink strokes). Return a bounding box tightly enclosing all of the visible ink strokes.\n"
    "If no writing is visible or it's ambiguous, return a bbox of null.\n\n"
    "Output format (return EXACTLY one JSON object and nothing else):\n"
    '{"bbox": [ymin, xmin, ymax, xmax] | null, "label": "whiteboard_writing"}\n'
    "The bbox coordinates MUST be normalized to 0-1000 and are in [ymin, xmin, ymax, xmax] order.\n"
)

DEFAULT_SPOT_HAND_CAMERA_NAME = "hand_color_image"
DEFAULT_IPHONE_EXTRINSICS_PATH = str((Path(__file__).resolve().parents[1] / "iphone_extrinsics.json"))


def compute_wall_normal(P1: np.ndarray, P2: np.ndarray, P3: np.ndarray) -> np.ndarray:
    """Compute normalized wall normal from 3 points on the surface.

    Args:
        P1: First point on wall (e.g., bottom-right)
        P2: Second point on wall (e.g., top-right)
        P3: Third point on wall (e.g., bottom-left)

    Returns:
        Normalized wall normal pointing away from wall (toward robot)
    """
    # Compute two edge vectors lying on the wall surface
    edge_up = P2 - P1      # vertical edge
    edge_left = P3 - P1    # horizontal edge

    # Wall normal via cross product (right-hand rule)
    wall_normal = np.cross(edge_left, edge_up)
    wall_normal_norm = np.linalg.norm(wall_normal)

    if wall_normal_norm < 1e-6:
        raise RuntimeError("Cannot compute wall normal: points are collinear")

    wall_normal = wall_normal / wall_normal_norm  # normalize

    # Ensure normal points AWAY from wall (toward robot)
    # For a wall in front of the robot, normal should have negative X component
    # If positive, we computed it backward, so flip
    if wall_normal[0] > 0:
        wall_normal = -wall_normal

    return wall_normal


def project_onto_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project vector onto plane perpendicular to normal.

    Args:
        vec: Vector to project
        normal: Normal vector of the plane (should be normalized)

    Returns:
        Projected vector lying in the plane
    """
    return vec - np.dot(vec, normal) * normal


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
    """Fetch the current BODY->hand camera transform using the snapshot attached to a hand-camera image.

    Note: Spot's robot-state transform snapshot often does NOT include camera sensor frames.
    The image response snapshot does, so we compute BODY->handcam from the image shot metadata.
    """
    image_client = robot.ensure_client(ImageClient.default_service_name)
    rgb_req = build_image_request(
        hand_camera_name,
        quality_percent=100,
        pixel_format=None,
    )
    responses = image_client.get_image([rgb_req])
    if not responses:
        raise RuntimeError(f"No image responses returned for camera '{hand_camera_name}'.")
    resp = responses[0]
    # Ensure decoding succeeds (also sanity-checks the response has data).
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
        # Small neighborhood fallback
        win = 3
        v0, v1 = max(0, v - win), min(H, v + win + 1)
        u0, u1 = max(0, u - win), min(W, u + win + 1)
        patch = depth_m[v0:v1, u0:u1]
        vals = patch[np.isfinite(patch) & (patch > 0)]
        if vals.size == 0:
            raise RuntimeError("No valid depth near pixel")
        z = float(np.median(vals))

    fx, fy = K_iphone[0, 0], K_iphone[1, 1]
    cx, cy = K_iphone[0, 2], K_iphone[1, 2]

    x_cam = (u - cx) / fx * z
    y_cam = (v - cy) / fy * z
    p_cam_h = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)

    p_body = (T_body_iphone @ p_cam_h)[:3]
    return p_body


def compute_target_pose_from_bbox_iphone(
    bbox_pixels: list[int],
    depth_m: np.ndarray,
    K_iphone: np.ndarray,
    T_body_iphone: np.ndarray,
    wall_normal: np.ndarray,
    z_clearance_m: float = 0.16,
) -> math_helpers.SE3Pose:
    """Compute BODY-frame target pose from bbox pixels in iPhone image.

    Args:
        bbox_pixels: Bounding box [ymin, xmin, ymax, xmax] in pixels
        depth_m: Depth image in meters
        K_iphone: iPhone camera intrinsics matrix
        T_body_iphone: Transform from iPhone frame to BODY frame
        wall_normal: Wall normal vector (should point toward robot)
        z_clearance_m: Clearance distance from wall surface in meters

    Returns:
        SE3Pose for the target position with clearance applied along wall normal
    """
    ymin, xmin, ymax, xmax = bbox_pixels
    u, v = int(xmax), int(ymax)  # bottom-right pixel

    p_body = _iphone_pixel_to_body_xyz(u, v, depth_m, K_iphone, T_body_iphone)

    # Apply clearance along wall normal (away from wall surface)
    p_cleared = p_body + z_clearance_m * wall_normal

    # Use forward-pointing gripper orientation (straight ahead in body X)
    # pitch=0 means pointing straight forward, no tilt
    return math_helpers.SE3Pose(
        x=float(p_cleared[0]),
        y=float(p_cleared[1]),
        z=float(p_cleared[2]),
        rot=math_helpers.Quat.from_pitch(0.0),
    )


def _compute_wipe_params_from_bbox_iphone(
    bbox: list[int],
    depth_m: np.ndarray,
    K_iphone: np.ndarray,
    T_body_iphone: np.ndarray,
    clearance: float = 0.05,
    spacing_m: float = 0.05,
    max_stroke_len: float = 0.35,
):
    """Compute wipe parameters from bbox using iPhone depth + T_body_iphone.

    This version computes the wall normal from the depth geometry and uses it
    to properly handle vertical surfaces (whiteboards) at arbitrary orientations.
    """
    ymin, xmin, ymax, xmax = bbox
    p_br = (int(xmax), int(ymax))
    p_tr = (int(xmax), int(ymin))
    p_bl = (int(xmin), int(ymax))

    P_br = _iphone_pixel_to_body_xyz(*p_br, depth_m, K_iphone, T_body_iphone)
    P_tr = _iphone_pixel_to_body_xyz(*p_tr, depth_m, K_iphone, T_body_iphone)
    P_bl = _iphone_pixel_to_body_xyz(*p_bl, depth_m, K_iphone, T_body_iphone)

    rr.log("debug/bbox_true",
        rr.Points3D([P_br, P_tr, P_bl], radii=0.01)
    )

    # Compute wall normal from the three points
    wall_normal = compute_wall_normal(P_br, P_tr, P_bl)

    rr.log(
        "debug/wall_normal",
        rr.Arrows3D(
            origins=[P_br],
            vectors=[wall_normal * 0.2],  # scale for visibility
            colors=[[255, 255, 0]],  # yellow
        ),
    )

    # Apply clearance along wall normal (away from wall surface)
    P_start = P_br + clearance * wall_normal

    # Use forward-pointing gripper orientation (straight ahead in body X)
    # pitch=0 means pointing straight forward, no tilt
    wipe_start_pose = math_helpers.SE3Pose(
        x=float(P_start[0]),
        y=float(P_start[1]),
        z=float(P_start[2]),
        rot=math_helpers.Quat.from_pitch(0.0),
    )

    # Stroke direction (up) - project onto wall plane
    up_vec = P_tr - P_br
    up_vec_projected = project_onto_plane(up_vec, wall_normal)

    rr.log(
        "debug/up_vec_raw",
        rr.Arrows3D(
            origins=[P_br],
            vectors=[up_vec],
            colors=[[255, 0, 0]],  # red
        ),
    )

    rr.log(
        "debug/up_vec_projected",
        rr.Arrows3D(
            origins=[P_br],
            vectors=[up_vec_projected],
            colors=[[0, 255, 0]],  # green
        ),
    )

    up_len = float(np.linalg.norm(up_vec_projected))
    if up_len < 1e-6:
        up_len = 0.0
        up_dir = np.array([0.0, 0.0, 0.0])
    else:
        up_dir = up_vec_projected / up_len

    stroke_len = min(up_len, max_stroke_len)
    stroke_dx = float(up_dir[0] * stroke_len)
    stroke_dy = float(up_dir[1] * stroke_len)
    stroke_dz = float(up_dir[2] * stroke_len)

    # Spacing across width (right -> left) - project onto wall plane
    side_vec = P_bl - P_br
    side_vec_projected = project_onto_plane(side_vec, wall_normal)

    rr.log(
        "debug/side_vec_projected",
        rr.Arrows3D(
            origins=[P_br],
            vectors=[side_vec_projected],
            colors=[[0, 0, 255]],  # blue
        ),
    )

    width_m = float(np.linalg.norm(side_vec_projected))
    if width_m > 1e-6:
        side_dir = side_vec_projected / width_m
    else:
        side_dir = np.array([0.0, 0.0, 0.0])

    delta_x_y_z_between_strokes = (
        float(side_dir[0] * spacing_m),
        float(side_dir[1] * spacing_m),
        float(side_dir[2] * spacing_m),
    )
    num_strokes = max(1, int(np.ceil(width_m / max(spacing_m, 1e-3))) + 1)

    end_look_pose = math_helpers.SE3Pose(
        x=0.65,
        y=0.0,
        z=0.4,
        rot=math_helpers.Quat.from_pitch(0.0),  # also point forward
    )

    return (
        wipe_start_pose,
        stroke_dx,
        stroke_dy,
        stroke_dz,
        delta_x_y_z_between_strokes,
        num_strokes,
        end_look_pose,
        wall_normal,  # return wall normal for use in other functions
    )

def draw_bounding_box(image_path, bbox_pixels, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box using pixel coordinates directly (no normalization).

    Args:
        image_path (str): Path to the image file.
        bbox_pixels (list|tuple): [ymin, xmin, ymax, xmax] in pixel units.
        color (tuple): BGR color for the rectangle.
        thickness (int): Line thickness.

    Returns:
        The annotated image (numpy array, BGR).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")

    H, W = img.shape[:2]
    ymin, xmin, ymax, xmax = map(int, bbox_pixels)

    # Clamp to image bounds
    x1 = max(0, min(xmin, W - 1))
    y1 = max(0, min(ymin, H - 1))
    x2 = max(0, min(xmax, W - 1))
    y2 = max(0, min(ymax, H - 1))

    # Ensure non-degenerate box
    if x2 <= x1:
        x2 = min(x1 + 1, W - 1)
    if y2 <= y1:
        y2 = min(y1 + 1, H - 1)

    img_out = img.copy()
    cv2.rectangle(img_out, (x1, y1), (x2, y2), color, thickness)

    output_path = image_path.replace(".png", "_annotated.png").replace(".jpg", "_annotated.jpg")
    cv2.imwrite(output_path, img_out)

    # return img_out
    return output_path


def wipe_multiple_strokes(
    robot: Robot,
    wipe_start_pose: math_helpers.SE3Pose,
    end_look_pose: math_helpers.SE3Pose,
    stroke_dx: float,
    stroke_dy: float,
    stroke_dz: float,
    delta_x_y_z_between_strokes: tuple[float, float, float],
    num_strokes: int,
    duration_per_stroke: float,
    num_attempts_per_stroke: int,
):
    """
    Execute multiple wipe strokes. After each stroke (and attempts) the start pose
    is shifted by delta_x_y_z_between_strokes in BODY frame.

    This version supports 3D motion to handle vertical surfaces properly.
    """
    curr = wipe_start_pose
    for _ in range(num_strokes):
        for _ in range(num_attempts_per_stroke):
            move_hand_to_relative_pose(robot, curr)
            first_move_pose = math_helpers.SE3Pose(
                x=curr.x + stroke_dx,
                y=curr.y + stroke_dy,
                z=curr.z + stroke_dz,
                rot=curr.rot,
            )
            move_hand_to_relative_pose_with_velocity(
                robot, curr, first_move_pose, duration_per_stroke
            )
            # Return to start of this stroke
            move_hand_to_relative_pose_with_velocity(
                robot, first_move_pose, curr, duration_per_stroke
            )
        # Shift to next stroke start
        curr = math_helpers.SE3Pose(
            x=curr.x + delta_x_y_z_between_strokes[0],
            y=curr.y + delta_x_y_z_between_strokes[1],
            z=curr.z + delta_x_y_z_between_strokes[2],
            rot=curr.rot,
        )
    # End look pose
    move_hand_to_relative_pose(robot, end_look_pose)
    

def get_bbox_from_gemini(
    vlm_query_str: str, pil_image: Image.Image
) -> list[int]:
    """
    Query Gemini VLM to get the bbox coordinates corresponding to the query.
    
    Args:
        vlm_query_str: Prompt asking Gemini to identify the spill
        pil_image: PIL Image to analyze
    
    Returns:
        List of [ymin, xmin, ymax, xmax] in pixel coordinates
    """
    # Ensure API key is set for Gemini
    # vlm = GoogleGeminiVLM("gemini-2.5-flash-preview-05-20")
    print(f'inside the function to get the bbox from gemini')
    # vlm = GoogleGeminiVLM("gemini-2.5-flash")
    # vlm = GoogleGeminiVLM("gemini-2.0-flash")
    vlm = GoogleGeminiVLM("gemini-2.5-pro")
    def _parse_bbox_list(raw: str) -> list[float]:
        """Parse a bbox dict {"bbox": [ymin, xmin, ymax, xmax]} from model output.
        Supports optional ```json fenced blocks. Returns raw numeric values
        (assumed normalized 0-1000) without scaling.
        """
        s = raw.strip()
        if "```" in s:
            parts = s.split("```")
            if len(parts) >= 2:
                block = parts[1]
                if block.startswith("json\n"):
                    block = "\n".join(block.splitlines()[1:])
                s = block.strip()
        # Load JSON object
        try:
            obj = json.loads(s)
        except Exception:
            l, r = s.find("{"), s.rfind("}")
            if l == -1 or r == -1 or r <= l:
                raise ValueError("Could not find JSON object in model response.")
            obj = json.loads(s[l:r + 1])

        if not isinstance(obj, dict) or "bbox" not in obj:
            raise ValueError("Expected a JSON object with key 'bbox'.")
        bbox = obj["bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4):
            raise ValueError("'bbox' must be a list of 4 numbers [ymin, xmin, ymax, xmax].")
        return [float(v) for v in bbox]
    
    # Query the VLM
    print(f'vlm: {vlm}, the query string is: {vlm_query_str}')
    vlm_output_list = vlm.sample_completions(
        prompt=vlm_query_str,
        imgs=[pil_image],
        temperature=0.0,
        seed=42,
        num_completions=1,
    )
    vlm_output_str = vlm_output_list[0]
    print(f'vlm_output_str: {vlm_output_str}')
    
    # Parse bbox and convert from normalized [0-1000] to pixel coordinates
    ymin_n, xmin_n, ymax_n, xmax_n = _parse_bbox_list(vlm_output_str)
    img_height = pil_image.height
    img_width = pil_image.width
    ymin = int(round(ymin_n * img_height / 1000.0))
    xmin = int(round(xmin_n * img_width / 1000.0))
    ymax = int(round(ymax_n * img_height / 1000.0))
    xmax = int(round(xmax_n * img_width / 1000.0))

    # Clamp to image bounds
    ymin = max(0, min(ymin, img_height - 1))
    xmin = max(0, min(xmin, img_width - 1))
    ymax = max(0, min(ymax, img_height - 1))
    xmax = max(0, min(xmax, img_width - 1))
    
    bbox = [ymin, xmin, ymax, xmax]
    return bbox

def gaze(robot, direction: str) -> None:
    """Move the hand to look in a certain direction."""
    look_pose = direction_to_pose[direction]
    move_hand_to_relative_pose(robot, look_pose)
    open_gripper(robot)

def gaze_without_open(robot, direction: str) -> None:
    """ Move the hand to look in a certain direction without opening the gripper."""
    look_pose = direction_to_pose[direction]
    move_hand_to_relative_pose(robot, look_pose)

def wipe_online(
    robot: Robot,
    lease_client: LeaseClient,
    lease_keepalive: LeaseKeepAlive,
    localizer=None,
    vlm_query_template: str = DEFAULT_WIPE_VLM_QUERY_TEMPLATE,
    z_offset: float = DEFAULT_WIPE_ONLINE_Z_OFFSET,
    expand_percentage: float = 0.0,
    iphone_extrinsics_path: str = DEFAULT_IPHONE_EXTRINSICS_PATH,
) -> None:
    
    # stow the arm
    stow_arm(robot)

    # have the robot look up to look at the whiteboard
    gaze_without_open(robot, "WALL")

    # Get the latest frame from the shared streaming process
    # (must be started before running this skill)
    time.sleep(0.5)
    frame = get_latest_frame()
    if frame is None:
        raise RuntimeError("No iPhone frame received yet. Ensure iPhone is streaming.")

    rgb_img = frame.rgb  # HxWx3 RGB (full resolution)
    depth_img = frame.depth  # HxW float32 (typically lower resolution)
    if depth_img is None:
        raise RuntimeError("iPhone depth image is missing; cannot compute 3D points.")
    K_full = np.asarray(frame.intrinsics, dtype=np.float32)  # intrinsics at RGB resolution

    # Depth and RGB have different resolutions; compute scale factors and
    # scale intrinsics so they are valid for the depth resolution.
    H_rgb, W_rgb = rgb_img.shape[:2]
    H_d, W_d = depth_img.shape[:2]
    scale_x = W_d / float(W_rgb)
    scale_y = H_d / float(H_rgb)

    K_iphone = K_full.copy()
    K_iphone[0, 0] *= scale_x  # fx
    K_iphone[1, 1] *= scale_y  # fy
    K_iphone[0, 2] *= scale_x  # cx
    K_iphone[1, 2] *= scale_y  # cy

    save_folderpath = "erase_online_images_iphone"
    os.makedirs(save_folderpath, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ## save the rgb and the depth image to the disk
    rgb_pil = Image.fromarray(rgb_img)
    rgb_image_path = os.path.join(save_folderpath, f"rgb_{timestamp}.png")
    depth_image_path = os.path.join(save_folderpath, f"depth_{timestamp}.npy")
    intrinsics_path = os.path.join(save_folderpath, f"intrinsics_{timestamp}.json")
    rgb_pil.save(rgb_image_path)
    np.save(depth_image_path, depth_img.astype(np.float32))
    # Save intrinsics for this iPhone frame
    H, W = rgb_img.shape[:2]
    with open(intrinsics_path, "w") as f:
        json.dump(
            {
                "K_rgb": K_full.tolist(),
                "K_depth": K_iphone.tolist(),
                "width": int(W),
                "height": int(H),
            },
            f,
            indent=2,
        )

    # Compose BODY<-iPhone if we were given hand-camera extrinsics from calibration.
    # Calibration typically produces T_handcam_iphone (aka T_spot_iphone in calibrate_iphone_will.py),
    # but the wipe pipeline needs T_body_iphone for BODY-frame motion planning.
    # CRITICAL: Capture T_body_hand HERE at the same robot position as the iPhone frame!
    T_hand_iphone = _load_T_hand_iphone(iphone_extrinsics_path)
    T_body_hand_at_retreat = _get_T_body_hand_camera(robot, DEFAULT_SPOT_HAND_CAMERA_NAME)
    T_body_retreat_iphone = (T_body_hand_at_retreat @ T_hand_iphone).astype(np.float64)

    # Now approach forward to the manipulation position
    approach_pose = math_helpers.SE2Pose(
        0.3,
        0,
        0
    )
    navigate_to_relative_pose(robot, approach_pose)

    # Account for robot motion: when robot moves forward 0.3m, points in old BODY frame
    # need to be adjusted. Create transform from new BODY frame to old BODY frame.
    # If robot moved forward 0.3m in X, then T_body_new_body_old is a translation of [-0.3, 0, 0]
    T_body_approach_body_retreat = np.eye(4, dtype=np.float64)
    T_body_approach_body_retreat[0, 3] = -0.3  # translation in X

    # Update transform to new BODY frame: T_body_approach_iphone = T_body_approach_body_retreat @ T_body_retreat_iphone
    T_body_iphone = (T_body_approach_body_retreat @ T_body_retreat_iphone).astype(np.float64)

    # Point cloud in iPhone camera frame
    points, colors = rgbd_to_point_cloud(rgb_img, depth_img, K_iphone)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    # o3d.visualization.draw_geometries([pcd])

    voxel_size = 0.005
    rgb = cv2.cvtColor(cv2.imread(rgb_image_path), cv2.COLOR_BGR2RGB)
    rr.log("camera/rgb", rr.Image(rgb))

    depth_m = depth_img.astype(np.float32)
    rr.log("camera/depth", rr.Image(depth_m))

    # Transform points from iPhone camera frame to BODY frame for visualization
    points_cam = points.astype(np.float32)
    num_pts = points_cam.shape[0]
    if num_pts > 0:
        points_cam_h = np.concatenate(
            [points_cam, np.ones((num_pts, 1), dtype=np.float32)], axis=1
        )
        points_body_h = (T_body_iphone @ points_cam_h.T).T
        points_body = points_body_h[:, :3].astype(np.float32)
    else:
        points_body = points_cam

    # Log the 3D points in BODY frame
    rr.log('scene/points3d_body', rr.Points3D(positions=points_body, colors=colors, radii=voxel_size/2))

    # Run VLM on the full-resolution RGB image and get bbox in RGB pixel coordinates.
    bbox_rgb = get_bbox_from_gemini(vlm_query_template, rgb_pil)
    print(f"The coordinates of the bounding box (RGB space) are: {bbox_rgb}")

    # Scale bbox from RGB resolution (H_rgb,W_rgb) to depth resolution (H_d,W_d)
    ymin_r, xmin_r, ymax_r, xmax_r = bbox_rgb
    ymin_d = int(round(ymin_r * scale_y))
    ymax_d = int(round(ymax_r * scale_y))
    xmin_d = int(round(xmin_r * scale_x))
    xmax_d = int(round(xmax_r * scale_x))

    # Clamp to depth image bounds
    ymin_d = max(0, min(ymin_d, H_d - 1))
    ymax_d = max(0, min(ymax_d, H_d - 1))
    xmin_d = max(0, min(xmin_d, W_d - 1))
    xmax_d = max(0, min(xmax_d, W_d - 1))

    bbox = [ymin_d, xmin_d, ymax_d, xmax_d]
    print(f"Scaled bbox in depth space: {bbox}")

    # Optionally expand bbox in image space by a percentage along all directions
    if expand_percentage and expand_percentage > 0.0:
        ymin, xmin, ymax, xmax = bbox
        H, W = depth_img.shape[0], depth_img.shape[1]
        height_px = max(1, (ymax - ymin))
        width_px = max(1, (xmax - xmin))
        dy = int(round(0.5 * expand_percentage * height_px))
        dx = int(round(0.5 * expand_percentage * width_px))
        ymin_exp = max(0, ymin - dy)
        ymax_exp = min(H - 1, ymax + dy)
        xmin_exp = max(0, xmin - dx)
        xmax_exp = min(W - 1, xmax + dx)
        bbox = [ymin_exp, xmin_exp, ymax_exp, xmax_exp]
        print(f"Expanded bbox by {expand_percentage*100:.1f}% -> {bbox}")

    ## log the annotated image with the bounding box 
    annotated_image_path = draw_bounding_box(os.path.join(save_folderpath, f"rgb_{timestamp}.png"), bbox_rgb)
    annotated_img = cv2.cvtColor(cv2.imread(annotated_image_path), cv2.COLOR_BGR2RGB)
    rr.log('results/annotated', rr.Image(annotated_img))

    ## compute the wipe parameters from the bounding box coordinates
    depth_m = depth_img.astype(np.float32)
    (
        wipe_start_pose,
        stroke_dx,
        stroke_dy,
        stroke_dz,
        delta_x_y_z_between_strokes,
        num_strokes,
        end_look_pose,
        wall_normal,
    ) = _compute_wipe_params_from_bbox_iphone(
        bbox,
        depth_m,
        K_iphone,
        T_body_iphone,
        clearance=z_offset,
        spacing_m=0.05,
        max_stroke_len=0.35,
    )

    ## move the hand to the bottom-right position of the bounding box
    # Compute target pose from bbox using iPhone geometry
    # target_pose = compute_target_pose_from_bbox_iphone(
    #     bbox,
    #     depth_m,
    #     K_iphone,
    #     T_body_iphone,
    #     wall_normal,
    #     z_clearance_m=z_offset,
    # )
    # Log a red sphere at the target pose position
    # rr.log(
    #     'results/target_pose_marker',
    #     rr.Points3D(
    #         positions=np.array([[target_pose.x, target_pose.y, target_pose.z]], dtype=np.float32),
    #         colors=np.array([[255, 0, 0]], dtype=np.uint8),
    #         radii=0.02,
    #     ),
    # )

    # move_hand_to_relative_pose(robot, target_pose)

    # Visualize the wipe surface in BODY frame: corners, mesh, and stroke paths
    def _as_np_pose(p):
        return np.array([p.x, p.y, p.z], dtype=np.float32)

    start = _as_np_pose(wipe_start_pose)
    stroke_vec = np.array([stroke_dx, stroke_dy, stroke_dz], dtype=np.float32)
    delta_vec = np.array([delta_x_y_z_between_strokes[0], delta_x_y_z_between_strokes[1], delta_x_y_z_between_strokes[2]], dtype=np.float32)

    # Corners A (start), B (start + stroke), D (last row start), C (last row end)
    A = start
    B = start + stroke_vec
    D = start + max(int(num_strokes) - 1, 0) * delta_vec
    C = D + stroke_vec

    corners_body = np.stack([A, B, C, D], axis=0).astype(np.float32)

    # i) visualize the 3D corners
    rr.log(
        'scene/wipe_surface/corners',
        rr.Points3D(
            positions=corners_body,
            colors=np.array([[0, 128, 255]] * 4, dtype=np.uint8),
            radii=0.01,
        ),
    )

    # ii) visualize the wipe surface polygon (two triangles)
    rr.log(
        'scene/wipe_surface/mesh',
        rr.Mesh3D(
            vertex_positions=corners_body,
            triangle_indices=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
            vertex_colors=np.array([[0, 255, 0, 80]] * 4, dtype=np.uint8),
        ),
    )

    # iii) Visualize the wipe strokes (paths)
    strokes = []
    curr = start.copy()
    for _ in range(int(max(num_strokes, 0))):
        s = curr
        e = curr + stroke_vec
        strokes.append(np.stack([s, e], axis=0))
        curr = curr + delta_vec

    if len(strokes) > 0:
        rr.log(
            'scene/wipe_surface/strokes',
            rr.LineStrips3D(
                strips=strokes,
                colors=np.array([[255, 0, 0]], dtype=np.uint8),
                radii=0.005,
            ),
        )
    
    # Run multi-stroke wipe
    wipe_multiple_strokes(
        robot=robot,
        wipe_start_pose=wipe_start_pose,
        end_look_pose=end_look_pose,
        stroke_dx=stroke_dx,
        stroke_dy=stroke_dy,
        stroke_dz=stroke_dz + 0.05,
        delta_x_y_z_between_strokes=delta_x_y_z_between_strokes,
        num_strokes=num_strokes,
        duration_per_stroke=1.5,
        num_attempts_per_stroke=1,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Online wiping controller.")
    parser.add_argument(
        "--hostname",
        type=str,
        required=True,
        help="Spot hostname/IP (e.g., 192.168.80.3)",
    )
    parser.add_argument(
        "--expand_percentage",
        type=float,
        default=0.0,
        help="Fraction to expand bbox in image space (e.g., 0.2 for +20%).",
    )
    args = parser.parse_args()
    robot, lease_client, lease_keepalive = init_robot(args.hostname, "")
    
    wipe_online(
        robot,
        lease_client,
        lease_keepalive,
        localizer=None,
        expand_percentage=args.expand_percentage
    )

if __name__ == "__main__":
    main()