"""Skill to detect and wipe spills on surfaces using Spot's arm."""

import argparse
import json
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.sdk import Robot
from bosdyn.client.util import authenticate
from PIL import Image

from skills.spot_hand_move import (
    move_hand_to_relative_pose,
    move_hand_to_relative_pose_with_velocity,
    open_gripper,
    stow_arm,
)
from spot_utils.perception.spot_cameras import capture_images
from spot_utils.pretrained_model_interface import GoogleGeminiVLM
from spot_utils.spot_localization import SpotLocalizer
from spot_utils.utils import get_graph_nav_dir, verify_estop


def init_robot(hostname: str, map_name: str) -> tuple[Robot, LeaseClient, LeaseKeepAlive, SpotLocalizer]:
    """Initialize the robot connection, authenticate, sync time, and localize."""
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

DEFAULT_WIPE_ONLINE_Z_OFFSET = 0.02
DEFAULT_WIPE_VLM_QUERY_TEMPLATE = (
    "I have an image with some text written on it, and I am interested in finding "
    "a bounding box for it. Can you give me the coordinates of the bounding box "
    "that encloses the written text? The answer should follow the json format: "
    '{"bbox": [ymin, xmin, ymax, xmax], "label": "spill"}. '
    "The coordinates are in [ymin, xmin, ymax, xmax] format normalized to 0-1000."
)


def visualize_bbox_normalized(image_path, bbox_norm, color=(0, 255, 0), thickness=2):
    """Visualize a normalized bounding box on an image.

    img: HxWx3 uint8 (BGR).
    bbox_norm: [ymin, xmin, ymax, xmax] with each in [0, 1000].
    """
    img = cv2.imread(image_path)
    assert img is not None, f"Failed to read image: {image_path}"
    H, W = img.shape[:2]
    ymin, xmin, ymax, xmax = map(float, bbox_norm)

    # Scale normalized [0,1000] → pixels
    x1 = int(np.clip(xmin * W / 1000.0, 0, W - 1))
    y1 = int(np.clip(ymin * H / 1000.0, 0, H - 1))
    x2 = int(np.clip(xmax * W / 1000.0, 0, W - 1))
    y2 = int(np.clip(ymax * H / 1000.0, 0, H - 1))

    # Ensure non-degenerate box
    if x2 <= x1:
        x2 = min(x1 + 1, W - 1)
    if y2 <= y1:
        y2 = min(y1 + 1, H - 1)

    # Rectangle
    img_out = img.copy()
    cv2.rectangle(img_out, (x1, y1), (x2, y2), color, thickness)

    output_path = image_path.replace(".png", "_annotated.png").replace(".jpg", "_annotated.jpg")
    cv2.imwrite(output_path, img_out)

    return img_out

def draw_bounding_box(image_path, bbox_pixels, color=(0, 255, 0), thickness=2):
    """Draw a bounding box using pixel coordinates directly (no normalization).

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

def compute_target_pose_from_bbox(
    rgbd,
    bbox_pixels: list[int],
    z_clearance_m: float = 0.02,
) -> math_helpers.SE3Pose:
    """Compute the target pose from the bbox pixels.
    bbox_pixels: [ymin, xmin, ymax, xmax] in pixel units.
    """
    ymin, xmin, ymax, xmax = bbox_pixels
    u, v = int(xmax), int(ymax)  # bottom-right pixel

    # Depth to meters
    depth = rgbd.depth
    depth_m = depth.astype(np.float32) / 1000.0 if depth.dtype == np.uint16 else depth.astype(np.float32)

    z = float(depth_m[v, u]) if 0 <= v < depth_m.shape[0] and 0 <= u < depth_m.shape[1] else 0.0
    if not np.isfinite(z) or z <= 0:
        win = 3
        v0, v1 = max(0, v - win), min(depth_m.shape[0], v + win + 1)
        u0, u1 = max(0, u - win), min(depth_m.shape[1], u + win + 1)
        patch = depth_m[v0:v1, u0:u1]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            raise RuntimeError("No valid depth at or near bbox bottom-right.")
        z = float(np.median(valid))

    cam = rgbd.camera_model.intrinsics
    fx, fy = cam.focal_length.x, cam.focal_length.y
    cx, cy = cam.principal_point.x, cam.principal_point.y

    x_cam = (u - cx) / fx * z
    y_cam = (v - cy) / fy * z
    p_cam_h = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)

    T_vision_cam = get_a_tform_b(
        rgbd.transforms_snapshot, VISION_FRAME_NAME, rgbd.frame_name_image_sensor
    ).to_matrix()
    T_body_vision = get_a_tform_b(
        rgbd.transforms_snapshot, BODY_FRAME_NAME, VISION_FRAME_NAME
    ).to_matrix()

    p_body = (T_body_vision @ (T_vision_cam @ p_cam_h))[:3]

    target_pose = math_helpers.SE3Pose(
        x=float(p_body[0]),
        y=float(p_body[1]),
        z=float(p_body[2] + z_clearance_m),
        rot=math_helpers.Quat.from_pitch(np.pi / 2),
    )

    # move_hand_to_relative_pose(robot, target_pose)
    return target_pose

def wipe_one_stroke(
    robot: Robot,
    wipe_start_pose: math_helpers.SE3Pose,
    move_dx: float,
    move_dy: float,
    duration: float,
):
    """Execute a single wipe stroke starting at wipe_start_pose and moving by
    (move_dx, move_dy) in the BODY frame, then returning to the start.
    """
    move_hand_to_relative_pose(robot, wipe_start_pose)
    first_move_pose = math_helpers.SE3Pose(
        x=wipe_start_pose.x + move_dx,
        y=wipe_start_pose.y + move_dy,
        z=wipe_start_pose.z,
        rot=wipe_start_pose.rot,
    )
    move_hand_to_relative_pose_with_velocity(
        robot, wipe_start_pose, first_move_pose, duration
    )
    # Return to start pose
    move_hand_to_relative_pose_with_velocity(
        robot, first_move_pose, wipe_start_pose, duration
    )

def wipe_multiple_strokes(
    robot: Robot,
    wipe_start_pose: math_helpers.SE3Pose,
    end_look_pose: math_helpers.SE3Pose,
    stroke_dx: float,
    stroke_dy: float,
    delta_x_y_between_strokes: tuple[float, float],
    num_strokes: int,
    duration_per_stroke: float,
    num_attempts_per_stroke: int,
):
    """Execute multiple wipe strokes. After each stroke (and attempts) the start pose
    is shifted by delta_x_y_between_strokes in BODY frame.
    """
    curr = wipe_start_pose
    for _ in range(num_strokes):
        for _ in range(num_attempts_per_stroke):
            move_hand_to_relative_pose(robot, curr)
            first_move_pose = math_helpers.SE3Pose(
                x=curr.x + stroke_dx,
                y=curr.y + stroke_dy,
                z=curr.z,
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
            x=curr.x + delta_x_y_between_strokes[0],
            y=curr.y + delta_x_y_between_strokes[1],
            z=curr.z,
            rot=curr.rot,
        )
    # End look pose
    move_hand_to_relative_pose(robot, end_look_pose)

def _pixel_to_body_xyz(u: int, v: int, rgbd) -> np.ndarray:
    """Back-project a pixel (u, v) to BODY frame using rgbd intrinsics and transforms."""
    depth = rgbd.depth
    depth_m = depth.astype(np.float32) / 1000.0 if depth.dtype == np.uint16 else depth.astype(np.float32)
    if v < 0 or v >= depth_m.shape[0] or u < 0 or u >= depth_m.shape[1]:
        raise ValueError("Pixel out of bounds")
    z = float(depth_m[v, u])
    if not np.isfinite(z) or z <= 0:
        # Small neighborhood fallback
        win = 3
        v0, v1 = max(0, v - win), min(depth_m.shape[0], v + win + 1)
        u0, u1 = max(0, u - win), min(depth_m.shape[1], u + win + 1)
        patch = depth_m[v0:v1, u0:u1]
        vals = patch[np.isfinite(patch) & (patch > 0)]
        if vals.size == 0:
            raise RuntimeError("No valid depth near pixel")
        z = float(np.median(vals))
    cam = rgbd.camera_model.intrinsics
    fx, fy = cam.focal_length.x, cam.focal_length.y
    cx, cy = cam.principal_point.x, cam.principal_point.y
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

def _compute_wipe_params_from_bbox(
    rgbd,
    bbox: list[int],
    clearance: float = 0.015,
    spacing_m: float = 0.05,
    max_stroke_len: float = 0.35,
):
    """Compute wipe parameters from a bounding box in pixels.

    Returns wipe_start_pose, stroke deltas, inter-stroke spacing,
    num_strokes, and end_look_pose.
    """
    ymin, xmin, ymax, xmax = bbox
    p_br = (int(xmax), int(ymax))
    p_tr = (int(xmax), int(ymin))
    p_bl = (int(xmin), int(ymax))

    P_br = _pixel_to_body_xyz(*p_br, rgbd)
    P_tr = _pixel_to_body_xyz(*p_tr, rgbd)
    P_bl = _pixel_to_body_xyz(*p_bl, rgbd)

    # Start pose
    wipe_start_pose = math_helpers.SE3Pose(
        x=float(P_br[0]),
        y=float(P_br[1]),
        z=float(P_br[2] + clearance),
        rot=math_helpers.Quat.from_pitch(np.pi / 2),
    )

    # Stroke direction (up)
    up_vec = P_tr - P_br
    up_vec[2] = 0.0
    up_len = float(np.linalg.norm(up_vec[:2]))
    if up_len < 1e-6:
        up_len = 0.0
        up_dir = np.array([0.0, 0.0])
    else:
        up_dir = up_vec[:2] / up_len
    # Stroke length from bbox height, limited by max_stroke_len
    stroke_len = min(up_len, max_stroke_len)
    stroke_dx = float(up_dir[0] * stroke_len)
    stroke_dy = float(up_dir[1] * stroke_len)

    # Spacing across width (right -> left)
    side_vec = P_bl - P_br
    side_vec[2] = 0.0
    width_m = float(np.linalg.norm(side_vec[:2]))
    if width_m > 1e-6:
        side_dir = side_vec[:2] / width_m
    else:
        side_dir = np.array([0.0, 0.0])
    delta_x_y_between_strokes = (float(side_dir[0] * spacing_m), float(side_dir[1] * spacing_m))
    num_strokes = max(1, int(width_m / max(spacing_m, 1e-3))+1)

    end_look_pose = math_helpers.SE3Pose(
        x=0.65, y=0.0, z=0.4, rot=math_helpers.Quat.from_pitch(np.pi / 2.5)
    )

    return (
        wipe_start_pose,
        stroke_dx,
        stroke_dy,
        delta_x_y_between_strokes,
        num_strokes,
        end_look_pose,
    )

def visualize_bbox_prediction(image_path, bbox):
    """Draws a SCALED bounding box on an image using OpenCV and displays it.

    Args:
        image_path (str): Path to the ORIGINAL image file.
        bbox (list): Bounding box [ymin, xmin, ymax, xmax] from the model,
                     relative to the model's input size.

    """
    try:
        # --- Dimensions of the image the model processed ---
        # (This is the key piece of information you were missing)
        model_height = 682
        model_width = 910

        # 1. Read the ORIGINAL image to get its true dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return

        original_height, original_width, _ = img.shape
        
        # 2. Calculate scaling factors
        y_scale = original_height / model_height
        x_scale = original_width / model_width
        
        # 3. Unpack and scale the model's bounding box coordinates
        ymin, xmin, ymax, xmax = bbox
        
        scaled_ymin = int(ymin * y_scale)
        scaled_xmin = int(xmin * x_scale)
        scaled_ymax = int(ymax * y_scale)
        scaled_xmax = int(xmax * x_scale)

        # 4. Define points for the rectangle using SCALED coordinates
        pt1 = (scaled_xmin, scaled_ymin)
        pt2 = (scaled_xmax, scaled_ymax)
        
        # Define color (OpenCV uses BGR format, not RGB)
        color_bgr = (0, 0, 255)  # Red
        thickness = 3
        
        # 5. Draw the rectangle on the original image
        cv2.rectangle(img, pt1, pt2, color_bgr, thickness)
        
        # 6. Save the annotated image
        output_path = image_path.replace(".png", "_annotated.png").replace(".jpg", "_annotated.jpg")
        cv2.imwrite(output_path, img)
        
        print(f"Successfully saved annotated image to: {output_path}")
        return output_path

    except Exception as e:
        print(f"An error occurred: {e}")
    

def get_bbox_from_gemini(
    vlm_query_str: str, pil_image: Image.Image
) -> list[int]:
    """Query Gemini VLM to get the bbox coordinates corresponding to the query.
    
    Args:
        vlm_query_str: Prompt asking Gemini to identify the spill
        pil_image: PIL Image to analyze
    
    Returns:
        List of [ymin, xmin, ymax, xmax] in pixel coordinates

    """
    # Ensure API key is set for Gemini
    # vlm = GoogleGeminiVLM("gemini-2.5-flash-preview-05-20")
    print('inside the function to get the bbox from gemini')
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
            left, right = s.find("{"), s.rfind("}")
            if left == -1 or right == -1 or right <= left:
                raise ValueError("Could not find JSON object in model response.")
            obj = json.loads(s[left:right + 1])

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


def get_points_from_pixels(rgb_image_path, depth_image_path, intrinsics):
    """Back-project RGB and depth images into a 3D point cloud with colors."""
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


def gaze(robot, direction: str) -> None:
    """Move the hand to look in a certain direction."""
    look_pose = direction_to_pose[direction]
    move_hand_to_relative_pose(robot, look_pose)
    open_gripper(robot)

def wipe_online(
    robot: Robot,
    lease_client: LeaseClient,
    lease_keepalive: LeaseKeepAlive,
    localizer: SpotLocalizer,
    vlm_query_template: Optional[str] = None,
    z_offset: float = DEFAULT_WIPE_ONLINE_Z_OFFSET,
    expand_percentage: float = 0.0,
) -> None:
    """Detect a spill using vision and execute wiping motions to clean it."""
    # stow the arm
    stow_arm(robot)
    # have the robot look ahead to look at the spill 
    gaze(robot, "AHEAD")
    # gaze(robot, "DOWN")
    
    # capture the image using the hand camera 
    rgbds = capture_images(robot, localizer, camera_names=["hand_color_image"])
    rgbd = rgbds["hand_color_image"]
    cam_model = rgbd.camera_model  

    fx = cam_model.intrinsics.focal_length.x
    fy = cam_model.intrinsics.focal_length.y
    cx = cam_model.intrinsics.principal_point.x
    cy = cam_model.intrinsics.principal_point.y

    intrinsics = [fx, fy, cx, cy]

    ## extract the rgb and the depth image 
    rgb_img = rgbd.rgb
    depth_img = rgbd.depth
    # depth_m = depth_img.astype(np.float32)
    # if depth_img.dtype == np.uint16:
    #     depth_m = depth_m / 1000.0

    save_folderpath = "wipe_online_images"
    os.makedirs(save_folderpath, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ## save the rgb and the depth image to the disk
    rgb_pil = Image.fromarray(rgb_img)
    depth_pil = Image.fromarray(depth_img)
    rgb_image_path = os.path.join(save_folderpath, f"rgb_{timestamp}.png")
    depth_image_path = os.path.join(save_folderpath, f"depth_{timestamp}.png")
    rgb_pil.save(rgb_image_path)
    depth_pil.save(depth_image_path)

    points, colors = get_points_from_pixels(rgb_image_path, depth_image_path, intrinsics)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    # o3d.visualization.draw_geometries([pcd])

    voxel_size = 0.005
    rgb_raw = cv2.imread(rgb_image_path)
    assert rgb_raw is not None, f"Failed to read image: {rgb_image_path}"
    rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
    rr.log('camera/rgb', rr.Image(rgb))

    depth_raw = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    assert depth_raw is not None, f"Failed to read image: {depth_image_path}"
    depth = depth_raw.astype(np.float32) / 1000.0
    print(f'printing the max and min values of the depth image : {depth.max()} and {depth.min()}')
    rr.log('camera/depth', rr.Image(depth))

    # Transform points from camera frame to BODY frame for consistent visualization with target_pose
    T_vision_cam = get_a_tform_b(
        rgbd.transforms_snapshot, VISION_FRAME_NAME, rgbd.frame_name_image_sensor
    ).to_matrix()
    T_body_vision = get_a_tform_b(
        rgbd.transforms_snapshot, BODY_FRAME_NAME, VISION_FRAME_NAME
    ).to_matrix()
    T_body_cam = T_body_vision @ T_vision_cam

    points_cam = points.astype(np.float32)
    num_pts = points_cam.shape[0]
    if num_pts > 0:
        points_cam_h = np.concatenate([points_cam, np.ones((num_pts, 1), dtype=np.float32)], axis=1)
        points_body_h = (T_body_cam @ points_cam_h.T).T
        points_body = points_body_h[:, :3].astype(np.float32)
    else:
        points_body = points_cam

    # Log the 3D points in BODY frame
    rr.log('scene/points3d_body', rr.Points3D(positions=points_body, colors=colors, radii=voxel_size/2))

    if vlm_query_template is None:
        vlm_query_template = DEFAULT_WIPE_VLM_QUERY_TEMPLATE
    # vlm_query_template = "I have an image with some text written on it, and I am interested in finding a bounding box for it. Can you give me the coordinates of the bounding box that encloses the written text?"
    bbox = get_bbox_from_gemini(vlm_query_template, rgb_pil)
    print(f"The coordinates of the bounding box are: {bbox}")

    # Optionally expand bbox in image space by a percentage along all directions
    if expand_percentage and expand_percentage > 0.0:
        ymin, xmin, ymax, xmax = bbox
        H, W = rgb_img.shape[0], rgb_img.shape[1]
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
    annotated_image_path = draw_bounding_box(os.path.join(save_folderpath, f"rgb_{timestamp}.png"), bbox)
    annotated_raw = cv2.imread(annotated_image_path)
    assert annotated_raw is not None, f"Failed to read image: {annotated_image_path}"
    annotated_img = cv2.cvtColor(annotated_raw, cv2.COLOR_BGR2RGB)
    rr.log('results/annotated', rr.Image(annotated_img))

    ## move the hand to the bottom-right position of the bounding box 
    target_pose = compute_target_pose_from_bbox(rgbd, bbox, z_clearance_m=z_offset)
    # Log a red sphere at the target pose position
    rr.log(
        'results/target_pose_marker',
        rr.Points3D(
            positions=np.array([[target_pose.x, target_pose.y, target_pose.z]], dtype=np.float32),
            colors=np.array([[255, 0, 0]], dtype=np.uint8),
            radii=0.03,
        ),
    )
    
    move_hand_to_relative_pose(robot, target_pose)

    ## compute the wipe parameters from the bounding box coordinates 
    wipe_start_pose, stroke_dx, stroke_dy, delta_x_y_between_strokes, num_strokes, end_look_pose = _compute_wipe_params_from_bbox(
        rgbd,
        bbox,
        clearance=z_offset,
        spacing_m=0.05,
        max_stroke_len=0.35,
    )

    # Visualize the wipe surface in BODY frame: corners, mesh, and stroke paths
    def _as_np_pose(p):
        return np.array([p.x, p.y, p.z], dtype=np.float32)

    start = _as_np_pose(wipe_start_pose)
    stroke_vec = np.array([stroke_dx, stroke_dy, 0.0], dtype=np.float32)
    delta_vec = np.array([delta_x_y_between_strokes[0], delta_x_y_between_strokes[1], 0.0], dtype=np.float32)

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
        delta_x_y_between_strokes=delta_x_y_between_strokes,
        num_strokes=num_strokes,
        duration_per_stroke=1.0,
        num_attempts_per_stroke=1,
    )

def main() -> None:
    """Parse arguments and run the wipe-online skill."""
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
    parser.add_argument(
        "--z_offset",
        type=float,
        default=0.00,
        help="Hand Z offset above surface in meters (clearance).",
    )
    parser.add_argument(
        "--expand_percentage",
        type=float,
        default=0.0,
        help="Fraction to expand bbox in image space (e.g., 0.2 for +20%).",
    )
    args = parser.parse_args()
    robot, lease_client, lease_keepalive, localizer = init_robot(args.hostname, args.map_name)
    # rr.init("wipe_online", spawn=True)

    vlm_query_template = """
I have an image with some text written on it, and I am interested in finding a bounding box for it. Can you give me the coordinates of the bounding box that encloses the written text? 
The answer should follow the json format: {"bbox": [ymin, xmin, ymax, xmax], "label": "spill"}. 
The coordinates are in [ymin, xmin, ymax, xmax] format normalized to 0-1000."""
    wipe_online(
        robot,
        lease_client,
        lease_keepalive,
        localizer,
        vlm_query_template,
        args.z_offset,
        args.expand_percentage,
    )

if __name__ == "__main__":
    rr.init("wipe_online", spawn=True)
    main()