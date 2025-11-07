"""Interface for opening a drawer."""

import argparse
import traceback
import time
import numpy as np
from PIL import Image
import json
from typing import Tuple

import open3d as o3d
from bosdyn.api import (
    arm_command_pb2,
    manipulation_api_pb2,
    robot_command_pb2,
    synchronized_command_pb2,
    trajectory_pb2,
)
from bosdyn.client.image import ImageClient
import cv2
from numpy.typing import NDArray
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b, VISION_FRAME_NAME, get_se2_a_tform_b
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.sdk import Robot
from bosdyn.util import seconds_to_duration
from google.protobuf.wrappers_pb2 import (
    DoubleValue,  # pylint: disable=no-name-in-module
)

from spot_utils.utils import verify_estop, get_graph_nav_dir, get_robot_state
from spot_utils.gemini_utils import get_pixel_from_gemini
from spot_utils.perception.perception_structs import RGBDImageWithContext
from skills.grasp import grasp_at_pixel
from skills.spot_navigation import navigate_to_relative_pose
from skills.spot_hand_move import move_hand_to_relative_pose, open_gripper, close_gripper, stow_arm
# from grasp import grasp_at_pixel
from spot_utils.perception.spot_cameras import capture_images
from spot_utils.spot_localization import SpotLocalizer
from spot_utils.pretrained_model_interface import GoogleGeminiVLM
from exec_plan import direction_to_pose, gaze

import rerun as rr

def move_hand_back(robot, dx):
    robot_state_client = robot.ensure_client('robot-state')
    state = robot_state_client.get_robot_state()

    # Get hand pose relative to body
    body_T_hand = get_a_tform_b(state.kinematic_state.transforms_snapshot,
                                BODY_FRAME_NAME,
                                HAND_FRAME_NAME)
    
    hand_offset_T_hand = math_helpers.SE3Pose(-dx, 0, 0, math_helpers.Quat())
    body_T_new_hand = body_T_hand * hand_offset_T_hand
    move_hand_to_relative_pose(robot, body_T_new_hand)


def pixels_to_vision_points(
    pixels: list[tuple[int, int]],
    rgbd: RGBDImageWithContext,
) -> NDArray[np.float64]:
    """
    Convert 2D pixels (u, v) to 3D points (X, Y, Z) in vision frame.

    Args:
        pixels: list of (u, v) pixel coordinates
        rgbd: an RGBDImageWithContext instance from Spot's capture_images()

    Returns:
        Nx3 array of 3D points in vision frame (in meters)
    """

    vision_T_camera = get_a_tform_b(
        rgbd.transforms_snapshot,
        VISION_FRAME_NAME,
        rgbd.frame_name_image_sensor
    )

    depth_img = rgbd.depth
    depth_m = depth_img.astype(np.float32)
    if depth_img.dtype == np.uint16:
        depth_m = depth_m / 1000.0
    cam_model = rgbd.camera_model  

    fx = cam_model.intrinsics.focal_length.x
    fy = cam_model.intrinsics.focal_length.y
    cx = cam_model.intrinsics.principal_point.x
    cy = cam_model.intrinsics.principal_point.y

    # print(f"Intrinsics : {fx, fy, cx, cy}")
    # print(f"Depth scale : {depth_scale}")

    pts = []
    for (u, v) in pixels:
        if v < 0 or v >= depth_m.shape[0] or u < 0 or u >= depth_m.shape[1]:
            continue

        z = float(depth_m[v, u])
        # We filter out points further than 2 meters away
        if z <= 0 or z > 2.0:
            continue

        x = (float(u) - cx) / fx * z
        y = (float(v) - cy) / fy * z
        pts.append([x, y, z, 1.0])

    pts_cam = np.array(pts).T  # shape 4xN

    # Transform to vision frame
    pts_vision = (vision_T_camera.to_matrix() @ pts_cam).T[:, :3]  # Nx3
    return np.asarray(pts_vision, dtype=np.float32)


def fit_plane_to_points(points_vision: NDArray[np.float64]):
    """
    Fit a plane to 3D points and return its centroid and normal vector.

    Args:
        points_vision: Nx3 array of 3D points (in vision frame)

    Returns:
        centroid: (3,) array, mean position of points
        normal: (3,) array, unit normal vector of best-fit plane
    """
    assert points_vision.shape[1] == 3, "Points must be Nx3"

    # Compute centroid
    centroid = np.mean(points_vision, axis=0)

    # Subtract centroid
    Q = points_vision - centroid

    # Compute covariance and its eigenvectors
    _, _, vh = np.linalg.svd(Q)  # SVD is numerically stable
    normal = vh[-1, :]  # last row of V^T (smallest singular value)

    # Normalize
    normal /= np.linalg.norm(normal)

    if normal[0] < 0:
        normal = -normal

    return centroid, normal


def grasp_orientation_from_normal(normal_vec: np.ndarray, world_up: np.ndarray = np.array([0, 0, 1])) -> math_helpers.Quat:
    """
    Given a normal vector (pointing outward from the drawer), compute a quaternion
    such that the gripper's +X axis points opposite the normal (toward the drawer).

    Args:
        normal_vec: (3,) array, unit normal vector in vision frame.
        world_up: (3,) array, unit vertical direction (default [0,0,1]) (assume vision frame z axis is roughly this)

    Returns:
        math_helpers.Quat representing the grasp orientation.
    """
    x_axis = -normal_vec  # gripper faces opposite the drawer normal

    # Construct y and z axes orthogonal to x
    y_axis = np.cross(world_up, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # Rotation matrix (columns are body axes in vision frame)
    R = np.column_stack((x_axis, y_axis, z_axis))

    # Convert rotation matrix to quaternion
    quat = math_helpers.Quat.from_matrix(R)
    return quat


def compute_body_pose_in_front_of_drawer(drawer_point: np.ndarray,
                                         drawer_normal: np.ndarray,
                                         standoff_dist: float = 0.8) -> math_helpers.SE2Pose:
    """
    Compute a 2D pose (x, y, yaw) for Spot's body to face the drawer.

    Args:
        drawer_point: (3,) coordinates of a point on the drawer surface in vision frame.
        drawer_normal: (3,) unit normal vector pointing out of the drawer in vision frame.
        standoff_dist: distance to stand off from the drawer surface (m).

    Returns:
        math_helpers.SE2Pose representing where the body should move.
    """
    # Compute body target position: move back along -normal by standoff_dist
    body_pos = drawer_point + drawer_normal * standoff_dist

    # Compute yaw angle so body faces *toward* the drawer (along +normal)
    yaw = np.arctan2(-drawer_normal[1], -drawer_normal[0])  # face opposite normal

    return math_helpers.SE2Pose(body_pos[0], body_pos[1], yaw)


def compute_rotated_body_pose(robot: Robot, normal_vector: Tuple[float, float]) -> math_helpers.SE2Pose:
    """
    Rotate Spot's body to align with opposite of normal vector.
    """
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    state = robot_state_client.get_robot_state()
    vision_T_body = get_a_tform_b(state.kinematic_state.transforms_snapshot,
                              VISION_FRAME_NAME,
                              BODY_FRAME_NAME)
    x = vision_T_body.x
    y = vision_T_body.y
    yaw = np.arctan2(-normal_vector[1], -normal_vector[0])
    se2 = vision_T_body.get_closest_se2_transform()
    # print("CURRENT POSE: ", math_helpers.SE2Pose(x, y, se2.angle))
    return math_helpers.SE2Pose(x, y, yaw)


def set_body_height(robot, height_offset_m: float):
    """
    Adjust Spot's body height up or down.
    
    Args:
        robot: an instance of bosdyn.client.sdk.Robot
        height_offset_m: positive to raise, negative to lower (in meters)
    """
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    # Assert height offset is in safe range
    assert height_offset_m <= 0.2 and height_offset_m >= -0.2

    # Build a stand command with a height offset
    cmd = RobotCommandBuilder.synchro_stand_command(body_height=height_offset_m)

    # Send command
    command_client.robot_command(cmd)


def navigate_to_vision_goal(robot, vision_tform_goal: math_helpers.SE2Pose):
    # 1. Get the current robot transforms
    robot_state = get_robot_state(robot)
    transforms = robot_state.kinematic_state.transforms_snapshot

    # 2. Get current body pose in the vision frame
    vision_tform_body = get_se2_a_tform_b(transforms, VISION_FRAME_NAME, BODY_FRAME_NAME)

    # 3. Compute desired relative motion in body frame
    body_tform_goal = vision_tform_body.inverse() * vision_tform_goal

    # 4. Command robot to move by that relative transform
    navigate_to_relative_pose(robot, body_tform_goal)


def get_multiple_pixels_from_gemini(
    vlm_query_str: str, pil_image: Image, num_pixels: int = 15
) -> list[Tuple[int, int]]:
    # Assuming create_vlm_by_name exists and works like create_llm_by_name
    # Use the specific model name from CFG or hardcode if necessary
    vlm = GoogleGeminiVLM("gemini-2.0-flash")

    # 2. Construct the query
    # Adjust prompt as needed for better VLM performance
    def parse_json_output(json_output_str):
        # Parsing out the markdown fencing
        lines = json_output_str.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                json_output_str = "\n".join(lines[i + 1 :])
                json_output_str = json_output_str.split("```")[0]
                break
        json_output_str = json_output_str.strip()
        return json_output_str

    # 3. Query the VLM
    # Assuming sample_completions takes a list of images
    vlm_output_list = vlm.sample_completions(
        prompt=vlm_query_str,
        imgs=[pil_image],
        temperature=0.0,  # Low temp for deterministic output
        seed=42,
        num_completions=1,
    )
    vlm_output_str = vlm_output_list[0]
    # 4. Parse the JSON string
    json_string_to_parse = parse_json_output(vlm_output_str)
    parsed_data = json.loads(json_string_to_parse)
    # 5. Extract and denormalize coordinates
    if not isinstance(parsed_data, list) or not parsed_data:
        raise ValueError("Parsed JSON is not a non-empty list.")
    if len(parsed_data) < num_pixels:
        raise ValueError(f"Parsed JSON has less than {num_pixels} points.")
    pixels = []
    for point_obj in parsed_data[:num_pixels]:  # limit to first num_pixels points
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
        # Denormalize from 0-1000 range to image pixel coordinates
        img_height = pil_image.height
        img_width = pil_image.width
        y = int(y_norm * img_height / 1000.0)
        x = int(x_norm * img_width / 1000.0)
        # Clamp coordinates to be within image bounds
        y = max(0, min(y, img_height - 1))
        x = max(0, min(x, img_width - 1))
        pixels.append((x, y))
    
    # print(pixels)
    return pixels


def draw_colored_pixels(image_pil: Image, pixels: list[Tuple[int, int]], path: str, color: str):
    pixels_obj = image_pil.load()
    for pixel in pixels:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                px = min(max(pixel[0] + dx, 0), image_pil.width - 1)
                py = min(max(pixel[1] + dy, 0), image_pil.height - 1)
                pixels_obj[px, py] = (255, 0, 0) if color == "red" else (0, 0, 255)
    image_pil.save(path)


DEFAULT_HAND_LOOK_FLOOR_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=-0.15, rot=math_helpers.Quat.from_pitch(0)
)

DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2)
)

DEFAULT_HAND_LOOK_INTO_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.3, rot=math_helpers.Quat.from_pitch(np.pi / 4)
)

direction_to_pose = {
    "DOWN": DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE,
    "AHEAD": DEFAULT_HAND_LOOK_FLOOR_POSE,
    "INTO": DEFAULT_HAND_LOOK_INTO_POSE
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

    # Build Open3D point cloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    # pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))

    # o3d.visualization.draw_geometries([pcd])

    # return pcd

def open_drawer(
    robot: Robot,
    localizer: SpotLocalizer,
    standoff_dist: float = 0.8,
    body_height_offset: float = 0.0,
    retreat_offset: float = 0.1,
    checkpoint: int = 7,
) -> None:
    """
    Reach toward a drawer handle, close the gripper to grasp it, 
    then return to a resting pose and open the gripper.
    
    Args:
        robot: Spot robot instance.
        approach_offset: Distance (m) to stop before touching the handle.
        timeout: Seconds to allow for each arm motion.
    """
    # Gaze at drawer ahead
    gaze(robot, "AHEAD")

    # Capture RGBD image from Spot hand camera
    rgbds = capture_images(robot, localizer, camera_names=["hand_color_image"])
    rgbd = rgbds["hand_color_image"]
    # rgbd=None

    # Extract RGB image and depth image
    rgb = rgbd.rgb
    depth = rgbd.depth
    depth_pil = Image.fromarray(depth)
    depth_pil.save("raw_hand_camera_depth.png")
    rr.log("drawer_rgb", rr.Image(rgb))
    image_pil = Image.fromarray(rgb)
    image_pil.save("raw_hand_camera_output.jpg") 

    # Get a 2D pixel on the handle, and convert to 3D point
    handle_pixel = get_pixel_from_gemini(prompt_get_handle_pixel, image_pil)
    draw_colored_pixels(image_pil, [handle_pixel], "annotated_hand_camera_output.jpg", "red")
    
    cam_model = rgbd.camera_model  

    fx = cam_model.intrinsics.focal_length.x
    fy = cam_model.intrinsics.focal_length.y
    cx = cam_model.intrinsics.principal_point.x
    cy = cam_model.intrinsics.principal_point.y

    intrinsics = [fx, fy, cx, cy]

    rgb_image_path = "raw_hand_camera_output.jpg"
    depth_image_path = "raw_hand_camera_depth.png"
    points, colors = get_points_from_pixels(rgb_image_path, depth_image_path, intrinsics)

    # Convert entire point cloud from camera → vision frame
    vision_T_camera = get_a_tform_b(
        rgbd.transforms_snapshot,
        VISION_FRAME_NAME,
        rgbd.frame_name_image_sensor
    )
    points_hom = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
    vision_T_camera_mat = vision_T_camera.to_matrix()
    points_vision = (vision_T_camera_mat @ points_hom.T).T[:, :3].astype(np.float32)

    handle_3d_point = pixels_to_vision_points([handle_pixel], rgbd)[0]
    voxel_size = 0.005
    rr.log("3D_points", rr.Points3D(positions=points_vision, colors=colors, radii=voxel_size/2))
    
    # Get pixels on surface of drawer via SAM (try just Gemini first, get 15 pixels on front of drawer)
    front_surface_pixels = get_multiple_pixels_from_gemini(prompt_get_drawer_surface_pixel, image_pil, 15)
    draw_colored_pixels(image_pil, front_surface_pixels, "annotated_hand_camera_output.jpg", "blue")
    rr.log("drawer_pixels", rr.Image(np.array(image_pil)))
    if checkpoint == 0:
        return

    # Convert to 3D points on surface of drawer
    front_surface_3d_points = pixels_to_vision_points(front_surface_pixels, rgbd)
    rr.log("surface_points", rr.Points3D(positions=front_surface_3d_points, colors=[255, 0, 0], radii=voxel_size*1.5))

    # Fit a plane to those points via SVD and get normal vector
    _, normal_vector = fit_plane_to_points(front_surface_3d_points)
    # print("NORMAL VECTOR IS: ", normal_vector)

    # Compute approach grasp pose, aligned to normal
    # grasp_rot = grasp_orientation_from_normal(normal_vector)

    # ACTION: Move Spot's body to be aligned to the front of the drawer normal FIRST
    # rotated_body_pose = compute_rotated_body_pose(robot, normal_vector)
    # print("ROTATED POSE IS: ", rotated_body_pose)
    # navigate_to_vision_goal(robot, rotated_body_pose)
    body_target_pose = compute_body_pose_in_front_of_drawer(handle_3d_point, normal_vector, standoff_dist)
    # print("BODY POSE IS: ", body_target_pose)
    navigate_to_vision_goal(robot, body_target_pose)
    if checkpoint == 1:
        return

    # ACTION: Adjust Spot's height up and down depending on comfortable grasping position, find this param
    set_body_height(robot, body_height_offset)
    if checkpoint == 2:
        return

    # ACTION: Grasp at pixel on handle
    grasp_at_pixel(robot, rgbd, handle_pixel, move_while_grasping=False)
    if checkpoint == 4:
        return
    
    # Compute retreat pose along normal vector
    retreat_pose = math_helpers.SE2Pose(
        -retreat_offset,
        0,
        0
    )
    # print("RETREAT POSE IS: ", retreat_pose)

    # ACTION: Walk backwards to open drawer
    navigate_to_relative_pose(robot, retreat_pose)
    if checkpoint == 5:
        return

    # ACTION: Open gripper
    open_gripper(robot)
    if checkpoint == 6:
        return
    
    move_hand_back(robot, 0.1)

    # ACTION: Stow arm
    stow_arm(robot)
    if checkpoint == 7:
        return
    
def look_into_drawer(robot: Robot, localizer: SpotLocalizer):
    """ 
    Look into an opened drawer and query Gemini for what objects Spot sees inside.

    Args:
        robot: Spot robot instance.
        localizer: SpotLocalizer instance.
    """
    # Move arm to look into drawer.
    gaze(robot, "INTO")

    # Get image of drawer and save it.
    rgbds = capture_images(robot, localizer, camera_names=["hand_color_image"])
    rgb = rgbds["hand_color_image"].rgb
    pil_image = Image.fromarray(rgb)
    pil_image.save("inside_drawer_camera_output.jpg") 
    rr.log("inside_drawer_rgb", rr.Image(rgb))

    # Call Gemini to ask what objects are in the drawer.
    vlm = GoogleGeminiVLM("gemini-2.0-flash")
    vlm_output_list = vlm.sample_completions(
        prompt=prompt_get_objects_inside_drawer,
        imgs=[pil_image],
        temperature=0.0,  # Low temp for deterministic output
        seed=42,
        num_completions=1,
    )
    vlm_output_str = vlm_output_list[0]
    print(vlm_output_str)


prompt_get_handle_pixel = """
    Point to the green handle of the drawer.
    The answer should follow the json format: [{"point": , "label": }, ...]. The points are in [y, x] format normalized to 0-1000.
    """
prompt_get_drawer_surface_pixel = """
    Point to 15 points on the front face of the drawer, but avoid the drawer handles (including the green handle) or the edges of the front face.
    Make sure the points are on the front face, not the side face.
    The answer should follow the json format: [{"point": , "label": }, ...]. The points are in [y, x] format normalized to 0-1000.
    """
prompt_get_objects_inside_drawer = """
    Give me a descriptive list of objects inside this drawer.
    The answer should follow the format: ["object1", "object2", ...].
    """

if __name__ == "__main__":
    # Run this file alone to test manually.
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    # Get constants.
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
        required=False,
        help="The name of the map folder to load (sub-folder under graph_nav_maps)",
    )
    args = parser.parse_args()

    hostname = args.hostname

    # --- Connect and authenticate ---
    sdk = create_standard_sdk("SpotOpenDrawerClient")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)

    # --- Acquire lease ---
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    lease_keepalive = LeaseKeepAlive(
        lease_client, must_acquire=True, return_at_exit=True
    )

    # --- Localize ---
    path = get_graph_nav_dir(args.map_name)
    localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
    robot.time_sync.wait_for_sync()
    localizer.localize()
    print("[INFO] Localization successful.")
    # robot = None
    # localizer = None

    rr.init("open-drawer-test", spawn=True)

    # --- Run open_drawer routine ---
    try:
        print("[INFO] Running open_drawer()...")
        open_drawer(robot, localizer, standoff_dist=1.1, body_height_offset=0.0, retreat_offset=0.4, checkpoint=7)
        look_into_drawer(robot, localizer)

    except Exception as e:
        print(e)
        traceback.print_exc()