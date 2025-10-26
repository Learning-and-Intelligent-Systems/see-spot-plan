"""Interface for opening a drawer."""

import argparse
import time
import numpy as np
from PIL import Image
import json
from typing import Tuple

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
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b
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

from spot_utils.utils import verify_estop, get_pixel_from_gemini, get_graph_nav_dir
from spot_utils.perception.perception_structs import RGBDImageWithContext
from skills.grasp import grasp_at_pixel
from skills.spot_navigation import navigate_to_absolute_pose
from skills.spot_hand_move import move_hand_to_relative_pose, open_gripper, close_gripper, stow_arm
from spot_utils.perception.spot_cameras import capture_images
from spot_utils.spot_localization import SpotLocalizer
from spot_utils.pretrained_model_interface import GoogleGeminiVLM



def move_hand_to_absolute_pose(
    robot: Robot,
    goal_pose_odom: math_helpers.SE3Pose,
) -> None:
    """
    Move Spot's hand to an absolute pose expressed in odometry frame.

    Args:
        robot: Spot robot instance.
        goal_pose_odom: Desired SE3Pose in odometry frame.
    """
    # Transform goal_pose into the robot's body frame
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_state = robot_state_client.get_robot_state()

    # Compute the transform from goal_frame to body frame
    body_tform_odom = get_a_tform_b(
        robot_state.kinematic_state.transforms_snapshot,
        BODY_FRAME_NAME,
        ODOM_FRAME_NAME,
    )

    # Apply the transform to get the pose relative to the body
    goal_pose_body = body_tform_odom * goal_pose_odom

    # Move the hand using the existing relative pose function
    move_hand_to_relative_pose(robot, goal_pose_body)


def get_gripper_pose_odom(robot):
    """Return Spot's hand pose as an SE3Pose in the odom frame."""
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    state = robot_state_client.get_robot_state()

    odom_tform_hand = get_a_tform_b(
        state.kinematic_state.transforms_snapshot,
        ODOM_FRAME_NAME,
        HAND_FRAME_NAME,
    )
    return odom_tform_hand


def pixels_to_world_points(
    pixels: list[tuple[int, int]],
    rgbd: RGBDImageWithContext,
) -> NDArray[np.float64]:
    """
    Convert 2D pixels (u, v) to 3D points (X, Y, Z) in the world frame.

    Args:
        pixels: list of (u, v) pixel coordinates
        rgbd: an RGBDImageWithContext instance from Spot's capture_images()

    Returns:
        Nx3 array of 3D points in world frame (in meters)
    """
    depth_img = rgbd.depth
    depth_scale = rgbd.depth_scale
    cam_model = rgbd.camera_model
    world_T_cam = rgbd.world_tform_camera  # SE3Pose

    fx = cam_model.intrinsics.focal_length.x
    fy = cam_model.intrinsics.focal_length.y
    cx = cam_model.intrinsics.principal_point.x
    cy = cam_model.intrinsics.principal_point.y

    points_world = []
    for (u, v) in pixels:
        # Get depth (skip invalid or zero)
        depth = depth_img[int(v), int(u)] * depth_scale
        if depth <= 0:
            continue

        # Back-project to camera frame
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])

        # Transform to world frame
        point_world = world_T_cam.to_matrix() @ point_cam
        points_world.append(point_world[:3])

    return np.array(points_world)


def fit_plane_to_points(points_world: NDArray[np.float64]):
    """
    Fit a plane to 3D points and return its centroid and normal vector.

    Args:
        points_world: Nx3 array of 3D points (in world frame)

    Returns:
        centroid: (3,) array, mean position of points
        normal: (3,) array, unit normal vector of best-fit plane
    """
    assert points_world.shape[1] == 3, "Points must be Nx3"

    # Compute centroid
    centroid = np.mean(points_world, axis=0)

    # Subtract centroid
    Q = points_world - centroid

    # Compute covariance and its eigenvectors
    _, _, vh = np.linalg.svd(Q)  # SVD is numerically stable
    normal = vh[-1, :]  # last row of V^T (smallest singular value)

    # Normalize
    normal /= np.linalg.norm(normal)

    return centroid, normal


def grasp_orientation_from_normal(normal_vec: np.ndarray, world_up: np.ndarray = np.array([0, 0, 1])) -> math_helpers.Quat:
    """
    Given a normal vector (pointing outward from the drawer), compute a quaternion
    such that the gripper's +X axis points opposite the normal (toward the drawer).

    Args:
        normal_vec: (3,) array, unit normal vector in world/odom frame.
        world_up: (3,) array, approximate unit vertical direction (default [0,0,1]).

    Returns:
        math_helpers.Quat representing the grasp orientation.
    """
    x_axis = -normal_vec  # gripper faces opposite the drawer normal

    # Construct y and z axes orthogonal to x
    y_axis = np.cross(world_up, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = world_up

    # Rotation matrix (columns are body axes in world frame)
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
        drawer_point: (3,) world/odom coordinates of a point on the drawer surface.
        drawer_normal: (3,) world/odom unit normal vector pointing out of the drawer.
        standoff_dist: distance to stand off from the drawer surface (m).

    Returns:
        math_helpers.SE2Pose representing where the body should move.
    """
    # Compute body target position: move back along -normal by standoff_dist
    body_pos = drawer_point + drawer_normal * standoff_dist

    # Compute yaw angle so body faces *toward* the drawer (along +normal)
    yaw = np.arctan2(drawer_normal[1], drawer_normal[0]) + np.pi  # face opposite normal

    return math_helpers.SE2Pose(body_pos[0], body_pos[1], yaw)


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


def get_multiple_pixels_from_gemini(
    vlm_query_str: str, pil_image: Image, num_pixels: int = 15
) -> list[Tuple[int, int]]:
    # Assuming create_vlm_by_name exists and works like create_llm_by_name
    # Use the specific model name from CFG or hardcode if necessary
    vlm = GoogleGeminiVLM("gemini-1.5-flash")

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
    return pixels


def open_drawer(
    robot: Robot,
    localizer: SpotLocalizer,
    retreat_offset: float = 0.1,
) -> None:
    """
    Reach toward a drawer handle, close the gripper to grasp it, 
    then return to a resting pose and open the gripper.
    
    Args:
        robot: Spot robot instance.
        approach_offset: Distance (m) to stop before touching the handle.
        timeout: Seconds to allow for each arm motion.
    """

    # Capture RGBD image from Spot hand camera
    rgbds = capture_images(robot, localizer, camera_names=["hand_color_image"])
    rgbd = rgbds["hand_color_image"]

    # Extract RGB image
    rgb = rgbd.rgb
    image_pil = Image.fromarray(rgb)

    # Get a 2D pixel on the handle, and convert to 3D point
    handle_pixel = get_pixel_from_gemini(prompt_get_handle_pixel, image_pil)
    handle_3d_point = pixels_to_world_points([handle_pixel], rgbd)[0]
        
    # Get pixels on surface of drawer via SAM (try just Gemini first, get 15 pixels on front of drawer)
    front_surface_pixels = get_multiple_pixels_from_gemini(prompt_get_drawer_surface_pixel, image_pil, 15)

    # Convert to 3D points on surface of drawer
    front_surface_3d_points = pixels_to_world_points(front_surface_pixels, rgbd)

    # Fit a plane to those points via SVD and get normal vector
    _, normal_vector = fit_plane_to_points(front_surface_3d_points)

    # Compute approach grasp pose, aligned to normal
    grasp_rot = grasp_orientation_from_normal(normal_vector)

    # ACTION: Move Spot's body to be aligned to the front of the drawer normal FIRST
    body_target_pose = compute_body_pose_in_front_of_drawer(handle_3d_point, normal_vector, standoff_dist=0.8)
    navigate_to_absolute_pose(robot, localizer, body_target_pose)

    # ACTION: Adjust Spot's height up and down depending on comfortable grasping position, find this param
    set_body_height(robot, 0.0)

    # TODO: Potentially change all frames to vision frame or world frame instead of odom

    # ACTION: Open gripper
    open_gripper(robot)

    # ACTION: Grasp at pixel on handle
    grasp_at_pixel(robot, rgbd, handle_pixel, grasp_rot, move_while_grasping=False)

    # ACTION: Get grasp pose of gripper
    grasp_pose = get_gripper_pose_odom(robot)

    # Compute retreat pose along normal vector
    offset_vec = normal_vector * retreat_offset
    retreat_pose = math_helpers.SE2Pose(
        body_target_pose.x + offset_vec[0],
        grasp_pose.y + offset_vec[1],
        grasp_pose.angle
    )

    # ACTION: Walk backwards to open drawer
    navigate_to_absolute_pose(robot, localizer, retreat_pose)

    # ACTION: Open gripper
    open_gripper(robot)

    # ACTION: Stow arm
    stow_arm(robot)
    

prompt_get_handle_pixel = """
    Point to the handle of the drawer.
    The answer should follow the json format: [{"point": , "label": }, ...]. The points are in [y, x] format normalized to 0-1000.
    """
prompt_get_drawer_surface_pixel = """
    Point to 15 points on the front face of the drawer, but avoid the drawer handles or the edges of the front face.
    The answer should follow the json format: [{"point": , "label": }, ...]. The points are in [y, x] format normalized to 0-1000.
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
        required=True,
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

    # --- Run open_drawer routine ---
    try:
        print("[INFO] Running open_drawer()...")
        open_drawer(robot, localizer, 0.1)
    except Exception as e:
        print(f"[ERROR] open_drawer() failed: {e}")
    finally:
        lease_keepalive.shutdown()
        print("[INFO] Lease returned, exiting cleanly.")