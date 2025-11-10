import argparse
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.util import authenticate
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b, VISION_FRAME_NAME, get_se2_a_tform_b
from spot_utils.gemini_utils import get_pixel_from_gemini

from spot_utils.utils import get_graph_nav_dir, verify_estop, get_robot_state
from spot_utils.spot_localization import SpotLocalizer
from spot_utils.perception.spot_cameras import capture_images
from skills.grasp import grasp_at_pixel
from skills.spot_hand_move import open_gripper, stow_arm
from skills.spot_navigation import navigate_to_relative_pose

from open_drawer import (
    compute_body_pose_in_front_of_drawer,
    draw_colored_pixels,
    fit_plane_to_points,
    gaze,
    get_multiple_pixels_from_gemini,
    get_points_from_pixels,
    pixels_to_vision_points,
    prompt_get_handle_pixel,
    set_body_height,
    move_hand_back,
    navigate_to_vision_goal,
)


prompt_get_drawer_surface_pixel = """
You are looking at a cabinet with several drawers, but only one drawer is open and it has a green handle wrapped in tape.
Select exactly 15 points on the flat front face of that open drawer only.
Rules:
- Ignore all closed drawers, cabinet edges, and background objects.
- Do not place points on the green handle, tape, hardware, or any side/top/bottom faces.
- The points must lie entirely on the visible rectangular face that moves when the drawer is opened.
- Do not mess up. IF you pick wrong points not on the face of the open drawer, the robot will slam into the cabinet and the lab will be down $20,000.
Return JSON of the form [{"point": [y, x], "label": "open_drawer_face"}] with coordinates normalized to 0-1000.
"""

def _find_valid_depth_pixel(
    pixel: Tuple[int, int], depth: np.ndarray, max_radius: int = 3
) -> Tuple[int, int]:
    """Return the closest pixel around `pixel` with valid depth (non-zero and within range)."""
    u0, v0 = pixel
    height, width = depth.shape
    for radius in range(max_radius + 1):
        for dv in range(-radius, radius + 1):
            for du in range(-radius, radius + 1):
                u = u0 + du
                v = v0 + dv
                if not (0 <= u < width and 0 <= v < height):
                    continue
                z = float(depth[v, u])
                if z > 0.0 and z <= 2.5:
                    return (u, v)
    return pixel


def close_drawer(
    robot,
    localizer,
    standoff_dist: float = 0.8,
    body_height_offset: float = 0.0,
    advance_offset: float = 0.5,
    checkpoint: int = 7,
) -> Optional[Image.Image]:
    # Retreat slightly to mirror the opening routine.
    retreat_pose = math_helpers.SE2Pose(-0.3, 0.0, 0.0)
    navigate_to_relative_pose(robot, retreat_pose)
    if checkpoint == 0:
        return None

    # Point the hand camera towards the drawer.
    gaze(robot, "AHEAD")


    # Capture RGBD image from Spot hand camera
    rgbds = capture_images(robot, localizer, camera_names=["hand_color_image"])
    rgbd = rgbds["hand_color_image"]
    # rgbd=None

    # Extract RGB image and depth image
    rgb = rgbd.rgb
    depth = rgbd.depth
    depth_pil = Image.fromarray(depth)
    depth_pil.save("close_raw_hand_camera_depth.png")
    image_pil = Image.fromarray(rgb)
    image_pil.save("close_raw_hand_camera_output.jpg")

    # Get a 2D pixel on the handle, and convert to 3D point
    handle_pixel = get_pixel_from_gemini(prompt_get_handle_pixel, image_pil)
    draw_colored_pixels(image_pil, [handle_pixel], "close_annotated_hand_camera_output.jpg", "red")
    
    cam_model = rgbd.camera_model  

    fx = cam_model.intrinsics.focal_length.x
    fy = cam_model.intrinsics.focal_length.y
    cx = cam_model.intrinsics.principal_point.x
    cy = cam_model.intrinsics.principal_point.y

    intrinsics = [fx, fy, cx, cy]

    rgb_image_path = "close_raw_hand_camera_output.jpg"
    depth_image_path = "close_raw_hand_camera_depth.png"
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
    # rr.log_points("3D_points", positions=points_vision, colors=colors, radii=voxel_size / 2)
    
    # Get pixels on surface of drawer via SAM (try just Gemini first, get 15 pixels on front of drawer)
    front_surface_pixels = get_multiple_pixels_from_gemini(prompt_get_drawer_surface_pixel, image_pil, 15)
    draw_colored_pixels(image_pil, front_surface_pixels, "close_annotated_hand_camera_output.jpg", "blue")
    if checkpoint == 0:
        return None

    # Convert to 3D points on surface of drawer
    front_surface_3d_points = pixels_to_vision_points(front_surface_pixels, rgbd)
    # rr.log_points("surface_points", positions=front_surface_3d_points, colors=[255, 0, 0], radii=voxel_size * 1.5)

    # Fit a plane to those points via SVD and get normal vector
    _, normal_vector = fit_plane_to_points(front_surface_3d_points)
    # print("NORMAL VECTOR IS: ", normal_vector)

    # Compute approach grasp pose, aligned to normal
    # grasp_rot = grasp_orientation_from_normal(normal_vector)

    # ACTION: Move Spot's body to be aligned to the front of the drawer normal FIRST
    # rotated_body_pose = compute_rotated_body_pose(robot, normal_vector)
    # print("ROTATED POSE IS: ", rotated_body_pose)
    # navigate_to_vision_goal(robot, rotated_body_pose)
    robot_state = get_robot_state(robot)
    transforms = robot_state.kinematic_state.transforms_snapshot
    vision_tform_body = get_se2_a_tform_b(transforms, VISION_FRAME_NAME, BODY_FRAME_NAME)
    current_body_xy = np.array([vision_tform_body.x, vision_tform_body.y])

    body_target_pose = compute_body_pose_in_front_of_drawer(
        handle_3d_point, normal_vector, current_body_xy, standoff_dist
    )
    # print("BODY POSE IS: ", body_target_pose)
    navigate_to_vision_goal(robot, body_target_pose)
    if checkpoint == 1:
        return None

    # # ACTION: Adjust Spot's height up and down depending on comfortable grasping position, find this param
    # set_body_height(robot, body_height_offset)

    # ACTION: Grasp at pixel on handle using the original frame.
    grasp_at_pixel(robot, rgbd, handle_pixel, move_while_grasping=False)
    if checkpoint == 2:
        return None

    # Push the drawer closed by walking straight forward.
    push_pose = math_helpers.SE2Pose(advance_offset, 0.0, 0.0)
    navigate_to_relative_pose(robot, push_pose)
    if checkpoint == 3:
        return None

    open_gripper(robot)
    if checkpoint == 4:
        return None

    retreat_pose = math_helpers.SE2Pose(-0.15, 0.0, 0.0)
    navigate_to_relative_pose(robot, retreat_pose)

    stow_arm(robot)
    if checkpoint == 5:
        return None

    return image_pil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Close an opened drawer with Spot.")
    parser.add_argument(
        "--hostname",
        type=str,
        required=True,
        help="Robot hostname or IP (e.g. 192.168.80.3)",
    )
    parser.add_argument(
        "--map_name",
        type=str,
        required=True,
        help="GraphNav map folder under spot_utils/graph_nav_maps",
    )
    parser.add_argument(
        "--standoff_dist",
        type=float,
        default=0.8,
        help="Stand-off distance from the drawer front (meters)",
    )
    parser.add_argument(
        "--body_height_offset",
        type=float,
        default=0.0,
        help="Body height offset before closing (meters)",
    )
    parser.add_argument(
        "--advance_offset",
        type=float,
        default=0.55,
        help="Distance to move while pulling/pushing the drawer (meters)",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=7,
        help="Checkpoint to stop after (mirrors open_drawer checkpoints)",
    )
    args = parser.parse_args()

    sdk = create_standard_sdk("SpotCloseDrawerClient")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    lease_keepalive = LeaseKeepAlive(
        lease_client, must_acquire=True, return_at_exit=True
    )

    path = get_graph_nav_dir(args.map_name)
    localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
    robot.time_sync.wait_for_sync()
    localizer.localize()

    try:
        close_drawer(
            robot,
            localizer,
            standoff_dist=args.standoff_dist,
            body_height_offset=args.body_height_offset,
            advance_offset=args.advance_offset,
            checkpoint=args.checkpoint,
        )
    finally:
        pass
        # stow_arm(robot)

