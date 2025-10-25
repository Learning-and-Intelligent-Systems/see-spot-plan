"""Interface for opening a drawer."""

import argparse
import time
import numpy as np
from PIL import Image

from bosdyn.api import (
    arm_command_pb2,
    manipulation_api_pb2,
    robot_command_pb2,
    synchronized_command_pb2,
    trajectory_pb2,
)
from bosdyn.client.image import ImageClient
import cv2
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

from spot_utils.utils import verify_estop, get_pixel_from_gemini
from skills.grasp import grasp_at_pixel
from skills.spot_hand_move import move_hand_to_relative_pose, open_gripper, close_gripper, stow_arm
from spot_utils.perception.spot_cameras import capture_images
from spot_utils.spot_localization import SpotLocalizer



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


vlm_query_template = """
    Point to the handle of the drawer.
    The answer should follow the json format: [{"point": , "label": }, ...]. The points are in [y, x] format normalized to 0-1000.
    """
localizer = None # edit this to define a localizer

def open_drawer(
    robot: Robot,
    localizer: SpotLocalizer,
    vlm_query_str: str,
    retreat_offset: float = 0.1,
    timeout: float = 15.0,
) -> None:
    """
    Reach toward a drawer handle, close the gripper to grasp it, 
    then return to a resting pose and open the gripper.
    
    Args:
        robot: Spot robot instance.
        approach_offset: Distance (m) to stop before touching the handle.
        timeout: Seconds to allow for each arm motion.
    """

    # Make branch of see-spot-plan to collab on

    # Move Spot's body to be aligned to the front of the drawer normal FIRST

    # Capture RGBD image from Spot hand camera
    rgbds = capture_images(robot, localizer, camera_names=["hand_color_image"])
    rgbd = rgbds["hand_color_image"]

    # Extract RGB, depth, and camera instrinsics
    rgb = rgbd.rgb
    depth = rgbd.depth
    fx, fy, cx, cy = (
        rgbd.camera_model.intrinsics.focal_length.x,
        rgbd.camera_model.intrinsics.focal_length.y,
        rgbd.camera_model.intrinsics.principal_point.x,
        rgbd.camera_model.intrinsics.principal_point.y,
    )
    image_pil = Image.fromarray(rgb)

    # Get a 2D pixel on the handle
    pixel = get_pixel_from_gemini(vlm_query_str, image_pil)
        
    # Get pixels on surface of drawer via SAM (try just Gemini first, get 5 pixels on front of drawer)

    # Convert to 3D points on surface of drawer

    # Fit a plane to those points via SVD

    # Compute normal vector to that plane
    normal_vector = None

    # Compute approach grasp pose, aligned to normal
    grasp_rot = math_helpers.Quat.from_pitch(np.pi/2)  # edit this to be correct

    # Adjust Spot's height up and down depending on comfortable grasping position arm rel to body

    # Potentially change all frames to vision frame instead of odom

    # Open gripper
    open_gripper(robot)

    # Grasp at pixel on handle
    grasp_at_pixel(robot, rgbd, pixel, grasp_rot, move_while_grasping=False)

    # Get grasp pose of gripper
    grasp_pose = get_gripper_pose_odom(robot)

    # Compute retreat pose along normal vector
    offset_vec = normal_vector * retreat_offset
    retreat_pos = math_helpers.SE3Pose(
        grasp_pose.x + offset_vec[0],
        grasp_pose.y + offset_vec[1],
        grasp_pose.z + offset_vec[2],
        grasp_pose.rot
    )

    # Pull back to open drawer
    move_hand_to_absolute_pose(robot, retreat_pos)

    # Open gripper
    open_gripper(robot)

    # Stow arm
    stow_arm(robot)
    