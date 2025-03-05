"""Helper functions for capturing and manipulating images from Spot's cameras."""

from typing import Collection, Dict, Optional, Type

import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.sdk import Robot
from numpy.typing import NDArray

from spot_utils.perception.perception_structs import RGBDImageWithContext
from spot_utils.spot_localization import SpotLocalizer


def _image_response_to_image(
    image_response: image_pb2.ImageResponse,
) -> NDArray:
    """Extract an image from an image response.

    The type of image (rgb, depth, etc.) is detected based on the
    format.
    """
    # pylint: disable=no-member
    pixel_format = image_response.shot.image.pixel_format
    if pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype: Type[np.unsignedinteger] = np.uint16
    else:
        dtype = np.uint8

    # Read from the buffer.
    raw_image = image_response.shot.image
    img = np.frombuffer(raw_image.data, dtype=dtype)
    if image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(raw_image.rows, raw_image.cols)
    else:
        img = cv2.imdecode(img, -1)

    # Convert BGR to RGB.
    if pixel_format in [
        image_pb2.Image.PIXEL_FORMAT_RGB_U8,
        image_pb2.Image.PIXEL_FORMAT_RGBA_U8,
    ]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Squeeze to remove last channel for depth and grayscale images.
    img = img.squeeze()

    return img


ROTATION_ANGLE = {
    "hand_color_image": 0,
    "back_fisheye_image": 0,
    "frontleft_fisheye_image": -78,
    "frontright_fisheye_image": -102,
    "left_fisheye_image": 0,
    "right_fisheye_image": 180,
}
RGB_TO_DEPTH_CAMERAS = {
    "hand_color_image": "hand_depth_in_hand_color_frame",
    "left_fisheye_image": "left_depth_in_visual_frame",
    "right_fisheye_image": "right_depth_in_visual_frame",
    "frontleft_fisheye_image": "frontleft_depth_in_visual_frame",
    "frontright_fisheye_image": "frontright_depth_in_visual_frame",
    "back_fisheye_image": "back_depth_in_visual_frame",
}

# Hack to avoid double image capturing when we want to (1) get object states
# and then (2) use the image again for pixel-based grasping.
_LAST_CAPTURED_IMAGES: Dict[str, RGBDImageWithContext] = {}


def capture_images(
    robot: Robot,
    localizer: SpotLocalizer,
    camera_names: Optional[Collection[str]] = None,
    quality_percent: int = 100,
    relocalize: bool = False,
) -> Dict[str, RGBDImageWithContext]:
    """Build an image request and get the responses.

    If no camera names are provided, all RGB cameras are used.
    """
    global _LAST_CAPTURED_IMAGES  # pylint: disable=global-statement

    if camera_names is None:
        camera_names = set(RGB_TO_DEPTH_CAMERAS)

    image_client = robot.ensure_client(ImageClient.default_service_name)

    rgbds: Dict[str, RGBDImageWithContext] = {}

    # Get the world->robot transform so we can store world->camera transforms
    # in the RGBDWithContexts.
    if relocalize:
        localizer.localize()
    world_tform_body = localizer.get_last_robot_pose()
    body_tform_world = world_tform_body.inverse()

    # Package all the requests together.
    img_reqs: image_pb2.ImageRequest = []
    for camera_name in camera_names:
        # Build RGB image request.
        if "hand" in camera_name:
            rgb_pixel_format = None
        else:
            rgb_pixel_format = image_pb2.Image.PIXEL_FORMAT_RGB_U8  # pylint: disable=no-member
        rgb_img_req = build_image_request(
            camera_name, quality_percent=quality_percent, pixel_format=rgb_pixel_format
        )
        img_reqs.append(rgb_img_req)
        # Build depth image request.
        depth_camera_name = RGB_TO_DEPTH_CAMERAS[camera_name]
        depth_img_req = build_image_request(
            depth_camera_name, quality_percent=quality_percent, pixel_format=None
        )
        img_reqs.append(depth_img_req)

    # Send the request.
    responses = image_client.get_image(img_reqs)
    name_to_response = {r.source.name: r for r in responses}

    # Build RGBDImageWithContexts.
    for camera_name in camera_names:
        rgb_img_resp = name_to_response[camera_name]
        depth_img_resp = name_to_response[RGB_TO_DEPTH_CAMERAS[camera_name]]
        rgb_img = _image_response_to_image(rgb_img_resp)
        depth_img = _image_response_to_image(depth_img_resp)
        # Create transform.
        camera_tform_body = get_a_tform_b(
            rgb_img_resp.shot.transforms_snapshot,
            rgb_img_resp.shot.frame_name_image_sensor,
            BODY_FRAME_NAME,
        )
        camera_tform_world = camera_tform_body * body_tform_world
        world_tform_camera = camera_tform_world.inverse()
        # Extract other context.
        rot = ROTATION_ANGLE[camera_name]
        depth_scale = depth_img_resp.source.depth_scale
        transforms_snapshot = rgb_img_resp.shot.transforms_snapshot
        frame_name_image_sensor = rgb_img_resp.shot.frame_name_image_sensor
        camera_model = rgb_img_resp.source.pinhole
        # Finish RGBDImageWithContext.
        rgbd = RGBDImageWithContext(
            rgb_img,
            depth_img,
            rot,
            camera_name,
            world_tform_camera,
            depth_scale,
            transforms_snapshot,
            frame_name_image_sensor,
            camera_model,
        )
        rgbds[camera_name] = rgbd

    _LAST_CAPTURED_IMAGES = rgbds

    return rgbds
