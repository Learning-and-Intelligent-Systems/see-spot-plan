from dataclasses import dataclass
from typing import Any

import numpy as np
from bosdyn.api.geometry_pb2 import FrameTreeSnapshot
from bosdyn.client import math_helpers
from numpy.typing import NDArray
from scipy import ndimage


@dataclass
class RGBDImageWithContext:
    """An RGBD image with context including the pose and intrinsics of the
    camera."""

    rgb: NDArray[np.uint8]
    depth: NDArray[np.uint16]
    image_rot: float
    camera_name: str
    world_tform_camera: math_helpers.SE3Pose
    depth_scale: float
    transforms_snapshot: FrameTreeSnapshot
    frame_name_image_sensor: str
    camera_model: Any  # bosdyn.api.image_pb2.PinholeModel, but not available

    @property
    def rotated_rgb(self) -> NDArray[np.uint8]:
        """The image rotated to be upright."""
        return ndimage.rotate(self.rgb, self.image_rot, reshape=False)
