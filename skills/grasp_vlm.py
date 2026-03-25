"""High-level VLM-based grasping skill.

This module implements a grasp skill that:
1) Captures an image from Spot's hand camera.
2) Uses a Google Gemini VLM to point to an object described by a text prompt.
3) Converts the VLM's [y, x] normalized coordinates into a pixel.
4) Executes a low-level grasp at that pixel using ``skills.grasp.grasp_at_pixel``.
"""

from __future__ import annotations

import json
from typing import Optional, Tuple

import cv2
import numpy as np
import rerun as rr
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot
from PIL import Image

from skills.grasp import grasp_at_pixel
from spot_utils.perception.spot_cameras import capture_images
from spot_utils.pretrained_model_interface import GoogleGeminiVLM
from spot_utils.spot_localization import SpotLocalizer


def _get_pixel_from_gemini(vlm_query_str: str, pil_image: Image.Image) -> Tuple[int, int]:
    """Query Gemini VLM to get a single pixel [y, x] normalized to 0-1000, then
    denormalize to image pixel coordinates.

    This mirrors the usage pattern used in the wipe skill: construct a
    ``GoogleGeminiVLM`` instance and call ``sample_completions`` directly.
    """
    vlm = GoogleGeminiVLM("gemini-2.5-pro")
    print(f"Using Gemini VLM model: {vlm.get_id()}")
    print(f"Querying Gemini VLM with prompt: {vlm_query_str}")

    def _strip_markdown_fence(json_output_str: str) -> str:
        """Remove ```json fences if present and return the inner JSON string."""
        lines = json_output_str.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                json_output_str = "\n".join(lines[i + 1 :])
                json_output_str = json_output_str.split("```")[0]
                break
        return json_output_str.strip()

    # 1) Query the VLM
    vlm_output_list = vlm.sample_completions(
        prompt=vlm_query_str,
        imgs=[pil_image],
        temperature=0.0,
        seed=42,
        num_completions=1,
    )
    vlm_output_str = vlm_output_list[0]

    # 2) Parse JSON output
    json_string_to_parse = _strip_markdown_fence(vlm_output_str)
    parsed_data = json.loads(json_string_to_parse)

    if not isinstance(parsed_data, list) or not parsed_data:
        raise ValueError("Parsed JSON is not a non-empty list.")

    first_point_obj = parsed_data[0]
    if (
        "point" not in first_point_obj
        or not isinstance(first_point_obj["point"], list)
        or len(first_point_obj["point"]) != 2
    ):
        raise ValueError(
            "First element in JSON does not contain a valid 'point' list [y, x]."
        )

    y_norm, x_norm = first_point_obj["point"]
    if not isinstance(y_norm, (int, float)) or not isinstance(x_norm, (int, float)):
        raise ValueError("Normalized coordinates are not numbers.")

    # 3) Denormalize from 0–1000 range to image pixel coordinates
    img_height = pil_image.height
    img_width = pil_image.width
    y = int(y_norm * img_height / 1000.0)
    x = int(x_norm * img_width / 1000.0)

    # Clamp to image bounds
    y = max(0, min(y, img_height - 1))
    x = max(0, min(x, img_width - 1))

    # Return as (x, y) pixel coordinate
    return (x, y)


def test_pointing(image_path: str, text_prompt: str) -> Tuple[int, int]:
    """Offline test helper: run Gemini pointing on a saved RGB image.

    Loads an image from disk, runs the same VLM prompt as ``grasp_with_vlm``,
    logs the raw and annotated images to Rerun, and returns the predicted pixel.
    """
    # Load image (BGR) and convert to RGB
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")
    rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Log raw image
    rr.log("test/rgb_raw", rr.Image(rgb_np))

    # Use the same prompt format as in grasp_with_vlm
    vlm_query_template = f"""
    Point to the {text_prompt}. If you cannot see the {text_prompt} fully, point to the best guess.
    The answer should follow the json format: [{{"point": , "label": }}, ...]. The points are in [y, x] format normalized to 0-1000.
    """

    pil_image = Image.fromarray(rgb_np)
    pixel = _get_pixel_from_gemini(vlm_query_template, pil_image)

    # Draw the predicted pixel on the image
    cv2.circle(bgr, pixel, 5, (0, 0, 255), -1)
    annotated_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rr.log("test/pointing", rr.Image(annotated_rgb))

    # Optionally save annotated image to disk for quick inspection
    out_path = image_path.replace(".png", "_pointed.png").replace(".jpg", "_pointed.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
    print(f"Gemini predicted pixel {pixel}, annotated image saved to: {out_path}")

    return pixel

def grasp_with_vlm(
    robot: Robot,
    localizer: SpotLocalizer,
    text_prompt: Optional[str],
) -> None:
    """High-level grasp skill that uses a VLM to choose a pixel from a prompt.

    Args:
        robot: The Spot ``Robot`` instance.
        localizer: The ``SpotLocalizer`` for the current map.
        text_prompt: Text description of the object to grasp (e.g., "red apple").

    """
    camera = "hand_color_image"
    print(f"Calling grasping with VLM for text prompt: {text_prompt}")

    # Capture an image from the hand camera.
    images = capture_images(robot, localizer, [camera])
    rgbd = images[camera]
    rgb_np = rgbd.rgb
    rr.log("rgb_raw", rr.Image(rgb_np))

    # Call Gemini to point to the object described by the prompt.
    vlm_query_template = f"""
    Point to the {text_prompt}. If you cannot see the {text_prompt} fully, point to the best guess.
    The answer should follow the json format: [{{"point": , "label": }}, ...]. The points are in [y, x] format normalized to 0-1000.
    """
    image_pil = Image.fromarray(rgb_np)
    pixel = _get_pixel_from_gemini(vlm_query_template, image_pil)

    # Draw pixel on the image for logging.
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    cv2.circle(bgr, pixel, 5, (0, 0, 255), -1)
    rgb_annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rr.log("pointing", rr.Image(rgb_annotated))

    if pixel is not None:
        # Grasp at the pixel with a top-down grasp.
        top_down_rot = math_helpers.Quat.from_pitch(np.pi / 2)
        grasp_at_pixel(robot, rgbd, pixel, grasp_rot=top_down_rot)
        return

    raise RuntimeError("Grasp failed: VLM did not return a valid pixel.")


def main() -> None:
    """Simple manual test entrypoint for the pointing VLM."""
    rr.init("grasp_vlm_pointing_test", spawn=True)
    image_path = "push_button_images/rgb_20251102_192102.png"
    pixel = test_pointing(image_path, text_prompt="brown button")
    print(f"Test pointing pixel: {pixel}")


if __name__ == "__main__":
    main()