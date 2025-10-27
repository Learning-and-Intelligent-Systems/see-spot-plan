import json
from typing import Tuple

import PIL

from spot_utils.pretrained_model_interface import GoogleGeminiVLM


def get_pixel_from_gemini(
    vlm_query_str: str, pil_image: PIL.Image.Image
) -> Tuple[int, int]:
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
    # Assuming the first point is the desired one
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
    # Denormalize from 0-1000 range to image pixel coordinates
    img_height = pil_image.height
    img_width = pil_image.width
    y = int(y_norm * img_height / 1000.0)
    x = int(x_norm * img_width / 1000.0)
    # Clamp coordinates to be within image bounds
    y = max(0, min(y, img_height - 1))
    x = max(0, min(x, img_width - 1))
    # Assign to the 'pixel' variable in (x, y) format
    pixel = (x, y)
    return pixel



if __name__ == "__main__":
    import logging
    import rerun as rr
    from PIL import Image
    import cv2
    import numpy as np


    rr.init("gemini_test", spawn=True)
    logging.basicConfig(level=logging.DEBUG)

    pil_image = Image.open("388885845015.jpg")

    vlm_query_template = """
    Point to the human.
    The answer should follow the json format: [{"point": , "label": }, ...]. The points are in [y, x] format normalized to 0-1000.
    """

    pixel = get_pixel_from_gemini(vlm_query_template, pil_image)

    image_np = np.asarray(pil_image)
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.circle(bgr, pixel, 5, (0, 0, 255), -1)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rr.log("rgb", rr.Image(rgb))
    print()
