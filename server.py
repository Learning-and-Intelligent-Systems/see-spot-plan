from flask import Flask, request, jsonify
from exec_plan import move_to, init, get_graph_nav_dir
from open_drawer import open_drawer
from close_drawer import close_drawer
from look_into_container import look_into_container
import argparse
import base64
from io import BytesIO
from pathlib import Path
import yaml

"""
Example use: 
    python move_to_api.py --hostname 192.168.80.3 --map_name most_recent_map
"""

app = Flask(__name__)

@app.route("/get_location", methods=["GET"])
def api_get_location():
    """
    Returns the robot's current location in the mapping world frame (GraphNav seed frame).
    Example response:
    {
        "x": 1.23,
        "y": 0.45,
        "z": 0.0,
        "yaw": 1.57
    }
    """
    try:
        # Re-localize to update Spot's position
        localizer.localize()

        # Get last known robot pose
        pose = localizer.get_last_robot_pose()

        # Extract translation and rotation (yaw)
        x, y, z, yaw = pose.x, pose.y, pose.z, pose.rot.to_yaw()

        return jsonify({"status": "ok", "x": x, "y": y, "z": z, "yaw": yaw})

    except Exception as e:
        import traceback
        print("ERROR", e)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/move_to", methods=["POST"])
def api_move_to():
    """
    Expects JSON:
    {
        "x": <float>,
        "y": <float>,
        "yaw": <float>
    }
    """
    data = request.get_json()
    x = data.get("x")
    y = data.get("y")
    yaw = data.get("yaw")
    print(x)
    print(y)
    print(yaw)

    if None in (x, y, yaw):
        return jsonify({"error": "Missing x, y, or yaw"}), 400

    try:
        move_to(float(x), float(y), float(yaw))
        return jsonify({"status": "ok", "message": f"Moved to ({x}, {y}, {yaw})"})
    except Exception as e:
        import traceback
        print("ERROR:", e)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
    

# open_drawer(robot, localizer, standoff_dist=1.1, body_height_offset=0.0, retreat_offset=0.4, checkpoint=7)
@app.route("/open_drawer", methods=["POST"])
def api_open_drawer():
    """
    Expects JSON:
    {
        "standoff_dist": <float>,
        "body_height_offset": <float>,
        "retreat_offset": <float>
    }
    """
    data = request.get_json()
    standoff_dist = data.get("standoff_dist")
    body_height_offset = data.get("body_height_offset")
    retreat_offset = data.get("retreat_offset")
    print(standoff_dist)
    print(body_height_offset)
    print(retreat_offset)

    if None in (standoff_dist, body_height_offset, retreat_offset):
        return jsonify({"error": "Missing standoff_dist, body_height_offset, or retreat_offset"}), 400

    try:
        drawer_image = open_drawer(
            robot, localizer, standoff_dist, body_height_offset, retreat_offset
        )

        if drawer_image is None:
            return jsonify({"status": "ok", "message": "Opened drawer, no image captured"})

        buffer = BytesIO()
        drawer_image.save(buffer, format="PNG")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return jsonify(
            {
                "status": "ok",
                "message": "Opened drawer!",
                "image": {
                    "mime_type": "image/png",
                    "data": encoded,
                },
            }
        )
    except Exception as e:
        import traceback
        print("ERROR:", e)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/look_into_container", methods=["POST"])
def api_look_into_container():
    data = request.get_json() or {}

    forward_offset = data.get("forward_offset", 0.8)
    vertical_offset = data.get("vertical_offset", 0.6)
    pitch_deg = data.get("pitch_deg", 85.0)
    settle_seconds = data.get("settle_seconds", 1.5)
    image_basename = data.get("image_basename", "container_inspection")
    stow_after = data.get("stow_after", True)
    open_before_capture = data.get("open_before_capture", True)
    center_on_container = data.get("center_on_container", True)

    try:
        image = look_into_container(
            robot,
            localizer,
            forward_offset=float(forward_offset),
            vertical_offset=float(vertical_offset),
            pitch_deg=float(pitch_deg),
            settle_seconds=float(settle_seconds),
            image_basename=image_basename,
            stow_after=bool(stow_after),
            open_before_capture=bool(open_before_capture),
            center_on_container=bool(center_on_container),
        )

        img_buffer = BytesIO()
        image.save(img_buffer, format="JPEG")
        img_buffer.seek(0)
        image_bytes = img_buffer.read()
        encoded = base64.b64encode(image_bytes).decode("utf-8")

        image_path = Path(f"{image_basename}.jpg")
        return jsonify(
            {
                "status": "ok",
                "message": "Captured container image",
                "image": {
                    "mime_type": "image/jpeg",
                    "data": encoded,
                },
                "image_path": str(image_path),
                "image_size": {"width": image.size[0], "height": image.size[1]},
            }
        )
    except Exception as e:
        import traceback

        print("ERROR:", e)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API for Spot")
    parser.add_argument(
        "--hostname",
        type=str,
        required=True,
        help="The robot's hostname (e.g. 192.168.80.3)",
    )
    parser.add_argument(
        "--map_name",
        type=str,
        required=True,
        help="The name of the map folder to load (sub-folder under graph_nav_maps)",
    )
    parser.add_argument(
        "--sam_endpoint",
        type=str,
        required=False,
        help="Address of endpoint hosting GroundedSAM",
    )
    parser.add_argument(
        "--port", 
        type=int, 
        required=False, 
        default=5001
    )
    args = parser.parse_args()

    # Initialize Spot connection
    robot, localizer, sam_endpoint = init(args.hostname, args.map_name, args.sam_endpoint)

    # Start Flask server
    app.run(host="0.0.0.0", port=args.port)