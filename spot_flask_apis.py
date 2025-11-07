from flask import Flask, request, jsonify
from exec_plan import move_to, init, get_graph_nav_dir
from open_drawer import open_drawer
import argparse
import yaml

"""
Example use: 
    python move_to_api.py --hostname 192.168.80.3 --map_name most_recent_map
"""

app = Flask(__name__)

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
        open_drawer(robot, localizer, standoff_dist, body_height_offset, retreat_offset)
        return jsonify({"status": "ok", "message": f"Opened drawer!"})
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
        default=5000
    )
    args = parser.parse_args()

    # Initialize Spot connection
    robot, localizer, sam_endpoint = init(args.hostname, args.map_name, args.sam_endpoint)

    # Start Flask server
    app.run(host="0.0.0.0", port=args.port)