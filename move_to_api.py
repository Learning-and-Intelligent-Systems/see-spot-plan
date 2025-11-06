from flask import Flask, request, jsonify
from exec_plan import move_to, init, get_graph_nav_dir
import argparse
import yaml

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

    if None in (x, y, yaw):
        return jsonify({"error": "Missing x, y, or yaw"}), 400

    try:
        move_to(float(x), float(y), float(yaw))
        return jsonify({"status": "ok", "message": f"Moved to ({x}, {y}, {yaw})"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API for Spot")
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
    parser.add_argument(
        "--plan", type=str, required=True, help="Path of the Plan to run"
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
    init(args.hostname, args.map_name, args.sam_endpoint)

    # Load SPOT_ROOM_POSE if available
    with open(get_graph_nav_dir(args.map_name) / "metadata.yaml", "rb") as f:
        metadata = yaml.safe_load(f)
        if "spot-room-pose" in metadata.keys():
            SPOT_ROOM_POSE = metadata["spot-room-pose"]
        else:
            print("spot-room-pose not found in metadata.yaml, using default val")
            SPOT_ROOM_POSE = {"x": 0.0, "y": 0.0, "angle": 0.0}

    # Start Flask server
    app.run(host="0.0.0.0", port=args.port)