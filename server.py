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
import time
import numpy as np
from bosdyn.api import arm_command_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.geometry import EulerZXY
from google.protobuf import wrappers_pb2

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


@app.route("/get_qpos", methods=["GET"])
def api_get_qpos():
    """
    Get current robot state (arm joints, gripper, body position/pitch).

    Returns:
    {
        "status": "ok",
        "qpos": [arm_j0, arm_j1, arm_j2, arm_j3, arm_j4, arm_j5, gripper_frac,
                  body_x, body_y, body_z, body_pitch]
    }
    """
    try:
        # Get robot state
        robot_state_client = robot.ensure_client("robot_state")
        state = robot_state_client.get_robot_state()

        # Extract arm joint positions (first 6 values)
        arm_joints = [float(state.kinematic_state.joint_states[i].position.value) for i in range(6)]

        # Extract gripper open fraction (normalize percentage to 0.0-1.0)
        gripper_fraction = state.manipulator_state.gripper_open_percentage / 100.0

        # Extract body position and pitch
        body_frame_state = state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get("body")
        body_x = body_frame_state.parent_tform_child.position.x
        body_y = body_frame_state.parent_tform_child.position.y
        body_z = body_frame_state.parent_tform_child.position.z
        body_pitch = body_frame_state.parent_tform_child.rotation.to_yaw()

        # Assemble qpos array
        qpos = arm_joints + [gripper_fraction, body_x, body_y, body_z, body_pitch]

        return jsonify({"status": "ok", "qpos": qpos})

    except Exception as e:
        import traceback
        print("ERROR:", e)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/execute_action", methods=["POST"])
def api_execute_action():
    """
    Execute a single action timestep (1/20 Hz = 50ms).

    Expects JSON:
    {
        "action": [arm_j0, arm_j1, arm_j2, arm_j3, arm_j4, arm_j5, gripper_frac,
                    body_z, body_vel_x, body_vel_y, body_pitch]
    }

    Logic:
    - If body velocity magnitude > 0.03 m/s: send velocity command (walking)
    - Otherwise: send arm movement with optional height adjustment (standing)
    """
    data = request.get_json()
    action = data.get("action")

    if not action or len(action) != 11:
        return jsonify({"error": "action must be an array of 11 elements"}), 400

    try:
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()

        # Extract action components
        arm_q = action[0:6]  # First 6: arm joint targets
        gripper_fraction = action[6]  # Element 6: gripper open fraction
        body_z = action[7]  # Element 7: body height offset
        body_vel_x = action[8]  # Element 8: body velocity x
        body_vel_y = action[9]  # Element 9: body velocity y
        body_pitch = action[10]  # Element 10: body pitch

        # Build gripper command
        gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(gripper_fraction)
        gripper_command = gripper_cmd.synchronized_command.gripper_command

        # Check if we should send velocity commands (walking) vs arm-only movement
        velocity_magnitude = np.sqrt(body_vel_x**2 + body_vel_y**2)
        velocity_threshold = 0.03  # 3 cm/s threshold

        arm_command = None
        mobility_command = None

        # Build arm command - only send if arm position changed significantly
        # For single timestep, we always build and send arm command
        trajectory_time = 0.05  # 50ms for 20 Hz
        point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            arm_q[0], arm_q[1], arm_q[2],
            arm_q[3], arm_q[4], arm_q[5],
            time_since_reference_secs=trajectory_time,
        )

        max_vel = wrappers_pb2.DoubleValue(value=15.0)
        max_acc = wrappers_pb2.DoubleValue(value=30.0)

        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(
            points=[point],
            maximum_velocity=max_vel,
            maximum_acceleration=max_acc,
        )

        joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(
            trajectory=arm_joint_traj
        )
        arm_command = arm_command_pb2.ArmCommand.Request(
            arm_joint_move_command=joint_move_command
        )

        # Determine if walking or standing with arm movement
        if velocity_magnitude > velocity_threshold:
            # WALKING: send velocity command
            max_velocity = 1.5
            final_v_x = max(-max_velocity, min(max_velocity, body_vel_x))
            final_v_y = max(-max_velocity, min(max_velocity, body_vel_y))
            final_v_rot = 0.0

            velocity_cmd = RobotCommandBuilder.synchro_velocity_command(
                v_x=final_v_x,
                v_y=final_v_y,
                v_rot=final_v_rot
            )
            mobility_command = velocity_cmd.synchronized_command.mobility_command

            # Send arm + gripper + velocity
            sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                arm_command=arm_command,
                gripper_command=gripper_command,
                mobility_command=mobility_command
            )
        else:
            # STANDING: check if height adjustment is needed
            if body_z is not None and abs(body_z) > 0.001:  # 1mm threshold
                # Send height adjustment with arm movement
                final_height_offset = max(-0.1, min(0.1, body_z))

                footprint_R_body = EulerZXY(yaw=0.0, roll=0.0, pitch=body_pitch)
                stand_cmd = RobotCommandBuilder.synchro_stand_command(
                    body_height=final_height_offset,
                    footprint_R_body=footprint_R_body
                )
                mobility_command = stand_cmd.synchronized_command.mobility_command

                sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                    arm_command=arm_command,
                    gripper_command=gripper_command,
                    mobility_command=mobility_command
                )
            else:
                # ARM ONLY: no walking, no height adjustment
                sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                    arm_command=arm_command,
                    gripper_command=gripper_command
                )

        # Send the synchronized command
        robot_command = robot_command_pb2.RobotCommand(synchronized_command=sync_command)
        command_client.robot_command(robot_command)

        return jsonify({"status": "ok"})

    except Exception as e:
        import traceback
        print("ERROR:", e)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/reset_robot", methods=["POST"])
def api_reset_robot():
    """
    Reset robot to safe position (stow arm, stand).
    """
    try:
        from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient

        command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        # Build stow command
        stow_cmd = RobotCommandBuilder.arm_stow_command()

        # Send to robot
        command_client.robot_command(stow_cmd)

        # Wait for command to execute
        time.sleep(1.0)

        return jsonify({"status": "ok"})

    except Exception as e:
        import traceback
        print("ERROR:", e)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/power_off", methods=["POST"])
def api_power_off():
    """
    Power off the robot safely.
    """
    try:
        robot.power_off()
        return jsonify({"status": "ok"})

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