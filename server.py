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
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.geometry import EulerZXY
from google.protobuf import wrappers_pb2
from google.protobuf.timestamp_pb2 import Timestamp

"""
Example use: 
    python move_to_api.py --hostname 192.168.80.3 --map_name most_recent_map
"""

app = Flask(__name__)

# Global variables for robot and localizer (set in main)
robot = None
localizer = None
nominal_height = -5.850433805135551  # Nominal standing height for body z offset calculation

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
    Get current robot state (arm joints, gripper, body height, velocity, pitch).

    Returns:
    {
        "status": "ok",
        "qpos": [arm_j0, arm_j1, arm_j2, arm_j3, arm_j4, arm_j5, gripper_frac,
                  body_z, body_velocity_x, body_velocity_y, body_pitch]
    }

    Note: body_z is absolute z position in odom frame.
    For height offset in commands, subtract nominal_height (-5.850433805135551).
    Velocities are in body frame (v_x forward/back, v_y left/right).
    """
    try:
        # Get robot state
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        state = robot_state_client.get_robot_state()

        arm_joint_names = ["arm0.sh0", "arm0.sh1", "arm0.el0", "arm0.el1", "arm0.wr0", "arm0.wr1"]
        joint_states = state.kinematic_state.joint_states
        joint_dict = {js.name: js for js in joint_states}

        arm_joints = []
        for joint_name in arm_joint_names:
                arm_joints.append(float(joint_dict[joint_name].position.value))

        # Extract gripper open fraction (normalize percentage to 0.0-1.0)
        gripper_fraction = state.manipulator_state.gripper_open_percentage / 100.0
        gripper_fraction = max(0.0, min(1.0, gripper_fraction))

        # Extract body position and pitch using proper frame helpers
        odom_tform_body = get_odom_tform_body(state.kinematic_state.transforms_snapshot)
        body_pos = odom_tform_body.position
        body_rot = odom_tform_body.rotation

        body_z = body_pos.z

        # Extract Euler angles from quaternion
        w, x, y, z = body_rot.w, body_rot.x, body_rot.y, body_rot.z
        body_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        body_pitch = np.arcsin(2*(w*y - z*x))

        # Extract body velocity in odom frame
        v_x = 0.0
        v_y = 0.0
        try:
            if hasattr(state.kinematic_state, 'velocity_of_body_in_odom'):
                vel = state.kinematic_state.velocity_of_body_in_odom
                if vel is not None:
                    if hasattr(vel, 'linear') and vel.linear is not None:
                        v_x = vel.linear.x if hasattr(vel.linear, 'x') else 0.0
                        v_y = vel.linear.y if hasattr(vel.linear, 'y') else 0.0
        except (AttributeError, KeyError, TypeError):
            pass

        # Transform velocity from odom frame to body frame
        cos_yaw = np.cos(body_yaw)
        sin_yaw = np.sin(body_yaw)
        v_x_body = v_x * cos_yaw + v_y * sin_yaw
        v_y_body = -v_x * sin_yaw + v_y * cos_yaw

        # Assemble qpos array: [6 arm joints + gripper + body_z + body_vel_x + body_vel_y + pitch]
        qpos = arm_joints + [gripper_fraction, body_z, v_x_body, v_y_body, body_pitch]

        return jsonify({"status": "ok", "qpos": qpos})

    except Exception as e:
        import traceback
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"ERROR in /get_qpos: {error_msg}")
        print(error_traceback)
        return jsonify({
            "status": "error",
            "message": error_msg,
            "error_type": type(e).__name__
        }), 500


@app.route("/execute_action", methods=["POST"])
def api_execute_action():
    """
    Execute a single action timestep (1/20 Hz = 50ms).

    Expects JSON:
    {
        "action": [arm_j0, arm_j1, arm_j2, arm_j3, arm_j4, arm_j5, gripper_frac,
                    body_z, body_vel_x, body_vel_y, body_pitch]
    }

    Note: body_z is absolute z position (same format as /get_qpos).
    It is converted to height offset internally for movement commands.

    Logic:
    - If body velocity magnitude > 0.03 m/s: send velocity command (walking)
    - Otherwise: send arm movement with optional height adjustment (standing)
    """
    data = request.get_json()
    action = data.get("action")

    if not action or len(action) != 11:
        return jsonify({"error": "action must be an array of 11 elements"}), 400

    try:
        if robot is None:
            return jsonify({"status": "error", "message": "Robot not initialized"}), 500
        
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        # Don't take lease - localizer already has one from init()

        # Extract action components
        arm_q = action[0:6]  # First 6: arm joint targets
        gripper_fraction = action[6]  # Element 6: gripper open fraction
        body_z_absolute = action[7]  # Element 7: body z (absolute position in odom frame)
        body_vel_x = action[8]  # Element 8: body velocity x
        body_vel_y = action[9]  # Element 9: body velocity y
        body_pitch = action[10]  # Element 10: body pitch

        # Convert absolute body_z to height offset relative to nominal standing height
        global nominal_height
        body_z_offset = body_z_absolute - nominal_height

        # Build gripper command
        gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(gripper_fraction)
        gripper_command = gripper_cmd.synchronized_command.gripper_command

        # Check if we should send velocity commands (walking) vs arm-only movement
        velocity_magnitude = np.sqrt(body_vel_x**2 + body_vel_y**2)
        velocity_threshold = 0.03  # 3 cm/s threshold

        arm_command = None
        mobility_command = None

        # Determine if walking or standing with arm movement
        if velocity_magnitude > velocity_threshold:
            # WALKING: send velocity command only (skip arm to avoid timing conflicts)
            max_velocity = 1.5
            final_v_x = max(-max_velocity, min(max_velocity, body_vel_x))
            final_v_y = max(-max_velocity, min(max_velocity, body_vel_y))
            final_v_rot = 0.0

            # Use velocities directly: v_x_body is forward/back, v_y_body is left/right
            # This matches the convention used in data collection (collect_body_arm_data_remote.py)
            velocity_cmd = RobotCommandBuilder.synchro_velocity_command(
                v_x=final_v_x,  # body_vel_x = forward/backward in body frame
                v_y=final_v_y,  # body_vel_y = left/right in body frame
                v_rot=final_v_rot
            )
            mobility_command = velocity_cmd.synchronized_command.mobility_command

            # Send gripper + velocity only (no arm command when walking to avoid timing conflicts)
            sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                gripper_command=gripper_command,
                mobility_command=mobility_command
            )
        else:
            # STANDING: build arm command and check if height or pitch adjustment is needed
            # At 20 Hz, each command executes for 50ms (0.05s)
            # Use larger buffer to account for network latency and prevent ExpiredError
            trajectory_time = 0.5  # 200ms (50ms execution + 150ms buffer for network latency)
            point = RobotCommandBuilder.create_arm_joint_trajectory_point(
                arm_q[0], arm_q[1], arm_q[2],
                arm_q[3], arm_q[4], arm_q[5],
                time_since_reference_secs=trajectory_time,
            )

            max_vel = wrappers_pb2.DoubleValue(value=2.0)
            max_acc = wrappers_pb2.DoubleValue(value=4.0)

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
            
            height_needed = body_z_offset is not None and abs(body_z_offset) > 0.001  # 1mm threshold
            pitch_needed = abs(body_pitch) > 0.001  # 1mm threshold in radians

            if height_needed or pitch_needed:
                # Send height/pitch adjustment with arm movement
                final_height_offset = max(-0.1, min(0.1, body_z_offset)) if height_needed else 0.0

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
                # ARM ONLY: no walking, no height adjustment, no pitch change
                sync_command = synchronized_command_pb2.SynchronizedCommand.Request(
                    arm_command=arm_command,
                    gripper_command=gripper_command
                )

        # Send the synchronized command
        robot_command = robot_command_pb2.RobotCommand(synchronized_command=sync_command)
        
        # Set end_time_secs to prevent ExpiredError
        # At 20 Hz, each command should execute for 50ms, but add buffer for network latency
        # For velocity commands, set a longer duration to avoid expiration
        if velocity_magnitude > velocity_threshold:
            # Walking: velocity commands need longer duration to avoid expiration
            end_time_secs = time.time() + 0.5  # 500ms buffer for network latency
        else:
            # Standing: arm commands have trajectory_time, so shorter duration is fine
            end_time_secs = time.time() + 0.3  # 300ms buffer for network latency
        
        command_client.robot_command(robot_command, end_time_secs=end_time_secs)

        return jsonify({"status": "ok"})

    except Exception as e:
        import traceback
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"ERROR in /execute_action: {error_msg}")
        print(error_traceback)
        return jsonify({
            "status": "error", 
            "message": error_msg,
            "error_type": type(e).__name__
        }), 500


@app.route("/reset_robot", methods=["POST"])
def api_reset_robot():
    """
    Reset robot to safe position (stow arm, stand).
    """
    try:
        if robot is None:
            return jsonify({"status": "error", "message": "Robot not initialized"}), 500
        
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        # Don't take lease - localizer already has one from init()

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

    # Initialize Spot connection (module-level variables)
    robot, localizer, sam_endpoint = init(args.hostname, args.map_name, args.sam_endpoint)

    # Start Flask server
    app.run(host="0.0.0.0", port=args.port)