import argparse
import os
import time
from datetime import datetime, timezone

from bosdyn.api import geometry_pb2
from bosdyn.client import create_standard_sdk
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.util import authenticate

from spot_utils.utils import get_robot_state, verify_estop


def collect_hand_pose_data(robot, rate_hz=50.0):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "teleoperation_data"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"hand_pose_{timestamp}.txt")
    
    print(f"\nCollecting hand pose data at {rate_hz} Hz")
    print(f"Output: {filename}")
    print("Press Ctrl+C to stop.\n")
    
    state_client = robot.ensure_client(RobotStateClient.default_service_name)
    
    dt = 1.0 / rate_hz
    timestep = 0
    
    with open(filename, 'w') as f:
        f.write("# Time is UTC UNIX epoch seconds (float)\n")
        f.write("# Format: utc_epoch_seconds, x, y, z, qw, qx, qy, qz, gripper_open_fraction\n")
        f.write("# Hand pose is in body frame (frozen body)\n\n")
        
        try:
            while True:
                start_time = time.time()
                
                robot_state = state_client.get_robot_state()
                utc_ts = datetime.now(timezone.utc).timestamp()
                
                hand_tform = get_a_tform_b(
                    robot_state.kinematic_state.transforms_snapshot,
                    BODY_FRAME_NAME,
                    "hand"
                )
                
                pos = hand_tform.position
                rot = hand_tform.rotation
                
                gripper_raw = robot_state.manipulator_state.gripper_open_percentage
                gripper_state = gripper_raw / 100.0
                gripper_state = max(0.0, min(1.0, gripper_state))
                if gripper_state < 0.02:
                    gripper_state = 0.0
                
                f.write(f"{utc_ts}, {pos.x}, {pos.y}, {pos.z}, {rot.w}, {rot.x}, {rot.y}, {rot.z}, {gripper_state}\n")
                f.flush()
                
                if timestep % 50 == 0:
                    print(f"[Timestep {timestep}]")
                    print(f"  hand x: {pos.x:.4f} m")
                    print(f"  hand y: {pos.y:.4f} m")
                    print(f"  hand z: {pos.z:.4f} m")
                    gripper_status = "OPEN" if gripper_state > 0.5 else "CLOSED"
                    print(f"  gripper: {gripper_state:.4f} [{gripper_status}]")
                
                timestep += 1
                
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print(f"\nStopped. Data saved to {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    parser.add_argument("--rate", type=float, default=50.0)
    args = parser.parse_args()
    
    sdk = create_standard_sdk("SpotHandPoseCollect")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()
    
    collect_hand_pose_data(robot, rate_hz=args.rate)


if __name__ == "__main__":
    main()

