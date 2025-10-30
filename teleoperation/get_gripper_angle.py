import argparse

from bosdyn.client import create_standard_sdk
from bosdyn.client.util import authenticate

from spot_utils.utils import get_robot_state, verify_estop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    args = parser.parse_args()
    
    sdk = create_standard_sdk("SpotGripperMonitor")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()
    
    print("Getting gripper state...\n")
    
    robot_state = get_robot_state(robot)
    
    print("=== Manipulator State ===")
    print(f"gripper_open_percentage: {robot_state.manipulator_state.gripper_open_percentage}")
    print(f"\nType: {type(robot_state.manipulator_state.gripper_open_percentage)}")
    
    print("\n=== All manipulator_state attributes ===")
    for attr in dir(robot_state.manipulator_state):
        if not attr.startswith('_'):
            try:
                value = getattr(robot_state.manipulator_state, attr)
                if not callable(value):
                    print(f"{attr}: {value}")
            except:
                pass
    
    print("\n=== Gripper Interpretation ===")
    gripper_pct = robot_state.manipulator_state.gripper_open_percentage
    print(f"Value: {gripper_pct}")
    print(f"Status: ", end="")
    if gripper_pct > 0.8:
        print("OPEN")
    elif gripper_pct > 0.2:
        print("PARTIALLY OPEN/CLOSED")
    else:
        print("CLOSED")
    
    normalized_gripper_pct = gripper_pct/100
    print(f"\nNormalized (0-1): {normalized_gripper_pct:.6f}")


if __name__ == "__main__":
    main()

