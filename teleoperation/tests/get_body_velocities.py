import argparse
import os
import time
from datetime import datetime

from bosdyn.client import create_standard_sdk
from bosdyn.client.util import authenticate

from spot_utils.utils import get_robot_state, verify_estop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    args = parser.parse_args()
    
    sdk = create_standard_sdk("SpotVelocityMonitor")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()
    
    rate_hz = 10.0
    dt = 1.0 / rate_hz
    
    os.makedirs("teleoperation_data/walking", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"teleoperation_data/walking/body_velocities_{timestamp}.txt"
    
    print(f"Reading body velocities at {rate_hz} Hz. Press Ctrl+C to stop.")
    print(f"Saving data to: {filename}\n")
    print("=" * 60)
    
    with open(filename, "w") as f:
        f.write(f"# Body velocities collected at {rate_hz} Hz\n")
        f.write(f"# Timestamp: {timestamp}\n")
        f.write(f"# Format: timestep, velocity_x, velocity_y, angular_velocity\n\n")
        
        try:
            timestep = 0
            while True:
                start_time = time.time()
                
                robot_state = get_robot_state(robot)
                
                velocity_state = robot_state.kinematic_state.velocity_of_body_in_odom
                vel_x = velocity_state.linear.x
                vel_y = velocity_state.linear.y
                angular_vel = velocity_state.angular.z
                
                print(f"[Timestep {timestep}]")
                print(f"  velocity_x:     {vel_x:8.4f} m/s")
                print(f"  velocity_y:     {vel_y:8.4f} m/s")
                print(f"  angular_vel:    {angular_vel:8.4f} rad/s")
                print()
                
                f.write(f"{timestep}, {vel_x:.6f}, {vel_y:.6f}, {angular_vel:.6f}\n")
                f.flush()
                
                timestep += 1
                
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print(f"\nStopped. Data saved to {filename}")


if __name__ == "__main__":
    main()
