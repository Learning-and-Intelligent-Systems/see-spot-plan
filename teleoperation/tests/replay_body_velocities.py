import argparse
import time

from bosdyn.client import create_standard_sdk
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.util import authenticate
from bosdyn.util import seconds_to_duration

from spot_utils.utils import verify_estop


def replay_walking_trajectory(robot, filename, rate_hz=50.0):
    print(f"\nReading velocity data from: {filename}")
    
    velocities_data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split(',')
            timestep = int(parts[0])
            vel_x = float(parts[1])
            vel_y = float(parts[2])
            ang_vel = float(parts[3])
            velocities_data.append((timestep, vel_x, vel_y, ang_vel))
    
    print(f"Loaded {len(velocities_data)} timesteps (collected at 10 Hz)")
    print(f"Replaying at {rate_hz} Hz (interpolated). Press Ctrl+C to stop.\n")
    
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    
    dt = 1.0 / rate_hz
    original_dt = 0.1
    
    try:
        for i in range(len(velocities_data) - 1):
            timestep_start, vel_x_start, vel_y_start, ang_vel_start = velocities_data[i]
            timestep_end, vel_x_end, vel_y_end, ang_vel_end = velocities_data[i + 1]
            
            num_interp_steps = int(original_dt / dt)
            
            for j in range(num_interp_steps):
                start_time = time.time()
                
                alpha = j / num_interp_steps
                vel_x = vel_x_start * (1 - alpha) + vel_x_end * alpha
                vel_y = vel_y_start * (1 - alpha) + vel_y_end * alpha
                ang_vel = ang_vel_start * (1 - alpha) + ang_vel_end * alpha
                
                end_time = time.time() + 0.2
                
                cmd = RobotCommandBuilder.synchro_velocity_command(
                    v_x=vel_x,
                    v_y=vel_y,
                    v_rot=ang_vel
                )
                
                cmd.synchronized_command.mobility_command.se2_velocity_request.end_time.CopyFrom(
                    robot.time_sync.robot_timestamp_from_local_secs(end_time)
                )
                
                command_client.robot_command(cmd)
                
                if (i * num_interp_steps + j) % 50 == 0:
                    print(f"Timestep {i}: vel_x={vel_x:.4f} m/s, vel_y={vel_y:.4f} m/s, ang_vel={ang_vel:.4f} rad/s")
                
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)
        
        cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0, v_y=0, v_rot=0)
        command_client.robot_command(cmd)
        print("\nStopping robot...")
        
    except KeyboardInterrupt:
        cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0, v_y=0, v_rot=0)
        command_client.robot_command(cmd)
        print("\nReplay stopped. Robot stopped.")
    
    print(f"\nReplay complete! Executed {len(velocities_data)} timesteps.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, required=True)
    parser.add_argument("--filename", type=str, required=True, help="Path to velocity data file")
    args = parser.parse_args()
    
    sdk = create_standard_sdk("SpotWalkingReplay")
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()
    
    replay_walking_trajectory(robot, args.filename)


if __name__ == "__main__":
    main()

