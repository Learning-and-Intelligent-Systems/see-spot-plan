import requests
import json
import numpy as np
from scipy.spatial.transform import Rotation

def request_get_location(endpoint_ip: str = "127.0.0.1", port: int = 5000):
    url = f"http://{endpoint_ip}:{port}/get_location"    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        print("Location received:", data)

        x, y, z, yaw = data["x"], data["y"], data["z"], data["yaw"]
        return convert_spot_to_polycam_point(x, y, z, yaw)

    else:
        print("Error:", response.status_code, response.text)


def send_move_to(x: float, y: float, yaw: float, endpoint_ip: str = "127.0.0.1", port: int = 5000):
    url = f"http://{endpoint_ip}:{port}/move_to"
    payload = {"x": x, "y": y, "yaw": yaw}
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("Move command accepted:", response.json())
    else:
        print("Error:", response.status_code, response.text)


def send_open_drawer(standoff_dist: float = 1.1, body_height_offset: float = 0.0, retreat_offset: float = 0.4, endpoint_ip: str = "127.0.0.1", port: int = 5000):
    url = f"http://{endpoint_ip}:{port}/open_drawer"
    payload = {
        "standoff_dist": standoff_dist, 
        "body_height_offset": body_height_offset, 
        "retreat_offset": retreat_offset
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print("Open drawer request accepted:", response.json())
    else:
        print("Error:", response.status_code, response.text)

def get_transformation_matrix(path: str = "transformation.json"):
    """
    Return 4x4 homogenous matrix that transforms polycam points to spot world points.
    """
    pose = json.load(open(path))
    t, q = np.array(pose[0]), np.array(pose[1])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T

def convert_spot_to_polycam_point(x, y, z, yaw):
    """
    Convert point in spot world mapping to polycam.
    """
    spot_world_point = np.array([x, y, z, 1])

     # Get spot to polycam transform
    T_polycam_to_spot = get_transformation_matrix()
    T_spot_to_polycam = np.linalg.inv(T_polycam_to_spot)

    # Get Spot's polycam location by applying spot to polycam transform
    spot_polycam_point = T_spot_to_polycam @ spot_world_point
    x_p, y_p, z_p = spot_polycam_point[0], spot_polycam_point[1], spot_polycam_point[2]

    # Get Spot's transformed yaw
    R_polycam_to_spot = T_polycam_to_spot[:3, :3]
    # This means we represent the quaternion as euler angles and get the rotation around z axis (yaw)
    yaw_offset = Rotation.from_matrix(R_polycam_to_spot).as_euler('zyx')[0]
    yaw_p = yaw - yaw_offset

    return x_p, y_p, z_p, yaw_p


# Transform point in polycam to spot world frame
polycam_point = np.array([-2.393238, -1.6113499, 0, 1])     # replace hardcoded point with generated point
T = get_transformation_matrix()
spot_world_point = T @ polycam_point

x_des, y_des = spot_world_point[0], spot_world_point[1]
yaw_des = 0
print(x_des, y_des, yaw_des)

# Example use:
send_move_to(x_des, y_des, yaw_des, endpoint="127.0.0.1")
send_open_drawer(endpoint="127.0.0.1")
x_p, y_p, z_p, yaw_p = request_get_location(endpoint="127.0.0.1")