import requests
import json
import numpy as np
from scipy.spatial.transform import Rotation

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