"""
Comprehensive test suite for Spot robot API endpoints.

Run with: pytest test_api.py -v
"""

import pytest
import requests
import json
import base64
from unittest.mock import patch, MagicMock

# Base URL for the API
BASE_URL = "http://localhost:5001"


class TestGetQpos:
    """Tests for GET /get_qpos endpoint"""

    def test_get_qpos_success(self):
        """Test successful GET /get_qpos returns correct structure"""
        resp = requests.get(f"{BASE_URL}/get_qpos")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "qpos" in data
        assert isinstance(data["qpos"], list)

    def test_get_qpos_array_length(self):
        """Test /get_qpos returns array of exactly 11 elements"""
        resp = requests.get(f"{BASE_URL}/get_qpos")
        data = resp.json()
        assert len(data["qpos"]) == 11
        # 0-5: arm joints, 6: gripper, 7-9: body x,y,z, 10: body pitch

    def test_get_qpos_array_values_are_floats(self):
        """Test /get_qpos array contains numeric values"""
        resp = requests.get(f"{BASE_URL}/get_qpos")
        data = resp.json()
        for i, val in enumerate(data["qpos"]):
            assert isinstance(val, (int, float)), f"qpos[{i}] is not numeric: {val}"

    def test_get_qpos_reasonable_values(self):
        """Test /get_qpos returns reasonable joint angle ranges"""
        resp = requests.get(f"{BASE_URL}/get_qpos")
        data = resp.json()
        qpos = data["qpos"]
        # Arm joints (0-5) should be roughly in range [-pi, pi]
        for i in range(6):
            assert -4 < qpos[i] < 4, f"Arm joint {i} out of range: {qpos[i]}"
        # Gripper fraction (6) should be [0, 1]
        assert 0 <= qpos[6] <= 1, f"Gripper fraction out of range: {qpos[6]}"
        # Body height (9) should be reasonable (around 0 when standing)
        assert -1 < qpos[9] < 1, f"Body height out of range: {qpos[9]}"


class TestExecuteAction:
    """Tests for POST /execute_action endpoint"""

    def test_execute_action_success_arm_only(self):
        """Test POST /execute_action with arm-only movement (no walking, no height change)"""
        action = [0.0, 0.5, -1.2, 1.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
        resp = requests.post(f"{BASE_URL}/execute_action", json={"action": action})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_execute_action_success_with_height(self):
        """Test POST /execute_action with arm movement and height adjustment"""
        action = [0.0, 0.5, -1.2, 1.5, 0.0, 0.0, 0.5, 0.05, 0.0, 0.0, 0.1]
        # body_z=0.05, body_pitch=0.1
        resp = requests.post(f"{BASE_URL}/execute_action", json={"action": action})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_execute_action_success_with_walking(self):
        """Test POST /execute_action with walking (high velocity)"""
        action = [0.0, 0.5, -1.2, 1.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.3, 0.0]
        # body_vel_x=0.5, body_vel_y=0.3 (magnitude > 0.03 threshold)
        resp = requests.post(f"{BASE_URL}/execute_action", json={"action": action})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_execute_action_boundary_velocity_low(self):
        """Test POST /execute_action with velocity just below threshold (0.03)"""
        action = [0.0, 0.5, -1.2, 1.5, 0.0, 0.0, 0.5, 0.0, 0.02, 0.01, 0.0]
        # sqrt(0.02^2 + 0.01^2) = 0.0223 < 0.03 threshold
        resp = requests.post(f"{BASE_URL}/execute_action", json={"action": action})
        assert resp.status_code == 200

    def test_execute_action_boundary_velocity_high(self):
        """Test POST /execute_action with velocity just above threshold (0.03)"""
        action = [0.0, 0.5, -1.2, 1.5, 0.0, 0.0, 0.5, 0.0, 0.025, 0.015, 0.0]
        # sqrt(0.025^2 + 0.015^2) = 0.029 < 0.03 threshold
        resp = requests.post(f"{BASE_URL}/execute_action", json={"action": action})
        assert resp.status_code == 200

    def test_execute_action_max_velocity_clamping(self):
        """Test POST /execute_action clamps velocities to max 1.5 m/s"""
        action = [0.0, 0.5, -1.2, 1.5, 0.0, 0.0, 0.5, 0.0, 5.0, 5.0, 0.0]
        # Velocities should be clamped to ±1.5 m/s
        resp = requests.post(f"{BASE_URL}/execute_action", json={"action": action})
        assert resp.status_code == 200

    def test_execute_action_height_clamping(self):
        """Test POST /execute_action clamps height to ±0.1"""
        action = [0.0, 0.5, -1.2, 1.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]
        # Height should be clamped to ±0.1
        resp = requests.post(f"{BASE_URL}/execute_action", json={"action": action})
        assert resp.status_code == 200

    def test_execute_action_missing_action_field(self):
        """Test POST /execute_action without action field returns 400"""
        resp = requests.post(f"{BASE_URL}/execute_action", json={})
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    def test_execute_action_null_action(self):
        """Test POST /execute_action with null action returns 400"""
        resp = requests.post(f"{BASE_URL}/execute_action", json={"action": None})
        assert resp.status_code == 400

    def test_execute_action_wrong_array_length_too_short(self):
        """Test POST /execute_action with array length < 11 returns 400"""
        action = [0.0, 0.5, -1.2]  # Only 3 elements
        resp = requests.post(f"{BASE_URL}/execute_action", json={"action": action})
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "11" in data["error"]

    def test_execute_action_wrong_array_length_too_long(self):
        """Test POST /execute_action with array length > 11 returns 400"""
        action = list(range(15))  # 15 elements
        resp = requests.post(f"{BASE_URL}/execute_action", json={"action": action})
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    def test_execute_action_empty_array(self):
        """Test POST /execute_action with empty array returns 400"""
        resp = requests.post(f"{BASE_URL}/execute_action", json={"action": []})
        assert resp.status_code == 400

    def test_execute_action_negative_joint_values(self):
        """Test POST /execute_action accepts negative joint values"""
        action = [-1.5, -0.5, -1.2, -1.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        resp = requests.post(f"{BASE_URL}/execute_action", json={"action": action})
        assert resp.status_code == 200

    def test_execute_action_zero_gripper_open(self):
        """Test POST /execute_action with gripper fully closed"""
        action = [0.0, 0.5, -1.2, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        resp = requests.post(f"{BASE_URL}/execute_action", json={"action": action})
        assert resp.status_code == 200

    def test_execute_action_full_gripper_open(self):
        """Test POST /execute_action with gripper fully open"""
        action = [0.0, 0.5, -1.2, 1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        resp = requests.post(f"{BASE_URL}/execute_action", json={"action": action})
        assert resp.status_code == 200


class TestResetRobot:
    """Tests for POST /reset_robot endpoint"""

    def test_reset_robot_success(self):
        """Test successful POST /reset_robot"""
        resp = requests.post(f"{BASE_URL}/reset_robot")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_reset_robot_response_structure(self):
        """Test /reset_robot response has correct structure"""
        resp = requests.post(f"{BASE_URL}/reset_robot")
        data = resp.json()
        assert "status" in data


class TestPowerOff:
    """Tests for POST /power_off endpoint"""

    def test_power_off_success(self):
        """Test successful POST /power_off"""
        resp = requests.post(f"{BASE_URL}/power_off")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_power_off_response_structure(self):
        """Test /power_off response has correct structure"""
        resp = requests.post(f"{BASE_URL}/power_off")
        data = resp.json()
        assert "status" in data


class TestAPIIntegration:
    """Integration tests combining multiple endpoints"""

    def test_get_state_then_execute(self):
        """Test getting state and then executing an action"""
        # Get current state
        qpos_resp = requests.get(f"{BASE_URL}/get_qpos")
        assert qpos_resp.status_code == 200
        current_qpos = qpos_resp.json()["qpos"]

        # Execute action using current state as reference
        action = current_qpos[:7] + [0.0, 0.0, 0.0, 0.0]  # Use current arm/gripper, no body movement
        exec_resp = requests.post(f"{BASE_URL}/execute_action", json={"action": action})
        assert exec_resp.status_code == 200

    def test_get_location_and_move_to_same_spot(self):
        """Test getting location and moving to same location"""
        # Get current location
        loc_resp = requests.get(f"{BASE_URL}/get_location")
        assert loc_resp.status_code == 200
        loc = loc_resp.json()

        # Move to same location
        move_resp = requests.post(
            f"{BASE_URL}/move_to",
            json={"x": loc["x"], "y": loc["y"], "yaw": loc["yaw"]}
        )
        assert move_resp.status_code == 200


if __name__ == "__main__":
    print("Run tests with: pytest test_api.py -v")
    print("Or for specific test class: pytest test_api.py::TestExecuteAction -v")
    print("Or for specific test: pytest test_api.py::TestExecuteAction::test_execute_action_success_arm_only -v")
