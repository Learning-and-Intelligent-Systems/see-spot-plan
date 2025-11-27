"""
run_plan_example.py

Example script for calling the Spot plan execution FastAPI service.

It:
1) Initializes the Spot robot and localizer via POST /init.
2) Sends a small Python "plan" (list of command strings) via POST /run_plan/string.
"""

from typing import List, Dict, Any
import os

import requests


# Base URL for the exec_plan_api FastAPI server
SPOT_PLAN_API_BASE_URL = os.getenv("SPOT_PLAN_API_BASE_URL", "http://0.0.0.0:8001")

SPOT_PLAN_HEALTH_URL = f"{SPOT_PLAN_API_BASE_URL}/health"
SPOT_PLAN_INIT_URL = f"{SPOT_PLAN_API_BASE_URL}/init"
SPOT_PLAN_RUN_STRING_URL = f"{SPOT_PLAN_API_BASE_URL}/run_plan/string"

# Default Spot connection + map; override via environment variables as needed.
SPOT_HOSTNAME = os.getenv("SPOT_HOSTNAME", "192.168.80.3")
SPOT_MAP_NAME = os.getenv("SPOT_MAP_NAME", "spot_apple_test")
SPOT_SAM_ENDPOINT = os.getenv("SPOT_SAM_ENDPOINT")  # optional


def ensure_spot_api_initialized() -> None:
    """
    Ensure the Spot plan execution API has an initialized robot and localizer.

    It first checks /health; if the robot/localizer are not initialized, it
    issues a POST /init with the configured hostname/map.
    """
    try:
        resp = requests.get(SPOT_PLAN_HEALTH_URL, timeout=5)
        resp.raise_for_status()
        data: Dict[str, Any] = resp.json()
        if data.get("robot_initialized") and data.get("localizer_initialized"):
            return
    except requests.RequestException:
        # If health check fails, fall through and attempt initialization anyway.
        pass

    init_body: Dict[str, Any] = {
        "hostname": SPOT_HOSTNAME,
        "map_name": SPOT_MAP_NAME,
    }
    if SPOT_SAM_ENDPOINT is not None:
        init_body["sam_endpoint"] = SPOT_SAM_ENDPOINT

    resp = requests.post(SPOT_PLAN_INIT_URL, json=init_body, timeout=60)
    resp.raise_for_status()
    print("Initialized Spot via /init")


def build_plan_source(plan_lines: List[str]) -> str:
    """
    Convert a list of plan command lines into a single Python source string.

    The FastAPI server expects:
        {"plan_source": "<python commands as a single string>"}
    """
    # Normalize lines: strip trailing newlines, then join with '\n' and
    # ensure a final newline at the end.
    stripped_lines = [line.rstrip("\n") for line in plan_lines if line.strip()]
    return "\n".join(stripped_lines) + "\n"


def run_plan_from_lines(plan_lines: List[str]) -> Dict[str, Any]:
    """
    Ensure the API is initialized and then execute the given plan lines.
    """
    # 1) Make sure Spot + localizer are initialized
    ensure_spot_api_initialized()

    # 2) Build the code string to send
    plan_source = build_plan_source(plan_lines)
    print("Sending plan_source to /run_plan/string:")
    print(plan_source)

    # 3) Call the FastAPI endpoint
    resp = requests.post(
        SPOT_PLAN_RUN_STRING_URL,
        json={"plan_source": plan_source},
        # Grasping can take a while (image capture, VLM call, planning, execution),
        # so allow a generous read timeout here.
        timeout=300,
    )
    resp.raise_for_status()
    result = resp.json()
    print("Plan execution result:", result)
    return result


if __name__ == "__main__":
    # Example plan you provided:
    example_plan_lines = [
        "move_to(x_abs=1.520925736602535, y_abs=-2.965914221681329, yaw_abs=1.078456328340842)\n",
        "gaze('AHEAD')",
        "grasp(text_prompt='red apple')",
        "stow_arm(ROBOT)",
    ]

    run_plan_from_lines(example_plan_lines)


