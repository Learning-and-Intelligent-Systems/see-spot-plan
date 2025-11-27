"""Simple HTTP API for executing Spot plans using the existing exec_plan module.

This exposes minimal endpoints to:
- initialize the robot and localization
- execute a plan from a file path
- execute a plan from a string containing plan commands

It is intentionally small and reuses the globals and functions defined in
`exec_plan.py` so that existing plan files continue to work unchanged.
"""

from __future__ import annotations

from threading import Lock
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import exec_plan
from spot_utils.utils import get_graph_nav_dir


app = FastAPI(title="Spot Plan Execution API")


_execution_lock = Lock()


class InitRequest(BaseModel):
    """Request body for initializing the robot and localizer."""

    hostname: str
    map_name: str
    sam_endpoint: Optional[str] = None


class PlanFromFileRequest(BaseModel):
    """Request body for executing a plan stored in a file."""

    plan_path: str


class PlanFromStringRequest(BaseModel):
    """Request body for executing a plan provided as a string."""

    plan_source: str


def _load_spot_room_pose(map_name: str) -> dict:
    """Load SPOT_ROOM_POSE from the graph_nav metadata for the given map."""
    metadata_path = get_graph_nav_dir(map_name) / "metadata.yaml"
    with open(metadata_path, "rb") as f:
        metadata = yaml.safe_load(f) or {}
    if "spot-room-pose" in metadata:
        return metadata["spot-room-pose"]
    return {"x": 0.0, "y": 0.0, "angle": 0.0}


def _ensure_initialized() -> None:
    """Raise an HTTPException if the robot/localizer are not initialized."""
    if exec_plan.ROBOT is None or exec_plan.LOCALIZER is None:
        raise HTTPException(
            status_code=400,
            detail="Robot is not initialized. Call /init first.",
        )


@app.get("/health")
def health() -> dict:
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "robot_initialized": exec_plan.ROBOT is not None,
        "localizer_initialized": exec_plan.LOCALIZER is not None,
    }


@app.post("/init")
def init(req: InitRequest) -> dict:
    """Initialize the robot, lease, and localizer."""
    with _execution_lock:
        exec_plan.init(req.hostname, req.map_name, req.sam_endpoint)
        exec_plan.SPOT_ROOM_POSE = _load_spot_room_pose(req.map_name)
    return {"status": "initialized"}


@app.post("/run_plan/file")
def run_plan_from_file(req: PlanFromFileRequest) -> dict:
    """Execute a plan from a file path.

    The plan file should contain the same Python commands you would pass to
    exec_plan.py (e.g., calls to move_to, grasp_at_pose, etc.).
    """
    _ensure_initialized()
    with _execution_lock:
        try:
            with open(req.plan_path, "r", encoding="utf-8") as plan_file:
                source = plan_file.read()
        except OSError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read plan file: {exc}",
            ) from exc

        # Execute the plan in the context of the exec_plan module so that
        # all the existing helper functions and globals are available.
        exec(source, exec_plan.__dict__, exec_plan.__dict__)

    return {"status": "completed"}


@app.post("/run_plan/string")
def run_plan_from_string(req: PlanFromStringRequest) -> dict:
    """Execute a plan provided directly as a string."""
    _ensure_initialized()
    with _execution_lock:
        exec(req.plan_source, exec_plan.__dict__, exec_plan.__dict__)
    return {"status": "completed"}


if __name__ == "__main__":
    # Convenience entrypoint for local development:
    #   python exec_plan_api.py --host 0.0.0.0 --port 8000
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Spot Plan Execution API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    uvicorn.run("exec_plan_api:app", host=args.host, port=args.port, reload=False)


