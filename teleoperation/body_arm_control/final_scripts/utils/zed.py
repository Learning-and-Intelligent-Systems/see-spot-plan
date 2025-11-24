#!/usr/bin/env python3
import os
import json
import time
from pathlib import Path

import numpy as np
import cv2
import pyzed.sl as sl
import open3d as o3d


def rgb_depth_to_pointcloud(rgb: np.ndarray, depth: np.ndarray, intrinsics: dict) -> o3d.geometry.PointCloud:
    """
    Convert RGB image, depth map, and camera intrinsics to a colored point cloud.

    Args:
        rgb: RGB image (H x W x 3) in uint8 format
        depth: Depth map (H x W) in meters (float32)
        intrinsics: Dictionary with keys 'fx', 'fy', 'cx', 'cy' for camera intrinsics

    Returns:
        o3d.geometry.PointCloud: Open3D point cloud with colors
    """
    h, w = depth.shape
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    # Create coordinate grids
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)

    # Unproject to 3D using pinhole camera model
    # X = (x - cx) * Z / fx
    # Y = (y - cy) * Z / fy
    # Z = depth
    z = depth
    x_3d = (xx - cx) * z / fx
    y_3d = (yy - cy) * z / fy
    z_3d = z

    # Stack into point cloud (flatten to Nx3)
    points = np.stack([x_3d, y_3d, z_3d], axis=-1).reshape(-1, 3)

    # Flatten RGB for colors (Nx3)
    colors = rgb.reshape(-1, 3) / 255.0  # Normalize to [0, 1]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def save_intrinsics(zed: sl.Camera, out_dir: Path):
    """
    Saves left-camera intrinsics + distortion in a JSON file.
    """
    cam_info = zed.get_camera_information()
    # In ZED SDK 5.x, calibration lives under camera_configuration
    calib = cam_info.camera_configuration.calibration_parameters
    left = calib.left_cam

    intrinsics = {
        "serial_number": cam_info.serial_number,
        "resolution": {
            "width": cam_info.camera_configuration.resolution.width,
            "height": cam_info.camera_configuration.resolution.height,
        },
        "left_cam": {
            "fx": left.fx,
            "fy": left.fy,
            "cx": left.cx,
            "cy": left.cy,
            "distortion": list(left.disto),  # [k1,k2,p1,p2,k3] for standard model
            "fov_horizontal_deg": left.h_fov,
        },
        "right_cam": {
            "fx": calib.right_cam.fx,
            "fy": calib.right_cam.fy,
            "cx": calib.right_cam.cx,
            "cy": calib.right_cam.cy,
            "distortion": list(calib.right_cam.disto),
            "fov_horizontal_deg": calib.right_cam.h_fov,
        },
        "stereo_baseline_mm": calib.get_camera_baseline(),  # baseline between sensors
    }

    out_path = out_dir / "intrinsics.json"
    with open(out_path, "w") as f:
        json.dump(intrinsics, f, indent=2)
    print(f"[OK] Wrote intrinsics to {out_path}")


def stream_zed_frames(
    resolution: int = sl.RESOLUTION.HD720,
    fps: int = 30,
    depth_mode: int = sl.DEPTH_MODE.NEURAL,
    depth_minimum_distance: float = 0.2,
):
    """
    Generator that streams RGB and depth frames from ZED camera.

    Args:
        resolution: ZED resolution (e.g., sl.RESOLUTION.HD720)
        fps: Frames per second
        depth_mode: Depth mode (QUALITY / PERFORMANCE / ULTRA)
        depth_minimum_distance: Minimum distance in meters

    Yields:
        Tuple[np.ndarray, np.ndarray]: (rgb_bgr, depth) where:
            - rgb_bgr: HxWx3 uint8 BGR image
            - depth: HxW float32 depth in meters
    """
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = resolution
    init.camera_fps = fps
    init.depth_mode = depth_mode
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = depth_minimum_distance

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED: {repr(status)}")
    print("[OK] ZED opened")

    runtime = sl.RuntimeParameters()
    rgb_mat = sl.Mat()
    depth_mat = sl.Mat()

    try:
        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(rgb_mat, sl.VIEW.LEFT)
                rgb_bgra = rgb_mat.get_data()
                rgb_bgr = cv2.cvtColor(rgb_bgra, cv2.COLOR_BGRA2BGR)

                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                depth = depth_mat.get_data()

                yield rgb_bgr, depth
            else:
                time.sleep(0.001)

    finally:
        zed.close()
        print("[OK] ZED closed")


def save_zed_frames(
    output_dir: Path,
    num_frames: int = None,
    print_fps: bool = True,
    fps_print_interval: int = 30,
    **stream_kwargs,
):
    """
    Stream frames from ZED and save to disk.

    Args:
        output_dir: Directory to save frames
        num_frames: Number of frames to capture (None = stream until interrupted)
        print_fps: Whether to print FPS statistics
        fps_print_interval: Print FPS every N frames
        **stream_kwargs: Arguments passed to stream_zed_frames()
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rgb").mkdir(exist_ok=True)
    (output_dir / "depth_png").mkdir(exist_ok=True)
    (output_dir / "depth_npy").mkdir(exist_ok=True)

    # Save intrinsics for the first frame's camera state
    zed_temp = sl.Camera()
    init = sl.InitParameters(**{k: v for k, v in stream_kwargs.items()
                                if k in ['resolution', 'fps', 'depth_mode', 'depth_minimum_distance']})
    if not hasattr(init, 'camera_resolution'):
        init.camera_resolution = stream_kwargs.get('resolution', sl.RESOLUTION.HD720)
    if not hasattr(init, 'camera_fps'):
        init.camera_fps = stream_kwargs.get('fps', 30)
    if not hasattr(init, 'depth_mode'):
        init.depth_mode = stream_kwargs.get('depth_mode', sl.DEPTH_MODE.ULTRA)
    if not hasattr(init, 'coordinate_units'):
        init.coordinate_units = sl.UNIT.METER
    if not hasattr(init, 'depth_minimum_distance'):
        init.depth_minimum_distance = stream_kwargs.get('depth_minimum_distance', 0.2)

    status = zed_temp.open(init)
    if status == sl.ERROR_CODE.SUCCESS:
        save_intrinsics(zed_temp, output_dir)
        zed_temp.close()

    frame_idx = 0
    t0 = time.time()

    print("Streaming... press Ctrl+C to stop.")
    try:
        for rgb_bgr, depth in stream_zed_frames(**stream_kwargs):
            rgb_path = output_dir / "rgb" / f"{frame_idx:06d}.png"
            depth_png_path = output_dir / "depth_png" / f"{frame_idx:06d}.png"
            depth_npy_path = output_dir / "depth_npy" / f"{frame_idx:06d}.npy"

            cv2.imwrite(str(rgb_path), rgb_bgr)
            # Convert depth to uint16 for visualization
            # Normalize depth to 0-65535 range for 16-bit PNG
            depth_valid = depth[~np.isnan(depth)]
            if len(depth_valid) > 0:
                depth_min = depth_valid.min()
                depth_max = depth_valid.max()
                depth_normalized = ((depth - depth_min) / (depth_max - depth_min + 1e-6) * 65535).astype(np.uint16)
                depth_normalized[np.isnan(depth)] = 0  # Set NaN to black
            else:
                depth_normalized = np.zeros_like(depth, dtype=np.uint16)
            cv2.imwrite(str(depth_png_path), depth_normalized)
            np.save(str(depth_npy_path), depth)

            frame_idx += 1

            if print_fps and frame_idx % fps_print_interval == 0:
                fps = frame_idx / (time.time() - t0)
                print(f"Frames: {frame_idx} | FPS: {fps:.1f}")

            if num_frames is not None and frame_idx >= num_frames:
                break

    except KeyboardInterrupt:
        print("\nStopping...")


def main():
    dirpath = "/home/kelly_lucy/policy/zed_examples/zed_capture"
    capture_index = 20
    rgb_filename = f"rgb/{capture_index:06d}.png"
    depth_png_filename = f"depth_png/{capture_index:06d}.png"
    intrinsics_filename = "intrinsics.json"

    rgb_path = os.path.join(dirpath, rgb_filename)
    depth_png_path = os.path.join(dirpath, depth_png_filename)
    intrinsics_path = os.path.join(dirpath, intrinsics_filename)

    rgb = cv2.imread(rgb_path)
    # Read depth PNG (saved as uint16 in millimeters)
    depth_png = cv2.imread(depth_png_path, cv2.IMREAD_UNCHANGED)
    # Convert from millimeters (uint16) to meters (float32)
    depth = depth_png.astype(np.float32) / 1000.0
    intrinsics_data = json.load(open(intrinsics_path))

    # Extract left camera intrinsics
    left_intrinsics = intrinsics_data["left_cam"]

    # pcd = rgb_depth_to_pointcloud(rgb, depth, left_intrinsics)
    # o3d.visualization.draw_geometries([pcd])
    # # o3d.io.write_point_cloud(os.path.join(dirpath, "pcd.ply"), pcd)
    # # print(f"[OK] Wrote point cloud to {os.path.join(dirpath, 'pcd.ply')}")
    # assert False

    out_dir = Path("./zed_capture")
    save_zed_frames(out_dir)


if __name__ == "__main__":
    main()
