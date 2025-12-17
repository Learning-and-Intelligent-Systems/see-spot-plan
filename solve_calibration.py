"""
Load and visualize calibration samples from will_test1 directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import rerun as rr
from cv2 import aruco

CHARUCOBOARD_ROWCOUNT = SQUARES_Y = 9
CHARUCOBOARD_COLCOUNT = SQUARES_X = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.015
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)

CHARUCO_BOARD = aruco.CharucoBoard(
    size=(SQUARES_X, SQUARES_Y),
    squareLength=CHARUCOBOARD_CHECKER_SIZE,
    markerLength=CHARUCOBOARD_MARKER_SIZE,
    dictionary=ARUCO_DICT,
)

# Detector Params
detector_params = cv2.aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
calib_flags = (
    cv2.CALIB_USE_INTRINSIC_GUESS
    + cv2.CALIB_FIX_PRINCIPAL_POINT
    + cv2.CALIB_FIX_FOCAL_LENGTH
)

charuco_params = aruco.CharucoParameters()
charuco_params.tryRefineMarkers = True


@dataclass
class SpotFrame:
    """RGBD + pose from Spot's in-hand camera at a single time."""

    rgb_path: str
    depth_path: str
    camera_matrix: np.ndarray  # 3x3 intrinsics
    T_body_hand: np.ndarray  # 4x4 BODY->hand camera
    rgb: np.ndarray
    depth: np.ndarray


@dataclass
class IphoneFrame:
    """RGBD + intrinsics from iPhone at a single time."""

    rgb_path: str
    depth_path: str
    camera_matrix_rgb: np.ndarray  # 3x3 RGB intrinsics
    camera_matrix_depth: np.ndarray  # 3x3 depth intrinsics
    rgb: np.ndarray
    depth: Optional[np.ndarray]


@dataclass
class CalibrationSample:
    """One paired observation of the Charuco board from both cameras."""

    sample_idx: int

    # BODY -> hand camera transform at capture time
    T_body_hand: np.ndarray

    # Frames
    spot_frame: Optional[SpotFrame] = None
    iphone_frame: Optional[IphoneFrame] = None


def rgbd_to_point_cloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float = 1000.0,
    max_depth: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an RGBD image and intrinsics into a point cloud in the camera frame.

    Args:
        rgb: HxWx3 uint8 array, assumed RGB.
        depth: HxW (or HxWx1) array, uint16 in depth_scale units or float32 meters.
        intrinsics: 3x3 matrix or array-like [fx, fy, cx, cy].
        depth_scale: Scale factor from uint16 depth units to meters (default: 1000).
        max_depth: Optional maximum depth in meters for filtering points.

    Returns:
        points: Nx3 float32 array of 3D points in the camera frame.
        colors: Nx3 float32 array of RGB colors in [0, 1].
    """
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if depth.dtype == np.uint16 or depth.dtype == np.int32:
        depth_m = depth.astype(np.float32) / float(depth_scale)
    else:
        depth_m = depth.astype(np.float32)

    H, W = depth_m.shape
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_NEAREST)

    K = np.asarray(intrinsics, dtype=np.float32)
    if K.shape == (3, 3):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
    elif K.size == 4:
        fx, fy, cx, cy = K.ravel().tolist()
    else:
        raise ValueError(f"Intrinsics must be 3x3 or length-4, got shape {K.shape}")

    u_coords, v_coords = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )

    z = depth_m
    valid = z > 0
    if max_depth is not None:
        valid &= z <= float(max_depth)

    x = (u_coords - cx) / fx * z
    y = (v_coords - cy) / fy * z

    points = np.stack((x, y, z), axis=-1)[valid]

    rgb_float = rgb.astype(np.float32) / 255.0
    colors = rgb_float.reshape(-1, 3)[valid.ravel()]

    return points.astype(np.float32), colors.astype(np.float32)


def detect_and_visualize_charuco(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    visual_types: Optional[List[str]] = None,
) -> tuple[np.ndarray, Optional[tuple]]:
    """Detect and visualize ChArUco board in an image.

    Args:
        image: RGB image (HxWx3 uint8 array)
        camera_matrix: 3x3 camera intrinsics matrix
        dist_coeffs: Optional distortion coefficients (can be None for zero distortion)
        visual_types: List of visualization types to draw. Options: "markers", "charuco", "axes"

    Returns:
        annotated_image: Image with visualizations drawn
        detection_data: Tuple of (corners, charuco_corners, charuco_ids, rvec, tvec) if detected, None otherwise
    """
    # Set default visual types if not provided
    if visual_types is None:
        visual_types = ["markers", "charuco", "axes"]

    # Convert to BGR for OpenCV processing if needed
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[2] == 3 else image.copy()
    annotated = image_bgr.copy()

    # Use zero distortion if not provided
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    # Create detector
    detector = aruco.CharucoDetector(CHARUCO_BOARD, charuco_params, detector_params)

    # Detect markers and ChArUco corners
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image_bgr)

    # If no detection, return original image
    if charuco_corners is None or len(charuco_corners) == 0:
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), None

    # Draw markers (without IDs)
    if "markers" in visual_types and marker_corners is not None:
        aruco.drawDetectedMarkers(annotated, marker_corners)

    # Draw ChArUco corners
    if "charuco" in visual_types:
        aruco.drawDetectedCornersCharuco(annotated, charuco_corners, charuco_ids)

    # Draw axes if requested and we have enough points
    rvec, tvec = None, None
    if "axes" in visual_types and len(charuco_corners) >= 4:
        # Get object and image points for PnP
        obj_points, img_points = CHARUCO_BOARD.matchImagePoints(charuco_corners, charuco_ids)

        # Solve PnP to get board pose
        valid_pose, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, camera_matrix, dist_coeffs
        )

        if valid_pose:
            cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    # Convert back to RGB
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    detection_data = (marker_corners, charuco_corners, charuco_ids, rvec, tvec)
    return annotated_rgb, detection_data


def compute_relative_transform(
    rvec1: np.ndarray,
    tvec1: np.ndarray,
    rvec2: np.ndarray,
    tvec2: np.ndarray,
) -> np.ndarray:
    """Compute relative transformation from camera1 to camera2 using common board observations.

    Both cameras observe the same board. We have:
    - T_cam1_board: camera1 → board (rvec1, tvec1)
    - T_cam2_board: camera2 → board (rvec2, tvec2)

    Returns T_cam1_cam2: transformation from camera1 to camera2
    T_cam1_cam2 = T_cam1_board @ inv(T_cam2_board)

    Args:
        rvec1: Rotation vector from camera1 to board
        tvec1: Translation vector from camera1 to board
        rvec2: Rotation vector from camera2 to board
        tvec2: Translation vector from camera2 to board

    Returns:
        T_cam1_cam2: 4x4 transformation matrix from camera1 to camera2
    """
    # Convert rvec to rotation matrices
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)

    # Build 4x4 transformation matrices
    T_cam1_board = np.eye(4, dtype=np.float64)
    T_cam1_board[:3, :3] = R1
    T_cam1_board[:3, 3] = tvec1.squeeze()

    T_cam2_board = np.eye(4, dtype=np.float64)
    T_cam2_board[:3, :3] = R2
    T_cam2_board[:3, 3] = tvec2.squeeze()

    # Compute relative transformation
    T_board_cam2 = np.linalg.inv(T_cam2_board)
    T_cam1_cam2 = T_cam1_board @ T_board_cam2

    return T_cam1_cam2


def load_sample(sample_dir: Path, sample_idx: int) -> CalibrationSample:
    """Load a single calibration sample from disk.

    Args:
        sample_dir: Path to the sample directory (e.g., will_test1/sample_0000)
        sample_idx: Sample index number

    Returns:
        CalibrationSample with loaded data
    """
    # Load Spot frame
    spot_rgb_path = sample_dir / "spot_rgb.png"
    spot_depth_path = sample_dir / "spot_depth.png"
    spot_intrinsics_path = sample_dir / "spot_intrinsics.json"

    spot_rgb = cv2.cvtColor(cv2.imread(str(spot_rgb_path)), cv2.COLOR_BGR2RGB)
    spot_depth = cv2.imread(str(spot_depth_path), cv2.IMREAD_UNCHANGED)

    with open(spot_intrinsics_path, "r") as f:
        spot_intrinsics = json.load(f)
    spot_K = np.array(spot_intrinsics["K"], dtype=np.float64)

    # Load iPhone frame
    iphone_rgb_path = sample_dir / "iphone_rgb.png"
    iphone_depth_path = sample_dir / "iphone_depth.npy"
    iphone_intrinsics_path = sample_dir / "iphone_intrinsics.json"

    iphone_rgb = cv2.cvtColor(cv2.imread(str(iphone_rgb_path)), cv2.COLOR_BGR2RGB)
    iphone_depth = np.load(str(iphone_depth_path))

    with open(iphone_intrinsics_path, "r") as f:
        iphone_intrinsics = json.load(f)
    iphone_K_rgb = np.array(iphone_intrinsics["K"], dtype=np.float64)

    # Rescale depth intrinsics based on depth image size vs RGB image size
    rgb_height, rgb_width = iphone_rgb.shape[:2]
    depth_height, depth_width = iphone_depth.shape[:2]
    scale_x = depth_width / rgb_width
    scale_y = depth_height / rgb_height

    iphone_K_depth = iphone_K_rgb.copy()
    iphone_K_depth[0, 0] *= scale_x  # fx
    iphone_K_depth[1, 1] *= scale_y  # fy
    iphone_K_depth[0, 2] *= scale_x  # cx
    iphone_K_depth[1, 2] *= scale_y  # cy

    # Create dataclass instances
    spot_frame = SpotFrame(
        rgb_path=str(spot_rgb_path),
        depth_path=str(spot_depth_path),
        camera_matrix=spot_K,
        T_body_hand=np.eye(4, dtype=np.float64),  # Will be filled from samples.json
        rgb=spot_rgb,
        depth=spot_depth,
    )

    iphone_frame = IphoneFrame(
        rgb_path=str(iphone_rgb_path),
        depth_path=str(iphone_depth_path),
        camera_matrix_rgb=iphone_K_rgb,
        camera_matrix_depth=iphone_K_depth,
        rgb=iphone_rgb,
        depth=iphone_depth,
    )

    # Create calibration sample (will fill in T_body_hand from samples.json)
    sample = CalibrationSample(
        sample_idx=sample_idx,
        T_body_hand=np.eye(4, dtype=np.float64),
        spot_frame=spot_frame,
        iphone_frame=iphone_frame,
    )

    return sample


def load_all_samples(base_dir: Path) -> List[CalibrationSample]:
    """Load all calibration samples from a directory.

    Args:
        base_dir: Base directory containing sample_XXXX subdirectories and samples.json

    Returns:
        List of CalibrationSample objects with loaded data
    """
    samples_json_path = base_dir / "samples.json"

    # Load samples.json to get metadata
    with open(samples_json_path, "r") as f:
        samples_data = json.load(f)

    samples = []

    for sample_idx, sample_data in enumerate(samples_data):
        sample_dir = base_dir / f"sample_{sample_idx:04d}"

        # Load the sample data
        sample = load_sample(sample_dir, sample_idx)

        # Fill in the metadata from samples.json
        T_body_hand = np.array(sample_data["T_body_hand"], dtype=np.float64)
        sample.spot_frame.T_body_hand = T_body_hand
        sample.T_body_hand = T_body_hand
        samples.append(sample)

    return samples


def visualize_samples_with_rerun(samples: List[CalibrationSample]) -> None:
    """Visualize calibration samples using rerun.

    Args:
        samples: List of CalibrationSample objects to visualize
    """
    rr.init("will_calibrate_iphone", spawn=True)

    # Collect all T_spot_iphone transformations
    transforms = []

    for sample in samples:
        idx = sample.sample_idx
        rr.set_time_sequence("sample", idx)

        # Initialize detection variables
        spot_detection = None
        iphone_detection = None

        # Visualize Spot frame
        if sample.spot_frame is not None:
            spot = sample.spot_frame

            # Log RGB image
            # rr.log("spot/hand/rgb", rr.Image(spot.rgb))

            # Detect and visualize ChArUco board
            spot_annotated, spot_detection = detect_and_visualize_charuco(
                spot.rgb,
                spot.camera_matrix,
                visual_types=["markers", "axes"],
            )
            rr.log("spot/hand/rgb_annotated", rr.Image(spot_annotated))

            # Generate and log point cloud
            points_spot_cam, colors_spot = rgbd_to_point_cloud(
                spot.rgb, spot.depth, spot.camera_matrix, depth_scale=1000.0
            )
            if points_spot_cam.size > 0:
                rr.log(
                    "spot/hand/points3d_cam",
                    rr.Points3D(
                        positions=points_spot_cam,
                        colors=(colors_spot * 255).astype(np.uint8),
                    ),
                )

            # If board detected, log the board pose
            if spot_detection is not None:
                marker_corners, charuco_corners, charuco_ids, rvec, tvec = spot_detection
                if rvec is not None and tvec is not None:
                    # Convert rvec to rotation matrix
                    R, _ = cv2.Rodrigues(rvec)
                    rr.log(
                        "spot/hand/charuco_board",
                        rr.Transform3D(
                            mat3x3=R.squeeze(),
                            translation=tvec.squeeze(),
                            axis_length=0.05,
                        ),
                    )

        # Visualize iPhone frame
        if sample.iphone_frame is not None:
            iphone = sample.iphone_frame

            # Log RGB image
            # rr.log("iphone/rgb", rr.Image(iphone.rgb))

            # Detect and visualize ChArUco board
            iphone_annotated, iphone_detection = detect_and_visualize_charuco(
                iphone.rgb,
                iphone.camera_matrix_rgb,
                visual_types=["markers", "axes"],
            )
            rr.log("iphone/rgb_annotated", rr.Image(iphone_annotated))

            # Generate and log point cloud
            if iphone.depth is not None:
                points_iphone_cam, colors_iphone = rgbd_to_point_cloud(
                    iphone.rgb,
                    iphone.depth,
                    iphone.camera_matrix_depth,
                    depth_scale=1.0,
                )
                if points_iphone_cam.size > 0:
                    rr.log(
                        "iphone/points3d_cam",
                        rr.Points3D(
                            positions=points_iphone_cam,
                            colors=(colors_iphone * 255).astype(np.uint8),
                        ),
                    )

            # If board detected, log the board pose
            if iphone_detection is not None:
                marker_corners, charuco_corners, charuco_ids, rvec, tvec = iphone_detection
                if rvec is not None and tvec is not None:
                    # Convert rvec to rotation matrix
                    R, _ = cv2.Rodrigues(rvec)
                    rr.log(
                        "iphone/charuco_board",
                        rr.Transform3D(
                            mat3x3=R.squeeze(),
                            translation=tvec.squeeze(),
                            axis_length=0.05,
                        ),
                    )

        # Compute relative transformation from Spot to iPhone if both detected the board
        if (
            spot_detection is not None
            and iphone_detection is not None
            and spot_detection[3] is not None
            and iphone_detection[3] is not None
        ):
            spot_rvec, spot_tvec = spot_detection[3], spot_detection[4]
            iphone_rvec, iphone_tvec = iphone_detection[3], iphone_detection[4]

            T_spot_iphone = compute_relative_transform(
                spot_rvec, spot_tvec, iphone_rvec, iphone_tvec
            )
            transforms.append((idx, T_spot_iphone))

            # Log the relative transformation
            rr.log(
                "spot/hand/T_spot_iphone",
                rr.Transform3D(
                    mat3x3=T_spot_iphone[:3, :3],
                    translation=T_spot_iphone[:3, 3],
                    axis_length=0.1,
                ),
            )

            # Transform iPhone point cloud to Spot camera frame
            if iphone.depth is not None:
                points_iphone_cam, colors_iphone = rgbd_to_point_cloud(
                    iphone.rgb,
                    iphone.depth,
                    iphone.camera_matrix_depth,
                    depth_scale=1.0,
                )
                if points_iphone_cam.size > 0:
                    # Transform points: points_spot = T_spot_iphone @ points_iphone
                    points_iphone_hom = np.hstack([points_iphone_cam, np.ones((len(points_iphone_cam), 1))])
                    points_iphone_in_spot = (T_spot_iphone @ points_iphone_hom.T).T[:, :3]

                    # Log transformed iPhone point cloud in Spot camera frame
                    rr.log(
                        "spot/hand/iphone_points3d_transformed",
                        rr.Points3D(
                            positions=points_iphone_in_spot,
                            colors=(colors_iphone * 255).astype(np.uint8),
                        ),
                    )

            print(f"[INFO] Sample {idx}: T_spot_iphone computed")
            print(f"  Translation: {T_spot_iphone[:3, 3]}")
            print(f"  Rotation:\n{T_spot_iphone[:3, :3]}")

        # Log the T_body_hand transform
        rr.log(
            "body/hand_camera",
            rr.Transform3D(
                mat3x3=sample.T_body_hand[:3, :3],
                translation=sample.T_body_hand[:3, 3],
                axis_length=0.1,
            ),
        )

        print(f"[INFO] Logged sample {idx} to rerun")

    # Compute statistics on T_spot_iphone transforms
    if len(transforms) > 0:
        print("\n" + "="*80)
        print(f"[INFO] T_spot_iphone statistics from {len(transforms)} samples:")
        print("="*80)

        # Extract translations and rotations
        translations = np.array([T[:3, 3] for _, T in transforms])
        rotations = [T[:3, :3] for _, T in transforms]

        # Compute mean and std of translation
        mean_translation = np.mean(translations, axis=0)
        std_translation = np.std(translations, axis=0)

        print(f"\nTranslation (meters):")
        print(f"  Mean: [{mean_translation[0]:.4f}, {mean_translation[1]:.4f}, {mean_translation[2]:.4f}]")
        print(f"  Std:  [{std_translation[0]:.4f}, {std_translation[1]:.4f}, {std_translation[2]:.4f}]")
        print(f"  Max std component: {np.max(std_translation):.4f} m")

        # Compute angular differences from the first rotation
        if len(rotations) > 1:
            R_ref = rotations[0]
            angular_errors = []

            for i, R in enumerate(rotations[1:], start=1):
                # Compute relative rotation
                R_rel = R_ref.T @ R
                # Convert to angle-axis
                angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
                angular_errors.append(np.degrees(angle))

            mean_angular_error = np.mean(angular_errors)
            max_angular_error = np.max(angular_errors)

            print(f"\nRotation errors (relative to sample {transforms[0][0]}):")
            print(f"  Mean angular error: {mean_angular_error:.3f} degrees")
            print(f"  Max angular error:  {max_angular_error:.3f} degrees")

        # Print individual transforms for inspection
        print(f"\nIndividual transforms:")
        for idx, T in transforms:
            t = T[:3, 3]
            print(f"  Sample {idx}: t=[{t[0]:7.4f}, {t[1]:7.4f}, {t[2]:7.4f}]")

        print("="*80)

        # Filter outliers and compute robust average
        print("\n" + "="*80)
        print("[INFO] Filtering outliers and computing robust average:")
        print("="*80)

        # Use median absolute deviation (MAD) for robust outlier detection
        median_translation = np.median(translations, axis=0)
        mad_translation = np.median(np.abs(translations - median_translation), axis=0)

        # Identify outliers (samples with translation deviation > 3 * MAD)
        outlier_threshold = 3.0
        deviations = np.abs(translations - median_translation)
        is_outlier = np.any(deviations > outlier_threshold * mad_translation, axis=1)

        inlier_indices = [i for i, outlier in enumerate(is_outlier) if not outlier]
        outlier_indices = [i for i, outlier in enumerate(is_outlier) if outlier]

        print(f"\nOutlier detection (threshold = {outlier_threshold} * MAD):")
        print(f"  Median translation: [{median_translation[0]:.4f}, {median_translation[1]:.4f}, {median_translation[2]:.4f}]")
        print(f"  MAD: [{mad_translation[0]:.4f}, {mad_translation[1]:.4f}, {mad_translation[2]:.4f}]")
        print(f"  Inliers: {len(inlier_indices)}/{len(transforms)}")
        print(f"  Outliers removed: {len(outlier_indices)} samples")

        if outlier_indices:
            print(f"\n  Removed samples:")
            for i in outlier_indices:
                idx, T = transforms[i]
                t = T[:3, 3]
                dev = deviations[i]
                print(f"    Sample {idx}: t=[{t[0]:7.4f}, {t[1]:7.4f}, {t[2]:7.4f}], dev=[{dev[0]:.4f}, {dev[1]:.4f}, {dev[2]:.4f}]")

        if len(inlier_indices) > 0:
            # Compute average translation from inliers
            inlier_translations = translations[inlier_indices]
            avg_translation = np.mean(inlier_translations, axis=0)
            std_translation_inliers = np.std(inlier_translations, axis=0)

            # Average rotation using SVD (Kabsch algorithm)
            inlier_rotations = [rotations[i] for i in inlier_indices]
            # Simple averaging (could use proper rotation averaging, but this is good enough)
            avg_rotation = np.mean(inlier_rotations, axis=0)
            # Orthogonalize using SVD
            U, _, Vt = np.linalg.svd(avg_rotation)
            avg_rotation = U @ Vt
            # Ensure right-handed coordinate system
            if np.linalg.det(avg_rotation) < 0:
                U[:, -1] *= -1
                avg_rotation = U @ Vt

            # Construct average transformation
            T_spot_iphone_avg = np.eye(4, dtype=np.float64)
            T_spot_iphone_avg[:3, :3] = avg_rotation
            T_spot_iphone_avg[:3, 3] = avg_translation

            print(f"\nRobust average T_spot_iphone (from {len(inlier_indices)} inliers):")
            print(f"\n  Translation: [{avg_translation[0]:.6f}, {avg_translation[1]:.6f}, {avg_translation[2]:.6f}]")
            print(f"  Std (inliers): [{std_translation_inliers[0]:.6f}, {std_translation_inliers[1]:.6f}, {std_translation_inliers[2]:.6f}]")
            print(f"\n  Rotation matrix:")
            for row in avg_rotation:
                print(f"    [{row[0]:9.6f}, {row[1]:9.6f}, {row[2]:9.6f}]")
            print(f"\n  Full 4x4 matrix:")
            for row in T_spot_iphone_avg:
                print(f"    [{row[0]:9.6f}, {row[1]:9.6f}, {row[2]:9.6f}, {row[3]:9.6f}]")

        print("="*80)


def visualize_accumulated_point_clouds(
    samples: List[CalibrationSample],
    use_robust_average: bool = True
) -> None:
    """Visualize all point clouds accumulated in the hand camera frame.

    Args:
        samples: List of CalibrationSample objects
        use_robust_average: If True, compute and use robust average T_spot_iphone.
                           If False, use per-frame transformations.
    """
    rr.init("will_calibrate_iphone_accumulated", spawn=True)

    # First pass: collect all transforms
    transforms = []
    for sample in samples:
        if sample.spot_frame is None or sample.iphone_frame is None:
            continue

        # Detect ChArUco in both frames
        spot_annotated, spot_detection = detect_and_visualize_charuco(
            sample.spot_frame.rgb,
            sample.spot_frame.camera_matrix,
            visual_types=["markers", "axes"],
        )

        iphone_annotated, iphone_detection = detect_and_visualize_charuco(
            sample.iphone_frame.rgb,
            sample.iphone_frame.camera_matrix_rgb,
            visual_types=["markers", "axes"],
        )

        if (spot_detection is not None and iphone_detection is not None
            and spot_detection[3] is not None and iphone_detection[3] is not None):

            spot_rvec, spot_tvec = spot_detection[3], spot_detection[4]
            iphone_rvec, iphone_tvec = iphone_detection[3], iphone_detection[4]

            T_spot_iphone = compute_relative_transform(
                spot_rvec, spot_tvec, iphone_rvec, iphone_tvec
            )
            transforms.append((sample.sample_idx, T_spot_iphone))

    # Compute robust average if requested
    T_to_use = None
    if use_robust_average and len(transforms) > 0:
        print("[INFO] Computing robust average transformation for accumulated view...")

        translations = np.array([T[:3, 3] for _, T in transforms])
        rotations = [T[:3, :3] for _, T in transforms]

        # Filter outliers
        median_translation = np.median(translations, axis=0)
        mad_translation = np.median(np.abs(translations - median_translation), axis=0)
        outlier_threshold = 3.0
        deviations = np.abs(translations - median_translation)
        is_outlier = np.any(deviations > outlier_threshold * mad_translation, axis=1)

        inlier_indices = [i for i, outlier in enumerate(is_outlier) if not outlier]

        if len(inlier_indices) > 0:
            inlier_translations = translations[inlier_indices]
            avg_translation = np.mean(inlier_translations, axis=0)

            inlier_rotations = [rotations[i] for i in inlier_indices]
            avg_rotation = np.mean(inlier_rotations, axis=0)
            U, _, Vt = np.linalg.svd(avg_rotation)
            avg_rotation = U @ Vt
            if np.linalg.det(avg_rotation) < 0:
                U[:, -1] *= -1
                avg_rotation = U @ Vt

            T_to_use = np.eye(4, dtype=np.float64)
            T_to_use[:3, :3] = avg_rotation
            T_to_use[:3, 3] = avg_translation

            print(f"[INFO] Using robust average from {len(inlier_indices)} inliers")

    # Second pass: visualize point clouds per frame
    print("[INFO] Visualizing point clouds per frame in hand camera frame...")

    for sample in samples:
        idx = sample.sample_idx
        rr.set_time_sequence("sample", idx)

        if sample.spot_frame is None:
            continue

        # Always add Spot point cloud (it's already in hand frame)
        points_spot, colors_spot = rgbd_to_point_cloud(
            sample.spot_frame.rgb,
            sample.spot_frame.depth,
            sample.spot_frame.camera_matrix,
            depth_scale=1000.0,
        )

        if points_spot.size > 0:
            rr.log(
                "hand/spot_points",
                rr.Points3D(
                    positions=points_spot,
                    colors=(colors_spot * 255).astype(np.uint8),
                ),
            )

        # Add iPhone point cloud if available and transform is available
        if sample.iphone_frame is not None and sample.iphone_frame.depth is not None:
            points_iphone, colors_iphone = rgbd_to_point_cloud(
                sample.iphone_frame.rgb,
                sample.iphone_frame.depth,
                sample.iphone_frame.camera_matrix_depth,
                depth_scale=1.0,
            )

            if points_iphone.size > 0:
                # Use robust average or per-frame transform
                T_transform = None
                if use_robust_average and T_to_use is not None:
                    T_transform = T_to_use
                else:
                    # Find this sample's transform
                    for transform_idx, T in transforms:
                        if transform_idx == idx:
                            T_transform = T
                            break

                if T_transform is not None:
                    # Transform to hand frame
                    points_iphone_hom = np.hstack([points_iphone, np.ones((len(points_iphone), 1))])
                    points_iphone_in_hand = (T_transform @ points_iphone_hom.T).T[:, :3]

                    rr.log(
                        "hand/iphone_points",
                        rr.Points3D(
                            positions=points_iphone_in_hand,
                            colors=(colors_iphone * 255).astype(np.uint8),
                        ),
                    )

    print("[INFO] Per-frame visualization complete! Check the rerun viewer.")


def main():
    """Main function to load and visualize calibration samples."""
    base_dir = Path("calib_data/adilucy_test1")

    print(f"[INFO] Loading samples from {base_dir}")
    samples = load_all_samples(base_dir)
    print(f"[INFO] Loaded {len(samples)} samples")

    print("[INFO] Visualizing samples with rerun...")
    visualize_samples_with_rerun(samples)
    print("[INFO] Done! Check the rerun viewer.")

    # Show accumulated point clouds in a new session
    print("\n[INFO] Creating accumulated point cloud visualization...")
    visualize_accumulated_point_clouds(samples, use_robust_average=True)
    print("[INFO] All visualizations complete!")


if __name__ == "__main__":
    main()
