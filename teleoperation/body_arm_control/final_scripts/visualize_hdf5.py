#!/usr/bin/env python3
"""
Visualize HDF5 dataset - view images and joint positions frame by frame.

Controls:
    - Press 'n' or right arrow for next frame
    - Press 'p' or left arrow for previous frame
    - Press 'q' to quit
    - Press 's' to save current frame
    - Press 'j' to toggle joint position display
    - Press 'h' to show help
"""

import h5py
import cv2
import numpy as np
from pathlib import Path
import sys

# ===== CONFIGURATION =====
HDF5_FILE = "/home/kelly_lucy/Downloads/see-spot-plan/teleoperation_data/episode_20251125_212002.hdf5"
# Change this to visualize different files
# ======================


class HDF5Viewer:
    def __init__(self, hdf5_path):
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"File not found: {hdf5_path}")

        # Open HDF5 file
        self.file = h5py.File(self.hdf5_path, 'r')

        # Load data
        self.qpos = self.file['observations']['qpos'][:]
        self.zed_images = self.file['observations']['images']['zed_camera']
        self.arm_images = self.file['observations']['images']['arm_camera']

        self.num_timesteps = self.qpos.shape[0]
        self.current_frame = 0
        self.show_joints = True
        self.window_name = "HDF5 Viewer - Press 'h' for help"

        print(f"\nOpened: {self.hdf5_path.name}")
        print(f"Total frames: {self.num_timesteps}")
        print(f"Joint dim: {self.qpos.shape[1]}")
        print(f"ZED camera: {self.zed_images.shape}")
        print(f"Arm camera: {self.arm_images.shape}")
        print("\nPress 'h' for help")

    def get_current_frame_display(self):
        """Create a display image with both cameras and joint info"""
        # Get images
        zed_img = self.zed_images[self.current_frame]
        arm_img = self.arm_images[self.current_frame]

        # Resize for display if needed
        display_height = 480
        zed_display = cv2.resize(zed_img, (int(zed_img.shape[1] * display_height / zed_img.shape[0]), display_height))
        arm_display = cv2.resize(arm_img, (int(arm_img.shape[1] * display_height / arm_img.shape[0]), display_height))

        # Concatenate images horizontally
        max_height = max(zed_display.shape[0], arm_display.shape[0])

        # Pad images to same height if needed
        if zed_display.shape[0] < max_height:
            pad = max_height - zed_display.shape[0]
            zed_display = np.pad(zed_display, ((pad//2, pad - pad//2), (0, 0), (0, 0)), mode='constant')
        if arm_display.shape[0] < max_height:
            pad = max_height - arm_display.shape[0]
            arm_display = np.pad(arm_display, ((pad//2, pad - pad//2), (0, 0), (0, 0)), mode='constant')

        # Concatenate images horizontally
        combined = np.concatenate([zed_display, arm_display], axis=1)

        # Convert RGB to BGR for OpenCV
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        # Add text overlay with frame info
        text_color = (0, 255, 0)  # Green in BGR
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Frame counter
        cv2.putText(combined, f"Frame: {self.current_frame} / {self.num_timesteps - 1}",
                   (10, 30), font, font_scale, text_color, thickness)

        # Show joint positions if enabled
        if self.show_joints:
            qpos = self.qpos[self.current_frame]
            y_offset = 60

            # Show first few joints (up to 8 to fit on screen)
            for i in range(min(8, len(qpos))):
                text = f"q{i}: {qpos[i]:7.3f}"
                cv2.putText(combined, text, (10, y_offset + i * 25),
                           font, 0.5, text_color, 1)

            if len(qpos) > 8:
                # Show remaining joints on the right side
                for i in range(8, len(qpos)):
                    text = f"q{i}: {qpos[i]:7.3f}"
                    cv2.putText(combined, text, (200, 60 + (i - 8) * 25),
                               font, 0.5, text_color, 1)

        # Add help text at bottom
        help_text = "n/→: next | p/←: prev | q: quit | s: save | j: toggle joints | h: help"
        cv2.putText(combined, help_text, (10, combined.shape[0] - 10),
                   font, 0.4, (255, 255, 0), 1)

        return combined

    def save_current_frame(self):
        """Save current frame as image"""
        output_dir = Path("hdf5_frames")
        output_dir.mkdir(exist_ok=True)

        display = self.get_current_frame_display()
        output_path = output_dir / f"frame_{self.current_frame:04d}.png"
        cv2.imwrite(str(output_path), display)

        print(f"Saved: {output_path}")

    def show_help(self):
        """Print help message"""
        print("\n" + "="*60)
        print("HDF5 Viewer - Controls")
        print("="*60)
        print("  n or Right Arrow  : Next frame")
        print("  p or Left Arrow   : Previous frame")
        print("  q                 : Quit")
        print("  s                 : Save current frame to 'hdf5_frames/'")
        print("  j                 : Toggle joint position display")
        print("  h                 : Show this help")
        print("="*60 + "\n")

    def run(self):
        """Main display loop"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self.window_name, 1400, 600)

        self.show_help()

        while True:
            # Get and display frame
            display = self.get_current_frame_display()
            cv2.imshow(self.window_name, display)

            # Wait for key press
            key = cv2.waitKey(0)
            key_ascii = key & 0xFF

            if key_ascii == ord('q'):
                break
            elif key_ascii == ord('n') or key == 65363 or key == 83:  # 'n' or right arrow
                self.current_frame = min(self.current_frame + 1, self.num_timesteps - 1)
            elif key_ascii == ord('p') or key == 65361 or key == 81:  # 'p' or left arrow
                self.current_frame = max(self.current_frame - 1, 0)
            elif key_ascii == ord('s'):
                self.save_current_frame()
            elif key_ascii == ord('j'):
                self.show_joints = not self.show_joints
                print(f"Joint display: {'ON' if self.show_joints else 'OFF'}")
            elif key_ascii == ord('h'):
                self.show_help()

        cv2.destroyAllWindows()
        self.file.close()
        print("Viewer closed")


def main():
    try:
        viewer = HDF5Viewer(HDF5_FILE)
        viewer.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
