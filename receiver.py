#!/usr/bin/env python3
"""Kiwi Frame Receiver.

Receives ARKit frame bundles from the Kiwi iOS app over TCP.
Uses Protocol Buffers for efficient binary serialization.
"""

import socket
import struct
from datetime import datetime
from io import BytesIO

import numpy as np
import rerun as rr
from PIL import Image

from frame_bundle_pb2 import FrameBundle

# Configuration
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 8888

def main():
    """Start the TCP server and receive ARKit frames from the iPhone."""
    # Create TCP socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)

    print("=" * 60)
    print("🥝 Kiwi Frame Receiver (TCP + Protobuf)")
    print("=" * 60)
    print(f"✅ Listening on {HOST}:{PORT}")
    print(f"📱 Configure iPhone to send to: {get_local_ip()}:{PORT}")
    print("⏳ Waiting for connection...\n")

    rr.init("kiwi_stream", spawn=True)

    # Accept connection
    conn, addr = server_sock.accept()
    print(f"📱 Connected from {addr[0]}:{addr[1]}\n")

    frame_count = 0
    start_time = datetime.now()

    try:
        while True:
            # Read length prefix (4 bytes, big-endian)
            length_data = recv_exact(conn, 4)
            if not length_data:
                print("\n🔌 Connection closed by client")
                break

            length = struct.unpack('>I', length_data)[0]

            # Read Protobuf payload
            protobuf_data = recv_exact(conn, length)
            if not protobuf_data:
                print("\n🔌 Connection closed by client")
                break

            # Decode Protobuf
            frame = FrameBundle()
            frame.ParseFromString(protobuf_data)

            # Update stats
            frame_count += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Decode RGB image from JPEG (binary data preserved natively in protobuf)
            rgb_data = frame.rgb_image_data
            rgb_image = Image.open(BytesIO(rgb_data))
            rgb = np.array(rgb_image)

            # Decode depth data if available
            depth = None
            depth_scale = 1000.0  # Default depth scale (1000 = millimeters)
            if frame.depth_data:
                depth_data = frame.depth_data  # Binary data preserved natively
                depth_width = frame.depth_width
                depth_height = frame.depth_height
                # Depth is stored as Float32 array
                depth = np.frombuffer(depth_data, dtype=np.float32)
                depth = depth.reshape((depth_height, depth_width))

            # Print frame info
            depth_status = '✅' if frame.depth_data else '❌'
            total_size = len(length_data) + len(protobuf_data)
            print(f"📦 Frame {frame.frame_number:5d} | "
                  f"Image: {frame.image_width:4d}x{frame.image_height:4d} | "
                  f"Depth: {depth_status} | "
                  f"Size: {total_size:6d}B | "
                  f"FPS: {fps:4.1f}")

            # Log to Rerun
            rr.set_time_sequence("frame", frame_count)
            rr.log("world/camera/rgb", rr.Image(rgb))
            if depth is not None:
                rr.log("world/camera/depth", rr.DepthImage(depth, meter=depth_scale))

            # Log camera pose (4x4 transform matrix)
            # ARKit uses column-major matrices, convert to row-major for Rerun
            transform = np.array(frame.transform, dtype=np.float32).reshape(4, 4).T

            # Extract rotation (3x3) and translation (3,)
            rotation = transform[:3, :3]
            translation = transform[:3, 3]

            # Debug: print translation to verify it's changing
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"📍 Position: x={translation[0]:.3f}m, y={translation[1]:.3f}m, z={translation[2]:.3f}m")

            rr.log("world/camera", rr.Transform3D(
                mat3x3=rotation,
                translation=translation
            ))

            # Log camera intrinsics (now properly scaled to match RGB image)
            intrinsics = np.array(frame.intrinsics, dtype=np.float32).reshape(3, 3)
            rr.log("world/camera", rr.Pinhole(
                image_from_camera=intrinsics,
                width=frame.image_width,
                height=frame.image_height
            ))

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 60}")
        print("📊 Session Stats")
        print(f"{'=' * 60}")
        print(f"Frames received: {frame_count}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Average FPS: {fps:.1f}")
        print("\n👋 Receiver stopped")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    finally:
        conn.close()
        server_sock.close()


def recv_exact(sock, num_bytes):
    """Receive exactly num_bytes from socket (TCP requires this)."""
    data = b''
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            return None  # Connection closed
        data += chunk
    return data


def get_local_ip():
    """Get local IP address for display purposes."""
    try:
        # Create a socket to find local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    main()
