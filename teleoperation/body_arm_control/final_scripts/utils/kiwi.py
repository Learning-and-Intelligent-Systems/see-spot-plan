#!/usr/bin/env python3
"""
Kiwi Frame Receiver
Receives ARKit frame bundles from the Kiwi iOS app over TCP
Uses Protocol Buffers for efficient binary serialization
"""

import socket
import struct
import numpy as np
from PIL import Image
from io import BytesIO

from datetime import datetime
from frame_bundle_pb2 import FrameBundle


def start_kiwi_server(host: str = '0.0.0.0', port: int = 8888):
    """
    Start Kiwi TCP server and wait for iPhone connection.

    Args:
        host: Host to listen on (default: all interfaces)
        port: Port to listen on (default: 8888)

    Returns:
        Tuple[socket.socket, str]: Connected socket and client address
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)

    print("=" * 60)
    print("🥝 Kiwi Frame Receiver (TCP + Protobuf)")
    print("=" * 60)
    print(f"✅ Listening on {host}:{port}")
    print(f"📱 Configure iPhone to send to: {get_local_ip()}:{port}")
    print(f"⏳ Waiting for connection...\n")

    conn, addr = server_sock.accept()
    print(f"📱 Connected from {addr[0]}:{addr[1]}\n")

    return conn, addr


def stream_kiwi_frames(conn: socket.socket, use_rerun: bool = False):
    """
    Generator that yields RGB and depth frames from Kiwi iPhone app.

    Args:
        conn: Connected socket from start_kiwi_server()
        use_rerun: Whether to log frames to Rerun (default: False)

    Yields:
        Tuple with frame data:
            - rgb: HxWx3 uint8 RGB image (from JPEG)
            - depth: HxW float32 depth in meters (or None if not available)
            - transform: 4x4 camera pose matrix
            - intrinsics: 3x3 camera intrinsics matrix
            - frame_number: Frame counter from iPhone
    """
    if use_rerun:
        import rerun as rr
        rr.init("kiwi_stream", spawn=True)

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

            # Debug: Log all fields in the frame
            print(f"\nDEBUG: Frame fields:")
            print(f"  - frame_number: {frame.frame_number}")
            print(f"  - image_width: {frame.image_width}")
            print(f"  - image_height: {frame.image_height}")
            print(f"  - rgb_image_data: {len(frame.rgb_image_data)} bytes")
            print(f"  - depth_width: {frame.depth_width}")
            print(f"  - depth_height: {frame.depth_height}")
            print(f"  - depth_data: {len(frame.depth_data)} bytes")
            print(f"  - transform: {len(frame.transform)} floats")
            print(f"  - intrinsics: {len(frame.intrinsics)} floats")

            # Update stats
            frame_count += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Decode RGB image from JPEG
            rgb_data = frame.rgb_image_data

            if rgb_data:
                try:
                    rgb_image = Image.open(BytesIO(rgb_data))
                    rgb = np.array(rgb_image)
                except Exception as e:
                    print(f"⚠️ Failed to decode image: {e}")
                    rgb = None
            else:
                # RGB data not available from iPhone app
                rgb = None

            # Decode depth data if available
            depth = None
            if frame.depth_data and frame.depth_width > 0 and frame.depth_height > 0:
                depth_data = frame.depth_data
                depth_width = frame.depth_width
                depth_height = frame.depth_height
                expected_size = depth_width * depth_height * 4
                actual_size = len(depth_data)
                if actual_size == expected_size:
                    depth = np.frombuffer(depth_data, dtype=np.float32)
                    depth = depth.reshape((depth_height, depth_width))
                elif actual_size % 4 == 0:
                    print(f"⚠️  Depth data size mismatch: expected {expected_size} bytes ({depth_width}x{depth_height}), got {actual_size} bytes")
                    num_elements = actual_size // 4
                    if num_elements >= depth_width * depth_height:
                        depth = np.frombuffer(depth_data, dtype=np.float32)[:depth_width * depth_height]
                        depth = depth.reshape((depth_height, depth_width))
                    else:
                        print(f"⚠️  Depth data too small: need {depth_width * depth_height} floats, got {num_elements}")
                        depth = None
                else:
                    print(f"⚠️  Depth data size not multiple of 4: got {actual_size} bytes")
                    depth = None

            # Extract camera pose
            if len(frame.transform) >= 16:
                transform = np.array(frame.transform[:16], dtype=np.float32).reshape(4, 4).T
            else:
                transform = np.eye(4, dtype=np.float32)

            # Extract camera intrinsics
            if len(frame.intrinsics) >= 9:
                intrinsics = np.array(frame.intrinsics[:9], dtype=np.float32).reshape(3, 3)
            else:
                intrinsics = np.eye(3, dtype=np.float32)

            # Print frame info
            rgb_status = '✅' if rgb is not None else '❌'
            depth_status = '✅' if frame.depth_data else '❌'
            total_size = len(length_data) + len(protobuf_data)
            print(f"📦 Frame {frame.frame_number:5d} | "
                  f"RGB: {rgb_status} | "
                  f"Depth: {depth_status} {frame.depth_width:4d}x{frame.depth_height:4d} | "
                  f"Size: {total_size:6d}B | "
                  f"FPS: {fps:4.1f}")

            # Log to Rerun if enabled
            if use_rerun:
                rr.set_time_sequence("frame", frame_count)
                if rgb is not None:
                    rr.log("world/camera/rgb", rr.Image(rgb))
                if depth is not None:
                    rr.log("world/camera/depth", rr.DepthImage(depth, meter=1000.0))

                rotation = transform[:3, :3]
                translation = transform[:3, 3]
                rr.log("world/camera", rr.Transform3D(
                    mat3x3=rotation,
                    translation=translation
                ))
                rr.log("world/camera", rr.Pinhole(
                    image_from_camera=intrinsics,
                    width=frame.image_width,
                    height=frame.image_height
                ))

            yield rgb, depth, transform, intrinsics, frame.frame_number

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 60}")
        print(f"📊 Session Stats")
        print(f"{'=' * 60}")
        print(f"Frames received: {frame_count}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Average FPS: {fps:.1f}")
        print(f"\n👋 Receiver stopped")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


def main(host: str = '0.0.0.0', port: int = 8888, use_rerun: bool = True):
    """
    Run Kiwi receiver in standalone mode (for testing).

    Args:
        host: Host to listen on
        port: Port to listen on
        use_rerun: Whether to visualize in Rerun
    """
    conn, addr = start_kiwi_server(host, port)
    server_sock = None

    try:
        for rgb, depth, transform, intrinsics, frame_number in stream_kiwi_frames(conn, use_rerun=use_rerun):
            pass  # Data is automatically processed by the generator
    finally:
        conn.close()


def recv_exact(sock, num_bytes):
    """Receive exactly num_bytes from socket (TCP requires this)"""
    data = b''
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            return None  # Connection closed
        data += chunk
    return data


def get_local_ip():
    """Get local IP address for display purposes"""
    try:
        # Create a socket to find local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"


if __name__ == "__main__":
    main()