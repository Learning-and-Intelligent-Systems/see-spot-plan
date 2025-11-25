#!/usr/bin/env python3
"""
Kiwi Frame Receiver
Receives ARKit frame bundles from the Kiwi iOS app over TCP
Uses Protocol Buffers for efficient binary serialization
"""

import socket
import struct
import numpy as np
import os
import time
from pathlib import Path
from queue import Queue
from threading import Thread
from io import BytesIO

from datetime import datetime
from frame_bundle_pb2 import FrameBundle

from PIL import Image


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
    
    # Optimize socket for high throughput
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)  # 4MB receive buffer
    
    print(f"📱 Connected from {addr[0]}:{addr[1]}\n")

    return conn, addr


def stream_kiwi_frames(conn: socket.socket, use_rerun: bool = False, save_images: bool = False, save_dir: str = None, print_interval: int = 30):
    """
    Generator that yields RGB frames from Kiwi iPhone app.

    Args:
        conn: Connected socket from start_kiwi_server()
        use_rerun: Whether to log frames to Rerun (default: False)
        save_images: Whether to save RGB images to disk (default: False)
        save_dir: Directory to save images (default: ./kiwi_frames)
        print_interval: Print stats every N frames (default: 30, set to 0 to disable)

    Yields:
        rgb: HxWx3 uint8 RGB image (from JPEG)
    """
    if use_rerun:
        import rerun as rr
        rr.init("kiwi_stream", spawn=True)

    # Background thread for saving images (non-blocking)
    save_queue = None
    save_thread = None
    if save_images:
        if save_dir is None:
            save_dir = "./kiwi_frames"
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving images to: {save_path.absolute()}")
        
        save_queue = Queue(maxsize=10)  # Buffer up to 10 frames
        
        def save_worker():
            while True:
                item = save_queue.get()
                if item is None:  # Poison pill
                    break
                rgb_data, frame_num = item
                # Save JPEG directly without decoding (much faster)
                with open(save_path / f"frame_{frame_num:06d}.jpg", 'wb') as f:
                    f.write(rgb_data)
                save_queue.task_done()
        
        save_thread = Thread(target=save_worker, daemon=True)
        save_thread.start()

    frame_count = 0
    start_time = datetime.now()
    rgb_field_num = None  # Cache the RGB field number after first frame

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

            # Extract RGB from wire format (skip full protobuf parse for speed)
            rgb_data_manual = None
            
            # Extract RGB from wire format (optimized: only parse if field number not cached)
            if rgb_field_num is None:
                # First frame: find RGB field
                i = 0
                while i < len(protobuf_data):
                    tag = protobuf_data[i]
                    field_num = tag >> 3
                    wire_type = tag & 0x7
                    i += 1
                    
                    if wire_type == 0:  # varint
                        while i < len(protobuf_data) and (protobuf_data[i] & 0x80):
                            i += 1
                        i += 1
                    elif wire_type == 2:  # length-delimited
                        length_field = 0
                        shift = 0
                        start_pos = i
                        while i < len(protobuf_data):
                            byte = protobuf_data[i]
                            length_field |= (byte & 0x7F) << shift
                            i += 1
                            if not (byte & 0x80):
                                break
                            shift += 7
                        
                        if length_field > 1000 and i + length_field <= len(protobuf_data):
                            data = protobuf_data[i:i+length_field]
                            if data[:2] == b'\xff\xd8' or data[:4] == b'\x89PNG':
                                rgb_data_manual = data
                                rgb_field_num = field_num
                                print(f"✅ Found RGB image in field {field_num} ({length_field} bytes)")
                                break
                        i += length_field
                    elif wire_type == 5:  # 32bit float
                        i += 4
                    elif wire_type == 1:  # 64bit fixed
                        i += 8
                    else:
                        break
            else:
                # Subsequent frames: directly extract from known field
                i = 0
                while i < len(protobuf_data):
                    tag = protobuf_data[i]
                    field_num = tag >> 3
                    wire_type = tag & 0x7
                    i += 1
                    
                    if field_num == rgb_field_num and wire_type == 2:  # length-delimited
                        length_field = 0
                        shift = 0
                        while i < len(protobuf_data):
                            byte = protobuf_data[i]
                            length_field |= (byte & 0x7F) << shift
                            i += 1
                            if not (byte & 0x80):
                                break
                            shift += 7
                        
                        if i + length_field <= len(protobuf_data):
                            rgb_data_manual = protobuf_data[i:i+length_field]
                        break
                    elif wire_type == 0:  # varint
                        while i < len(protobuf_data) and (protobuf_data[i] & 0x80):
                            i += 1
                        i += 1
                    elif wire_type == 2:  # length-delimited (skip)
                        length_field = 0
                        shift = 0
                        while i < len(protobuf_data):
                            byte = protobuf_data[i]
                            length_field |= (byte & 0x7F) << shift
                            i += 1
                            if not (byte & 0x80):
                                break
                            shift += 7
                        i += length_field
                    elif wire_type == 5:  # 32bit float
                        i += 4
                    elif wire_type == 1:  # 64bit fixed
                        i += 8
                    else:
                        break

            # Update stats
            frame_count += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Decode RGB image using PIL (faster)
            if not rgb_data_manual:
                continue

            try:
                pil_image = Image.open(BytesIO(rgb_data_manual))
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                elif pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                rgb = np.array(pil_image)
            except Exception:
                continue

            # Print frame info (reduced frequency)
            if print_interval > 0 and frame_count % print_interval == 0:
                total_size = len(length_data) + len(protobuf_data)
                print(f"📦 Frame {frame_count:5d} | "
                      f"Image: {rgb.shape[1]:4d}x{rgb.shape[0]:4d} | "
                      f"Size: {total_size:6d}B | "
                      f"FPS: {fps:4.1f}")

            # Save RGB image if requested (non-blocking via background thread)
            # Put raw JPEG data instead of decoded image to avoid expensive copy
            if save_images and save_queue:
                try:
                    save_queue.put_nowait((rgb_data_manual, frame_count))
                except:
                    pass  # Queue full, skip saving this frame

            # Log to Rerun if enabled (do this last as it can be slow)
            if use_rerun:
                rr.set_time("frame", sequence=frame_count)
                rr.log("world/camera/rgb", rr.Image(rgb))

            yield rgb

    except KeyboardInterrupt:
        pass  # Will print stats below

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

    finally:
        # Always report stats at the end (print first, before cleanup)
        if frame_count > 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"\n\n{'=' * 60}")
            print(f"📊 Session Stats")
            print(f"{'=' * 60}")
            print(f"Frames received: {frame_count}")
            print(f"Duration: {elapsed:.1f}s")
            print(f"Average FPS: {fps:.1f}")
            print(f"\n👋 Receiver stopped")
        
        # Stop background save thread (with timeout to avoid blocking)
        if save_queue:
            try:
                save_queue.put_nowait(None)  # Poison pill
            except:
                pass  # Queue full, thread will exit when it processes current items
            # Wait for remaining saves with timeout (don't block indefinitely)
            start_wait = time.time()
            while save_queue.unfinished_tasks > 0 and (time.time() - start_wait) < 2.0:
                time.sleep(0.1)


def main(host: str = '0.0.0.0', port: int = 8888, use_rerun: bool = False, save_images: bool = False, save_dir: str = None, print_interval: int = 30):
    """
    Run Kiwi receiver in standalone mode (for testing).

    Args:
        host: Host to listen on
        port: Port to listen on
        use_rerun: Whether to visualize in Rerun
        save_images: Whether to save RGB images to disk
        save_dir: Directory to save images (default: ./kiwi_frames)
        print_interval: Print stats every N frames (default: 30, set to 0 to disable)
    """
    conn, addr = start_kiwi_server(host, port)

    try:
        for rgb in stream_kiwi_frames(conn, use_rerun=use_rerun, save_images=save_images, save_dir=save_dir, print_interval=print_interval):
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
    main(save_images=True, save_dir='./kiwi_frames')