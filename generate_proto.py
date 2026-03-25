"""Generate Python protobuf files from .proto sources."""

import subprocess
import sys


def generate():
    subprocess.check_call(
        [sys.executable, "-m", "grpc_tools.protoc", "--python_out=.", "-I.", "frame_bundle.proto"]
    )


if __name__ == "__main__":
    generate()
