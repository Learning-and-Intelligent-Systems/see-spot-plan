"""Setup script."""

from setuptools import find_packages, setup

path_to_myproject = "."

setup(
    name="see_spot_run",
    version="0.1.0",
    packages=find_packages(include=["spot_utils", "spot_utils.*", "skills", "skills.*"]),
    install_requires=[
        "numpy>=1.23.5",
        "pytest==7.1.3",
        "pyyaml==6.0",
        "types-PyYAML",
        "bosdyn-client >= 3.1",
        "opencv-python >= 4.8.0",
        "dill",
        "scipy",
        "open3d",
        "rerun-sdk",
        "pillow",
        "google-generativeai",
        "imagehash",
        "openai",
        "tenacity",
        "rich",
        "pydantic",
        "fastapi",
        "uvicorn",
    ],
    include_package_data=True,
    extras_require={"develop": ["ruff==0.9.8", "ty"]},
)
