"""Setup script."""

from setuptools import find_packages, setup

path_to_myproject = "."

setup(
    name="see_spot_run",
    version="0.1.0",
    packages=find_packages(include=["spot_utils", "spot_utils.*"]),
    install_requires=[
        "numpy==1.23.5",
        "pytest==7.1.3",
        "mypy==1.8.0",
        "pyyaml==6.0",
        "pylint==2.14.5",
        "types-PyYAML",
        "bosdyn-client >= 3.1",
        "opencv-python == 4.7.0.72",
        "scipy",
    ],
    include_package_data=True,
    extras_require={"develop": ["ruff==0.9.8"]},
)
