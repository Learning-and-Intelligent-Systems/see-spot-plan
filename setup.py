"""Setup script."""

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install


def _generate_proto():
    from generate_proto import generate
    generate()


class PostInstall(install):
    """Run proto generation after install."""

    def run(self):
        """Run install and generate proto files."""
        super().run()
        _generate_proto()


class PostDevelop(develop):
    """Run proto generation after develop install."""

    def run(self):
        """Run develop install and generate proto files."""
        super().run()
        _generate_proto()

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
        "google-genai",
        "imagehash",
        "openai",
        "tenacity",
        "rich",
        "pydantic",
        "fastapi",
        "uvicorn",
        "grpcio-tools",
    ],
    include_package_data=True,
    cmdclass={"install": PostInstall, "develop": PostDevelop},
    extras_require={"develop": ["ruff==0.9.8", "ty"]},
)
