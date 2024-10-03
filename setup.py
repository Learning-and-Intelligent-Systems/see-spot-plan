"""Setup script."""
from setuptools import find_packages, setup

path_to_myproject = "."

setup(name="see_spot_run",
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
      ],
      include_package_data=True,
      extras_require={
          "develop": [
              "pytest-pylint==0.18.0",
              "yapf==0.32.0",
              "docformatter==1.4",
              "isort==5.10.1",
          ]
      })
