import os
from glob import glob

from setuptools import find_packages, setup


package_name = "tracking"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "samples"), glob("samples/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="spencer",
    maintainer_email="20426598+roshambo919@users.noreply.github.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "boxtracker3d = tracking.boxtracker3d:main",
            "multi_platform_boxtracker3d = tracking.multi_platform_boxtracker3d:main",
            "metrics_evaluator = tracking.metrics_evaluator:main",
            "metrics_visualizer = tracking.metrics_visualizer:main",
            "track_and_truth_sample = samples.TrackAndTruthSample:main",
        ],
    },
)
