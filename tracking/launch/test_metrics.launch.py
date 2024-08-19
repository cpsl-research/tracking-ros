import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    metrics_config = os.path.join(
        get_package_share_directory("tracking"),
        "config",
        "metrics.yaml",
    )

    metrics_node = Node(
        package="tracking",
        executable="metrics_evaluator",
        name="metrics",
        parameters=[metrics_config],
        arguments=["--ros-args", "--log-level", "INFO"],
    )

    track_and_truth_pub = Node(
        package="tracking",
        executable="track_and_truth_sample",
        name="track_and_truth_sample",
    )

    return LaunchDescription([metrics_node, track_and_truth_pub])
