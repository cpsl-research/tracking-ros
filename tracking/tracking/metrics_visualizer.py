#! /usr/bin/env python3

import os
import threading
from collections import deque

#################################################################
# fmt: off
# this is needed to allow for importing of avstack/bridge things
# while also using matplotlib/pyqt5
import cv2  # noqa # pylint: disable=unused-import


os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
# fmt: on
#################################################################

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from avstack_msgs.msg import TrackingMetrics
from rclpy.node import Node


line_colors = {
    "n_truths": "#7BC8F6",
    "n_tracks": "#76FF7B",
    "n_assign": "#01153E",
    "ospa": "#029386",
}


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18


plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("lines", linewidth=4)
plt.rc("grid", linestyle="--", color="black", alpha=0.5)


def rostime_to_time(msg):
    return float(msg.sec + msg.nanosec / 1e9)


class MetricsVisualizer(Node):
    """Trust Visualizer for showing how to use matplotlib within ros 2 node

    Inspiration from: https://github.com/timdodge54/matplotlib_ros_tutorials

    Attributes:
        fig: Figure object for matplotlib
        ax: Axes object for matplotlib
        x: x values for matplotlib
        y: y values for matplotlib
        lock: lock for threading
        _sub: Subscriber for node
    """

    def __init__(self):
        """Initialize."""
        super().__init__("trust_visualizer")
        self.declare_parameter("history_length", value=50)
        self.history_length = self.get_parameter("history_length").value

        # Initialize figure and axes and save to class
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 6))
        self._data = {}
        self._lines = {}

        # (0, 0) -- assignments
        self.axs[0, 0].set_title(f"Tracking to Truth Assignments")
        self.axs[0, 0].set_xlabel("Time (s)")
        self.axs[0, 0].set_ylabel("Number")
        self.axs[0, 0].grid()
        for identifier in ["n_truths", "n_tracks", "n_assign"]:
            self._data[identifier] = deque([], maxlen=self.history_length)
            self._lines[identifier] = self.axs[0, 0].plot(
                [],
                [],
                label=identifier,
                color=line_colors[identifier],
                marker="o",
                linestyle="dashed",
            )[0]
        self.axs[0, 0].legend(loc="upper left")  # this assume no new lines

        # (0, 1) -- ospa
        self.axs[0, 1].set_title(f"Optimal SubPattern Assignment (OSPA)")
        self.axs[0, 1].set_xlabel("Time (s)")
        self.axs[0, 1].set_ylabel("Metric")
        self.axs[0, 1].grid()
        self._data["ospa"] = deque([], maxlen=self.history_length)
        self._lines["ospa"] = self.axs[0, 1].plot(
            [],
            [],
            label="OSPA",
            color=line_colors["ospa"],
            marker="o",
            linestyle="dashed",
        )[0]
        self.axs[0, 1].legend(loc="upper left")  # this assume no new lines

        # (1, 0) -- nothing
        # (1, 1) -- nothing

        plt.tight_layout()

        # create Thread lock to prevent multiaccess threading errors
        self._lock = threading.Lock()

        # subscribe to the metrics outputs
        qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )
        self.cbg = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self.sub_tracking_metrics = self.create_subscription(
            TrackingMetrics,
            "tracking_metrics",
            self.receive_tracking_metrics,
            qos_profile=qos,
            callback_group=self.cbg,
        )

    def receive_tracking_metrics(self, msg: TrackingMetrics):
        """Callback to receive tracking metrics

        Args:
            msg: message with tracking metrics
        """
        # lock thread
        with self._lock:
            timestamp = rostime_to_time(msg.header.stamp)
            for identifier in ["n_truths", "n_tracks", "n_assign", "ospa"]:
                self._data[identifier].append((timestamp, getattr(msg, identifier)))

    def plt_func(self, _):
        """Function for for adding data to axis.

        Args:
            _ : Dummy variable that is required for matplotlib animation.
            dynamic_ylim: boolean on whether to enable dynamic ylim adjustment

        Returns:
            Axes object for matplotlib
        """
        ax_id_sets = [
            (self.axs[0, 0], ["n_truths", "n_tracks", "n_assign"]),
            (self.axs[0, 1], ["ospa"]),
        ]
        for ax, id_set in ax_id_sets:
            xmin = np.inf
            xmax = 1
            ymin = 0
            ymax = 1
            for identifier in id_set:
                x_time = [d[0] for d in self._data[identifier]]
                y_data = [d[1] for d in self._data[identifier]]
                self._lines[identifier].set_xdata(x_time)
                self._lines[identifier].set_ydata(y_data)
                if len(x_time) > 0:
                    xmin = min(xmin, min(x_time) - 1)
                    xmax = max(xmax, max(x_time) + 1)
                    ymax = max(ymax, max(y_data) + 1)
                else:
                    xmin = 0
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])

    def _plt(self, interval_ms=1000 / 10):
        """Function for initializing and showing matplotlib animation."""
        self.ani = anim.FuncAnimation(self.fig, self.plt_func, interval=interval_ms)
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = MetricsVisualizer()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    node._plt()


if __name__ == "__main__":
    main()
