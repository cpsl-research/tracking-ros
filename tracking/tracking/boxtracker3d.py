import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection3DArray

from avstack_bridge import Bridge
from avstack_bridge.detections import DetectionBridge
from avstack_bridge.tracks import TrackBridge

from avstack.modules.tracking.tracker3d import BasicBoxTracker3D
from avstack_msgs.msg import BoxTrackArray


class BoxTracker(Node):
    def __init__(self):
        super().__init__("tracker")
        self.model = BasicBoxTracker3D(check_reference=False)

        qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE
        )

        # subscribe to 3d detections
        self.subscriber_dets = self.create_subscription(
            Detection3DArray,
            "detections_3d",
            self.dets_callback,
            qos_profile=qos,
        )

        # publish 3d tracks
        self.publisher_trks = self.create_publisher(
            BoxTrackArray,
            "tracks_3d",
            qos_profile=qos,
        )

    def dets_callback(self, dets_msg: Detection3DArray) -> BoxTrackArray:
        dets_avstack = DetectionBridge.detectionarray_to_avstack(dets_msg)
        platform = Bridge.header_to_reference(dets_msg.header)
        trks_avstack = self.model(dets_avstack, platform=platform, check_reference=False)
        trks_ros = TrackBridge.avstack_to_tracks(trks_avstack, header=dets_msg.header)
        self.publisher_trks.publish(trks_ros)


def main(args=None):
    rclpy.init(args=args)

    tracker = BoxTracker()

    rclpy.spin(tracker)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    tracker.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
