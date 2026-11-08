#!/usr/bin/env python3
"""SIM-ONLY: the sim's stereo depth on the drone's topics, /oak/stereo/image_raw + camera_info (640x400 16UC1 mm),
in the OAK's frame name. The sim depth is SGBM on the sim's stereo pair, already stamped at capture with 30 ms latency."""
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster

FRAME = 'oak_rgb_camera_optical_frame'
LEFT_CAMERA = (0.1161, 0.0373, 0.0288)      # the sim's oak_left, which the sim depth is registered to
OPTICAL = (0.5, -0.5, 0.5, -0.5)            # base_link FLU <- optical, xyzw


class OakDepth(Node):
    def __init__(self):
        super().__init__('sim_oak_depth')
        qos = QoSProfile(depth=5)
        image, info = (self.create_publisher(t, f'/oak/stereo/{n}', qos) for t, n in ((Image, 'image_raw'), (CameraInfo, 'camera_info')))
        self.create_subscription(Image, '/camera/depth/image_raw', lambda m: self.relay(image, m), qos)
        self.create_subscription(CameraInfo, '/camera/depth/camera_info', lambda m: self.relay(info, m), qos)
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = 'base_link', FRAME
        t, r = tf.transform.translation, tf.transform.rotation
        t.x, t.y, t.z = LEFT_CAMERA
        r.x, r.y, r.z, r.w = OPTICAL
        self.static = StaticTransformBroadcaster(self)
        self.static.sendTransform(tf)

    def relay(self, pub, m):
        m.header.frame_id = FRAME
        pub.publish(m)


def main():
    rclpy.init()
    rclpy.spin(OakDepth())


if __name__ == '__main__':
    main()
