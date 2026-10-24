"""Camera viewer: /camera/image_raw with AprilTag detections and camera height drawn on top.

    ros2 run drone_sim camera_view                    # live window (needs X11), q / Esc to quit
    ros2 run drone_sim camera_view --save out.png     # save one annotated frame and exit
"""
import argparse
import os
import sys

import cv2
import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image

from drone_sim.config_loader import camera_offset_in_base, load_all, quat_to_matrix

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
WINDOW = 'drone_sim camera'


class CameraView(Node):
    def __init__(self, save_path=None, show_detections=True):
        super().__init__('camera_view')
        self.declare_parameter('config_dir', '')
        cfg = load_all(self.get_parameter('config_dir').value)
        cam = cfg['camera']
        self.r_cam = camera_offset_in_base(cfg)
        self.save_path = save_path
        self.show_detections = show_detections
        self.bridge = CvBridge()
        self.detections = []
        self.cam_z = None
        self.done = False
        self.window_open = False
        self.create_subscription(Image, cam['image_topic'], self.on_image, 1)
        self.create_subscription(AprilTagDetectionArray, '/detections', self.on_detections, 1)
        self.create_subscription(Odometry, cfg['controller']['odom_topic'], self.on_odom, 1)
        self.get_logger().info('waiting for %s' % cam['image_topic'])

    def on_detections(self, msg):
        self.detections = msg.detections

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.cam_z = p.z + float((quat_to_matrix(q.x, q.y, q.z, q.w) @ self.r_cam)[2])

    def annotate(self, img):
        if self.show_detections:
            for d in self.detections:
                pts = [(int(c.x), int(c.y)) for c in d.corners]
                for i in range(4):
                    cv2.line(img, pts[i], pts[(i + 1) % 4], GREEN, 2)
                cv2.putText(img, str(d.id), (int(d.centre.x) - 10, int(d.centre.y) + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
        hud = 'tags: %d' % len(self.detections)
        if self.cam_z is not None:
            hud += '   camera height: %.3f m' % self.cam_z
        cv2.putText(img, hud, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, YELLOW, 2)
        return img

    def on_image(self, msg):
        if self.done:
            return
        img = self.annotate(self.bridge.imgmsg_to_cv2(msg, 'bgr8'))
        if self.save_path:
            cv2.imwrite(self.save_path, img)
            self.get_logger().info('saved %s' % os.path.abspath(self.save_path))
            self.done = True
            return
        if not self.window_open:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)   # resizable; opens at half resolution
            cv2.resizeWindow(WINDOW, img.shape[1] // 2, img.shape[0] // 2)
            self.window_open = True
        cv2.imshow(WINDOW, img)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            self.done = True


def main(args=None):
    ap = argparse.ArgumentParser(description='View the drone camera with AprilTag detections')
    ap.add_argument('--save', metavar='PNG', help='save one annotated frame to this file and exit')
    ap.add_argument('--no-detections', action='store_true', help='do not draw tag outlines')
    opts = ap.parse_args(remove_ros_args(sys.argv)[1:])
    rclpy.init(args=args)
    node = CameraView(opts.save, not opts.no_detections)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
