"""TF publisher: static mounts/anchors/ground-truth tag frames + dynamic world->base_link."""
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from drone_sim.config_loader import load_all, rpy_to_quat


def make_tf(parent, child, xyz, quat, stamp=None):
    t = TransformStamped()
    if stamp is not None:
        t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = float(xyz[0])
    t.transform.translation.y = float(xyz[1])
    t.transform.translation.z = float(xyz[2])
    t.transform.rotation.x = float(quat[0])
    t.transform.rotation.y = float(quat[1])
    t.transform.rotation.z = float(quat[2])
    t.transform.rotation.w = float(quat[3])
    return t


class TfNode(Node):
    def __init__(self):
        super().__init__('tf_node')
        self.declare_parameter('config_dir', '')
        cfg = load_all(self.get_parameter('config_dir').value)
        self.base = cfg['drone']['odometry']['robot_base_frame']
        cam = cfg['camera']
        imu = cfg['imu']
        mounts = cfg['drone']['mounts']

        frame_names = {
            'com': 'com_link',
            'camera': 'camera_link',
            'camera_optical': cam['frame_id'],
            'imu_ideal': imu['ideal_frame_id'],
            'imu_noisy': imu['noisy_frame_id'],
            'uwb_tag': cfg['uwb']['tag_frame'],
        }
        static = []
        for key, child in frame_names.items():
            m = mounts[key]
            parent = frame_names[m['parent']] if 'parent' in m else self.base
            static.append(make_tf(parent, child, m['xyz'], rpy_to_quat(*m['rpy'])))
        for a in cfg['uwb']['anchors']:
            static.append(make_tf('world', a['name'], (a['x'], a['y'], a['z']), (0, 0, 0, 1)))
        for tag in cfg['apriltags']['tags']:
            # frame at the tag centre on its top surface, axes aligned with world (x = artwork up)
            static.append(make_tf('world', 'apriltag_gt_%d' % tag['id'],
                                  (tag['x'], tag['y'], cfg['apriltags']['thickness_m']), (0, 0, 0, 1)))
        self.static_bc = StaticTransformBroadcaster(self)
        self.static_bc.sendTransform(static)

        self.bc = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, cfg['controller']['pose_topic'], 10)
        self.create_subscription(Odometry, cfg['controller']['odom_topic'], self.on_odom, 50)

    def on_odom(self, m):
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        self.bc.sendTransform(make_tf('world', self.base, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w),
                                      m.header.stamp))
        ps = PoseStamped()
        ps.header.stamp = m.header.stamp
        ps.header.frame_id = 'world'
        ps.pose = m.pose.pose
        self.pose_pub.publish(ps)


def main(args=None):
    rclpy.init(args=args)
    node = TfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
