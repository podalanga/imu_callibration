"""Load the YAML configuration files and small rotation helpers (no ROS dependency)."""
import math
import os

import numpy as np
import yaml

CONFIG_NAMES = ['world', 'apriltags', 'uwb', 'drone', 'camera', 'imu', 'controller', 'trajectory']


def default_config_dir():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('drone_sim'), 'config')
    except Exception:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(here, 'config')


def load_all(config_dir=None):
    """Return {name: dict} for every config file; config_dir '' or None -> installed share dir."""
    config_dir = config_dir or default_config_dir()
    cfg = {}
    for name in CONFIG_NAMES:
        with open(os.path.join(config_dir, name + '.yaml'), 'r') as f:
            cfg[name] = yaml.safe_load(f)
    cfg['_config_dir'] = os.path.abspath(config_dir)
    return cfg


def package_root(cfg):
    """Directory holding config/ and models/ (parent of the config dir)."""
    return os.path.dirname(cfg['_config_dir'])


def rpy_to_matrix(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll) (SDF / ROS fixed-axis convention)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def rpy_to_quat(roll, pitch, yaw):
    """Quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def quat_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_to_yaw(x, y, z, w):
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def mount(cfg, name):
    """(xyz ndarray, rotation matrix) of a mount, relative to its parent (default base_link)."""
    m = cfg['drone']['mounts'][name]
    return np.array(m['xyz'], dtype=float), rpy_to_matrix(*m['rpy'])


def camera_offset_in_base(cfg):
    """Camera (optical centre) position in base_link; the optical frame shares camera_link's origin."""
    return mount(cfg, 'camera')[0] + mount(cfg, 'camera')[1] @ mount(cfg, 'camera_optical')[0]
