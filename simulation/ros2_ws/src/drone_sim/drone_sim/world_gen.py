"""Generate the Gazebo world SDF, the ros_gz_bridge YAML and the apriltag_ros params from config.

Everything (tag positions, anchors, drone geometry, topics) is derived from the YAML files.
"""
import argparse
import os

import yaml

from drone_sim.config_loader import load_all, package_root


def _f(v):
    return '%.9g' % float(v)


def _pose(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    return '<pose>%s</pose>' % ' '.join(_f(v) for v in list(xyz) + list(rpy))


def _vec(v):
    return ' '.join(_f(x) for x in v)


def _rgba(c):
    return ' '.join(_f(x) for x in c)


def _material(rgba):
    return ('<material><ambient>%s</ambient><diffuse>%s</diffuse><specular>0.1 0.1 0.1 1</specular>'
            '</material>' % (_rgba(rgba), _rgba(rgba)))


def _world_header(cfg):
    w = cfg['world']
    lt = w['lighting']
    ph = w['physics']
    amb = lt['ambient']
    out = []
    out.append('<?xml version="1.0"?>')
    out.append('<sdf version="1.9">')
    out.append('<world name="%s">' % w['world_name'])
    out.append('<physics name="drone_physics" type="ignored">'
               '<max_step_size>%s</max_step_size><real_time_factor>%s</real_time_factor>'
               '<real_time_update_rate>%s</real_time_update_rate></physics>'
               % (_f(ph['max_step_size']), _f(ph['real_time_factor']),
                  _f(ph['real_time_factor'] / ph['max_step_size'])))
    out.append('<gravity>%s</gravity>' % _vec(w['gravity']))
    out.append('<magnetic_field>%s</magnetic_field><atmosphere type="adiabatic"/>' % _vec(w['magnetic_field']))
    sc = w['spherical_coordinates']
    out.append('<spherical_coordinates><surface_model>EARTH_WGS84</surface_model>'
               '<world_frame_orientation>ENU</world_frame_orientation><latitude_deg>%s</latitude_deg>'
               '<longitude_deg>%s</longitude_deg><elevation>%s</elevation></spherical_coordinates>'
               % (_f(sc['latitude_deg']), _f(sc['longitude_deg']), _f(sc['elevation'])))
    for fn, nm, extra in [
        ('gz-sim-physics-system', 'gz::sim::systems::Physics', ''),
        ('gz-sim-user-commands-system', 'gz::sim::systems::UserCommands', ''),
        ('gz-sim-scene-broadcaster-system', 'gz::sim::systems::SceneBroadcaster', ''),
        ('gz-sim-sensors-system', 'gz::sim::systems::Sensors', '<render_engine>ogre2</render_engine>'),
        ('gz-sim-imu-system', 'gz::sim::systems::Imu', ''),
        ('gz-sim-air-pressure-system', 'gz::sim::systems::AirPressure', ''),
        ('gz-sim-magnetometer-system', 'gz::sim::systems::Magnetometer', ''),
        ('gz-sim-navsat-system', 'gz::sim::systems::NavSat', ''),
    ]:
        out.append('<plugin filename="%s" name="%s">%s</plugin>' % (fn, nm, extra))
    out.append('<scene><ambient>%s</ambient><background>%s</background><shadows>%s</shadows></scene>'
               % (_rgba([amb, amb, amb, 1.0]), _rgba(lt['background']),
                  'true' if lt['cast_shadows'] else 'false'))
    d = lt['sun_diffuse']
    s = lt['sun_specular']
    out.append('<light type="directional" name="sun">%s<cast_shadows>%s</cast_shadows>'
               '<diffuse>%s</diffuse><specular>%s</specular><direction>%s</direction></light>'
               % (_pose((0, 0, 10)), 'true' if lt['cast_shadows'] else 'false',
                  _rgba([d, d, d, 1]), _rgba([s, s, s, 1]), _vec(lt['sun_direction'])))
    return out


def _floor(cfg):
    fl = cfg['world']['floor']
    size = '%s %s' % (_f(fl['size'][0]), _f(fl['size'][1]))
    return ('<model name="floor"><static>true</static>%s<link name="link">'
            '<collision name="collision"><pose>0 0 -5 0 0 0</pose><geometry><box><size>%s 10</size></box></geometry>'
            '<surface><friction><ode><mu>%s</mu><mu2>%s</mu2></ode></friction></surface></collision>'
            '<visual name="visual"><cast_shadows>false</cast_shadows>'
            '<geometry><plane><normal>0 0 1</normal><size>%s</size></plane></geometry>%s</visual>'
            '</link></model>'
            % (_pose((fl['center'][0], fl['center'][1], 0)), size, _f(fl['mu']), _f(fl['mu2']),
               size, _material(fl['rgba'])))


def _tags(cfg):
    at = cfg['apriltags']
    tex_dir = at['texture_dir']
    if not os.path.isabs(tex_dir):
        tex_dir = os.path.join(package_root(cfg), tex_dir)
    tex_dir = os.path.abspath(tex_dir)
    s = at['outer_size_m']
    t = at['thickness_m']
    yaw = at['texture_yaw_offset_rad']
    out = []
    for tag in at['tags']:
        tex = 'file://' + os.path.join(tex_dir, tag['name'] + '.png')
        out.append(
            '<model name="apriltag_%d"><static>true</static>%s<link name="link">'
            '<visual name="visual"><cast_shadows>false</cast_shadows>'
            '<geometry><box><size>%s %s %s</size></box></geometry>'
            '<material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular>'
            '<pbr><metal><albedo_map>%s</albedo_map><metalness>0.0</metalness><roughness>1.0</roughness>'
            '</metal></pbr></material></visual></link></model>'
            % (tag['id'], _pose((tag['x'], tag['y'], t / 2.0), (0, 0, yaw)),
               _f(s), _f(s), _f(t), tex))
    return out


def _anchors(cfg):
    u = cfg['uwb']
    m = u['marker']
    out = []
    for a in u['anchors']:
        # floor anchors stand on the floor, raised anchors hang below their point
        dz = m['length'] / 2.0 if a['z'] <= 0.0 else -m['length'] / 2.0
        out.append(
            '<model name="%s"><static>true</static>%s<link name="link"><visual name="visual">'
            '<cast_shadows>false</cast_shadows>'
            '<geometry><cylinder><radius>%s</radius><length>%s</length></cylinder></geometry>%s'
            '</visual></link></model>'
            % (a['name'], _pose((a['x'], a['y'], a['z'] + dz)),
               _f(m['radius']), _f(m['length']), _material(m['rgba'])))
    return out


def _imu_noise(n):
    return ('<noise type="gaussian"><mean>%s</mean><stddev>%s</stddev><bias_mean>%s</bias_mean>'
            '<bias_stddev>%s</bias_stddev><dynamic_bias_stddev>%s</dynamic_bias_stddev>'
            '<dynamic_bias_correlation_time>%s</dynamic_bias_correlation_time></noise>'
            % (_f(n['mean']), _f(n['stddev']), _f(n['bias_mean']), _f(n['bias_stddev']),
               _f(n['dynamic_bias_stddev']), _f(n['dynamic_bias_correlation_time'])))


def _imu_sensor(name, topic, frame_id, rate, mount, noisy_cfg=None):
    body = ''
    if noisy_cfg is not None:
        g = _imu_noise(noisy_cfg['gyro'])
        a = _imu_noise(noisy_cfg['accel'])
        body = ('<imu><angular_velocity><x>%s</x><y>%s</y><z>%s</z></angular_velocity>'
                '<linear_acceleration><x>%s</x><y>%s</y><z>%s</z></linear_acceleration></imu>'
                % (g, g, g, a, a, a))
    return ('<sensor name="%s" type="imu">%s<always_on>true</always_on><update_rate>%s</update_rate>'
            '<visualize>false</visualize><topic>%s</topic><gz_frame_id>%s</gz_frame_id>%s</sensor>'
            % (name, _pose(mount['xyz'], mount['rpy']), _f(rate), topic, frame_id, body))


def _carrier_link(name, mount, drone, sensors_xml):
    """Tiny-mass link rigidly attached to base_link (the x500 CoM) at the given mount; carries sensors."""
    m = drone['carrier_link_mass_kg']
    i = drone['carrier_link_inertia']
    return ('<link name="%s"><pose relative_to="%s">%s</pose><inertial><mass>%s</mass><inertia>'
            '<ixx>%s</ixx><ixy>0</ixy><ixz>0</ixz><iyy>%s</iyy><iyz>0</iyz><izz>%s</izz></inertia></inertial>'
            '%s</link>'
            '<joint name="%s_joint" type="fixed"><parent>%s</parent><child>%s</child></joint>'
            % (name, drone['base_link'], ' '.join(_f(v) for v in list(mount['xyz']) + list(mount['rpy'])),
               _f(m), _f(i), _f(i), _f(i), sensors_xml, name, drone['base_link'], name))


def build_drone_model_sdf(cfg):
    """x500_tagcam: model://x500 (merged) + camera/IMU carrier links on fixed joints + OdometryPublisher.

    Same pattern as PX4's x500_mono_cam_down. All mounts are relative to base_link, which is the x500 CoM
    (x500_base has an identity inertial pose).
    """
    d = cfg['drone']
    cam = cfg['camera']
    imu = cfg['imu']
    mounts = d['mounts']
    o = ['<?xml version="1.0"?>', '<sdf version="1.9">', '<model name="%s">' % d['model_dir_name'],
         '<self_collide>false</self_collide>',
         '<include merge="true"><uri>%s</uri></include>' % d['base_model_uri']]
    cm = mounts['camera']
    cam_sensor = (
        '<sensor name="camera" type="camera"><always_on>true</always_on><update_rate>%s</update_rate>'
        '<visualize>false</visualize><topic>%s</topic><gz_frame_id>%s</gz_frame_id>'
        '<camera><camera_info_topic>%s</camera_info_topic><horizontal_fov>%s</horizontal_fov>'
        '<image><width>%d</width><height>%d</height><format>R8G8B8</format></image>'
        '<clip><near>%s</near><far>%s</far></clip></camera></sensor>'
        % (_f(cam['rate_hz']), cam['image_topic'], cam['frame_id'], cam['info_topic'], _f(cam['hfov']),
           cam['width'], cam['height'], _f(cam['clip_near']), _f(cam['clip_far'])))
    o.append(_carrier_link('camera_link', cm, d, cam_sensor))
    for key, topic_key, fid_key, noisy in (('imu_ideal', 'ideal_topic', 'ideal_frame_id', None),
                                           ('imu_noisy', 'noisy_topic', 'noisy_frame_id', imu['noisy'])):
        sensor = _imu_sensor(key, imu[topic_key], imu[fid_key], imu['rate_hz'],
                             {'xyz': (0, 0, 0), 'rpy': (0, 0, 0)}, noisy)
        o.append(_carrier_link(key + '_link', mounts[key], d, sensor))
    od = d['odometry']
    off = d['x500']['com_in_model_frame']
    o.append(
        '<plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">'
        '<odom_frame>%s</odom_frame><robot_base_frame>%s</robot_base_frame><dimensions>3</dimensions>'
        '<xyz_offset>%s</xyz_offset><odom_publish_frequency>%s</odom_publish_frequency>'
        '<odom_topic>%s</odom_topic></plugin>'
        % (od['odom_frame'], od['robot_base_frame'], _vec(off), _f(od['publish_frequency']), od['gz_topic']))
    o.append('</model></sdf>')
    return '\n'.join(o) + '\n'


def build_drone_model_config(cfg):
    d = cfg['drone']
    return ('<?xml version="1.0"?>\n<model><name>%s</name><version>1.0</version><sdf version="1.9">model.sdf</sdf>'
            '<description>PX4 x500 with downward AprilTag camera, ideal and noisy IMUs at the CoM and a GT '
            'odometry publisher (generated by drone_sim world_gen).</description></model>\n' % d['model_dir_name'])


def _drone_include(cfg):
    d = cfg['drone']
    sp = d['spawn']
    return ('<include><uri>model://%s</uri><name>%s</name>%s</include>'
            % (d['model_dir_name'], d['model_name'], _pose((sp['x'], sp['y'], sp['z']), (0, 0, sp['yaw']))))


def build_world_sdf(cfg):
    out = _world_header(cfg)
    out.append(_floor(cfg))
    out.extend(_tags(cfg))
    out.extend(_anchors(cfg))
    out.append(_drone_include(cfg))
    out.append('</world></sdf>')
    return '\n'.join(out) + '\n'


def build_bridge(cfg):
    d = cfg['drone']
    c = cfg['controller']
    cam = cfg['camera']
    imu = cfg['imu']
    wname = cfg['world']['world_name']

    def entry(ros, gz, rt, gt, direction):
        return {'ros_topic_name': ros, 'gz_topic_name': gz, 'ros_type_name': rt,
                'gz_type_name': gt, 'direction': direction}

    return [
        entry('/clock', '/world/%s/clock' % wname, 'rosgraph_msgs/msg/Clock', 'gz.msgs.Clock', 'GZ_TO_ROS'),
        entry(c['odom_topic'], d['odometry']['gz_topic'], 'nav_msgs/msg/Odometry', 'gz.msgs.Odometry',
              'GZ_TO_ROS'),
        entry(cam['image_topic'], cam['image_topic'], 'sensor_msgs/msg/Image', 'gz.msgs.Image', 'GZ_TO_ROS'),
        entry(cam['info_topic'], cam['info_topic'], 'sensor_msgs/msg/CameraInfo', 'gz.msgs.CameraInfo',
              'GZ_TO_ROS'),
        entry(imu['ideal_topic'], imu['ideal_topic'], 'sensor_msgs/msg/Imu', 'gz.msgs.IMU', 'GZ_TO_ROS'),
        entry(imu['noisy_topic'], imu['noisy_topic'], 'sensor_msgs/msg/Imu', 'gz.msgs.IMU', 'GZ_TO_ROS'),
    ]


def build_apriltag_params(cfg):
    """Parameters for christianrauch/apriltag_ros (3.2.x) AprilTagNode: family, size, max_hamming,
    detector.*, pose_estimation_method, tag.ids/frames/sizes. Topics are image_rect / camera_info (remapped)."""
    at = cfg['apriltags']
    det = at['detector']
    ids = [int(t['id']) for t in at['tags']]
    return {'/**': {'ros__parameters': {
        'image_transport': 'raw',
        'family': at['family'].replace('tag', ''),
        'size': float(at['size_m']),
        'max_hamming': int(at['max_hamming']),
        'pose_estimation_method': at['pose_estimation_method'],
        'detector': {'threads': int(det['threads']), 'decimate': float(det['decimate']),
                     'blur': float(det['blur']), 'refine': bool(det['refine']),
                     'sharpening': float(det['sharpening']), 'debug': bool(det['debug'])},
        'tag': {'ids': ids,
                'frames': ['%s%d' % (at['detection_frame_prefix'], i) for i in ids],
                'sizes': [float(at['size_m'])] * len(ids)},
    }}}


def generate(out_dir, config_dir=None):
    """Write world.sdf, bridge.yaml, apriltag.yaml and models/x500_tagcam into out_dir. Returns their paths."""
    cfg = load_all(config_dir)
    model_dir = os.path.join(out_dir, 'models', cfg['drone']['model_dir_name'])
    os.makedirs(model_dir, exist_ok=True)
    paths = {
        'world': os.path.join(out_dir, cfg['world']['world_name'] + '.sdf'),
        'bridge': os.path.join(out_dir, 'bridge.yaml'),
        'apriltag': os.path.join(out_dir, 'apriltag.yaml'),
        'models_dir': os.path.join(out_dir, 'models'),
        'world_name': cfg['world']['world_name'],
        'model_name': cfg['drone']['model_name'],
    }
    with open(paths['world'], 'w') as f:
        f.write(build_world_sdf(cfg))
    with open(os.path.join(model_dir, 'model.sdf'), 'w') as f:
        f.write(build_drone_model_sdf(cfg))
    with open(os.path.join(model_dir, 'model.config'), 'w') as f:
        f.write(build_drone_model_config(cfg))
    with open(paths['bridge'], 'w') as f:
        yaml.safe_dump(build_bridge(cfg), f, sort_keys=False)
    with open(paths['apriltag'], 'w') as f:
        yaml.safe_dump(build_apriltag_params(cfg), f, sort_keys=False)
    return paths


def main(out_dir, config_dir=None):
    return generate(out_dir, config_dir)


def cli():
    ap = argparse.ArgumentParser(description='Generate drone_sim world SDF + bridge YAML')
    ap.add_argument('out_dir')
    ap.add_argument('--config-dir', default=None)
    args = ap.parse_args()
    for k, v in generate(args.out_dir, args.config_dir).items():
        print('%s: %s' % (k, v))


if __name__ == '__main__':
    cli()
