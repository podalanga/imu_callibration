import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from drone_sim import world_gen
from drone_sim.config_loader import default_config_dir

PX4_RC_BUILD = 'build/px4_sitl_default'


def _flag(context, name):
    return LaunchConfiguration(name).perform(context).lower() == 'true'


def _setup(context, *args, **kwargs):
    headless = _flag(context, 'headless')
    gui = _flag(context, 'gui')
    detector = _flag(context, 'detector')
    use_px4 = _flag(context, 'px4')
    cfg_dir = LaunchConfiguration('config_dir').perform(context) or default_config_dir()
    px4_dir = LaunchConfiguration('px4_dir').perform(context) or os.environ.get('PX4_DIR', '/opt/PX4-Autopilot')

    out_dir = os.path.join(os.path.expanduser('~/.ros/drone_sim'), 'run_%d' % os.getpid())
    paths = world_gen.main(out_dir, cfg_dir)

    if headless:
        gz_args = '-s -r --headless-rendering %s' % paths['world']
    elif not gui:
        gz_args = '-s -r %s' % paths['world']
    else:
        gz_args = '-r %s' % paths['world']

    px4_build = os.path.join(px4_dir, PX4_RC_BUILD)
    actions = [
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', paths['models_dir']),
        # x500 / x500_base / mono_cam models and PX4's MotorFailurePlugin
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(px4_dir, 'Tools/simulation/gz/models')),
        AppendEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH',
                                  os.path.join(px4_build, 'src/modules/simulation/gz_plugins')),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items()),
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='gz_bridge', output='screen',
             parameters=[{'config_file': paths['bridge'], 'use_sim_time': True}]),
    ]

    if use_px4:
        rootfs = os.path.join(px4_build, 'rootfs')
        os.makedirs(rootfs, exist_ok=True)
        if _flag(context, 'reset_params'):
            for f in ('parameters.bson', 'parameters_backup.bson'):
                try:
                    os.remove(os.path.join(rootfs, f))
                except OSError:
                    pass
        actions.append(ExecuteProcess(
            cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'], name='microxrce_agent', output='screen'))
        # PX4 attaches to the already running model (PX4_GZ_STANDALONE=1) and waits for the world to be ready.
        actions.append(ExecuteProcess(
            cmd=[os.path.join(px4_build, 'bin/px4'), '-i', '0', '-d', os.path.join(px4_build, 'etc')],
            cwd=rootfs, name='px4_sitl', output='screen',
            additional_env={
                'PX4_GZ_STANDALONE': '1',
                'PX4_SYS_AUTOSTART': '4099',
                'PX4_GZ_WORLD': paths['world_name'],
                'PX4_GZ_MODEL_NAME': paths['model_name'],
                'PX4_SIM_MODEL': 'gz_x500',
            }))

    common = [{'use_sim_time': True, 'config_dir': cfg_dir}]
    nodes = ['tf_node', 'uwb_node', 'uwb_multilateration_node']
    if use_px4:
        nodes += ['gt_to_px4_ev_node', 'px4_mission_node']
    for exe in nodes:
        actions.append(Node(package='drone_sim', executable=exe, name=exe, output='screen', parameters=common))
    if detector:
        actions.append(Node(
            package='apriltag_ros', executable='apriltag_node', name='apriltag', output='screen',
            remappings=[('image_rect', '/camera/image_raw'), ('camera_info', '/camera/camera_info')],
            parameters=[paths['apriltag'], {'use_sim_time': True}]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false', description='server-only, headless rendering'),
        DeclareLaunchArgument('gui', default_value='true', description='start the Gazebo GUI (ignored if headless)'),
        DeclareLaunchArgument('detector', default_value='true', description='run apriltag_ros'),
        DeclareLaunchArgument('px4', default_value='true',
                              description='start MicroXRCEAgent + PX4 SITL + EV bridge + mission node'),
        DeclareLaunchArgument('reset_params', default_value='true',
                              description='delete PX4 parameters.bson so airframe defaults apply'),
        DeclareLaunchArgument('px4_dir', default_value='', description='PX4-Autopilot dir (default $PX4_DIR)'),
        DeclareLaunchArgument('config_dir', default_value='', description='override config directory'),
        OpaqueFunction(function=_setup),
    ])
