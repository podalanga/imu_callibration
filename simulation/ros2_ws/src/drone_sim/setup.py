import os
from setuptools import setup

package_name = 'drone_sim'


def tree(src, dst):
    """Recursively map every file under src to an install directory under dst."""
    out = {}
    for root, _, files in os.walk(src):
        if not files:
            continue
        rel = os.path.relpath(root, src)
        d = dst if rel == '.' else os.path.join(dst, rel)
        out.setdefault(d, []).extend(os.path.join(root, f) for f in files)
    return list(out.items())


data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]
data_files += tree('config', os.path.join('share', package_name, 'config'))
data_files += tree('launch', os.path.join('share', package_name, 'launch'))
data_files += tree('models', os.path.join('share', package_name, 'models'))  # tag textures

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='podalanga',
    maintainer_email='joshuajohn.lvj@gmail.com',
    description='Gazebo Harmonic drone sim over an AprilTag track',
    license='MIT',
    entry_points={
        'console_scripts': [
            'px4_mission_node = drone_sim.px4_mission_node:main',
            'gt_to_px4_ev_node = drone_sim.gt_to_px4_ev_node:main',
            'uwb_multilateration_node = drone_sim.uwb_multilateration_node:main',
            'uwb_node = drone_sim.uwb_node:main',
            'tf_node = drone_sim.tf_node:main',
            'world_gen = drone_sim.world_gen:cli',
            'camera_view = drone_sim.camera_view:main',
        ],
    },
)
