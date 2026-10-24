# syntax=docker/dockerfile:1
# Stage "base": ROS 2 Jazzy + Gazebo (ros_gz) + apriltag_ros. Enough for the world/render checks.
FROM osrf/ros:jazzy-desktop AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-gz ros-jazzy-apriltag-ros ros-jazzy-apriltag-msgs ros-jazzy-rqt-image-view \
        python3-numpy python3-yaml python3-pip python3-colcon-common-extensions \
        git curl wget ca-certificates build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------------------------
# Stage "px4": PX4-Autopilot SITL (built), Micro-XRCE-DDS-Agent, px4_msgs underlay.
FROM base AS px4
ARG PX4_TAG=v1.16.2
ARG XRCE_TAG=v2.4.3
ARG PX4_MSGS_BRANCH=release/1.16
ARG PX4_BUILD_JOBS=6
ENV PX4_DIR=/opt/PX4-Autopilot

# ubuntu.sh calls sudo and lsb_release; Gazebo already comes from ros_gz, so skip its sim tools.
RUN apt-get update && apt-get install -y --no-install-recommends sudo lsb-release \
    && rm -rf /var/lib/apt/lists/*

# --shallow-submodules drops NuttX's tags, but px_update_git_header.py reads them even for SITL
# (IndexError otherwise). Placeholder tag; only the version string in the header is affected.
RUN git clone --depth 1 --branch ${PX4_TAG} --recurse-submodules --shallow-submodules \
        https://github.com/PX4/PX4-Autopilot.git ${PX4_DIR} \
    && git -C ${PX4_DIR}/platforms/nuttx/NuttX/nuttx tag nuttx-0.0.0
# ubuntu.sh pip-installs PX4's python deps; without a pin pip tries to replace Debian's numpy
# (fails: "RECORD file not found") and a numpy 2.x upgrade would break ROS's cv_bridge ABI.
# PIP_IGNORE_INSTALLED: install into /usr/local alongside Debian packages instead of uninstalling
# them; the constraint keeps numpy on the Debian version (same ABI as the ROS debs).
RUN python3 -c "import numpy; print('numpy==' + numpy.__version__)" > /etc/pip-constraints.txt \
    && cat /etc/pip-constraints.txt
RUN cd ${PX4_DIR} \
    && PIP_CONSTRAINT=/etc/pip-constraints.txt PIP_BREAK_SYSTEM_PACKAGES=1 PIP_IGNORE_INSTALLED=1 \
       bash Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools \
    && rm -rf /var/lib/apt/lists/*

# Custom airframe 4099 (ROMFS airframes are listed explicitly in CMake, they are not globbed).
COPY airframes/4099_gz_x500_tagcam ${PX4_DIR}/ROMFS/px4fmu_common/init.d-posix/airframes/4099_gz_x500_tagcam
RUN sed -i 's/^\(\s*\)4001_gz_x500$/\14001_gz_x500\n\14099_gz_x500_tagcam/' \
        ${PX4_DIR}/ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt \
    && grep -q 4099_gz_x500_tagcam ${PX4_DIR}/ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt

# Build SITL only (the gz_x500 run target is NOT used: our launch starts gz and PX4 separately).
# Limited parallelism: the host has 15 GB RAM.
# Gazebo comes from the ROS vendor packages (/opt/ros/jazzy/opt/gz_*_vendor, gz-transport13 + gz-msgs10), so PX4's
# gz_bridge module finds it through CMAKE_PREFIX_PATH and links it via LD_LIBRARY_PATH at run time (ROS setup.bash).
RUN cd ${PX4_DIR} \
    && export CMAKE_PREFIX_PATH="$(ls -d /opt/ros/jazzy/opt/*_vendor | tr '\n' ':')" \
    && CMAKE_BUILD_PARALLEL_LEVEL=${PX4_BUILD_JOBS} make px4_sitl \
    && test -d build/px4_sitl_default/src/modules/simulation/gz_bridge

# Micro XRCE-DDS Agent (PX4 <-> ROS 2 bridge, UDP 8888)
RUN git clone --depth 1 --branch ${XRCE_TAG} https://github.com/eProsima/Micro-XRCE-DDS-Agent.git /opt/xrce \
    && cmake -S /opt/xrce -B /opt/xrce/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/xrce/build --parallel ${PX4_BUILD_JOBS} \
    && cmake --install /opt/xrce/build && ldconfig && rm -rf /opt/xrce

# px4_msgs underlay (must match the PX4 release)
RUN mkdir -p /opt/px4_ws/src && git clone --depth 1 --branch ${PX4_MSGS_BRANCH} \
        https://github.com/PX4/px4_msgs.git /opt/px4_ws/src/px4_msgs \
    && cd /opt/px4_ws && . /opt/ros/jazzy/setup.sh \
    && colcon build --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log

ENV GZ_SIM_RESOURCE_PATH=${PX4_DIR}/Tools/simulation/gz/models:${PX4_DIR}/Tools/simulation/gz/worlds
ENV GZ_SIM_SYSTEM_PLUGIN_PATH=${PX4_DIR}/build/px4_sitl_default/src/modules/simulation/gz_plugins
RUN echo 'source /opt/ros/jazzy/setup.bash; source /opt/px4_ws/install/setup.bash; \
[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' >> /root/.bashrc

# Sources ROS + px4_msgs + /ws overlay for every command (docker run, and run.sh's docker exec).
COPY --chmod=0755 entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
WORKDIR /ws
