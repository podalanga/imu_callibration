"""Convert an EuRoC MAV sequence (ASL format) to the common dataset format.

  python scripts/convert_euroc.py ~/datasets/V1_01_easy/mav0 data/euroc_V101
imu0/data.csv: timestamp [ns], w_RS_S_x,y,z [rad/s], a_RS_S_x,y,z [m/s^2]
state_groundtruth_estimate0/data.csv: timestamp, p_RS_R xyz, q_RS wxyz, v_RS_R xyz, b_w, b_a
The ground truth is the pose of the IMU frame in a gravity-aligned world frame, so roll and pitch
can be compared directly; yaw depends on the world frame and is not compared.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import io

src, out = sys.argv[1], sys.argv[2]
imu = np.loadtxt(os.path.join(src, "imu0", "data.csv"), delimiter=",", comments="#")
gt = np.loadtxt(os.path.join(src, "state_groundtruth_estimate0", "data.csv"), delimiter=",", comments="#")
t_us = (imu[:, 0] // 1000).astype(np.int64)
ds = io.Dataset(imu=io.Imu(t_us=t_us, gyro=imu[:, 1:4], acc=imu[:, 4:7], temp=np.full(len(imu), 25.0),
                           seq=np.arange(len(imu))),
                gt=np.column_stack([gt[:, 0] // 1000, gt[:, 1:4], gt[:, 4:8], gt[:, 8:11]]),
                config=dict(source=f"EuRoC {src}", rate_hz=200))
io.save(out, ds)
print(f"wrote {out}: {len(t_us)} IMU samples, {len(gt)} ground-truth poses")
