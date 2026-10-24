"""Embedded filter (firmware/main/eskf.c + nav.c, float32) must match the Python reference (float64).
This is test E6 / gate M5's numerical-agreement check, run on the host."""
import os, sys, subprocess, shutil
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nav import synth, io, runner


@pytest.mark.skipif(shutil.which("cc") is None and shutil.which("gcc") is None, reason="no C compiler")
@pytest.mark.parametrize("flags,kw", [([], {}), (["--line"], dict(use_line=True))])
def test_c_matches_python(tmp_path, flags, kw):
    subprocess.run(["make", "-s", "-C", os.path.join(ROOT, "firmware", "host"), "host_replay"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ds = synth.make_track_dataset(synth.ErrorModel(ba=np.array([0.02, -0.015, 0.01]), bg=np.radians([0.02] * 3), seed=4),
                                  bump_deg=0.3)
    io.save(tmp_path / "d", ds)
    subprocess.run([os.path.join(ROOT, "firmware", "host", "host_replay"), str(tmp_path / "d" / "imu.csv"),
                    str(tmp_path / "c.csv")] + flags, check=True, stderr=subprocess.DEVNULL)
    c = io.load_est(str(tmp_path / "c.csv"))
    p = runner.run(io.load(tmp_path / "d").imu, runner.NavConfig(**kw))["est"]
    assert c.shape == p.shape
    assert np.max(np.linalg.norm(c[:, 1:4] - p[:, 1:4], axis=1)) < 0.01      # 1 cm over the whole run
    assert np.mean(c[:, 17] != p[:, 17]) < 0.001                              # same stop decisions


@pytest.mark.skipif(shutil.which("cc") is None and shutil.which("gcc") is None, reason="no C compiler")
def test_fixed_point_matches_float_mirror(tmp_path):
    """Tier 3 bonus check: integer pipeline vs the same algorithm in float64 (nav/fixedpoint.py)."""
    from nav import fixedpoint
    from nav.quat import euler2q, tilt_from_accel
    subprocess.run(["make", "-s", "-C", os.path.join(ROOT, "firmware", "host"), "host_fixed"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ds = synth.make_track_dataset(synth.ErrorModel(seed=7), bump_deg=0.3)
    io.save(tmp_path / "d", ds)
    imu = io.load(tmp_path / "d").imu
    r, p = tilt_from_accel(imu.acc[:1600].mean(axis=0))
    q0 = euler2q(r, p, 0.0)
    subprocess.run([os.path.join(ROOT, "firmware", "host", "host_fixed"), str(tmp_path / "d" / "imu.csv"),
                    str(tmp_path / "fx.csv"), *map(str, q0)], check=True)
    c = np.loadtxt(tmp_path / "fx.csv", delimiter=",", skiprows=1)
    f = fixedpoint.run(imu, q0)
    assert c.shape[0] == f.shape[0]
    assert np.max(np.linalg.norm(c[:, 1:3] - f[:, 1:3], axis=1)) < 0.2     # quantisation only
    assert np.mean(c[:, 11] != f[:, 11]) < 0.01
