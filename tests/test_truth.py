"""Truth-known test suite (task C2). Every test feeds inputs whose correct output is known exactly.

Run:  pytest -q tests
A failure here is almost always a frame, sign or unit bug. Fix it before tuning anything.
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import quat, synth, runner, calib, attitude
from nav.eskf import ESKF

G = synth.G0
NOISELESS = dict(Na=0.0, Ng=0.0, lsb_acc=0.0, lsb_gyro=0.0)


# ---------------------------------------------------------------- conventions
def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        phi = rng.normal(size=3)
        phi *= min(1.0, 3.0 / np.linalg.norm(phi))
        assert np.allclose(quat.qlog(quat.qexp(phi)), phi, atol=1e-9)
        R = quat.q2R(quat.qexp(phi))
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_body_to_world_direction():
    # yaw +90 deg: body x axis must point along world +y
    q = quat.euler2q(0, 0, np.pi / 2)
    assert np.allclose(quat.q2R(q) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_gravity_sign_at_rest():
    # level phone at rest reads +g on its up axis (acquisition test A3 in software)
    t, w, f, gt = synth.static_session(duration_s=1.0)
    assert np.allclose(f[0], [0, 0, G], atol=1e-9)


def test_tilt_from_accel():
    for r, p in [(0.1, -0.2), (-0.3, 0.25), (0.0, 0.0)]:
        f = quat.q2R(quat.euler2q(r, p, 0.7)).T @ np.array([0, 0, G])
        assert np.allclose(quat.tilt_from_accel(f), (r, p), atol=1e-12)


# ---------------------------------------------------------------- propagation
def test_static_box_stays_put():
    f = ESKF(quat.euler2q(0, 0, 0))
    for _ in range(200 * 60):
        f.predict(np.array([0, 0, G]), np.zeros(3), 0.005)
    assert np.linalg.norm(f.p) < 1e-9 and np.linalg.norm(f.v) < 1e-9


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_constant_rotation_90deg(axis):
    w = np.zeros((201, 3)); w[:, axis] = np.pi / 2       # 90 deg/s for 1 s
    t = np.arange(201) * 0.005
    q = attitude.gyro_only([1, 0, 0, 0], w, t)[-1]
    e = np.zeros(3); e[axis] = np.pi / 2
    assert np.allclose(quat.qlog(q), e, atol=1e-9)


def test_constant_acceleration():
    # 0.2 m/s^2 along body x for 5 s from rest: x = 0.5*0.2*25 = 2.5 m, v = 1.0 m/s
    f = ESKF(quat.euler2q(0, 0, 0))
    for _ in range(1000):
        f.predict(np.array([0.2, 0, G]), np.zeros(3), 0.005)
    assert np.allclose(f.p, [2.5, 0, 0], atol=1e-9)
    assert np.allclose(f.v, [1.0, 0, 0], atol=1e-12)


def test_noise_free_static_session_does_not_drift():
    # generator and strapdown agree: a noise-free stationary IMU reintegrates to exactly zero
    ds = synth.make_static_dataset(synth.ErrorModel(**NOISELESS), duration_s=60.0)
    r = runner.run(ds.imu, runner.NavConfig(use_zupt=False))
    assert np.linalg.norm(r["est"][-1, 1:4]) < 1e-6              # position
    assert np.linalg.norm(r["est"][-1, 4:7]) < 1e-6              # velocity


def test_tilt_error_couples_as_derived():
    # pitch error d_theta_y produces x acceleration error g*d_theta_y (Chapter 12 check)
    dth = np.radians(0.1)
    f = ESKF(quat.euler2q(0, dth, 0))
    for _ in range(200 * 10):
        f.predict(np.array([0, 0, G]), np.zeros(3), 0.005)
    assert np.isclose(f.p[0], 0.5 * G * np.sin(dth) * 100, rtol=1e-3)   # nominal-minus-true = -0.5 g dtheta_y t^2


# ---------------------------------------------------------------- covariance and updates
def test_covariance_growth_matches_noise_density():
    Na, Ng = 4e-3, 1e-4
    f = ESKF(quat.euler2q(0, 0, 0), Na=Na, Ng=Ng, Kba=0, Kbg=0,
             P0_sigmas=dict(p=0, v=0, att=0, ba=0, bg=0))
    for _ in range(2000):                                  # 10 s
        f.predict(np.array([0, 0, G]), np.zeros(3), 0.005)
    assert np.isclose(f.P[5, 5], Na**2 * 10, rtol=1e-6)    # vertical velocity: no tilt coupling
    assert np.isclose(f.P[8, 8], Ng**2 * 10, rtol=1e-6)    # yaw


def test_scalar_zupt_worked_example():
    # Chapter 11: P- = 1.54e-4, R = 1e-4  ->  K = 0.606, P+ = 6.07e-5
    f = ESKF(quat.euler2q(0, 0, 0), P0_sigmas=dict(p=0, v=0, att=0, ba=0, bg=0))
    f.P[3, 3] = 1.54e-4
    f.v = np.array([0.0123, 0, 0])
    H = np.zeros((1, 15)); H[0, 3] = 1
    ok, nis = f.update(np.array([-f.v[0]]), H, np.array([[1e-4]]), gate=False)
    assert np.isclose(f.P[3, 3], 1.54e-4 * 1e-4 / 2.54e-4, rtol=1e-9)
    assert np.isclose(f.v[0], 0.0123 * (1 - 1.54 / 2.54), rtol=1e-9)


def test_covariance_stays_symmetric_positive():
    ds = synth.make_static_dataset(synth.ErrorModel(seed=2), duration_s=60.0)
    r = runner.run(ds.imu, runner.NavConfig())
    P = r["filter"].P
    assert np.allclose(P, P.T, atol=1e-15)
    assert np.all(np.linalg.eigvalsh(P) > 0)


# ---------------------------------------------------------------- aided accuracy on synthetic data
def test_zupt_bounds_drift_on_static_session():
    # a phone sitting still: free inertial integration drifts away with accel bias, ZUPT/ZARU bounds it
    err = synth.ErrorModel(ba=np.array([2e-3, -1.5e-3, 1e-3]) * G, bg=np.radians([0.02, -0.02, 0.02]), seed=5)
    ds = synth.make_static_dataset(err, duration_s=120.0)
    free_p = np.linalg.norm(runner.run(ds.imu, runner.NavConfig(use_zupt=False))["est"][-1, 1:4])
    aided_p = np.linalg.norm(runner.run(ds.imu, runner.NavConfig(use_zupt=True))["est"][-1, 1:4])
    assert free_p > 1.0           # unaided position error grows unbounded even at rest
    assert aided_p < 0.05          # ZUPT/ZARU keeps it near zero (synthetic, modelled errors only)


# ---------------------------------------------------------------- calibration round trip
def test_calibration_recovers_known_parameters():
    err = synth.ErrorModel.random_mpu6050(seed=3)
    ds = synth.make_calibration_dataset(err, n_poses=40, seed=3)
    cp, iv = calib.calibrate(ds.imu, G, t_init=40.0)
    assert np.allclose(np.diag(cp.Ka), np.diag(err.Ka), atol=3e-4)
    assert np.allclose(cp.ba, err.ba, atol=5e-3)
    assert np.allclose(np.diag(cp.Kg), np.diag(err.Kg), atol=1e-3)
    assert np.allclose(cp.bg, err.bg, atol=1e-4)
    assert np.allclose(cp.Tg, err.Tg, atol=1e-3)
