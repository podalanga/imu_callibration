# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Extracts raw IMU readings from an Android phone over a local WebSocket, calibrates out systematic
sensor error, and characterises how the phone's own IMU drifts: noise density / bias instability
(Allan deviation), attitude drift under free integration, and how much ZUPT/ZARU (stationary
aiding) bounds position/velocity drift while the phone sits still.

This started as an ESP32 + MPU6050 trolley-on-a-track indoor-positioning project (firmware, UART
protocol, hardware-in-the-loop replay, wheel/line-constraint aiding, a fixed-point pipeline). That
hardware-specific code has been removed; the focus now is phone acquisition + calibration +
drift/noise analysis in pure Python. `Strapdown Field Guide.html` still has the full math
derivations as background theory, but its "Track and ground truth", "Real-time embedded
implementation", "Team plan" and "Demonstration and judge defence" chapters describe the old
hardware scope and no longer apply.

## Commands

```bash
pip install -r requirements.txt
pytest -q tests                      # full suite
pytest -q tests/test_truth.py::test_zupt_bounds_drift_on_static_session   # single test
```

Phone acquisition (requires the phone's Sensor Server app running; `localhost:8080` is reached
via `adb forward`, or point `--host` at the phone's Wi-Fi IP:8080):

```bash
adb forward tcp:8080 tcp:8080
python phone_data/websocket_sensor.py                           # quick terminal view, nothing saved
python phone_data/record_phone.py runs/phone_calib_s1            # logs imu.csv, Ctrl+C to stop
python phone_data/record_phone.py runs/phone_static_10min --minutes 10
```

Analysis pipeline (same scripts work on real recordings or synthetic data):

```bash
python scripts/make_synth.py calib data/syn_calib_s1 --seed 1
python scripts/make_synth.py static data/syn_static --minutes 30

python scripts/run_calib.py <session_dir> --lat <latitude> [--holdout <dirs>] --out calib.yaml [--plots results/calib]
python scripts/run_allan.py <static_session_dir> [--calib calib.yaml] --out noise.yaml [--plot results/allan.png]
python scripts/run_filter.py <static_session_dir> [--calib calib.yaml] [--noise noise.yaml] --ablation [--plots results/drift]

python scripts/check_acq.py <session_dir> [--faces]             # acquisition sanity checks (timing, noise floor, 6-face test)
python scripts/compare_attitude.py <dataset_with_groundtruth.csv> [--calib calib.yaml] [--plot out.png]
python scripts/convert_euroc.py <euroc_mav0_dir> <out_dir>       # import a public ground-truthed benchmark
```

All `scripts/*.py` are standalone CLIs: each does `sys.path.insert(0, repo_root)` then
`from nav import ...` — there is no package install step.

## Architecture

**`phone_data/`** — the only layer that touches the network. Both scripts open three parallel
`websocket.WebSocketApp` connections (`ws://<host>/sensor/connect?type=<android.sensor.*>`) for
accelerometer/gyroscope/magnetometer *_uncalibrated* streams, each on its own daemon thread
overwriting a shared `latest` dict. `websocket_sensor.py` just prints the latest values at ~60 Hz;
`record_phone.py` samples that same shared state at a fixed `--rate` and writes it to
`imu.csv` in the common dataset format below, which is what makes a phone recording usable by
everything in `nav/` and `scripts/`. Phone accel/gyro values are already SI (m/s^2, rad/s) — no
datasheet/LSB conversion is needed, only calibration.

**`nav/`** — pure-Python reference implementation, no I/O beyond `io.py`:
- `io.py` — the common on-disk dataset format every script reads/writes: a directory with
  `imu.csv` (`t_us,seq,gx,gy,gz,ax,ay,az,temp_c`), optional `groundtruth.csv`/`markers.csv`/`est.csv`,
  and `config.yaml`. `Imu`/`Dataset` dataclasses wrap it.
- `quat.py` — the rotation convention used *everywhere* in this repo: Hamilton quaternions,
  scalar-first (`[w,x,y,z]`), `q` maps body→world, attitude error is a local/right perturbation
  (`q_true = q ⊗ Exp(dtheta)`). Do not introduce a different convention anywhere else.
- `calib.py` — multi-position calibration (Tedaldi, Pretto, Menegatti, ICRA 2014): a static-interval
  detector (`static_intervals`, parameterless threshold sweep) feeds nonlinear least-squares fits
  for accel and gyro. Error model: `a_O = Ta·Ka·(a_S + ba)`, `w_O = Tg·Kg·(w_S + bg)` (units m/s^2,
  rad/s) — `CalibParams` holds/persists this to/from YAML. `six_position()` is the naive baseline
  used to show the multi-position fit is better.
- `allan.py` — overlapping Allan deviation (`adev`) and noise-parameter extraction (`N` = angle/
  velocity random walk at τ=1s, `B` = bias instability at the ADEV minimum, `K` = rate random walk)
  from a long static recording. This is the primary drift-characterisation tool.
- `attitude.py` — baseline attitude integrators (gyro-only, Mahony, Madgwick) for comparing attitude
  drift against each other or against ground truth.
- `eskf.py` — 15-state error-state Kalman filter, float64 reference. Error state order is
  `dp(0:3) dv(3:6) dtheta(6:9) dba(9:12) dbg(12:15)`. `predict()` is plain strapdown integration
  (this is where drift comes from); `update()` is the generic EKF correction with chi-squared
  gating; `zupt()`/`zaru()` are the only measurement models (zero-velocity/zero-angular-rate update
  during detected stationary periods — this is what bounds drift at rest).
- `runner.py` — the causal pipeline wiring the above together: `NavConfig` (noise/detector/aiding
  parameters) + `StopDetector` (a causal SHOE-style stillness test) + `run()`, which aligns on the
  initial static period, predicts every sample, and applies ZUPT/ZARU while still.
- `synth.py` — synthetic data with *exact* ground truth, built from an `ErrorModel` (MPU6050-like
  bias/scale/non-orthogonality/noise injector) applied to ideal signals: `calibration_session()`
  (static poses joined by smooth rotations, for testing `calib.py`) and `static_session()` (zero
  motion the whole run — p=0, v=0, q=identity truth — for testing the drift/Allan pipeline without
  a real phone).

**`scripts/`** — thin CLIs over `nav/`. The key design point: `run_filter.py` assumes the input
dataset is a *static* (phone-at-rest) session, so true position/velocity are zero and attitude is
constant throughout — meaning the filter's own estimate drifting away from that zero baseline *is*
the drift error, with no ground truth required. That's why real recordings for drift analysis
should be taken with the phone sitting still, and why `synth.static_session` (not a moving-track
generator) is the synthetic stand-in. `compare_attitude.py` + `convert_euroc.py` are a separate,
optional validation path: they check the attitude-drift code against the public EuRoC MAV benchmark,
which does have real ground truth, independent of the (ground-truth-free) phone pipeline.

**`tests/test_truth.py`** — "truth-known" suite: every test's correct output is known exactly
(quaternion round-trips, static-box/constant-acceleration ESKF propagation, a worked covariance
example, calibration round-trip on synthetic data, free-inertial-vs-ZUPT drift bound). A failure
here is almost always a frame/sign/unit bug — fix it before tuning anything.

## Conventions (do not change one without the others)

* Hamilton quaternions, scalar-first; `q` maps body frame to world frame; attitude error is a
  local (right) perturbation.
* Calibration: `a_O = Ta Ka (a_S + ba)`, `w_O = Tg Kg (w_S + bg)`; units m/s^2 and rad/s.
* `imu.csv` always stores raw sensor-frame values as received (phone `*_uncalibrated` streams are
  already SI; calibration is applied downstream, never baked into the recording).
