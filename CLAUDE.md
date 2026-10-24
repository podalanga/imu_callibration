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
python phone_data/app.py                                        # test console on :8050 (does adb forward itself)
python phone_data/analysis.py runs/<run> --kind static|quick|turnon [--extra <runs>] [--calib calib.yaml] [--out <dir>]
adb forward tcp:8080 tcp:8080
python phone_data/websocket_sensor.py                           # quick terminal view, nothing saved
python phone_data/phone_logger.py runs/static_2h --minutes 120 --dashboard   # record + live dashboard on :8050
python phone_data/phone_logger.py --dashboard                   # live dashboard only, nothing saved
python phone_data/phone_logger.py --replay runs/static_2h --speed 20 --dashboard
python phone_data/record_phone.py runs/phone_calib_s1            # older command line, same logger underneath
python phone_data/merge.py runs/static_2h                       # rebuild imu.csv from raw_*.csv (e.g. after a crash)
```

Analysis pipeline (same scripts work on real recordings or synthetic data):

```bash
python scripts/make_synth.py calib data/syn_calib_s1 --seed 1
python scripts/make_synth.py static data/syn_static --minutes 30

python scripts/run_calib.py <session_dir> --lat <latitude> [--holdout <dirs>] --out calib.yaml [--plots results/calib]
python scripts/run_allan.py <static_session_dir> [--calib calib.yaml] --out noise.yaml [--plot results/allan.png]
python scripts/characterise_noise.py <long_static_dir> [--extra <short sessions>] [--calib calib.yaml] --out phone_noise.yaml [--plots results/noise]
python scripts/make_synth.py static data/phone_sim --minutes 30 --noise-model phone_noise.yaml   # simulate this phone
python scripts/run_filter.py <static_session_dir> [--calib calib.yaml] [--noise noise.yaml] --ablation [--plots results/drift]

python scripts/check_acq.py <session_dir> [--faces]             # acquisition sanity checks (timing, noise floor, 6-face test)
python scripts/compare_attitude.py <dataset_with_groundtruth.csv> [--calib calib.yaml] [--plot out.png]
python scripts/convert_euroc.py <euroc_mav0_dir> <out_dir>       # import a public ground-truthed benchmark
```

All `scripts/*.py` are standalone CLIs: each does `sys.path.insert(0, repo_root)` then
`from nav import ...` — there is no package install step.

## Architecture

**`phone_data/`** — the only layer that touches the network. The phone runs the Sensor Server app
(package `github.umer0586.sensorserver`); `stream.py` opens ONE aiohttp WebSocket to
`ws://<host>/sensors/connect?types=[...]` (URL-encoded JSON list) carrying accelerometer/gyroscope
(and optionally magnetometer) *_uncalibrated* events. Every event is kept with its
`SensorEvent.timestamp` (hardware ns, phone boot clock); never use laptop arrival time, and never
sample a "latest value" (that was the old design and it dropped about half the samples).
On the CPH2661 (Bosch BMI26x) the stream is ~395 Hz with ~6 µs timing std and accel/gyro on one clock.
- `phone_logger.py` — records `raw_<sensor>.csv` (`t_ns,x,y,z,bx,by,bz`; bx..bz is Android's own bias
  estimate, not applied), `battery.csv` (battery temperature polled over adb every 10 s; the phone
  does not expose IMU temperature), `meta.yaml`, then calls `merge.py`. `--dashboard` serves
  `dashboard.html` on :8050 from the same asyncio loop; the file writer never waits for it.
- `merge.py` — raw streams → `imu.csv` + `config.yaml`: gyro timestamps are the master clock, accel
  interpolated onto them, `seq` = sample index implied by the timestamp (so dropped samples show as
  sequence gaps in `check_acq.py`), `temp_c` = battery temperature. Writes an acquisition report.
- `live_stats.py` — O(1)-per-event running metrics for the dashboard: timing health, running bias,
  free static drift after a `--align` still period with Allan-based 1σ envelopes, live Allan.
- `record_phone.py` / `websocket_sensor.py` — old command lines, now thin wrappers over the above.
- `app.py` + `web/app.html` — the test console (aiohttp + one page, uPlot and three.js from CDNs). One
  phone WebSocket fans out to a monitor (rates, plots, stillness), a `Demo` (pure-Python Mahony for the 3D
  view; drift race) and, during a test, a recorder (`RawWriter` + `LiveStats`). An adb loop keeps the port
  forward set and checks plugged in / Stay awake / Sensor Server in the foreground. Stopping a test starts
  `analysis.py` as a subprocess; the UI polls `<run>/progress.json`. Runs and reports are served at /runs/.
- `analysis.py` — the automatic pipeline (merge → quality verdicts → noise model on the longest undisturbed
  stretch → drift segments vs prediction → ESKF ablation on ≤ 20 min → report), all plots and
  `results.json`. `report.py` renders a self-contained HTML report (plots base64-embedded).
  ZARU sigma is set to the measured per-sample gyro noise N·√fs; the MPU6050 default is too tight for a phone.
Phone accel/gyro values are already SI (m/s^2, rad/s) — no datasheet/LSB conversion is needed, only calibration.
Android device frame: x right, y toward the top edge, z out of the screen.

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
- `noise_model.py` — per-sensor, per-axis noise model from static sessions: N/B/K from a weighted
  least-squares fit of the Allan variance over τ ≤ T/10 (`fit_allan`; reading B at the raw ADEV minimum
  fails when the curve is still falling), quantisation step, temperature slope, turn-on repeatability,
  timing jitter. `predicted_drift()` gives 1σ heading/position drift including the error of the
  initial bias average (N/√t_align, usually the dominant term); `drift_segments()` measures the same
  thing on back-to-back segments of a real recording. `error_model_params()` → `synth.ErrorModel.from_noise_yaml`.
  The fit weights each τ by its Allan error bar and flags B/K as resolved only when the term carries ≥ 50 %
  of the fitted variance somewhere; otherwise reports show "< B_limit / K_limit" and an unresolved K is not
  passed to the filter (`filter_params` emits Kba/Kbg only when resolved). `temp_compensate()` removes the
  measured linear temperature slope (analysis.py applies it before drift segments and the filter).
  `predicted_drift_axes()` uses gyro z for heading and gyro/accel x, y for position (axes differ on phones).
  `disturbances()` flags 1 s windows whose spread exceeds 4× the run's median (or a floor) or whose gravity
  direction moved > 0.5°; `disturbance_rule()` is shared with the live detector in `live_stats.py`.
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
  a real phone). `ErrorModel` takes per-axis arrays and an optional Gauss-Markov bias instability
  (`B_*`, `tau_*`); `ErrorModel.from_noise_yaml()` builds the model of a real, characterised phone.
  Sign convention: reading = truth − bias, so the yaml's measured mean reading enters as `bg = -mean`.

**`scripts/`** — thin CLIs over `nav/`. The key design point: `run_filter.py` assumes the input
dataset is a *static* (phone-at-rest) session, so true position/velocity are zero and attitude is
constant throughout — meaning the filter's own estimate drifting away from that zero baseline *is*
the drift error, with no ground truth required. That's why real recordings for drift analysis
should be taken with the phone sitting still, and why `synth.static_session` (not a moving-track
generator) is the synthetic stand-in. `compare_attitude.py` + `convert_euroc.py` are a separate,
optional validation path: they check the attitude-drift code against the public EuRoC MAV benchmark,
which does have real ground truth, independent of the (ground-truth-free) phone pipeline.

**`tests/test_noise_model.py`** — simulate known noise, characterise it, compare (white noise per
axis, bias instability within 2×, yaml → simulator round trip incl. bias sign, free drift vs prediction).

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
