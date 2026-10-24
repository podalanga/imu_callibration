# imu-nav

Extract raw IMU readings from an Android phone over a local WebSocket, calibrate out
systematic sensor error, and characterise how the phone's own IMU drifts: noise density and bias
instability (Allan deviation), attitude drift under free integration, and how much ZUPT/ZARU
(stationary aiding) bounds position/velocity drift while the phone sits still.

This started as a trolley/track indoor-positioning exercise built around an ESP32 + MPU6050
(see "History" below); it has been restructured to work directly off a phone's built-in IMU and
to focus on extraction + drift/noise analysis rather than embedded delivery.

## What is verified and what is not

| Part | Status |
|---|---|
| Python package `nav/`, scripts | Runs end to end on synthetic data; `pytest -q tests` passes |
| Calibration (`nav/calib.py`) | Recovers known parameters from synthetic sessions |
| Allan deviation (`nav/allan.py`) | Standard overlapping-ADEV estimator, unit-tested on synthetic noise |
| Drift pipeline (`nav/eskf.py`, `nav/runner.py`) | Free inertial integration vs. ZUPT/ZARU-bounded drift, matches theory on synthetic data |
| Phone acquisition (`phone_data/`) | Live-tested on a CPH2661 over `adb forward`: every event kept with its hardware timestamp, 395 Hz, ~6 µs timing std, 0 gaps in 59 s |
| Test console (`phone_data/app.py`) | Tested against the phone: demo, checklist, a 1 min recording → analysis → report, combined report |
| Disturbance detection | Offline and live detectors tested on injected taps / tilt changes (`tests/test_noise_model.py`) |
| Live dashboard (`phone_logger.py --dashboard`) | Tested live and in replay of a recorded run |
| Noise model (`nav/noise_model.py`) | Round-trip tested: simulate known noise → characterise → recovered (`tests/test_noise_model.py`) |
| Real phone drift numbers | Only a 1-minute test so far. Needs the ≥1 h static recording below |

## Layout

```
phone_data/     app.py + web/app.html (test console), analysis.py + report.py (automatic analysis, HTML report),
                phone_logger.py (CLI recorder + simple dashboard), stream, merge, live_stats;
                websocket_sensor.py (terminal view), record_phone.py (old CLI, same logger)
nav/            Python reference: quat, io, synth, calib, allan, noise_model, attitude, eskf, runner
scripts/        make_synth, run_calib, run_allan, characterise_noise, run_filter, check_acq, compare_attitude, convert_euroc
tests/          truth-known suite (test_truth.py)
```

## Test console (the easy way)

```bash
pip install -r requirements.txt
python phone_data/app.py                            # open http://localhost:8050
```

The console sets up `adb forward` itself and checks the phone over adb (connected, streaming rate,
plugged in, Stay awake, Sensor Server in the foreground, phone still and flat).

- **Live demo**: raw accelerometer / gyroscope / magnetometer, sample rate and jitter, a 3D phone driven
  by a Mahony filter, and a drift race (raw gyro vs bias-removed gyro vs Mahony). Nothing is saved.
- **Drift analysis**: setup checklist → choose Static drift / Turn-on repeatability / Quick check and a
  duration → recording with live data quality, a stillness verdict and disturbance log → automatic
  analysis (`phone_data/analysis.py`, separate process) → `runs/<name>/report.html`.
- **Sessions**: every run with its verdicts and report; pick a main session plus extra sessions for a
  combined report (e.g. the 2 h static run + three turn-on sessions → turn-on repeatability).

## Recordings in this repo

`runs/` holds the real phone recordings with their reports (`runs/<name>/report.html`), `data/` the
synthetic datasets. CSVs over 50 MB are committed gzipped (`imu.csv.gz`, `raw_gyro.csv.gz`, …) to stay
under GitHub's 100 MB file limit; every loader reads the `.gz` directly, so nothing needs unpacking.
Before pushing a new long recording, run `python scripts/pack_for_git.py`.

## Quick start (command line)

```bash
pip install -r requirements.txt
pytest -q tests                                     # everything must pass before trusting real data

# 1. Extraction: live view, or record to disk
adb forward tcp:8080 tcp:8080                        # USB, once per (re)connect to the phone's sensor server
python phone_data/websocket_sensor.py                # quick terminal view, nothing saved
python phone_data/phone_logger.py runs/phone_calib_s1                       # records every sample, Ctrl+C to stop
python phone_data/phone_logger.py runs/static_2h --minutes 120 --dashboard  # dashboard at http://localhost:8050
python scripts/check_acq.py runs/static_2h --rate 395                       # timing: rate, jitter, gaps

# 2. Calibration (hold the phone still in several orientations during the recording)
python scripts/run_calib.py runs/phone_calib_s1 --lat 10.76 --out calib.yaml     # 10.76 = NIT Trichy

# 3. Noise / drift characterisation (phone flat on a solid surface, untouched, >= 1 h; plus a few
#    5-minute sessions each after restarting the app, for turn-on repeatability)
python scripts/characterise_noise.py runs/static_2h --extra runs/static_s1 runs/static_s2 \
       --calib calib.yaml --out phone_noise.yaml --plots results/noise
python scripts/run_filter.py runs/static_2h --calib calib.yaml --noise phone_noise.yaml \
       --ablation --plots results/drift

# 4. Simulate this phone: synthetic sessions with the measured noise, rate and jitter
python scripts/make_synth.py static data/phone_sim --minutes 30 --noise-model phone_noise.yaml

# synthetic rehearsal of the same pipeline, no phone needed
python scripts/make_synth.py calib data/syn_calib_s1 --seed 1
python scripts/make_synth.py static data/syn_static --minutes 30
python scripts/run_calib.py data/syn_calib_s1 --lat 0 --out syn_calib.yaml
python scripts/run_allan.py data/syn_static --out syn_noise.yaml
python scripts/run_filter.py data/syn_static --noise syn_noise.yaml --ablation --plots results/syn_drift
```

`run_filter.py` assumes the phone was at rest for the whole run: true position/velocity are
zero and attitude is constant, so the filter's own estimate moving away from that *is* the
drift error — no ground truth needed. `compare_attitude.py` + `convert_euroc.py` are optional:
they validate the attitude-drift code against the public EuRoC benchmark, which does have real
ground truth, as a cross-check independent of the phone-specific pipeline.

## Conventions (do not change one without the others)

* Quaternions: Hamilton, scalar first, body to world. Attitude error: local (right) perturbation.
* Calibration: `a_O = Ta Ka (a_S + ba)`, `w_O = Tg Kg (w_S + bg)`; units m/s^2 and rad/s.
* `imu.csv` always stores raw sensor-frame values, straight from the phone's `*_uncalibrated`
  accelerometer/gyroscope streams (already SI units; no datasheet conversion needed).

## History

The original repo targeted an ESP32 + MPU6050 trolley-on-a-track indoor-positioning challenge
(firmware, UART protocol, hardware-in-the-loop replay, wheel/line-constraint aiding, a fixed-point
pipeline). That hardware-specific code has been removed; `Strapdown Field Guide.html` still
contains the full derivations and is kept as background theory, but its "Track and ground
truth", "Real-time embedded implementation", "Team plan" and "Demonstration and judge defence"
chapters describe the old hardware scope and no longer apply here.
