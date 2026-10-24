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
| Phone acquisition (`phone_data/`) | Live-tested against a real phone over `adb forward` / Wi-Fi |
| Real phone drift numbers | Not yet collected — run `scripts/run_allan.py` and `scripts/run_filter.py` on a real recording |

## Layout

```
phone_data/     websocket_sensor.py (quick terminal view), record_phone.py (logs imu.csv)
nav/            Python reference: quat, io, synth, calib, allan, attitude, eskf, runner
scripts/        make_synth, run_calib, run_allan, run_filter, check_acq, compare_attitude, convert_euroc
tests/          truth-known suite (test_truth.py)
```

## Quick start

```bash
pip install -r requirements.txt
pytest -q tests                                     # everything must pass before trusting real data

# 1. Extraction: live view, or record to disk
adb forward tcp:8080 tcp:8080                        # USB, once per (re)connect to the phone's sensor server
python phone_data/websocket_sensor.py                # quick terminal view, nothing saved
python phone_data/record_phone.py runs/phone_calib_s1          # records imu.csv
python phone_data/record_phone.py runs/phone_static_10min --minutes 10

# 2. Calibration (hold the phone still in several orientations during the recording)
python scripts/run_calib.py runs/phone_calib_s1 --lat <your latitude> --out calib.yaml

# 3. Noise / drift characterisation (record the phone sitting still for as long as practical)
python scripts/run_allan.py runs/phone_static_10min --calib calib.yaml --out noise.yaml --plot results/allan.png
python scripts/run_filter.py runs/phone_static_10min --calib calib.yaml --noise noise.yaml \
       --ablation --plots results/drift

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
