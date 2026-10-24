# imu-nav

Starter repository for the IMU indoor-positioning hardware round: MPU6050 + ESP32 (original),
multi-position calibration, 15-state error-state Kalman filter with stop, wheel and line aiding,
host tools, and a fixed-point pipeline for the Tier 3 bonus. The companion guide ("Strapdown
Field Guide") explains every equation; its Chapter 21 runbook says which file each step uses.

## What is verified and what is not

| Part | Status |
|---|---|
| Python package `nav/`, scripts | Runs end to end on synthetic data; 19 tests pass (`pytest -q tests`) |
| Calibration (`nav/calib.py`) | Recovers known parameters from synthetic sessions; held-out gravity-norm error 43 mg raw, 7 mg six-position, 0.3 mg multi-position (synthetic) |
| Embedded ESKF (`firmware/main/eskf.c`, `nav.c`) | Compiled on the host, matches the Python float64 reference to 0.1 mm over a full run |
| Fixed-point pipeline (`mahony_q.c`) | Matches its float64 mirror to centimetres; its *accuracy* is far below the ESKF (metres), see Chapter 16 |
| ESP32 firmware (`firmware/`) | Builds with ESP-IDF v5.3.1 with no warnings. **Not yet run on hardware** |
| Host tools (`tools/`) | Protocol codec unit-checked. Logger, viewer and HIL replay **not yet run against a board** |
| Accuracy numbers | Synthetic only. Real performance is unknown until gate M2 and the first track runs |

## Layout

```
nav/            Python reference: quat, io, synth, calib, allan, attitude, eskf, runner, evaluate, fixedpoint
scripts/        make_synth, run_calib, run_allan, run_filter (ablation), tune_detector
tests/          truth-known suite (test_truth.py), C-vs-Python checks (test_c_vs_python.py)
tools/          protocol, logger, viewer, hil_replay, compare_est, gen_calib_header
firmware/       ESP-IDF project (main/) and host builds of the same C files (host/)
```

## Quick start (Ubuntu)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q tests                                     # everything must pass before you touch hardware

# synthetic end-to-end rehearsal of the whole pipeline
for i in 1 2 3; do python scripts/make_synth.py calib data/syn_calib_s$i --seed $i; done
python scripts/make_synth.py static data/syn_static --minutes 30
python scripts/make_synth.py track  data/syn_track --seed 5 --stress
python scripts/run_calib.py data/syn_calib_s1 --holdout data/syn_calib_s2 data/syn_calib_s3 --lat 0 --out syn_calib.yaml --plots results/syn_calib
python scripts/run_allan.py data/syn_static --out syn_noise.yaml --plot results/syn_allan.png
python scripts/run_filter.py data/syn_track --noise syn_noise.yaml --ablation --plots results/syn_track
```

## Firmware

```bash
# ESP-IDF v5.3.x installed and exported (. $IDF_PATH/export.sh)
cd firmware
idf.py set-target esp32
idf.py build flash            # then close any serial monitor: UART0 carries binary frames
cd host && make               # host builds of the same filter for comparison tests
```

Wiring (GY-521): VCC 3V3, GND, SDA GPIO21, SCL GPIO22, INT GPIO4. Marker button: the board's BOOT button (GPIO0).

## Conventions (do not change one without the others)

* World frame: origin at the start marker, x along the track, z up. Body frame: x forward, y left, z up.
* Quaternions: Hamilton, scalar first, body to world. Attitude error: local (right) perturbation.
* Calibration: `a_O = Ta Ka (a_S + ba)`, `w_O = Tg Kg (w_S + bg)`; units m/s^2 and rad/s.
* `imu.csv` always stores raw sensor-frame values converted only by the datasheet sensitivity.
