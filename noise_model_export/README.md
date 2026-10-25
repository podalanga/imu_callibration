# Phone IMU noise model (CPH2661, Bosch BMI26x)

This folder describes how this phone's IMU errors behave, measured from a 1-hour static recording. Use it to
make synthetic IMU readings that are as noisy as the real phone, for example to test an EKF.

| File | What it is |
|---|---|
| `phone_imu_noise_model.yaml` | The numbers (per axis, SI units) |
| `synth_imu_noise.py` | `add_noise(t, gyro_true, acc_true, model)`: ideal signals in, phone-like readings out. Needs only numpy + pyyaml |
| `measured_allan.csv` | The real phone's Allan deviation curves (tau, gx..az) |
| `allan_validation.png` | Real vs synthetic Allan curves, 1 h each |

## The model

Every axis of both sensors gets the same four error terms, plus rounding:

```
reading = truth + b_on + b_gm(t) + b_rw(t) + w(t)          then rounded to 1 LSB
```

| Term | What it means | How to generate (dt = 1/fs) |
|---|---|---|
| `w` white noise | sample-to-sample jitter | `normal(0, N*sqrt(fs))` every sample |
| `b_gm` bias instability | bias that wanders and comes back, over tens to hundreds of seconds | `b[k+1] = exp(-dt/tau_gm)*b[k] + normal(0, sigma_gm*sqrt(1-exp(-2dt/tau_gm)))` |
| `b_rw` bias random walk | bias that wanders and doesn't come back | `b[k+1] = b[k] + normal(0, K*sqrt(dt))` |
| `b_on` turn-on bias | a constant, different at each power-up | constant |

### Values (fs = 395.1 Hz)

| | x | y | z | units |
|---|---|---|---|---|
| **Gyro N** | 9.29e-5 | 9.14e-5 | 8.39e-5 | rad/s/√Hz |
| Gyro sigma_gm | 2.48e-4 | 1.17e-4 | 2.28e-4 | rad/s |
| Gyro tau_gm | 52 | 67 | 236 | s |
| Gyro K | 1.58e-5 | 1.00e-5 | 1.67e-5 | rad/s²/√Hz |
| Gyro turn-on bias | 3.64e-3 | −5.97e-3 | 1.94e-3 | rad/s |
| **Accel N** | 1.10e-3 | 1.17e-3 | 1.18e-3 | m/s²/√Hz |
| Accel sigma_gm | 2.83e-3 | 8.0e-4 | 1.65e-3 | m/s² |
| Accel tau_gm | 236 | 10.7 | 6.8 | s |
| Accel K | 0 | 9.9e-5 | 2.26e-4 | m/s³/√Hz |

- Gyro LSB = 1.065e-3 rad/s (2000 °/s range); accel LSB = 4.785e-3 m/s² (16 g range).
- Timestamps: 395.1 Hz, 3 µs jitter, no dropped samples.
- Per-sample std at 395 Hz: gyro about 1.7e-3 rad/s (0.1 °/s), accel about 0.023 m/s².
- In datasheet units: gyro ARW about 0.32 °/√h and bias instability about 0.01 °/s; accel about 115 µg/√Hz.

### Using it

```python
import numpy as np
from synth_imu_noise import load_model, add_noise

m = load_model()                                   # phone_imu_noise_model.yaml
t = np.arange(0, 600, 1 / m["fs_hz"])              # 10 min at the phone's rate
gyro_true = np.zeros((len(t), 3))                  # your trajectory's ideal body rates, rad/s
acc_true = np.tile([0, 0, 9.78], (len(t), 1))      # your ideal specific force, m/s^2
gyro, acc, t_meas = add_noise(t, gyro_true, acc_true, m, seed=1,
                              gyro_turn_on=0.005,   # draw a new turn-on bias (std 0.005 rad/s), or "measured"/"zero"
                              acc_turn_on=0.05)     # see "accel turn-on bias" below
```

The truth signals are in the Android device frame: x to the right, y toward the top edge, z out of the screen.
A phone lying flat and still reads about +9.8 m/s² on z.

To check the generator, run `python synth_imu_noise.py --validate`. It simulates 1 h static and redraws
`allan_validation.png`. Out to τ = 600 s the synthetic curves match the real ones within a median of
2–4 % (gyro) and 5–9 % (accel).

## For the EKF (continuous-time PSDs → discrete Q)

With IMU-driven propagation, the noise terms go into Q like this:

| EKF state / input | Continuous PSD | Discrete variance per step dt |
|---|---|---|
| gyro white noise → attitude | N_g² | N_g² · dt |
| accel white noise → velocity | N_a² | N_a² · dt |
| gyro bias as GM state | 2·sigma_gm²/tau_gm | sigma_gm² · (1 − e^(−2dt/tau_gm)) |
| gyro bias as random walk only | K_g² | K_g² · dt |
| accel bias: same, with the accel values | | |

The usual simple EKF models the bias as a random walk only. Then a good bias PSD is the larger of K² and
2·sigma_gm²/tau_gm. Otherwise the filter is too confident about the bias.

## Caveats (please read)

1. **Turn-on bias came from one power-up.** The gyro values are this session's average reading. We have not
   yet measured how much they change between power-ups. For Monte-Carlo runs, draw a random one. About
   0.005 rad/s (0.3 °/s) is the size we saw.
2. **Accel turn-on bias is unknown.** Flat on a table, accel bias can't be told apart from tilt and scale
   error. The measured |a| was 9.839 m/s², while local g is about 9.78 m/s², so the combined error is about
   0.06 m/s². Draw the accel turn-on bias with std ≈ 0.05 m/s² to be safe.
3. **Temperature.** The phone warmed from 33.1 to 35.6 °C during the hour, mostly after minute 25. The
   biases follow the temperature: correlation 0.6–0.85, slopes in the yaml as `temp_slope`. Pass
   `dtemp=` to add this effect. Part of the slow wander seen beyond τ ≈ 600 s, especially accel z, comes
   from this warming. The Gauss-Markov / random-walk terms do not reproduce it.
4. **Beyond 10 minutes the data are thin.** Values at τ > 600 s are averaged over only a few clusters.
   Trust the model up to about 10 minutes of free integration.
5. **Very short timescales.** Below about 10 ms the real gyro is slightly smoother than white noise. This
   is the chip's internal low-pass filter. Real gyro readings are also mostly whole numbers of LSB
   (per-sample std ≈ 1.6 LSB). The generator rounds to the LSB, so both effects are close enough for an
   EKF.
6. **Scale factor and misalignment are not included.** A static test can't measure them. Leave them out
   (identity), or add about 0.5 % scale and 0.5° misalignment for a stress test.

## How the numbers were obtained

- Data: `runs/One_hr_drift_test/imu.csv`. Uncalibrated accel + gyro, phone flat on a table and plugged in,
  2026-10-24 21:01–22:01, 1,422,283 samples per sensor.
- 465 of the 3600 s showed bumps where some axis's 1 s std went above 1.25× its median, mostly because
  people were near the table. Each of those seconds, ±3 s, was replaced by resampled clean noise from the
  20 s on either side, keeping the local mean. After that, kurtosis was about 0 on every axis.
- The overlapping Allan deviation was computed per axis. N was read off the −½ slope at τ = 0.05–0.5 s.
  sigma_gm, tau_gm and K were then fitted to the exact Allan variance of white + Gauss-Markov +
  random-walk, using τ = 0.05–600 s (fit rms 3–8 %).
