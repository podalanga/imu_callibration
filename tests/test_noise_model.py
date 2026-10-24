"""Noise characterisation round trip: simulate a sensor with known noise, characterise it, compare.

Run:  pytest -q tests/test_noise_model.py
"""
import os, sys
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import synth, noise_model as nm, io

FS = 100.0
TRUE = dict(Na=np.array([2e-3, 3e-3, 4e-3]), Ng=np.radians([0.004, 0.006, 0.008]),
            B_gyro=np.radians([0.002, 0.002, 0.002]), tau_gyro=np.array([30.0, 30.0, 30.0]))


def _static(minutes=20, seed=1, **kw):
    err = synth.ErrorModel(lsb_acc=0.0, lsb_gyro=0.0, seed=seed, **kw)
    return synth.make_static_dataset(err, duration_s=minutes * 60, fs=FS).imu


def test_white_noise_recovered_per_axis():
    imu = _static(Na=TRUE["Na"], Ng=TRUE["Ng"])
    res, _ = nm.characterise([imu])
    assert np.allclose(nm.axis_array(res, "acc", "N"), TRUE["Na"], rtol=0.15)
    assert np.allclose(nm.axis_array(res, "gyro", "N"), TRUE["Ng"], rtol=0.15)
    assert abs(res["timing"]["fs"] - FS) < 0.01


def test_bias_instability_order_of_magnitude():
    imu = _static(minutes=40, Ng=TRUE["Ng"], B_gyro=TRUE["B_gyro"], tau_gyro=TRUE["tau_gyro"])
    res, _ = nm.characterise([imu])
    em = nm.error_model_params(res)
    ratio = np.array(em["B_gyro"]) / TRUE["B_gyro"]
    assert np.all((ratio > 0.5) & (ratio < 2.0)), ratio


def test_noise_yaml_feeds_simulator(tmp_path):
    imu = _static(Na=TRUE["Na"], Ng=TRUE["Ng"], bg=np.radians([0.2, -0.3, 0.1]))
    res, _ = nm.characterise([imu])
    reading = nm.axis_array(res, "gyro", "mean")
    p = tmp_path / "noise.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(io._plain(dict(nm.filter_params(res), error_model=nm.error_model_params(res))), f)
    err = synth.ErrorModel.from_noise_yaml(str(p), seed=3)
    sim = synth.make_static_dataset(err, duration_s=600, fs=FS).imu
    res2, _ = nm.characterise([sim])
    assert np.allclose(nm.axis_array(res2, "gyro", "N"), TRUE["Ng"], rtol=0.25)
    assert np.allclose(nm.axis_array(res2, "gyro", "mean"), reading, atol=np.radians(0.005))   # same bias sign


def test_disturbance_detector():
    imu = _static(minutes=5, Na=TRUE["Na"], Ng=TRUE["Ng"])
    assert nm.disturbances(imu)["validity"] == "valid"
    i0, i1 = np.searchsorted(imu.t, [100.0, 103.0])
    imu.gyro[i0:i1] += np.radians(5.0) * np.sin(np.linspace(0, 30, i1 - i0))[:, None]   # a tap: vibration
    j0 = np.searchsorted(imu.t, 200.0)
    imu.acc[j0:] = imu.acc[j0:] @ np.array([[1, 0, 0], [0, np.cos(0.02), -np.sin(0.02)], [0, np.sin(0.02), np.cos(0.02)]]).T
    d = nm.disturbances(imu)                       # ...and from 200 s the phone rests 1.1 deg tilted: moved
    starts = [e[0] for e in d["events"]]
    assert any(99 <= s <= 101 for s in starts) and any(199 <= s <= 201 for s in starts)
    assert d["validity"] == "invalid"
    i0, i1 = nm.longest_clean(imu.t, d["mask"])
    assert imu.t[i1 - 1] - imu.t[i0] > 90


def test_live_detector_agrees_with_offline():
    from phone_data.live_stats import LiveDisturbance
    imu = _static(minutes=4, Na=TRUE["Na"], Ng=TRUE["Ng"])
    i0, i1 = np.searchsorted(imu.t, [100.0, 103.0])
    imu.gyro[i0:i1] += np.radians(5.0) * np.sin(np.linspace(0, 30, i1 - i0))[:, None]
    live = LiveDisturbance()
    for t, a, g in zip(imu.t, imu.acc, imu.gyro):
        live.update_acc(a)
        live.update_gyro(t, g)
    s = live.snapshot()
    assert s["n_events"] == 1 and 99 <= s["events"][0]["start_s"] <= 101
    assert s["validity"] == nm.disturbances(imu)["validity"] == "partly valid"


def test_temperature_slope_sign_and_compensation():
    slope = np.radians([0.01, -0.005, 0.0])                   # rad/s per degC in the simulator's bias
    imu = _static(minutes=20, Ng=TRUE["Ng"], temp_slope_bg=slope, temp_rate=0.005)   # warms 6 degC
    res, _ = nm.characterise([imu])
    measured = nm.axis_array(res, "gyro", "temp_slope")
    assert np.allclose(measured, -slope, atol=np.radians(0.001))    # reading = truth - bias
    assert np.allclose(nm.error_model_params(res)["temp_slope_bg"], slope, atol=np.radians(0.001))
    before = np.ptp(nm.window_means(imu.t, imu.gyro, 60.0)[1][:, 0])
    assert nm.temp_compensate(imu, res) is not None
    after = np.ptp(nm.window_means(imu.t, imu.gyro, 60.0)[1][:, 0])
    assert after < 0.2 * before


def test_short_run_reports_bias_instability_as_limit():
    imu = _static(minutes=1, Ng=TRUE["Ng"], B_gyro=TRUE["B_gyro"], tau_gyro=np.full(3, 600.0))
    res, _ = nm.characterise([imu])
    for a in "xyz":
        r = res["gyro"][a]
        assert not r["B_resolved"] and r["B_limit"] > 0
    assert "Kbg" not in nm.filter_params(res)                 # an unseen K must not reach the filter


def test_gzipped_dataset_loads(tmp_path):
    import gzip, shutil
    ds = synth.make_static_dataset(synth.ErrorModel(Ng=TRUE["Ng"], seed=2), duration_s=5, fs=FS)
    io.save(str(tmp_path), ds)
    p = tmp_path / "imu.csv"
    with open(p, "rb") as s, gzip.open(str(p) + ".gz", "wb") as d:
        shutil.copyfileobj(s, d)
    p.unlink()
    back = io.load(str(tmp_path)).imu
    assert np.array_equal(back.t_us, ds.imu.t_us) and np.allclose(back.gyro, ds.imu.gyro, rtol=1e-8)


def test_free_drift_matches_prediction_on_white_noise():
    imu = _static(minutes=30, Na=TRUE["Na"], Ng=TRUE["Ng"])
    tt, head, pos = nm.drift_segments(imu, segment_s=60.0)
    i = int(np.searchsorted(tt, 30.0))
    measured = np.sqrt(np.mean(head[:, i] ** 2))
    predicted = nm.predicted_drift(tt[i], TRUE["Ng"][2], 0.0, 0.0, 0.0, t_align=10.0)["heading"]
    assert 0.6 < measured / predicted < 1.6


def test_gz_noise_params_reproduce_measured_noise():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import noise_to_gz
    imu = _static(minutes=40, Na=TRUE["Na"], Ng=TRUE["Ng"], B_gyro=TRUE["B_gyro"], tau_gyro=TRUE["tau_gyro"])
    res, _ = nm.characterise([imu])
    noise = dict(error_model=nm.error_model_params(res))
    p = noise_to_gz.gz_params(noise, FS)
    rng = np.random.default_rng(5)
    sim = _static(minutes=40)                                  # same timestamps, signal replaced by gz noise
    sim.gyro = np.stack([noise_to_gz.gz_sample(p["gyro"], len(sim.t), FS, rng) for _ in range(3)], axis=1)
    sim.acc = np.stack([noise_to_gz.gz_sample(p["accel"], len(sim.t), FS, rng) for _ in range(3)], axis=1) + [0, 0, 9.80665]
    res2, _ = nm.characterise([sim])
    worst = lambda r, s, k: np.max(nm.axis_array(r, s, k))
    assert abs(worst(res2, "gyro", "N") / worst(res, "gyro", "N") - 1) < 0.15
    assert abs(worst(res2, "acc", "N") / worst(res, "acc", "N") - 1) < 0.15
    ratio = worst(res2, "gyro", "B") / worst(res, "gyro", "B")
    assert 0.5 < ratio < 2.0, ratio
