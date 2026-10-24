"""
IMU EKF Tuning Workbench with Magnetometer
------------------------------------------
Run with:
    pip install streamlit numpy pandas plotly
    streamlit run imu_ekf_tuning_workbench_v3_mag.py

CSV upload format (column names are case-insensitive):
    time, gx, gy, gz, ax, ay, az, mx, my, mz
Optional ground-truth columns:
    true_roll_deg, true_pitch_deg, true_yaw_deg

Gyroscope units: rad/s. Accelerometer units: m/s^2. Magnetometer units: any consistent field unit (commonly µT).
This is an offline research/visualization tool, not flight-certified software.
"""

import io
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

G = 9.81

st.set_page_config(page_title="IMU EKF Tuning Workbench", layout="wide")
st.title("IMU EKF Tuning Workbench")
st.caption("3-axis synthetic testing • CSV replay • manual tuning • parameter search • ground-truth comparison")

# ---------------------------- Quaternion utilities ----------------------------
def q_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])

def q_to_matrix(q):
    qw, qx, qy, qz = q_normalize(q)
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])

def q_to_euler_deg(q):
    qw, qx, qy, qz = q_normalize(q)
    roll = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    sinp = 2*(qw*qy - qz*qx)
    pitch = math.asin(float(np.clip(sinp, -1.0, 1.0)))
    yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    return np.degrees([roll, pitch, yaw])

def euler_to_q(roll, pitch, yaw):
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return q_normalize(np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ]))

def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def make_synthetic(duration, rate, amplitude, frequency, gyro_std, accel_std, seed, motion="Combined manoeuvres", roll_amp=25, pitch_amp=20, yaw_amp=60, bias_std=0.002, linear_accel=0.0, mag_noise=1.0, mag_bias_std=0.4, mag_strength=48.0, mag_dip_deg=60.0):
    """Create 3-axis truth and corresponding idealized IMU samples.

    Angles are generated in radians internally. Gyro channels are body angular
    rates (rad/s); accelerometer channels are specific-force-like gravity cues
    plus optional synthetic translational acceleration and noise.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / rate
    t = np.arange(0, duration, dt)
    w = 2*np.pi*frequency
    ar, ap, ay = np.radians([roll_amp, pitch_amp, yaw_amp])
    if motion == "Single-axis roll":
        roll, pitch, yaw = ar*np.sin(w*t), np.zeros_like(t), np.zeros_like(t)
    elif motion == "Single-axis pitch":
        roll, pitch, yaw = np.zeros_like(t), ap*np.sin(w*t), np.zeros_like(t)
    elif motion == "Single-axis yaw":
        roll, pitch, yaw = np.zeros_like(t), np.zeros_like(t), ay*np.sin(w*t)
    elif motion == "Smooth random manoeuvres":
        # Low-pass filtered random control points make a continuous, non-periodic path.
        knot_t = np.linspace(t[0], t[-1], max(5, int(duration*1.5)))
        def random_path(amp):
            return np.interp(t, knot_t, rng.uniform(-amp, amp, len(knot_t)))
        roll, pitch, yaw = random_path(ar), random_path(ap), random_path(ay)
    elif motion == "Step and hold manoeuvres":
        phase = (t / max(duration, 1e-9))
        roll = ar*np.select([phase < .25, phase < .5, phase < .75], [0, .7, -.6], default=.3)
        pitch = ap*np.select([phase < .2, phase < .45, phase < .7], [0, -.6, .8], default=-.2)
        yaw = ay*np.select([phase < .3, phase < .6], [0, .4], default=-.5)
        # Smooth transitions rather than physically impossible instantaneous jumps.
        roll = np.convolve(roll, np.ones(5)/5, mode='same')
        pitch = np.convolve(pitch, np.ones(5)/5, mode='same')
        yaw = np.convolve(yaw, np.ones(5)/5, mode='same')
    else:  # Combined, multi-frequency 3-axis manoeuvres
        roll = ar*(0.72*np.sin(w*t) + 0.28*np.sin(2.3*w*t + .4))
        pitch = ap*(0.75*np.sin(.73*w*t + .8) + 0.25*np.sin(1.8*w*t))
        yaw = ay*(0.7*np.sin(.43*w*t + .2) + 0.3*np.sin(1.4*w*t + 1.1))
    # Euler angle derivatives -> body-frame p,q,r (ZYX convention).
    rd = np.gradient(roll, dt); pitch_dot = np.gradient(pitch, dt); yd = np.gradient(yaw, dt)
    p_rate = rd - yd*np.sin(pitch)
    q_rate = pitch_dot*np.cos(roll) + yd*np.sin(roll)*np.cos(pitch)
    r_rate = -pitch_dot*np.sin(roll) + yd*np.cos(roll)*np.cos(pitch)
    gyro = np.column_stack([p_rate, q_rate, r_rate])
    gyro_bias = rng.normal(0, bias_std, 3)
    gyro += gyro_bias + rng.normal(0, gyro_std, gyro.shape)
    # Specific-force gravity direction in body coordinates for a stationary origin.
    accel = np.column_stack([-G*np.sin(pitch), G*np.sin(roll)*np.cos(pitch), G*np.cos(roll)*np.cos(pitch)])
    if linear_accel > 0:
        accel += rng.normal(0, linear_accel, accel.shape)
    accel += rng.normal(0, accel_std, accel.shape)
    # Earth magnetic field expressed in the navigation frame. The dip angle is
    # positive down from the horizontal; body readings are obtained via R.T.
    dip = np.radians(mag_dip_deg)
    earth_field = mag_strength * np.array([np.cos(dip), 0.0, np.sin(dip)])
    mag = np.empty((len(t), 3))
    for i, (rr, pp, yy) in enumerate(zip(roll, pitch, yaw)):
        mag[i] = q_to_matrix(euler_to_q(rr, pp, yy)).T @ earth_field
    mag += rng.normal(0, mag_bias_std, 3)[None, :]
    mag += rng.normal(0, mag_noise, mag.shape)
    return pd.DataFrame({"time": t, "gx": gyro[:,0], "gy": gyro[:,1], "gz": gyro[:,2],
        "ax": accel[:,0], "ay": accel[:,1], "az": accel[:,2],
        "mx": mag[:,0], "my": mag[:,1], "mz": mag[:,2],
        "true_roll_deg": np.degrees(roll), "true_pitch_deg": np.degrees(pitch), "true_yaw_deg": np.degrees(yaw)})

def prepare_data(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {
        "timestamp": "time", "t": "time", "time_s": "time",
        "gyro_x": "gx", "gyro_y": "gy", "gyro_z": "gz",
        "gyr_x": "gx", "gyr_y": "gy", "gyr_z": "gz",
        "accel_x": "ax", "accel_y": "ay", "accel_z": "az",
        "acc_x": "ax", "acc_y": "ay", "acc_z": "az",
        "mag_x": "mx", "mag_y": "my", "mag_z": "mz",
        "magnetometer_x": "mx", "magnetometer_y": "my", "magnetometer_z": "mz",
        "roll": "true_roll_deg", "pitch": "true_pitch_deg", "yaw": "true_yaw_deg",
    }
    df = df.rename(columns={k:v for k,v in aliases.items() if k in df.columns})
    required = ["time", "gx", "gy", "gz", "ax", "ay", "az"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    for c in required + [c for c in ["mx", "my", "mz"] if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    mag_cols = [c for c in ["mx", "my", "mz"] if c in df.columns]
    if mag_cols and len(mag_cols) != 3:
        raise ValueError("Magnetometer columns must be supplied together: mx, my, mz.")
    if mag_cols:
        df = df.dropna(subset=mag_cols)
    df = df.dropna(subset=required).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("Need at least 3 valid rows.")
    if np.any(np.diff(df["time"].to_numpy()) <= 0):
        raise ValueError("Time values must be strictly increasing.")
    return df

# ------------------------------- EKF core -------------------------------------
def run_ekf(df, q_scale, r_scale, p0, accel_gate, gyro_bias, initial_from_accel,
            mag_r_scale=0.03, mag_norm_tolerance=0.45, mag_dip_deg=60.0):
    """Quaternion EKF with sequential gravity and magnetic-field vector updates.

    Accelerometer measurements are in m/s^2. Magnetometer readings are normalized
    before update, so mag_r_scale is a dimensionless practical tuning parameter.
    """
    n = len(df)
    times = df["time"].to_numpy(float)
    gyro = df[["gx", "gy", "gz"]].to_numpy(float) - np.asarray(gyro_bias)[None, :]
    accel = df[["ax", "ay", "az"]].to_numpy(float)
    has_mag = all(c in df.columns for c in ["mx", "my", "mz"])
    mag = df[["mx", "my", "mz"]].to_numpy(float) if has_mag else None
    dip = np.radians(mag_dip_deg)
    earth_mag_dir = np.array([np.cos(dip), 0.0, np.sin(dip)])

    if initial_from_accel:
        a0 = accel[0]
        roll0 = math.atan2(a0[1], a0[2])
        pitch0 = math.atan2(-a0[0], max(1e-9, math.sqrt(a0[1]**2 + a0[2]**2)))
        yaw0 = 0.0
        if has_mag and np.linalg.norm(mag[0]) > 1e-9:
            # Tilt-compensate the first magnetic vector, then estimate heading.
            m_level = q_to_matrix(euler_to_q(roll0, pitch0, 0.0)) @ (mag[0] / np.linalg.norm(mag[0]))
            yaw0 = math.atan2(-m_level[1], m_level[0])
        q = euler_to_q(roll0, pitch0, yaw0)
    else:
        q = np.array([1.0, 0.0, 0.0, 0.0])

    P = np.eye(4) * p0
    est = np.zeros((n, 3))
    accel_innov = np.zeros(n)
    mag_innov = np.zeros(n)
    accel_norm = np.linalg.norm(accel, axis=1)
    mag_norm = np.linalg.norm(mag, axis=1) if has_mag else np.zeros(n)

    def accel_h(qv):
        return q_to_matrix(qv).T @ np.array([0.0, 0.0, G])

    def mag_h(qv):
        return q_to_matrix(qv).T @ earth_mag_dir

    def measurement_update(qv, Pv, z, hfun, R):
        pred = hfun(qv)
        residual = z - pred
        H = np.zeros((len(z), 4))
        eps = 1e-6
        for j in range(4):
            qp, qm = qv.copy(), qv.copy()
            qp[j] += eps
            qm[j] -= eps
            H[:, j] = (hfun(q_normalize(qp)) - hfun(q_normalize(qm))) / (2 * eps)
        S = H @ Pv @ H.T + R
        try:
            K = np.linalg.solve(S, H @ Pv).T
        except np.linalg.LinAlgError:
            K = Pv @ H.T @ np.linalg.pinv(S)
        qnew = q_normalize(qv + K @ residual)
        I = np.eye(4)
        Pnew = (I - K @ H) @ Pv @ (I - K @ H).T + K @ R @ K.T
        return qnew, 0.5 * (Pnew + Pnew.T), residual

    for k in range(n):
        dt = (times[k] - times[k-1]) if k else (times[1] - times[0])
        dt = float(np.clip(dt, 1e-5, 0.2))
        wx, wy, wz = gyro[k]
        Omega = np.array([[0, -wx, -wy, -wz], [wx, 0, wz, -wy],
                          [wy, -wz, 0, wx], [wz, wy, -wx, 0]], dtype=float)
        F = np.eye(4) + 0.5 * dt * Omega
        q = q_normalize(F @ q)
        Q = np.eye(4) * (q_scale * max(dt, 1e-5))
        P = F @ P @ F.T + Q

        # Use accelerometer only when its magnitude is reasonably gravity-like.
        if abs(accel_norm[k] - G) <= accel_gate:
            q, P, residual = measurement_update(q, P, accel[k], accel_h, np.eye(3) * r_scale)
            accel_innov[k] = np.linalg.norm(residual)

        # Magnetometer constrains heading. Reject near-zero or strongly disturbed
        # field magnitudes; direction-only update avoids dependence on field units.
        if has_mag and mag_norm[k] > 1e-9:
            reference_mag_norm = np.median(mag_norm[max(0, k-20):k+1])
            if reference_mag_norm > 1e-9 and abs(mag_norm[k] - reference_mag_norm) / reference_mag_norm <= mag_norm_tolerance:
                zmag = mag[k] / mag_norm[k]
                q, P, residual = measurement_update(q, P, zmag, mag_h, np.eye(3) * mag_r_scale)
                mag_innov[k] = np.linalg.norm(residual)
        est[k] = q_to_euler_deg(q)

    return est, accel_innov, accel_norm, mag_innov, mag_norm

def calculate_metrics(df, est):
    truth_cols = ["true_roll_deg", "true_pitch_deg", "true_yaw_deg"]
    if not all(c in df.columns for c in truth_cols):
        return None
    truth = df[truth_cols].to_numpy(float)
    err = np.column_stack([wrap_deg(est[:,i] - truth[:,i]) for i in range(3)])
    return {
        "truth": truth, "error": err,
        "rmse": np.sqrt(np.mean(err**2, axis=0)),
        "mae": np.mean(np.abs(err), axis=0),
        "max": np.max(np.abs(err), axis=0),
        "overall_rmse": float(np.sqrt(np.mean(err**2)))
    }

# -------------------------------- Sidebar -------------------------------------
with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose input", ["Synthetic test signal", "Upload recorded CSV"])
    uploaded = None
    if source == "Upload recorded CSV":
        uploaded = st.file_uploader("Upload IMU CSV", type=["csv"])
    st.divider()
    st.header("Synthetic signal")
    duration = st.slider("Duration (s)", 2.0, 30.0, 5.0, 1.0)
    rate = st.select_slider("Sample rate (Hz)", options=[25, 50, 100, 200, 400], value=100)
    motion = st.selectbox("Motion profile", ["Combined manoeuvres", "Single-axis roll", "Single-axis pitch", "Single-axis yaw", "Smooth random manoeuvres", "Step and hold manoeuvres"])
    roll_amp = st.slider("Roll excursion (deg)", 0, 80, 25)
    pitch_amp = st.slider("Pitch excursion (deg)", 0, 80, 20)
    yaw_amp = st.slider("Yaw excursion (deg)", 0, 180, 60)
    frequency = st.slider("Base motion frequency (Hz)", 0.1, 3.0, 0.8, 0.1)
    gyro_std = st.slider("Simulated gyro white noise (rad/s)", 0.0, 0.15, 0.02, 0.005)
    accel_std = st.slider("Simulated accelerometer noise (m/s²)", 0.0, 1.5, 0.2, 0.05)
    mag_noise = st.slider("Simulated magnetometer noise (µT)", 0.0, 8.0, 1.0, 0.25)
    mag_bias_std = st.slider("Synthetic magnetometer bias std (µT per axis)", 0.0, 5.0, 0.4, 0.1)
    mag_strength = st.slider("Synthetic Earth-field magnitude (µT)", 20.0, 70.0, 48.0, 1.0)
    mag_dip_deg = st.slider("Synthetic magnetic dip angle (deg)", -80.0, 80.0, 60.0, 1.0)
    bias_std = st.slider("Synthetic gyro bias std (rad/s per axis)", 0.0, 0.02, 0.002, 0.0005)
    linear_accel = st.slider("Synthetic translational acceleration disturbance (m/s²)", 0.0, 4.0, 0.0, 0.1)
    seed = st.number_input("Random seed", 0, 100000, 7, 1)
    st.divider()
    st.header("EKF tuning")
    q_scale = st.select_slider("Process noise Q scale", options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], value=1e-3, format_func=lambda x: f"{x:g}")
    r_scale = st.select_slider("Measurement noise R scale", options=[1e-3, 1e-2, 5e-2, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0], value=0.2, format_func=lambda x: f"{x:g}")
    p0 = st.select_slider("Initial covariance P₀", options=[1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0], value=0.1, format_func=lambda x: f"{x:g}")
    accel_gate = st.slider("Accel norm gate (m/s² from 9.81)", 0.1, 5.0, 2.0, 0.1)
    mag_r_scale = st.select_slider("Magnetometer measurement noise R", options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.5], value=0.03, format_func=lambda x: f"{x:g}")
    mag_norm_tolerance = st.slider("Magnetic-field magnitude disturbance gate (fraction)", 0.1, 1.0, 0.45, 0.05)
    init_from_accel = st.checkbox("Initialize roll/pitch from first accelerometer sample", value=True)
    st.caption("Optional gyro bias correction (rad/s)")
    bx = st.number_input("Gyro bias X", value=0.0, format="%.5f")
    by = st.number_input("Gyro bias Y", value=0.0, format="%.5f")
    bz = st.number_input("Gyro bias Z", value=0.0, format="%.5f")

# -------------------------------- Load data -----------------------------------
try:
    if source == "Upload recorded CSV":
        if uploaded is None:
            st.info("Upload a CSV to start. Required columns: time, gx, gy, gz, ax, ay, az. Magnetometer columns mx, my, mz enable heading correction (recommended). Optional reference columns: true_roll_deg, true_pitch_deg, true_yaw_deg.")
            st.stop()
        df = prepare_data(pd.read_csv(uploaded))
    else:
        df = make_synthetic(duration, rate, 0.0, frequency, gyro_std, accel_std, int(seed), motion, roll_amp, pitch_amp, yaw_amp, bias_std, linear_accel, mag_noise, mag_bias_std, mag_strength, mag_dip_deg)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

est, innovation, accel_norm, mag_innovation, mag_norm = run_ekf(
    df, q_scale, r_scale, p0, accel_gate, (bx, by, bz), init_from_accel,
    mag_r_scale, mag_norm_tolerance, mag_dip_deg
)
metrics = calculate_metrics(df, est)
truth_available = metrics is not None

# -------------------------------- Summary -------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", f"{len(df):,}")
c2.metric("Duration", f"{df['time'].iloc[-1] - df['time'].iloc[0]:.2f} s")
c3.metric("Median sample rate", f"{1/np.median(np.diff(df['time'])):.1f} Hz")
if truth_available:
    c4.metric("Overall attitude RMSE", f"{metrics['overall_rmse']:.3f}°")
else:
    c4.metric("Ground truth", "Not provided")

tab1, tab2, tab3, tab4 = st.tabs(["Attitude", "Error & metrics", "Sensor diagnostics", "Auto-tune"])

with tab1:
    fig = go.Figure()
    labels = ["Roll", "Pitch", "Yaw"]
    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter(x=df["time"], y=est[:,i], mode="lines", name=f"EKF {label}"))
        if truth_available:
            fig.add_trace(go.Scatter(x=df["time"], y=metrics["truth"][:,i], mode="lines",
                                     name=f"True {label}", line=dict(dash="dash")))
    fig.update_layout(title="Attitude estimate vs reference", xaxis_title="Time (s)",
                      yaxis_title="Angle (degrees)", hovermode="x unified", height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("When mx, my, mz are present, the magnetometer provides a magnetic-heading reference that helps constrain yaw drift. Hard/soft-iron effects, nearby current-carrying wires, steel, and incorrect magnetic-dip assumptions can still bias heading.")

with tab2:
    if truth_available:
        err = metrics["error"]
        fig = go.Figure()
        for i, label in enumerate(["Roll", "Pitch", "Yaw"]):
            fig.add_trace(go.Scatter(x=df["time"], y=err[:,i], mode="lines", name=f"{label} error"))
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(title="Estimation error (EKF − ground truth)", xaxis_title="Time (s)",
                          yaxis_title="Error (degrees)", hovermode="x unified", height=420)
        st.plotly_chart(fig, use_container_width=True)
        rows = []
        for i, label in enumerate(["Roll", "Pitch", "Yaw"]):
            rows.append({"Axis": label, "RMSE (°)": metrics["rmse"][i],
                         "MAE (°)": metrics["mae"][i], "Max |error| (°)": metrics["max"][i]})
        st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)
    else:
        st.warning("No ground-truth orientation columns found. Error metrics cannot be computed from IMU data alone. Upload reference angles or use the synthetic dataset.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=innovation, name="Accelerometer innovation norm"))
    if all(c in df.columns for c in ["mx", "my", "mz"]):
        fig.add_trace(go.Scatter(x=df["time"], y=mag_innovation, name="Magnetometer innovation norm"))
    fig.update_layout(title="Sensor innovation norms", xaxis_title="Time (s)", yaxis_title="Residual norm", height=320)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = go.Figure()
    for col in ["gx", "gy", "gz"]:
        fig.add_trace(go.Scatter(x=df["time"], y=df[col], mode="lines", name=col))
    fig.update_layout(title="Gyroscope measurements", xaxis_title="Time (s)", yaxis_title="Angular rate (rad/s)", height=350)
    st.plotly_chart(fig, use_container_width=True)
    fig = go.Figure()
    for col in ["ax", "ay", "az"]:
        fig.add_trace(go.Scatter(x=df["time"], y=df[col], mode="lines", name=col))
    fig.add_trace(go.Scatter(x=df["time"], y=accel_norm, mode="lines", name="|accel|", line=dict(dash="dash")))
    fig.add_hline(y=G, line_dash="dot", annotation_text="9.81 m/s²")
    fig.update_layout(title="Accelerometer measurements and magnitude", xaxis_title="Time (s)", yaxis_title="Acceleration (m/s²)", height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("During translation or strong manoeuvres, accelerometer magnitude/direction may not represent gravity alone. The norm gate is a simple heuristic, not a complete dynamic-acceleration detector.")
    if all(c in df.columns for c in ["mx", "my", "mz"]):
        fig = go.Figure()
        for col in ["mx", "my", "mz"]:
            fig.add_trace(go.Scatter(x=df["time"], y=df[col], mode="lines", name=col))
        fig.add_trace(go.Scatter(x=df["time"], y=mag_norm, mode="lines", name="|mag|", line=dict(dash="dash")))
        fig.update_layout(title="Magnetometer measurements and magnitude", xaxis_title="Time (s)", yaxis_title="Magnetic field (input units)", height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Magnetometer calibration is important: correct hard-iron offsets and soft-iron scaling, and keep the sensor away from motors, power wiring, steel, and magnets.")
    else:
        st.warning("No mx, my, mz columns found. This run is using gyro + accelerometer only, so yaw will not have an absolute magnetic reference.")

with tab4:
    st.write("Search a coarse grid of Q and R scales on this dataset. Auto-tuning is meaningful only when reference attitude is available.")
    if not truth_available:
        st.warning("Upload ground-truth attitude columns or switch to the synthetic signal to enable supervised auto-tuning.")
    else:
        q_candidates = st.multiselect("Q candidates", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                                      default=[1e-5, 1e-4, 1e-3, 1e-2])
        r_candidates = st.multiselect("R candidates", [1e-3, 1e-2, 5e-2, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
                                      default=[0.05, 0.1, 0.2, 0.5, 1.0])
        if st.button("Run Q/R grid search", type="primary"):
            if not q_candidates or not r_candidates:
                st.error("Select at least one Q candidate and one R candidate.")
            else:
                results = []
                progress = st.progress(0)
                total = len(q_candidates) * len(r_candidates)
                count = 0
                for qv in q_candidates:
                    for rv in r_candidates:
                        candidate_est, _, _, _, _ = run_ekf(
                            df, qv, rv, p0, accel_gate, (bx, by, bz), init_from_accel,
                            mag_r_scale, mag_norm_tolerance, mag_dip_deg
                        )
                        m = calculate_metrics(df, candidate_est)
                        results.append({"Q": qv, "R": rv, "Overall RMSE (°)": m["overall_rmse"],
                                        "Roll RMSE (°)": m["rmse"][0],
                                        "Pitch RMSE (°)": m["rmse"][1],
                                        "Yaw RMSE (°)": m["rmse"][2]})
                        count += 1
                        progress.progress(count / total)
                result_df = pd.DataFrame(results).sort_values("Overall RMSE (°)").reset_index(drop=True)
                st.success(f"Evaluated {len(result_df)} parameter combinations.")
                st.dataframe(result_df.round(5), use_container_width=True, hide_index=True)
                best = result_df.iloc[0]
                st.write(f"Best tested combination: **Q = {best['Q']:g}, R = {best['R']:g}**, overall RMSE = **{best['Overall RMSE (°)']:.4f}°**.")
                st.caption("This is the best combination within the selected grid for this dataset, not a guarantee of best real-flight performance. Validate on separate motion sequences before choosing it.")

# -------------------------------- Export ---------------------------------------
out = df.copy()
out[["ekf_roll_deg", "ekf_pitch_deg", "ekf_yaw_deg"]] = est
out["accel_innovation_norm"] = innovation
out["accel_magnitude"] = accel_norm
out["mag_innovation_norm"] = mag_innovation
if all(c in df.columns for c in ["mx", "my", "mz"]):
    out["mag_magnitude"] = mag_norm
if truth_available:
    out[["roll_error_deg", "pitch_error_deg", "yaw_error_deg"]] = metrics["error"]
st.download_button(
    "Download replay results as CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="imu_ekf_replay_results.csv",
    mime="text/csv"
)

with st.expander("Important interpretation notes"):
    st.markdown("""
- The current filter estimates **attitude**, not drone position. Position estimation requires a motion/velocity model and additional observations; double-integrating accelerometer data drifts rapidly.
- Accelerometer + gyroscope can constrain roll and pitch when acceleration is gravity-dominated, but cannot provide an absolute yaw reference. Magnetometer vectors mx, my, mz help constrain heading when calibrated and undisturbed.
- For recorded data, ground-truth columns are needed to compute true RMSE/MAE. Without them, the app can show residuals and sensor diagnostics but cannot know the true attitude error.
- Q and R values depend on the state/error representation and units. This app exposes practical tuning scales; before flight use, derive covariance matrices from sensor noise specifications/logs and validate the filter equations.
- This is an offline experimental workbench, not flight-ready code. Test on a bench, validate on held-out logs, and retain the existing flight-controller estimator as a fallback.
""")
