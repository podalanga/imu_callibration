"""IMU + camera (AprilTag) fusion on a recorded walk, with a 3D trajectory view for the demo.

  python scripts/fuse_camera_imu.py runs/walk_01 --lr-half-gap 0.0475

The filter is the 15-error-state ESKF of 7_4_ESKF_EKF_Comparision_PhoneIMUNoise.py (nominal p, v, q, bg, ba; error
[dp, dv, dtheta, dbg, dba]; right/body attitude error; gravity-direction update that fades out while accelerating),
with the UWB range update replaced by a camera pose update: whenever the camera sees tags, the tag-based camera
pose gives the phone's position and attitude, and the IMU carries the solution between tag sightings.

Steps
  1. camera pose per video frame from the floor tags (scripts/camera_track.py)
  2. camera <-> IMU: the video's first frame on the phone's sensor clock comes from session.yaml; the exact offset
     and the camera's mounting rotation in the phone come from matching the camera's rotation (frame to frame)
     against the gyro
  3. runs: camera only, IMU only (dead reckoning from the same start), IMU + camera, and IMU + camera with the camera
     blanked out part of the time (the blanked frames are the reference: this measures how well the IMU bridges)
  4. outputs in <run>/fusion/: trajectory_3d.html (interactive), trajectory_3d.png and trajectory_panels.png (slides), fused.csv, summary.yaml

Tag layout: tag_layout.csv (checkpoints 1 m apart along x); --lr-half-gap sets the left/right tag centres to
y = +-value (0.20 m in the printed plan; 0.0475 m when the cut pieces are laid edge to edge).
"""
import argparse, math, os, sys
import numpy as np
import yaml
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from nav import io
from camera_track import LAYOUT, TagPose, intrinsics, tag_corners

G = 9.80665


# ---------------------------------------------------------------- rotation helpers (Hamilton, w first)
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_so3(phi):
    return cv2.Rodrigues(np.asarray(phi, float).reshape(3, 1))[0]


def log_so3(R):
    return cv2.Rodrigues(R)[0].ravel()


# ---------------------------------------------------------------- 1. camera poses
def layout_rows(half_gap):
    import csv
    with open(LAYOUT) as f:
        rows = list(csv.DictReader(f))
    if half_gap is not None:
        for r in rows:
            r["y_m"] = half_gap if float(r["y_m"]) > 0 else -half_gap
    return rows


def camera_poses(video, world, cache, max_err_mm=2.0):
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    cap = cv2.VideoCapture(video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    K, dist = intrinsics(W, H)
    tp = TagPose(world, K, dist)
    t, p, R, err, ntag = [], [], [], [], []
    k = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        d = tp.detect(img)
        r = tp.pose(d)
        good = r is not None and r[3] <= max_err_mm
        t.append(k / fps)
        p.append(r[0] if good else np.full(3, np.nan))
        R.append(r[1] if good else np.full((3, 3), np.nan))
        err.append(r[3] if good else np.nan)
        ntag.append(len(d))
        k += 1
    out = dict(t=np.array(t), p=np.array(p), R=np.array(R), err_mm=np.array(err), ntag=np.array(ntag), fps=fps)
    np.savez(cache, **out)
    return out


# ---------------------------------------------------------------- 2. camera <-> IMU sync and mounting
def sync(cam, imu_t, gyro, t0_guess, search=0.4, window_s=0.3):
    """Offset so that imu time = video time + offset, and R_bc (camera frame -> phone body frame).

    Over windows of ~window_s the camera's own rotation (from two tag poses) is compared with the gyro integrated
    over the same window: the rotation angles must match in time (offset) and the rotation axes differ by the
    camera's mounting rotation (R_bc, by Kabsch). Frame-to-frame rotations are too noisy for this at 30 fps."""
    ok = np.isfinite(cam["p"][:, 0]) & (cam["ntag"] >= 2)
    m = max(1, int(round(window_s * cam["fps"])))
    pairs = [(k, k + m) for k in range(len(ok) - m) if ok[k] and ok[k + m]]
    a = np.array([log_so3(cam["R"][i].T @ cam["R"][j]) for i, j in pairs])
    ta = np.array([(cam["t"][i], cam["t"][j]) for i, j in pairs])
    n_still = int(1.0 / np.median(np.diff(imu_t)))
    w = gyro - gyro[:n_still].mean(axis=0)
    Rg = np.empty((len(imu_t), 3, 3)); Rg[0] = np.eye(3)
    for k in range(1, len(imu_t)):                            # gyro orientation, integrated once
        Rg[k] = Rg[k - 1] @ exp_so3(w[k] * (imu_t[k] - imu_t[k - 1]))

    def gyro_rot(off):
        i = np.clip(np.searchsorted(imu_t, ta[:, 0] + off), 0, len(imu_t) - 1)
        j = np.clip(np.searchsorted(imu_t, ta[:, 1] + off), 0, len(imu_t) - 1)
        return np.array([log_so3(Rg[x].T @ Rg[y]) for x, y in zip(i, j)])

    na = np.linalg.norm(a, axis=1)
    best = (-2, t0_guess)
    for grid in (np.arange(t0_guess - search, t0_guess + search, 0.01), None):
        if grid is None:
            grid = np.arange(best[1] - 0.012, best[1] + 0.012, 0.001)
        for off in grid:
            r = np.corrcoef(np.linalg.norm(gyro_rot(off), axis=1), na)[0, 1]
            if r > best[0]:
                best = (r, off)
    corr, off = best
    b = gyro_rot(off)
    sel = (na > 0.05) & (np.linalg.norm(b, axis=1) > 0.05)
    U, _, Vt = np.linalg.svd(b[sel].T @ a[sel])               # Kabsch: b = R_bc a
    R_fit = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    # a phone camera sits on the body axes to well under a degree: snap to the nearest axis permutation
    R_bc = np.zeros((3, 3))
    for i in range(3):
        j = int(np.argmax(np.abs(R_fit[i])))
        R_bc[i, j] = np.sign(R_fit[i, j])
    snap_deg = float(np.degrees(np.linalg.norm(log_so3(R_bc.T @ R_fit))))
    if abs(np.linalg.det(R_bc) - 1) > 1e-6 or snap_deg > 20:
        R_bc, snap_deg = R_fit, 0.0                           # no clean axis match: keep the fit
    cost = lambda o: np.median(np.linalg.norm(gyro_rot(o)[sel] - a[sel] @ R_bc.T, axis=1))
    grid = np.arange(off - 0.15, off + 0.15, 0.002)           # refine the offset with full rotation vectors
    off = float(grid[int(np.argmin([cost(o) for o in grid]))])
    resid = float(np.degrees(cost(off)))
    return off, R_bc, dict(rotation_correlation=float(corr), windows=int(sel.sum()), window_s=float(m / cam["fps"]),
                           axis_residual_deg=resid, fit_to_axes_deg=snap_deg)


# ---------------------------------------------------------------- 3. ESKF (from 7_4_ESKF_EKF_Comparision_PhoneIMUNoise.py)
def eskf_update(P, nominal, residual, H, Rm, gate=None):
    S = H @ P @ H.T + Rm
    nis = float(residual @ np.linalg.solve(S, residual))
    if gate is not None and nis > gate:
        return P, nominal, False, nis
    K = np.linalg.solve(S, H @ P).T
    d = K @ residual
    I = np.eye(15)
    P = (I - K @ H) @ P @ (I - K @ H).T + K @ Rm @ K.T
    p, v, R, bg, ba = nominal
    p, v = p + d[0:3], v + d[3:6]
    R = R @ exp_so3(d[6:9])                                   # right / body-frame error
    bg, ba = bg + d[9:12], ba + d[12:15]
    J = np.eye(15); J[6:9, 6:9] -= 0.5 * skew(d[6:9])
    P = J @ P @ J.T
    return 0.5 * (P + P.T), (p, v, R, bg, ba), True, nis


def camera_sigmas(ntag, err_mm):
    """Camera pose noise: one checkpoint (2 tags) leaves a shift/tilt trade-off, so it is trusted less."""
    if ntag >= 3:
        sp, sa = 0.004, 0.4
    elif ntag == 2:
        sp, sa = 0.015, 1.0
    else:
        sp, sa = 0.04, 3.0
    sp = max(sp, 3e-3 * err_mm)
    return sp, np.radians(sa)


def run_eskf(imu_t, gyro, acc, cam_meas, start, noise, use_camera=True, use_gravity=True,
             grav_sigma=0.03, grav_acc_scale=1.0, grav_gate=2.0, cam_gate=40.0, reset_after=5, cam_att="yaw"):
    """cam_meas: list of (t, p_body, R_wb, ntag, err_mm) on the IMU clock. start: (t0, p0, R0, bg0).

    Camera fixes go through a chi-square gate. While no tag is in view the IMU drifts faster than its white-noise
    model says (uncalibrated scale, walking dynamics), so after `reset_after` gated-out fixes in a row from 2+ tags
    the pose restarts from the camera, with the velocity from the last two camera fixes and the learnt biases.
    cam_att: which attitude the camera corrects: "yaw" (default; roll and pitch come from gravity, because a camera
    seeing one checkpoint can be a few degrees off in tilt and every degree leaks gravity into position), "full", or
    "none" (position only)."""
    t0, p0, R0, bg0 = start
    k0 = int(np.searchsorted(imu_t, t0))
    nominal = (p0.copy(), np.zeros(3), R0.copy(), bg0.copy(), np.zeros(3))
    P = np.diag([0.01**2] * 3 + [0.05**2] * 3 + [np.radians(1)**2] * 3 + [0.01**2] * 3 + [0.1**2] * 3)
    fs = 1.0 / np.median(np.diff(imu_t))
    Sigma = np.diag(np.r_[np.full(3, (noise["Na"] * math.sqrt(fs))**2), np.full(3, (noise["Ng"] * math.sqrt(fs))**2),
                          np.full(3, noise["Kbg"]**2), np.full(3, noise["Kba"]**2)])
    gw = np.array([0.0, 0.0, -G])
    j = int(np.searchsorted([m[0] for m in cam_meas], t0)) if cam_meas else 0
    out, used, rejected, streak, resets = [], 0, 0, 0, 0
    prev = None                                               # last camera fix with 2+ tags: velocity on restart
    for k in range(k0 + 1, len(imu_t)):
        dt = imu_t[k] - imu_t[k - 1]
        p, v, R, bg, ba = nominal
        w, f = gyro[k] - bg, acc[k] - ba
        a = R @ f + gw
        p = p + v * dt + 0.5 * a * dt * dt
        v = v + a * dt
        R = R @ exp_so3(w * dt)
        nominal = (p, v, R, bg, ba)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -0.5 * R @ skew(f) * dt * dt
        F[0:3, 12:15] = -0.5 * R * dt * dt
        F[3:6, 6:9] = -R @ skew(f) * dt
        F[3:6, 12:15] = -R * dt
        F[6:9, 6:9] = np.eye(3) - skew(w) * dt
        F[6:9, 9:12] = -np.eye(3) * dt
        Gm = np.zeros((15, 12))
        Gm[0:3, 0:3] = -0.5 * R * dt * dt
        Gm[3:6, 0:3] = -R * dt
        Gm[6:9, 3:6] = -np.eye(3) * dt
        Gm[9:12, 6:9] = np.eye(3) * math.sqrt(dt)
        Gm[12:15, 9:12] = np.eye(3) * math.sqrt(dt)
        P = F @ P @ F.T + Gm @ Sigma @ Gm.T
        if use_gravity:                                       # tilt from the gravity direction, faded while accelerating
            p, v, R, bg, ba = nominal
            fm = acc[k] - ba
            fn = np.linalg.norm(fm)
            if abs(fn - G) <= grav_gate:
                h = R.T @ np.array([0.0, 0.0, 1.0])
                Hg = np.zeros((3, 15)); Hg[:, 6:9] = skew(h)
                infl = math.sqrt(1 + (np.linalg.norm(R @ fm + gw) / grav_acc_scale)**2) if grav_acc_scale > 0 else 1.0
                P, nominal, _, _ = eskf_update(P, nominal, fm / fn - h, Hg, np.eye(3) * (grav_sigma * infl)**2, 12.0)
        while j < len(cam_meas) and cam_meas[j][0] <= imu_t[k]:
            tc, pc, Rc, nt, em = cam_meas[j]
            j += 1
            if not use_camera:
                continue
            p, v, R, bg, ba = nominal
            sp, sa = camera_sigmas(nt, em)
            if cam_att == "full":
                r = np.r_[pc - p, log_so3(R.T @ Rc)]
                H = np.zeros((6, 15)); H[0:3, 0:3] = np.eye(3); H[3:6, 6:9] = np.eye(3)
                Rm = np.diag([sp**2] * 3 + [sa**2] * 3)
            elif cam_att == "yaw":                            # heading about world z; body error -> world: R dtheta
                dyaw = math.atan2(Rc[1, 0], Rc[0, 0]) - math.atan2(R[1, 0], R[0, 0])
                r = np.r_[pc - p, (dyaw + math.pi) % (2 * math.pi) - math.pi]
                H = np.zeros((4, 15)); H[0:3, 0:3] = np.eye(3); H[3, 6:9] = R[2, :]
                Rm = np.diag([sp**2] * 3 + [sa**2])
            else:
                r = pc - p
                H = np.zeros((3, 15)); H[0:3, 0:3] = np.eye(3)
                Rm = np.eye(3) * sp**2
            P, nominal, ok, _ = eskf_update(P, nominal, r, H, Rm, cam_gate)
            used += ok; rejected += not ok
            streak = 0 if ok else streak + (nt >= 2)
            if streak >= reset_after:
                if prev is not None and 0 < tc - prev[0] < 0.5:
                    v = (pc - prev[1]) / (tc - prev[0])          # velocity from the last two camera fixes
                nominal = (pc.copy(), v, Rc.copy(), bg, ba)
                P[0:3, :] = P[:, 0:3] = P[6:9, :] = P[:, 6:9] = P[3:6, :] = P[:, 3:6] = 0.0
                P[0:3, 0:3], P[6:9, 6:9], P[3:6, 3:6] = np.eye(3) * sp**2, np.eye(3) * sa**2, np.eye(3) * 0.3**2
                streak, resets = 0, resets + 1
            if nt >= 2:
                prev = (tc, pc)
        p, v, R, bg, ba = nominal
        out.append(np.r_[imu_t[k], p, v, R[:, 0], R[:, 1], R[:, 2], bg, ba, np.sqrt(np.diag(P)[0:3])])
    return np.array(out), dict(camera_used=int(used), camera_rejected=int(rejected), resets=int(resets))


# ---------------------------------------------------------------- 4. 3D view
def figure_html(run, layout_world, cam_pts, fused, dropout, imu_only, gaps, stats, path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = go.Figure()
    xs = np.concatenate([c[:, 0] for c in layout_world.values()])
    x0, x1 = xs.min() - 0.6, xs.max() + 0.6
    fig.add_trace(go.Mesh3d(x=[x0, x1, x1, x0], y=[-0.6, -0.6, 0.6, 0.6], z=[0, 0, 0, 0], i=[0, 0], j=[1, 2], k=[2, 3],
                            color="#d9d4c7", opacity=0.55, hoverinfo="skip", name="floor", showlegend=False))
    for i, c in layout_world.items():
        ctr, h = c.mean(axis=0), np.ptp(c[:, 0]) / 2
        px = [ctr[0] - 0.69 * h * 2, ctr[0] + 0.69 * h * 2]          # the white cut piece (95 x 138.5 mm) around it
        py = [ctr[1] - 0.68 * h * 2, ctr[1] + 0.68 * h * 2]
        fig.add_trace(go.Mesh3d(x=[px[0], px[1], px[1], px[0]], y=[py[0], py[0], py[1], py[1]], z=[0.001] * 4,
                                i=[0, 0], j=[1, 2], k=[2, 3], color="white", hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Mesh3d(x=c[:, 0], y=c[:, 1], z=c[:, 2] + 0.003, i=[0, 0], j=[1, 2], k=[2, 3], color="black",
                                hovertext=f"AprilTag {i}", hoverinfo="text", showlegend=False))
    for k, x in enumerate(sorted({round(float(c[:, 0].mean()), 3) for c in layout_world.values()})):
        fig.add_trace(go.Scatter3d(x=[x], y=[-0.35], z=[0.0], mode="text", text=[f"{x:.0f} m"], textfont=dict(size=11, color="#555"),
                                   hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=cam_pts[:, 1], y=cam_pts[:, 2], z=cam_pts[:, 3], mode="markers",
                               marker=dict(size=2.2, color="#f28e2b"), name="camera fix (AprilTag pose, 2+ tags)",
                               hovertemplate="camera fix<br>t %{customdata:.2f} s<extra></extra>", customdata=cam_pts[:, 0]))
    fig.add_trace(go.Scatter3d(x=fused[:, 1], y=fused[:, 2], z=np.full(len(fused), 0.004), mode="lines",
                               line=dict(width=3, color="rgba(60,60,60,0.45)"), name="shadow on the floor", hoverinfo="skip"))
    sp = np.linalg.norm(fused[:, 4:7], axis=1)
    fig.add_trace(go.Scatter3d(x=fused[:, 1], y=fused[:, 2], z=fused[:, 3], mode="lines",
                               line=dict(width=7, color=fused[:, 0], colorscale="Viridis",
                                         colorbar=dict(title="time [s]", len=0.6, x=0.98)),
                               name="ESKF: IMU + camera", customdata=np.c_[fused[:, 0], sp],
                               hovertemplate="ESKF<br>t %{customdata[0]:.2f} s<br>speed %{customdata[1]:.2f} m/s<br>"
                                             "x %{x:.3f}  y %{y:.3f}  z %{z:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=dropout[:, 1], y=dropout[:, 2], z=dropout[:, 3], mode="lines",
                               line=dict(width=4, color="#4e79a7", dash="dash"),
                               name=f"ESKF with camera off {stats['dropout']['off_s']:.0f} s of every {stats['dropout']['period_s']:.0f} s",
                               visible="legendonly"))
    io_ = imu_only[np.all(np.abs(imu_only[:, 1:4] - imu_only[0, 1:4]) < [12, 2, 2], axis=1)]
    io_ = io_[: np.argmax(np.r_[np.abs(np.diff(io_[:, 0])) > 0.05, True]) + 1] if len(io_) else io_
    fig.add_trace(go.Scatter3d(x=io_[:, 1], y=io_[:, 2], z=io_[:, 3], mode="lines", line=dict(width=4, color="#e15759"),
                               name=f"IMU only (leaves the track after {stats['imu_only_1m_s']:.1f} s)"))
    step = max(1, int(round(1.0 / np.median(np.diff(fused[:, 0])))))
    sel = fused[::step]
    fwd = sel[:, 10:13]                                        # body y axis = top edge of the phone
    fig.add_trace(go.Cone(x=sel[:, 1], y=sel[:, 2], z=sel[:, 3], u=fwd[:, 0], v=fwd[:, 1], w=fwd[:, 2], sizemode="absolute",
                          sizeref=0.3, anchor="tail", colorscale=[[0, "#59a14f"], [1, "#59a14f"]], showscale=False,
                          name="phone heading (1 s)", showlegend=True, hoverinfo="skip"))
    long = [g for g in gaps if g[1] - g[0] > 3.0]
    for a, b in long:                                          # label the long stretches without any tag in view
        k = int(np.searchsorted(fused[:, 0], 0.5 * (a + b)))
        k = min(k, len(fused) - 1)
        fig.add_trace(go.Scatter3d(x=[fused[k, 1]], y=[fused[k, 2]], z=[fused[k, 3] + 0.35], mode="text",
                                   text=[f"{b - a:.0f} s with no tag in view"], textfont=dict(size=12, color="#b07aa1"),
                                   hoverinfo="skip", showlegend=False))
    for (pt, sym, col, lab) in ((fused[0], "diamond", "#2ca02c", "start"), (fused[-1], "x", "#d62728", "end")):
        fig.add_trace(go.Scatter3d(x=[pt[1]], y=[pt[2]], z=[pt[3]], mode="markers+text", text=[lab], textposition="top center",
                                   marker=dict(size=7, color=col, symbol=sym), name=lab, showlegend=False))
    fig.update_layout(
        scene=dict(xaxis_title="x along the track [m]", yaxis_title="y [m]", zaxis_title="height [m]",
                   aspectmode="manual", aspectratio=dict(x=3.2, y=0.55, z=0.55),
                   yaxis=dict(range=[-0.8, 0.8]), zaxis=dict(range=[0, 1.6]),
                   camera=dict(eye=dict(x=-1.1, y=-1.9, z=0.9))),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"), margin=dict(l=0, r=0, t=10, b=0), height=640)

    f2 = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.08,
                       subplot_titles=("Top view", "Where the IMU carries the solution"))
    for i, c in layout_world.items():
        f2.add_trace(go.Scatter(x=np.r_[c[:, 0], c[0, 0]], y=np.r_[c[:, 1], c[0, 1]], fill="toself", fillcolor="black",
                                line=dict(width=0), hoverinfo="skip", showlegend=False), 1, 1)
    f2.add_trace(go.Scatter(x=cam_pts[:, 1], y=cam_pts[:, 2], mode="markers", marker=dict(size=3, color="#f28e2b"),
                            name="camera fix"), 1, 1)
    f2.add_trace(go.Scatter(x=fused[:, 1], y=fused[:, 2], mode="lines", line=dict(color="#3b528b", width=2), name="ESKF"), 1, 1)
    f2.add_trace(go.Scatter(x=dropout[:, 0], y=np.full(len(dropout), np.nan), showlegend=False), 1, 2)
    for a, b in gaps:
        f2.add_vrect(x0=a, x1=b, fillcolor="#bab0ac", opacity=0.35, line_width=0, row=1, col=2)
    e = stats["_dropout_err"]
    f2.add_trace(go.Scatter(x=e[:, 0], y=e[:, 1] * 100, mode="markers", marker=dict(size=3, color="#4e79a7"),
                            name="error while camera blanked [cm]"), 1, 2)
    f2.update_xaxes(title_text="x [m]", row=1, col=1); f2.update_yaxes(title_text="y [m]", scaleanchor="x", row=1, col=1)
    f2.update_xaxes(title_text="time [s]  (grey: no tag in view)", row=1, col=2)
    f2.update_yaxes(title_text="ESKF vs camera [cm]", row=1, col=2)
    f2.update_layout(height=380, margin=dict(l=40, r=10, t=40, b=40), legend=dict(orientation="h", y=-0.25))

    f3 = separate_scenes(layout_world, cam_pts, fused, imu_only, stats)
    s = stats
    cards = [("Path walked", f"{s['path_m']:.1f} m"),
             ("End point, ESKF vs camera", f"{s['end_error_m'] * 100:.1f} cm"),
             ("Start to end (ESKF / camera)", f"{s['return_to_start_m']:.2f} / {s['camera_return_to_start_m']:.2f} m"),
             ("Camera fixes", f"{s['camera_fix_pct']:.0f} % of frames"),
             ("IMU bridging error", f"{s['dropout']['rmse_m'] * 100:.1f} cm RMS"),
             ("IMU only", f"off track after {s['imu_only_1m_s']:.1f} s")]
    head = "".join(f'<div class="card"><div class="v">{v}</div><div class="k">{k}</div></div>' for k, v in cards)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{run}: IMU + camera ESKF</title>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#fafaf7;color:#222}}
header{{padding:18px 24px 6px}}h1{{margin:0;font-size:22px}}.sub{{color:#666;margin:4px 0 0}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;padding:10px 24px}}.card{{background:#fff;border:1px solid #e5e2da;border-radius:8px;padding:10px 14px;min-width:150px}}
.v{{font-size:22px;font-weight:600}}.k{{color:#666;font-size:13px}}section{{background:#fff;margin:10px 24px;border:1px solid #e5e2da;border-radius:8px}}
.note{{color:#555;font-size:13px;padding:6px 24px 18px;max-width:1100px}}</style></head><body>
<header><h1>IMU + camera ESKF on the AprilTag track ({run})</h1>
<p class="sub">Phone IMU at {s['imu_rate_hz']:.0f} Hz drives a 15-state error-state Kalman filter; the camera's AprilTag pose corrects it whenever a checkpoint is in view.</p></header>
<div class="cards">{head}</div>
<section>{fig.to_html(full_html=False, include_plotlyjs=True)}</section>
<section>{f2.to_html(full_html=False, include_plotlyjs=False)}</section>
<h2 style="margin:22px 24px 0;font-size:18px">Each source on its own</h2>
<section>{f3.to_html(full_html=False, include_plotlyjs=False)}</section>
<p class="note">Drag to rotate, scroll to zoom, click legend entries to show or hide. Orange dots: camera poses from the tags.
Coloured line: the fused estimate (colour = time). Green cones: where the phone's top edge points, once a second.
Red: the same IMU integrated with no camera, from the same start. Blue dashed (legend): the filter with the camera
switched off {s['dropout']['off_s']:.0f} s out of every {s['dropout']['period_s']:.0f} s; its error against the
camera in those blanked stretches is the bridging error above. Vertical axis not to the same scale as the track.</p>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


PANELS = (("camera", "1. Camera only: AprilTag pose (2+ tags)", "#f28e2b"),
          ("imu", "2. IMU only: dead reckoning from the same start", "#e15759"),
          ("eskf", "3. ESKF: IMU + camera", None))


def _panel_series(name, cam_pts, fused, imu_only):
    if name == "camera":
        return cam_pts
    if name == "imu":
        keep = np.all(np.abs(imu_only[:, 1:4] - imu_only[0, 1:4]) < [12, 1.5, 1.5], axis=1)
        return imu_only[: np.argmin(keep)] if not keep.all() else imu_only
    return fused


def separate_scenes(layout_world, cam_pts, fused, imu_only, stats):
    """Plotly: one 3D scene per source, same axes and view, stacked."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    titles = [t for _, t, _ in PANELS]
    titles[1] += f" (leaves the track after {stats['imu_only_1m_s']:.1f} s)"
    f = make_subplots(rows=3, cols=1, specs=[[{"type": "scene"}]] * 3, subplot_titles=titles, vertical_spacing=0.03)
    for r, (name, _, col) in enumerate(PANELS, start=1):
        for c in layout_world.values():
            f.add_trace(go.Mesh3d(x=c[:, 0], y=c[:, 1], z=c[:, 2] + 0.003, i=[0, 0], j=[1, 2], k=[2, 3], color="black",
                                  hoverinfo="skip", showlegend=False), r, 1)
        d = _panel_series(name, cam_pts, fused, imu_only)
        f.add_trace(go.Scatter3d(x=d[:, 1], y=d[:, 2], z=np.zeros(len(d)), mode="lines",
                                 line=dict(width=2, color="rgba(60,60,60,0.4)"), hoverinfo="skip", showlegend=False), r, 1)
        if name == "camera":
            tr = go.Scatter3d(x=d[:, 1], y=d[:, 2], z=d[:, 3], mode="markers", marker=dict(size=2.5, color=col))
        elif name == "imu":
            tr = go.Scatter3d(x=d[:, 1], y=d[:, 2], z=d[:, 3], mode="lines", line=dict(width=6, color=col))
        else:
            tr = go.Scatter3d(x=d[:, 1], y=d[:, 2], z=d[:, 3], mode="lines",
                              line=dict(width=6, color=d[:, 0], colorscale="Viridis"))
        tr.update(showlegend=False, hovertemplate="t %{customdata:.2f} s<br>x %{x:.2f}  y %{y:.2f}  z %{z:.2f}<extra></extra>",
                  customdata=d[:, 0])
        f.add_trace(tr, r, 1)
    xs = np.concatenate([c[:, 0] for c in layout_world.values()])
    scene = dict(xaxis=dict(title="x [m]", range=[xs.min() - 2, xs.max() + 1.6]), yaxis=dict(title="y [m]", range=[-0.8, 0.8]),
                 zaxis=dict(title="z [m]", range=[0, 1.6]), aspectmode="manual", aspectratio=dict(x=3.2, y=0.55, z=0.55),
                 camera=dict(eye=dict(x=-1.1, y=-1.9, z=0.9)))
    f.update_layout(scene=scene, scene2=scene, scene3=scene, height=1250, margin=dict(l=0, r=0, t=30, b=0))
    return f


def figure_panels_png(layout_world, cam_pts, fused, imu_only, stats, path):
    """Slides: the three sources in separate 3D panels, same axes and view."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 15))
    xs = np.concatenate([c[:, 0] for c in layout_world.values()])
    for r, (name, title, col) in enumerate(PANELS):
        ax = fig.add_axes([-0.06, 0.66 - r * 0.32, 1.08, 0.34], projection="3d")
        for c in layout_world.values():
            ax.plot_trisurf(c[:, 0], c[:, 1], c[:, 2] + 0.003, color="black")
        d = _panel_series(name, cam_pts, fused, imu_only)
        ax.plot(d[:, 1], d[:, 2], 0 * d[:, 3], color="0.6", lw=0.8)
        if name == "camera":
            ax.scatter(d[:, 1], d[:, 2], d[:, 3], s=4, color=col)
        elif name == "imu":
            ax.plot(d[:, 1], d[:, 2], d[:, 3], color=col, lw=2.5)
            title += f"  (leaves the track after {stats['imu_only_1m_s']:.1f} s)"
        else:
            sc = ax.scatter(d[:, 1], d[:, 2], d[:, 3], c=d[:, 0], cmap="viridis", s=2)
        ax.set_xlim(xs.min() - 2, xs.max() + 1.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(0, 1.5)
        ax.set_box_aspect((6, 1.0, 1.0), zoom=1.3)
        ax.set_yticks([-0.5, 0, 0.5]); ax.set_zticks([0, 0.5, 1.0, 1.5])
        ax.set_xlabel("x along the track [m]", labelpad=10); ax.set_zlabel("height [m]")
        ax.view_init(elev=24, azim=-62)
        ax.set_title(title, loc="left", fontsize=14, y=0.92)
    cax = fig.add_axes([0.93, 0.1, 0.012, 0.18])
    fig.colorbar(sc, cax=cax, label="time [s]")
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def figure_png(layout_world, cam_pts, fused, imu_only, stats, path):
    """Static 3D view + top view for slides (matplotlib)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([-0.06, 0.22, 1.08, 0.80], projection="3d")
    for c in layout_world.values():
        ax.plot_trisurf(c[:, 0], c[:, 1], c[:, 2] + 0.003, color="black")
    sc = ax.scatter(fused[:, 1], fused[:, 2], fused[:, 3], c=fused[:, 0], cmap="viridis", s=2, label="ESKF: IMU + camera")
    ax.plot(fused[:, 1], fused[:, 2], 0 * fused[:, 3], color="0.55", lw=0.8, label="shadow on the floor")
    ax.scatter(cam_pts[:, 1], cam_pts[:, 2], cam_pts[:, 3], s=3, color="#f28e2b", label="camera fix (AprilTag)")
    io_ = imu_only[np.linalg.norm(imu_only[:, 1:4] - imu_only[0, 1:4], axis=1) < 2.0]
    ax.plot(io_[:, 1], io_[:, 2], io_[:, 3], color="#e15759", lw=2, label="IMU only")
    ax.set_box_aspect((6, 1.0, 1.0), zoom=1.25); ax.set_ylim(-0.6, 0.6); ax.set_zlim(0, 1.5)
    ax.set_xlabel("x along the track [m]", labelpad=12); ax.set_ylabel("y [m]"); ax.set_zlabel("height [m]")
    ax.set_yticks([-0.5, 0, 0.5]); ax.set_zticks([0, 0.5, 1.0, 1.5])
    ax.view_init(elev=24, azim=-62); ax.legend(loc="upper left", bbox_to_anchor=(0.06, 0.92))
    cax = fig.add_axes([0.93, 0.45, 0.012, 0.35])
    fig.colorbar(sc, cax=cax, label="time [s]")
    ax2 = fig.add_axes([0.05, 0.07, 0.9, 0.17])
    for c in layout_world.values():
        ax2.fill(c[:, 0], c[:, 1], color="black")
    ax2.plot(cam_pts[:, 1], cam_pts[:, 2], ".", ms=2, color="#f28e2b")
    ax2.plot(fused[:, 1], fused[:, 2], lw=1.2, color="#3b528b")
    ax2.set_aspect("equal"); ax2.set_ylim(-0.5, 0.5); ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]"); ax2.grid(alpha=0.3)
    ax2.set_title("Top view")
    fig.suptitle(f"IMU + camera ESKF on the AprilTag track: path {stats['path_m']:.1f} m, end point "
                 f"{stats['end_error_m'] * 100:.0f} cm from the camera, IMU bridging {stats['dropout']['rmse_m'] * 100:.0f} cm RMS",
                 fontsize=13)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--lr-half-gap", type=float, default=None, help="left/right tag centres at y = +-this (m)")
    ap.add_argument("--noise", default=os.path.join(ROOT, "runs", "One_hr_drift_test", "phone_noise.yaml"))
    ap.add_argument("--dropout", type=float, nargs=2, default=[2.0, 4.0], metavar=("OFF_S", "PERIOD_S"),
                    help="blank the camera OFF_S seconds out of every PERIOD_S for the bridging test")
    a = ap.parse_args()
    run, out = a.run, os.path.join(a.run, "fusion")
    os.makedirs(out, exist_ok=True)
    world = tag_corners(rows=layout_rows(a.lr_half_gap))
    cam = camera_poses(os.path.join(run, "video.mkv"), world, os.path.join(out, "camera_poses.npz"))
    imu = io.load(run).imu
    sess = yaml.safe_load(open(os.path.join(run, "session.yaml")))
    t_imu0 = imu.t_us[0] * 1e-6
    imu_t = imu.t_us * 1e-6 - t_imu0                           # IMU clock, s from the first sample
    t0_guess = sess["video_first_frame_sensor_ns"] * 1e-9 - t_imu0
    off, R_bc, sinfo = sync(cam, imu_t, imu.gyro, t0_guess)
    print(f"camera fixes {np.isfinite(cam['p'][:, 0]).sum()} / {len(cam['t'])} frames; video starts {off:.3f} s after the IMU "
          f"(clock estimate {t0_guess:.3f} s); rotation correlation {sinfo['rotation_correlation']:.3f}, "
          f"residual {sinfo['axis_residual_deg']:.2f} deg over {sinfo['window_s']:.2f} s windows")

    meas = [(cam["t"][k] + off, cam["p"][k], cam["R"][k] @ R_bc.T, int(cam["ntag"][k]), float(cam["err_mm"][k]))
            for k in range(len(cam["t"])) if np.isfinite(cam["p"][k, 0])]
    nz = yaml.safe_load(open(a.noise))
    em = nz.get("error_model", {})
    rw = lambda B, tau: float(np.max(np.asarray(B) * np.sqrt(2.0 / np.asarray(tau))))
    noise = dict(Na=float(nz["Na"]), Ng=float(nz["Ng"]), Kba=rw(em["B_acc"], em["tau_acc"]), Kbg=rw(em["B_gyro"], em["tau_gyro"]))
    first = next(m for m in meas if m[3] >= 2)
    k_first = int(np.searchsorted(imu_t, first[0]))
    n_still = int(1.0 / np.median(np.diff(imu_t)))
    start = (first[0], first[1], first[2], imu.gyro[max(0, k_first - n_still):k_first + 1].mean(axis=0))

    fused, finfo = run_eskf(imu_t, imu.gyro, imu.acc, meas, start, noise)
    imu_only, _ = run_eskf(imu_t, imu.gyro, imu.acc, meas, start, noise, use_camera=False, use_gravity=False)
    off_s, period = a.dropout
    blank = lambda t: ((t - first[0]) % period) >= (period - off_s)
    kept = [m for m in meas if not blank(m[0])]
    dropout, _ = run_eskf(imu_t, imu.gyro, imu.acc, kept, start, noise)
    held = np.array([np.r_[m[0], m[1]] for m in meas if blank(m[0])])
    est = np.column_stack([np.interp(held[:, 0], dropout[:, 0], dropout[:, i]) for i in (1, 2, 3)])
    derr = np.c_[held[:, 0], np.linalg.norm(est - held[:, 1:4], axis=1)]

    cam_pts = np.array([np.r_[m[0], m[1]] for m in meas if m[3] >= 2])        # plotted reference: 2+ tags
    have = np.zeros(len(cam["t"]), bool); have[np.isfinite(cam["p"][:, 0])] = True
    gaps, s0 = [], None
    for k, h in enumerate(have):
        if not h and s0 is None:
            s0 = cam["t"][k] + off
        if h and s0 is not None:
            gaps.append((s0, cam["t"][k] + off)); s0 = None
    io_dev = np.linalg.norm(imu_only[:, 1:3] - np.column_stack([np.interp(imu_only[:, 0], fused[:, 0], fused[:, i]) for i in (1, 2)]), axis=1)
    f10 = fused[:: max(1, int(round(0.1 / np.median(np.diff(fused[:, 0])))))]      # 10 Hz: path without jitter
    last2 = max((m for m in meas if m[3] >= 2), key=lambda m: m[0])
    k_last = int(np.searchsorted(fused[:, 0], last2[0]))
    stats = dict(run=os.path.basename(os.path.normpath(run)), imu_rate_hz=float(1 / np.median(np.diff(imu_t))),
                 path_m=float(np.sum(np.linalg.norm(np.diff(f10[:, 1:3], axis=0), axis=1))),
                 return_to_start_m=float(np.linalg.norm(fused[min(k_last, len(fused) - 1), 1:3] - fused[0, 1:3])),
                 camera_return_to_start_m=float(np.linalg.norm(last2[1][:2] - first[1][:2])),
                 end_error_m=float(np.linalg.norm(fused[min(k_last, len(fused) - 1), 1:3] - last2[1][:2])),
                 camera_fix_pct=float(100 * have.mean()), sync=dict(video_offset_s=float(off), **sinfo),
                 R_bc=R_bc.tolist(), filter=finfo, imu_only_1m_s=float(imu_only[np.argmax(io_dev > 1.0), 0] - first[0]) if (io_dev > 1).any() else float("nan"),
                 dropout=dict(off_s=off_s, period_s=period, rmse_m=float(np.sqrt(np.mean(derr[:, 1] ** 2))),
                              max_m=float(derr[:, 1].max()), samples=int(len(derr))),
                 layout_lr_half_gap_m=a.lr_half_gap)
    print(f"ESKF: {finfo['camera_used']} camera updates used, {finfo['camera_rejected']} gated out, {finfo['resets']} restarts; path {stats['path_m']:.1f} m, "
          f"end point {stats['end_error_m'] * 100:.1f} cm from the camera; start-to-end {stats['return_to_start_m']:.2f} m "
          f"(camera {stats['camera_return_to_start_m']:.2f} m)")
    print(f"camera blanked {off_s:.0f} s of every {period:.0f} s: error vs camera {stats['dropout']['rmse_m'] * 100:.1f} cm RMS, "
          f"max {stats['dropout']['max_m'] * 100:.1f} cm; IMU only is 1 m off after {stats['imu_only_1m_s']:.1f} s")
    hdr = "t,x,y,z,vx,vy,vz,r11,r21,r31,r12,r22,r32,r13,r23,r33,bgx,bgy,bgz,bax,bay,baz,sx,sy,sz"
    np.savetxt(os.path.join(out, "fused.csv"), fused, delimiter=",", header=hdr, comments="", fmt="%.6f")
    with open(os.path.join(out, "summary.yaml"), "w") as f:
        yaml.safe_dump({k: v for k, v in stats.items()}, f, sort_keys=False)
    stats["_dropout_err"] = derr
    dec = max(1, int(round(0.05 / np.median(np.diff(fused[:, 0])))))         # plot at 20 Hz
    figure_html(stats["run"], world, cam_pts, fused[::dec], dropout[::dec], imu_only[::dec], gaps, stats, os.path.join(out, "trajectory_3d.html"))
    figure_png(world, cam_pts, fused[::dec], imu_only[::dec], stats, os.path.join(out, "trajectory_3d.png"))
    figure_panels_png(world, cam_pts, fused[::dec], imu_only[::dec], stats, os.path.join(out, "trajectory_panels.png"))
    print(f"wrote {out}/trajectory_3d.html, trajectory_3d.png, trajectory_panels.png, fused.csv, summary.yaml")


if __name__ == "__main__":
    main()
