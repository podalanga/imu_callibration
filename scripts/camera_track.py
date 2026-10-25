"""Phone position from the floor AprilTags: the camera-based ground truth.

  python scripts/camera_track.py walk.mp4                        # video file -> results next to it
  python scripts/camera_track.py walk.mp4 --show                 # also watch the detections frame by frame
  python scripts/camera_track.py http://192.168.1.5:8080/video --show    # live phone stream (e.g. IP Webcam app)
  python scripts/camera_track.py clip.mkv --a4-page 1            # one printed A4 page instead of the floor track

For every frame: detect the tag36h11 tags, match their corners to the surveyed tag positions (tag_layout.csv,
track frame: x along the track, y to the left, z up, floor at z = 0), and solve the camera pose from all visible
tags at once (cv2.solvePnP). The camera position is the phone position.

--a4-page N: the tags of print page N (print_A4/page_N_*.pdf, 4 tags in a 2 x 2 grid, made by make_tags.py) as
they sit on that one sheet. Positions are then in the sheet frame: origin at the page centre, x towards the top of
the page (the arrows' +x), y towards the left edge, z up out of the paper.

Outputs in --out (default <video>_track/): track.csv (t, x, y, z, heading, tilt, tags, reprojection error) and
track.png (top view over the tag layout, position against time).

How to hold the phone: portrait, camera tilted about 58 deg below the horizontal, looking ahead along the track.
It then sees 2-3 checkpoints (about 5 tags) and the position is good to about 1 mm on rendered test images. Pointing
straight down it sees one checkpoint (2 tags, 0.4 m apart): a small sideways shift then looks almost like a small
tilt, and the error grows to 1-3 cm typically and up to 10 cm or more.

Camera intrinsics: --calib camera.yaml (fx, fy, cx, cy, dist from a checkerboard calibration) is most accurate;
without it the focal length comes from --hfov-deg, the field of view across the image's long side (OnePlus Nord 4
main camera, 16:9 video: about 69.4 deg; the same value works for portrait and landscape video), and lens
distortion is ignored.
"""
import argparse, csv, os, time
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYOUT = os.path.join(ROOT, "simulation", "apriltags_imu_track_70mm", "apriltags_imu_track", "tag_layout.csv")


def a4_page_rows(page, tag_mm=70.0):
    """Tag centres on print page `page` (make_tags.py: A4, 10 mm margins, 2 x 2 cells of 95 x 138.5 mm, tag top
    14 mm below the cell top, ids in reading order). Sheet frame: origin at the page centre, x up the page, y left."""
    rows = []
    for k in range(4):
        col, row = k % 2, k // 2
        x_mm = 10 + col * 95 + 47.5                      # from the page's left edge
        y_mm = 10 + row * 138.5 + 14 + tag_mm / 2        # from the page's top edge
        rows.append(dict(tag_id=4 * (page - 1) + k, x_m=(148.5 - y_mm) / 1000, y_m=(105 - x_mm) / 1000,
                         tag_size_m=tag_mm / 1000))
    return rows


def tag_corners(layout_csv=None, rows=None):
    """{id: (4, 3)} world corners of each tag's black square, in OpenCV's order TL, TR, BR, BL of the artwork.
    Every tag's arrow (artwork up) points to +x, so artwork right is -y."""
    if rows is None:
        with open(layout_csv) as f:
            rows = list(csv.DictReader(f))
    up, right = np.array([1.0, 0, 0]), np.array([0, -1.0, 0])
    out = {}
    for r in rows:
        c = np.array([float(r["x_m"]), float(r["y_m"]), 0.0])
        h = float(r["tag_size_m"]) / 2
        out[int(r["tag_id"])] = np.array([c - h * right + h * up, c + h * right + h * up,
                                          c + h * right - h * up, c - h * right - h * up])
    return out


def intrinsics(width, height, hfov_deg=69.4, calib=None):
    if calib:
        import yaml
        with open(calib) as f:
            c = yaml.safe_load(f)
        K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])
        return K, np.array(c.get("dist", [0, 0, 0, 0, 0]), float)
    f = max(width, height) / (2 * np.tan(np.radians(hfov_deg) / 2))      # FOV across the long side
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]]), np.zeros(5)


class TagPose:
    def __init__(self, world, K, dist):
        self.world, self.K, self.dist = world, K, dist
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX      # sub-pixel corners: about 2x better here
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)

    def detect(self, img):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is None:
            return {}
        return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners) if int(i) in self.world}

    def pose(self, dets):
        """Camera position (3,), camera-to-world rotation, rms reprojection error in pixels and the same error as a
        distance on the tags (mm, pixels scaled by range / focal length); None without tags. All tags lie on the floor plane, so IPPE gives both planar solutions
        and the one that reprojects better is kept."""
        if not dets:
            return None
        obj = np.concatenate([self.world[i] for i in dets]).astype(np.float64)
        img = np.concatenate([dets[i] for i in dets]).astype(np.float64)
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
        if n == 0:
            return None
        k = int(np.argmin(np.ravel(errs)))
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, self.K, self.dist, rvecs[k], tvecs[k])
        R_cw = cv2.Rodrigues(rvec)[0]                       # world -> camera
        pos = (-R_cw.T @ tvec).ravel()
        proj = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))
        err_mm = rms * np.linalg.norm(obj - pos, axis=1).mean() / self.K[0, 0] * 1000
        return pos, R_cw.T, rms, float(err_mm)


def heading_tilt(R_wc):
    """Where the camera looks: heading of the optical axis in the floor plane (deg, 0 = +x, 90 = +y) and how far
    it points below the horizontal (deg, 90 = straight down)."""
    z = R_wc[:, 2]
    return float(np.degrees(np.arctan2(z[1], z[0]))), float(np.degrees(np.arcsin(np.clip(-z[2], -1, 1))))


def draw(img, dets, res, t):
    for i, c in dets.items():
        cv2.polylines(img, [c.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(img, str(i), tuple(c.mean(axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    txt = f"t {t:6.2f} s  tags {len(dets)}"
    if res:
        p, R, rms, err_mm = res
        txt += f"  x {p[0]:+.3f}  y {p[1]:+.3f}  z {p[2]:.3f} m  fit {err_mm:.1f} mm"
    cv2.putText(img, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
    cv2.putText(img, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def plot(rows, world, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r = np.array([x for x in rows if np.isfinite(x[1])]) if rows else np.zeros((0, 9))
    pts = np.concatenate(list(world.values()))
    wide = np.ptp(pts[:, 0]) > 3 * np.ptp(pts[:, 1])           # long floor track: stack; small sheet: side by side
    fig = plt.figure(figsize=(14, 7) if wide else (15, 5.5))
    ax = fig.add_subplot(2, 1, 1) if wide else fig.add_subplot(1, 2, 1)
    for i, c in world.items():
        ax.fill(c[:, 0], c[:, 1], color="0.2"); ax.text(c[:, 0].mean(), c[:, 1].max() + 0.2 * np.ptp(c[:, 1]), str(i), ha="center", fontsize=7)
    if len(r):
        sc = ax.scatter(r[:, 1], r[:, 2], c=r[:, 0], s=6, cmap="viridis"); fig.colorbar(sc, ax=ax, label="time [s]")
        ax.plot(r[:, 1], r[:, 2], lw=0.5, color="tab:blue")
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(alpha=0.3)
    ax.set_title(f"{title}: camera position from the tags (top view)")
    ax2 = fig.add_subplot(2, 1, 2) if wide else fig.add_subplot(1, 2, 2)
    if len(r):
        for k, lab in ((1, "x"), (2, "y"), (3, "z (height)")):
            ax2.plot(r[:, 0], r[:, k], ".", ms=2, label=lab)
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("position [m]"); ax2.grid(alpha=0.3)
    if len(r):
        ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="video file, camera index (0) or stream URL")
    ap.add_argument("--layout", default=LAYOUT)
    ap.add_argument("--a4-page", type=int, default=None, help="use the 4 tags of this printed A4 page instead of --layout")
    ap.add_argument("--hfov-deg", type=float, default=69.4)
    ap.add_argument("--calib", default=None, help="camera.yaml with fx, fy, cx, cy, dist")
    ap.add_argument("--show", action="store_true", help="show each frame with the detections (q quits)")
    ap.add_argument("--max-err-mm", type=float, default=2.0,
                    help="drop poses whose corners fit worse than this, as a distance on the tags (mm)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = int(a.source) if a.source.isdigit() else a.source
    live = not (isinstance(src, str) and os.path.isfile(src))
    out = a.out or (os.path.splitext(a.source)[0] + "_track" if not live else os.path.join(ROOT, "runs", time.strftime("camera_%Y%m%d_%H%M%S")))
    os.makedirs(out, exist_ok=True)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.source}")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    world = tag_corners(rows=a4_page_rows(a.a4_page)) if a.a4_page else tag_corners(a.layout)
    K, dist = intrinsics(W, H, a.hfov_deg, a.calib)
    tp = TagPose(world, K, dist)
    print(f"{a.source}: {W}x{H} at {fps:.1f} fps, f = {K[0, 0]:.0f} px, {len(world)} tags in the layout")

    rows, n, t0 = [], 0, time.time()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = (time.time() - t0) if live else n / fps
        n += 1
        dets = tp.detect(img)
        res = tp.pose(dets)
        if res and res[3] > a.max_err_mm:
            res = None
        if res:
            hd, tl = heading_tilt(res[1])
            rows.append([t, *res[0], hd, tl, len(dets), res[2], res[3]])
        else:
            rows.append([t, np.nan, np.nan, np.nan, np.nan, np.nan, len(dets), np.nan, np.nan])
        if a.show:
            cv2.imshow("camera_track", cv2.resize(draw(img, dets, res, t), None, fx=0.6, fy=0.6))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()

    with open(os.path.join(out, "track.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z", "heading_deg", "tilt_deg", "tags", "rms_px", "fit_mm"])
        for r in rows:
            w.writerow([f"{r[0]:.4f}"] + ["" if not np.isfinite(v) else f"{v:.4f}" for v in r[1:6]] + [r[6]]
                       + ["" if not np.isfinite(v) else f"{v:.2f}" for v in r[7:9]])
    plot(rows, world, os.path.join(out, "track.png"), os.path.basename(str(a.source)))
    good = sum(np.isfinite(r[1]) for r in rows)
    print(f"{n} frames, position in {good} ({100 * good / max(n, 1):.0f} %); wrote {out}/track.csv and track.png")


if __name__ == "__main__":
    main()
