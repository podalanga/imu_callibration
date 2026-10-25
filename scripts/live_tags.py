"""Live AprilTag check: detect the tags in the phone camera and show how much the position moves.

  python scripts/live_tags.py --a4-page 1        # tags of the printed A4 page 1
  python scripts/live_tags.py                    # the floor track layout

Two windows:
  tags             the camera image with the detected tags, plus image brightness and sharpness
  position change  x, y, z of the camera relative to a reference (the first fix; press r to reset), last 15 s,
                   with the jitter (std over the last 2 s) and the detection rate
Hold the phone still over the tags: in good light the jitter stays well under 1 mm and every frame is detected.
Rising jitter, dropped frames or a dark / blurry image mean the lighting needs work.
Keys: r = reset the reference, q = quit.

The camera comes straight from the phone over USB (the scrcpy server streams the back camera, PyAV decodes it):
needs scrcpy (winget install Genymobile.scrcpy) and `pip install --user --no-deps av==12.3.0`.
"""
import argparse, collections, os, shutil, socket, subprocess, sys, threading, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_track import LAYOUT, TagPose, a4_page_rows, intrinsics, tag_corners

PORT = 27183


def scrcpy_dir():
    exe = shutil.which("scrcpy")
    if exe:
        return os.path.dirname(exe)
    base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    for root, _, files in os.walk(base):
        if "scrcpy-server" in files:
            return root
    raise SystemExit("scrcpy not found: winget install Genymobile.scrcpy")


class PhoneCamera:
    """Latest frame of the phone's camera: runs the scrcpy server on the phone in raw H.264 mode and decodes it."""

    def __init__(self, camera_id=2, size="1920x1080", fps=30):
        import av
        d = scrcpy_dir()
        ver = subprocess.run([os.path.join(d, "scrcpy"), "--version"], capture_output=True, text=True).stdout.split()[1]
        adb = lambda *a: subprocess.run(["adb", *a], capture_output=True, text=True, timeout=20)
        adb("push", os.path.join(d, "scrcpy-server"), "/data/local/tmp/scrcpy-server.jar")
        adb("forward", f"tcp:{PORT}", "localabstract:scrcpy")
        self.srv = subprocess.Popen(
            ["adb", "shell", "CLASSPATH=/data/local/tmp/scrcpy-server.jar", "app_process", "/", "com.genymobile.scrcpy.Server",
             ver, "tunnel_forward=true", "audio=false", "control=false", "raw_stream=true", "video_source=camera",
             f"camera_id={camera_id}", f"camera_size={size}", f"camera_fps={fps}", "video_codec=h264"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.sock, first = None, b""
        for _ in range(50):                       # the server needs a moment before the stream starts
            time.sleep(0.2)
            try:
                s = socket.create_connection(("127.0.0.1", PORT), timeout=5)
                first = s.recv(1 << 16)
                if first:
                    self.sock = s
                    break
                s.close()
            except OSError:
                pass
        if self.sock is None:
            self.close()
            raise SystemExit("no camera stream from the phone (is it on adb? is the camera free?)")
        self.codec = av.CodecContext.create("h264", "r")
        self.frame, self.n, self.ok = None, 0, True
        self.lock = threading.Lock()
        threading.Thread(target=self._run, args=(first,), daemon=True).start()

    def _run(self, data):
        while self.ok and data:
            for pkt in self.codec.parse(data):
                for fr in self.codec.decode(pkt):
                    img = fr.to_ndarray(format="bgr24")
                    with self.lock:
                        self.frame, self.n = img, self.n + 1
            try:
                data = self.sock.recv(1 << 16)
            except OSError:
                break
        self.ok = False

    def read(self):
        with self.lock:
            return self.n, self.frame

    def close(self):
        self.ok = False
        if self.sock:
            self.sock.close()
        self.srv.terminate()
        subprocess.run(["adb", "forward", "--remove", f"tcp:{PORT}"], capture_output=True)


def chart(hist, ref, W=900, H=520, span=15.0):
    """Position change (mm) against time on a plain canvas: x red, y green, z blue."""
    img = np.full((H, W, 3), 255, np.uint8)
    now = time.time()
    pts = [(t, p) for t, p in hist if now - t <= span]
    fixes = [(t, (p - ref) * 1000) for t, p in pts if p is not None]
    lim = max(2.0, max((np.abs(d).max() for _, d in fixes), default=0) * 1.2)
    top, bot, left, right = 70, H - 40, 70, W - 20
    ymap = lambda v: int((top + bot) / 2 - v / lim * (bot - top) / 2)
    xmap = lambda t: int(right - (now - t) / span * (right - left))
    for v in (-lim, -lim / 2, 0, lim / 2, lim):
        cv2.line(img, (left, ymap(v)), (right, ymap(v)), (225, 225, 225) if v else (150, 150, 150), 1)
        cv2.putText(img, f"{v:+.1f}", (5, ymap(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    cols = [(0, 0, 220), (0, 160, 0), (220, 0, 0)]
    for k in range(3):
        for (t0, d0), (t1, d1) in zip(fixes, fixes[1:]):
            if t1 - t0 < 0.3:
                cv2.line(img, (xmap(t0), ymap(d0[k])), (xmap(t1), ymap(d1[k])), cols[k], 2)
    recent = [p for t, p in pts if now - t <= 2.0]
    good = np.array([p for p in recent if p is not None])
    rate = 100 * len(good) / max(len(recent), 1)
    jit = good.std(axis=0) * 1000 if len(good) > 2 else np.full(3, np.nan)
    cur = fixes[-1][1] if fixes else np.full(3, np.nan)
    cv2.putText(img, "position change from reference [mm]   (r = reset, q = quit)", (left, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    for k, name in enumerate("xyz"):
        cv2.putText(img, f"{name} {cur[k]:+7.2f}  jitter {jit[k]:5.2f}", (left + k * 270, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cols[k], 2)
    verdict = "no tags" if not len(good) else ("good" if rate > 95 and np.nanmax(jit) < 1.0 else
                                               "ok" if rate > 80 and np.nanmax(jit) < 3.0 else "poor")
    cv2.putText(img, f"detected {rate:3.0f} % of frames (2 s)   lighting check: {verdict}", (left, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a4-page", type=int, default=None, help="tags of this printed A4 page (else the floor track)")
    ap.add_argument("--layout", default=LAYOUT)
    ap.add_argument("--camera-id", type=int, default=2)
    ap.add_argument("--hfov-deg", type=float, default=69.4)
    ap.add_argument("--source", default=None, help="instead of the phone over USB: camera index or stream URL")
    a = ap.parse_args()
    world = tag_corners(rows=a4_page_rows(a.a4_page)) if a.a4_page else tag_corners(a.layout)

    if a.source is None:
        cam = PhoneCamera(a.camera_id)
        read = cam.read
    else:
        cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
        cnt = [0]
        def read():
            ok, img = cap.read()
            cnt[0] += ok
            return cnt[0], img if ok else None
        cam = None
    tp, last_n, ref = None, -1, None
    hist = collections.deque(maxlen=2000)
    t_fps, n_fps, fps = time.time(), 0, 0.0
    try:
        while True:
            n, img = read()
            if img is None or n == last_n:
                if cam is not None and not cam.ok:
                    break
                cv2.waitKey(5)
                continue
            last_n = n
            if tp is None:
                K, dist = intrinsics(img.shape[1], img.shape[0], a.hfov_deg)
                tp = TagPose(world, K, dist)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = tp.detect(gray)
            res = tp.pose(dets)
            res = res if res and res[3] <= 2.0 else None
            pos = res[0] if res else None
            if pos is not None and ref is None:
                ref = pos.copy()
            hist.append((time.time(), pos))
            n_fps += 1
            if time.time() - t_fps > 1.0:
                fps, n_fps, t_fps = n_fps / (time.time() - t_fps), 0, time.time()
            sharp = cv2.Laplacian(cv2.resize(gray, None, fx=0.5, fy=0.5), cv2.CV_64F).var()
            for i, c in dets.items():
                cv2.polylines(img, [c.astype(np.int32)], True, (0, 255, 0), 3)
                cv2.putText(img, str(i), tuple(c.mean(axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            txt = f"{fps:4.1f} fps  tags {len(dets)}  brightness {gray.mean():5.1f}/255  sharpness {sharp:6.0f}"
            txt += f"  fit {res[3]:.2f} mm" if res else ""
            for col, w in (((0, 0, 0), 6), ((255, 255, 255), 2)):
                cv2.putText(img, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, w)
            cv2.imshow("tags", cv2.resize(img, None, fx=0.5, fy=0.5))
            cv2.imshow("position change", chart(hist, ref if ref is not None else np.zeros(3)))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                ref = pos.copy() if pos is not None else None
    except KeyboardInterrupt:
        pass
    finally:
        if cam is not None:
            cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
