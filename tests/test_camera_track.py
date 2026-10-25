"""Camera position from the floor tags: render the real tag artwork at a known camera pose, detect, solve, compare.

Run:  pytest -q tests/test_camera_track.py
"""
import csv, os, sys
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from camera_track import LAYOUT, tag_corners, intrinsics, TagPose, heading_tilt

PNG_DIR = os.path.join(ROOT, "simulation", "apriltags_imu_track_70mm", "apriltags_imu_track", "tags_png")
W, H = 1920, 1080


def look(pitch_deg, heading_deg=0.0):
    """Camera-to-world rotation: optical axis pitch_deg below horizontal towards heading_deg, image upright."""
    th, ps = np.radians(pitch_deg), np.radians(heading_deg)
    z = np.array([np.cos(th) * np.cos(ps), np.cos(th) * np.sin(ps), -np.sin(th)])
    x = np.array([np.sin(ps), -np.cos(ps), 0.0])
    return np.column_stack([x, np.cross(z, x), z])


def render(pos, R_wc, K, size=(W, H)):
    """Grey floor with every tag's PNG (white border included) warped in at its surveyed place."""
    with open(LAYOUT) as f:
        rows = list(csv.DictReader(f))
    w, h_img = size
    img = np.full((h_img, w, 3), 150, np.uint8)
    R_cw = R_wc.T
    rvec, tvec = cv2.Rodrigues(R_cw)[0], -R_cw @ pos
    up, right = np.array([1.0, 0, 0]), np.array([0, -1.0, 0])
    for r in rows:
        c = np.array([float(r["x_m"]), float(r["y_m"]), 0.0])
        h = float(r["tag_size_m"]) * 1.25 / 2                  # PNG extent = black square + 1-cell white border
        outer = np.array([c - h * right + h * up, c + h * right + h * up, c + h * right - h * up, c - h * right - h * up])
        if np.any((outer - pos) @ R_wc[:, 2] < 0.1):
            continue
        uv = cv2.projectPoints(outer, rvec, tvec, K, None)[0].reshape(4, 2).astype(np.float32)
        tag = cv2.imread(os.path.join(PNG_DIR, r["name"] + ".png"))
        s = tag.shape[0] - 1
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [s, 0], [s, s], [0, s]]), uv)
        warped = cv2.warpPerspective(tag, M, size, flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(np.full(tag.shape[:2], 255, np.uint8), M, size) > 128
        img[mask] = warped[mask]
    return img


def portrait(R_wc):
    x, y, z = R_wc.T
    return np.column_stack([y, -x, z])


def _solve(pos, R_wc):
    K, dist = intrinsics(W, H, 69.4)
    tp = TagPose(tag_corners(LAYOUT), K, dist)
    dets = tp.detect(render(np.array(pos), R_wc, K))
    return dets, tp.pose(dets)


@pytest.mark.parametrize("x", [0.3, 2.6, 5.1, 7.7])
def test_recommended_hold_is_millimetre_accurate(x):
    """Portrait, 58 deg below horizontal, looking along +x: several checkpoints in view."""
    pos = np.array([x, 0.03, 1.10])
    dets, (p, R, rms, _) = _solve(pos, portrait(look(58.0)))
    hd, tl = heading_tilt(R)
    assert len(dets) >= 4 and np.linalg.norm(p - pos) < 0.005 and abs(tl - 58.0) < 0.3 and abs(hd) < 0.3 and rms < 1.0


def test_top_down_single_checkpoint_is_centimetre_level():
    """Straight down only one checkpoint (2 tags) is in view: still a fix, but only to several cm (9 cm here)."""
    pos = np.array([3.0, 0.05, 1.10])
    dets, (p, R, rms, _) = _solve(pos, look(90.0))
    assert len(dets) == 2 and np.linalg.norm(p - pos) < 0.15 and rms < 2.0
