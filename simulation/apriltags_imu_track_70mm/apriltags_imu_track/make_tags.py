"""Generate AprilTag tag36h11 markers for the IMU validation track (v2, small tags).

Layout: 11 checkpoints, 1.0 m apart along the track (x), each with a
left and right tag at y = +0.20 m and y = -0.20 m (0.40 m center to center).
Tag black-border edge = 70 mm, printed 4 per A4 page (2 x 2, with cut lines).
IDs: left_k = 2(k-1), right_k = 2(k-1)+1.  IDs 22, 23 = test tags for the
detection check (fill the last page).
"""
import csv
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

OUT = "/home/claude/apriltags_imu_track"
PNG_DIR = os.path.join(OUT, "tags_png")
PDF_DIR = os.path.join(OUT, "print_A4")

N_CP = 11
SPACING_M = 1.0
HALF_GAP_M = 0.20
TAG_MM = 70.0             # black-border edge length (detector tag_size)
CELLS = 8                 # tag36h11: 6 data + 1 black border each side
QUIET_CELLS = 1           # white border, 1 cell each side
CELL_PX = 100             # PNG resolution per cell

PAGE_MARGIN_MM = 10.0     # keep everything away from printer edge distortion
COLS, ROWS = 2, 2

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def tag_image(tag_id: int) -> np.ndarray:
    """Black-border tag with exact integer pixels per cell (no white border)."""
    return cv2.aruco.generateImageMarker(dictionary, tag_id, CELLS * CELL_PX, borderBits=1)


def track_tags():
    for k in range(1, N_CP + 1):
        x = (k - 1) * SPACING_M
        yield f"left_{k}", 2 * (k - 1), k, x, +HALF_GAP_M
        yield f"right_{k}", 2 * (k - 1) + 1, k, x, -HALF_GAP_M


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(PNG_DIR)
    os.makedirs(PDF_DIR)

    rows = list(track_tags())
    n_track = len(rows)
    per_page = COLS * ROWS
    n_slots = -(-n_track // per_page) * per_page
    test = [(f"test_{i}", i, None, None, None) for i in range(n_track, n_slots)]
    all_tags = rows + test

    # ---------- 1. PNG folder ----------
    total_cells = CELLS + 2 * QUIET_CELLS
    total_mm = TAG_MM / CELLS * total_cells
    dpi = total_cells * CELL_PX / (total_mm / 25.4)
    for name, tid, *_ in all_tags:
        img = tag_image(tid)
        pad = QUIET_CELLS * CELL_PX
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        Image.fromarray(img).save(os.path.join(PNG_DIR, f"{name}.png"), dpi=(dpi, dpi))

    # ---------- 2. A4 print pages, 2 x 2 ----------
    W, H = A4
    m = PAGE_MARGIN_MM * mm
    cw = (W - 2 * m) / COLS            # 95 mm
    ch = (H - 2 * m) / ROWS            # 138.5 mm
    tag = TAG_MM * mm
    tag_top_pad = 14 * mm              # from cell top edge to tag top edge

    def draw_cell(c, col, row, name, tid, k, x, y):
        left = m + col * cw
        top = H - m - row * ch
        cx = left + cw / 2
        tx, ty = cx - tag / 2, top - tag_top_pad - tag
        c.drawImage(ImageReader(Image.fromarray(tag_image(tid))), tx, ty, tag, tag)

        # alignment ticks at the cut-piece edges, on the tag centre lines
        c.setLineWidth(0.5)
        tcy = ty + tag / 2
        c.line(cx, top, cx, top - 3 * mm)
        c.line(left, tcy, left + 2.5 * mm, tcy)
        c.line(left + cw, tcy, left + cw - 2.5 * mm, tcy)

        # labels (start 12 mm below the tag, outside the 8.75 mm quiet zone)
        ly = ty - 12 * mm
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(cx, ly, name)
        c.setFont("Helvetica", 7.5)
        if k is not None:
            c.drawCentredString(cx, ly - 5.5 * mm, f"tag36h11 ID {tid} | checkpoint {k}")
            c.drawCentredString(cx, ly - 9.5 * mm, f"x = {x:.2f} m, y = {y:+.2f} m | edge 70 mm")
        else:
            c.drawCentredString(cx, ly - 5.5 * mm, f"tag36h11 ID {tid} | TEST tag (not on track)")
            c.drawCentredString(cx, ly - 9.5 * mm, "for the 10 s detection check | edge 70 mm")

        # +x arrow (tag 'up' = track forward)
        ay = ly - 24 * mm
        c.setLineWidth(1.2)
        c.line(cx, ay, cx, ay + 7 * mm)
        p = c.beginPath()
        p.moveTo(cx - 2.5 * mm, ay + 7 * mm)
        p.lineTo(cx + 2.5 * mm, ay + 7 * mm)
        p.lineTo(cx, ay + 11 * mm)
        p.close()
        c.drawPath(p, fill=1, stroke=0)
        c.setFont("Helvetica", 6.5)
        c.drawString(cx + 4 * mm, ay + 3 * mm, "+x (start -> end)")

        # 50 mm scale bar
        sy = ay - 7 * mm
        sx = cx - 25 * mm
        c.setLineWidth(0.8)
        c.line(sx, sy, sx + 50 * mm, sy)
        for t in range(0, 51, 10):
            c.line(sx + t * mm, sy, sx + t * mm, sy + (2.5 if t % 50 == 0 else 1.2) * mm)
        c.setFont("Helvetica", 6)
        c.drawCentredString(cx, sy - 3.5 * mm, "must measure 50 mm")

    def draw_cut_lines(c):
        c.setLineWidth(0.3)
        c.setDash(2, 2)
        for i in range(COLS + 1):
            xx = m + i * cw
            c.line(xx, m, xx, H - m)
        for j in range(ROWS + 1):
            yy = H - m - j * ch
            c.line(m, yy, W - m, yy)
        c.setDash()
        c.setFont("Helvetica", 6)
        c.drawCentredString(W / 2, m / 2 - 1, "print at 100 % / Actual size, no 'Fit to page'  |  cut along dashed lines")

    combined = canvas.Canvas(os.path.join(PDF_DIR, "ALL_tags_A4_4_per_page.pdf"), pagesize=A4)
    for p in range(0, len(all_tags), per_page):
        chunk = all_tags[p:p + per_page]
        fname = "page_%d__%s.pdf" % (p // per_page + 1, "_".join(t[0] for t in chunk))
        single = canvas.Canvas(os.path.join(PDF_DIR, fname), pagesize=A4)
        for c in (single, combined):
            draw_cut_lines(c)
            for i, t in enumerate(chunk):
                draw_cell(c, i % COLS, i // COLS, *t)
            c.showPage()
        single.save()
    combined.save()

    # ---------- 3. layout CSV + floor plan ----------
    with open(os.path.join(OUT, "tag_layout.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "tag_id", "checkpoint", "x_m", "y_m", "tag_size_m"])
        for name, tid, k, x, y in rows:
            w.writerow([name, tid, k, f"{x:.2f}", f"{y:+.2f}", TAG_MM / 1000])

    PW, PH = A4[1], A4[0]
    c = canvas.Canvas(os.path.join(OUT, "floor_layout.pdf"), pagesize=(PW, PH))
    s = (PW - 40 * mm) / 10.6
    ox, oy = 25 * mm, PH / 2
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, PH - 15 * mm, "Floor layout: 11 checkpoints x 2 tags (top view), 70 mm tags")
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, PH - 21 * mm,
                 "Checkpoint spacing 1.00 m along x (center to center). Left/right tag centers at y = +0.20 m / -0.20 m "
                 "(0.40 m apart). Tag 'up' points along +x.")
    c.setDash(3, 3)
    c.line(ox - 5 * mm, oy, ox + 10.3 * s, oy)
    c.setDash()
    for name, tid, k, x, y in rows:
        cx, cy = ox + x * s, oy + y * s * 3
        side = TAG_MM / 1000 * s * 3
        c.rect(cx - side / 2, cy - side / 2, side, side, fill=0)
        c.setFont("Helvetica", 7)
        c.drawCentredString(cx, cy + side / 2 + 3, name)
        c.drawCentredString(cx, cy - side / 2 - 9, f"id {tid}")
    for k in range(N_CP):
        c.setFont("Helvetica", 8)
        c.drawCentredString(ox + k * s, oy - 0.2 * s * 3 - 30, f"x={k}.0 m")
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(20 * mm, 12 * mm, "Note: drawn with y (and tag size) exaggerated 3x for readability. Dashed line = track centerline.")
    c.save()


if __name__ == "__main__":
    main()
