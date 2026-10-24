AprilTag floor track for IMU validation (OnePlus Nord 4 at 1.10 m), v2: small tags
=================================================================================

Family      : tag36h11
Tag size    : 70 mm black-border edge (use tag_size = 0.070 m in the detector)
White border: 8.75 mm (1 cell) around every tag is part of the tag, keep it clear
Tags        : 22 track tags = 11 checkpoints x (left + right), plus 2 test tags
IDs         : left_k = 2(k-1), right_k = 2(k-1)+1   (left_1 = 0 ... right_11 = 21)
              test_22, test_23 = for the 10 s detection check, not placed on the track

Folders
-------
tags_png/   one PNG per tag (left_1.png ... right_11.png, test_22.png, test_23.png)
            1000 x 1000 px incl. white border = 87.5 mm at 290 dpi; usable as Gazebo textures
print_A4/   4 tags per A4 page (2 x 2), 10 mm page margins, dashed cut lines, names under each tag
            ALL_tags_A4_4_per_page.pdf = all 6 pages; page_N__<names>.pdf = individual pages
tag_layout.csv    tag positions in the track frame (x along track, y to the left)
floor_layout.pdf  top-view placement plan

Placement
---------
- Checkpoint spacing along the track: 1.00 m, tag center to tag center.
- Left/right: tag centers at y = +0.20 m and y = -0.20 m -> 0.40 m apart (center to center).
  Inner black edges are 0.33 m apart; outer black edges span 0.47 m.
- Every arrow points the same way: +x, start -> end.
- Tick marks on each cut piece mark the tag center lines for aligning with tape.

Camera
------
- Shutter 1/250 s or faster at walking speed (~0.5 m/s): blur must stay below ~4 mm.
- Video stabilization (EIS) off, focus and exposure locked.

Printing
--------
- Print at 100 % / "Actual size". Disable "Fit to page".
- Check the 50 mm scale bar with a ruler before printing everything.
- Matte paper. Tape all four edges of each cut piece flat.
