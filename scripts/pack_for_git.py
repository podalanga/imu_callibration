"""Prepare recordings for a git push: gzip every CSV over the size limit and keep the original out of git.

  python scripts/pack_for_git.py                 # runs/ and data/, files over 50 MB
  python scripts/pack_for_git.py --limit-mb 20

GitHub rejects files over 100 MB (and warns over 50 MB); a long recording's imu.csv / raw_gyro.csv are
~2 MB per minute. Numeric CSV compresses about 3-4x, and every loader in this repo reads `<name>.csv.gz`
when `<name>.csv` is absent (nav.io.csv_path), so a fresh clone works without unpacking anything.
The uncompressed files stay on this machine and are listed in .gitignore.
"""
import argparse, gzip, os, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARK = "# large CSVs, committed as .csv.gz (scripts/pack_for_git.py)"

ap = argparse.ArgumentParser()
ap.add_argument("--limit-mb", type=float, default=50.0)
ap.add_argument("dirs", nargs="*", default=["runs", "data"])
a = ap.parse_args()

big = []
for d in a.dirs:
    for base, _, files in os.walk(os.path.join(ROOT, d)):
        for f in files:
            p = os.path.join(base, f)
            if f.endswith(".csv") and os.path.getsize(p) > a.limit_mb * 1e6:
                gz = p + ".gz"
                if not os.path.exists(gz) or os.path.getmtime(gz) < os.path.getmtime(p):
                    with open(p, "rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                print(f"{os.path.relpath(p, ROOT)}: {os.path.getsize(p) / 1e6:.0f} MB -> {os.path.getsize(gz) / 1e6:.0f} MB gz")
                big.append(os.path.relpath(p, ROOT).replace(os.sep, "/"))

gi = os.path.join(ROOT, ".gitignore")
lines = open(gi).read().splitlines() if os.path.exists(gi) else []
known = set(lines)
new = [b for b in big if b not in known]
if new:
    if MARK not in known:
        lines += ["", MARK]
    lines += new
    with open(gi, "w") as f:
        f.write("\n".join(lines) + "\n")
print(f"{len(big)} large file(s); {len(new)} added to .gitignore")
