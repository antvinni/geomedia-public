"""Which 1-hour recording is which column of environment_class_mappings.csv?

The label file has one column per recording and one row per 5 s, with 0 (no aircraft),
1 (aircraft) and "ignore". check_mapping() measures, for every (column, file) pair, the mean
100–1 500 Hz band level of the column's aircraft rows minus its no-aircraft rows; each column's
own file must beat every other file by `min_gap_db`, and each file must be as long as its
column. Plain stdlib + numpy; no geomedia.

  python -I envmap.py ENV_DIR MAPPINGS_CSV   prints the matrix and the diagonal, exits 1 on failure
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # siblings, also under python -I
import wavio  # noqa: E402

ROW_S = 5.0
BAND = (100.0, 1500.0)


def env_files(env_dir):
    """The recordings in label-column order: WAVs sorted by name."""
    return sorted(f for f in os.listdir(env_dir) if f.lower().endswith(".wav") and not f.startswith("."))


def read_mappings(path):
    """Columns of stripped labels."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    body = [[c.strip() for c in r] for r in rows[1:] if r]
    return [[r[k] for r in body] for k in range(len(rows[0]))]


def row_levels(path, n_rows):
    """Mean band power (dB) of each 5-s row of the recording, and its duration."""
    sr, x = wavio.read(path)
    n = int(ROW_S * sr)
    out = []
    for r in range(n_rows):
        seg = x[r * n:(r + 1) * n]
        if len(seg) < n // 2:
            out.append(np.nan)
            continue
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        f = np.fft.rfftfreq(len(seg), 1 / sr)
        band = spec[(f >= BAND[0]) & (f <= BAND[1])]
        out.append(10 * np.log10(band.mean() + 1e-20))
    return np.array(out), len(x) / sr


def check_mapping(env_dir, mappings_csv, min_gap_db=3.0, verbose=False):
    """[(column, file, own aircraft-minus-silence dB)]; ValueError unless every column's own file
    beats every other file by `min_gap_db` and every file matches its column's length (±10 s)."""
    cols = read_mappings(mappings_csv)
    files = env_files(env_dir)
    if len(files) != len(cols):
        raise ValueError(f"{len(files)} recordings for {len(cols)} label columns")
    n_rows = len(cols[0])
    levels, durs = [], []
    for fn in files:
        lv, d = row_levels(os.path.join(env_dir, fn), n_rows)
        levels.append(lv)
        durs.append(d)
    m = np.full((len(cols), len(files)), np.nan)
    for k, c in enumerate(cols):
        a = np.array([v == "1" for v in c])
        s = np.array([v == "0" for v in c])
        for j, lv in enumerate(levels):
            m[k, j] = np.nanmean(lv[a]) - np.nanmean(lv[s])
    if verbose:
        print("column × file (aircraft − silence, dB):")
        print("      " + " ".join(f"{fn[:10]:>10}" for fn in files))
        for k in range(len(cols)):
            print(f"  {k:>2}  " + " ".join(f"{v:10.2f}" for v in m[k]))
    problems = []
    for k in range(len(cols)):
        others = np.delete(m[k], k)
        if not np.all(m[k, k] >= others + min_gap_db):
            problems.append(f"column {k}: own file {files[k]} {m[k, k]:.2f} dB vs best other {np.nanmax(others):.2f} dB")
        want = len(cols[k]) * ROW_S
        if abs(durs[k] - want) > 10.0:
            problems.append(f"{files[k]}: {durs[k]:.0f} s, column {k} has {want:.0f} s of rows")
    if problems:
        raise ValueError("; ".join(problems))
    return [(k, files[k], float(m[k, k])) for k in range(len(cols))]


if __name__ == "__main__":
    try:
        for k, fn, d in check_mapping(sys.argv[1], sys.argv[2], verbose=True):
            print(f"column {k} → {fn}: {d:+.2f} dB")
    except ValueError as e:
        print(f"mapping check FAILED: {e}")
        sys.exit(1)
