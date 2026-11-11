"""Marks: snap eye points onto the plain spectrogram, densify, fit, write, check. Stdlib + numpy
only; never geomedia.

  python -I marks.py --check MEDIA_DIR MANIFEST      every case's marks file exists and parses,
                                                     every labelled aircraft run is covered
  python -I marks.py --hash MEDIA_DIR MANIFEST OUT   sha256 of every marks file (marking order)
"""
import datetime
import hashlib
import json
import os
import sys
import tomllib

import numpy as np

SNAP_WIN = 0.02      # ± fraction searched around an eye mark
SNAP_WIDE = 0.06     # ± second pass for a mark whose ridge lies beyond SNAP_WIN
AVG_FRAMES = 2       # snap on power averaged over ±this many frames (±0.25 s)
PEAK_DB = 6.0        # a snap target must stand this far above the median of its ±BG_WIN neighbourhood
BG_WIN = 0.10        # (fixed, wider than a smeared ridge, so the ridge itself doesn't raise it)
JUMP_MAX = 0.01      # a mark's correction may differ from its neighbours' by at most this
C = 343.0
ROW_S = 5.0


def _peak(db, times, freqs, t, f, window):
    """(f, flag) for one mark: the brightest bin within ±`window` of f in the power averaged over
    ±AVG_FRAMES frames around t, refined by a parabola; flag `edge` when that bin is the window's edge with the spectrum still
    rising beyond it (the ridge lies outside the window), `no_peak` when nothing in the window
    stands PEAK_DB above the median of the ±BG_WIN neighbourhood (the brightest bin is noise), `out_of_range` off the
    spectrogram."""
    hop = times[1] - times[0] if len(times) > 1 else 0.125
    if not len(times) or t < times[0] - hop or t > times[-1] + hop or f <= 0 or f > freqs[-1]:
        return f, "out_of_range"
    j = int(np.argmin(np.abs(times - t)))
    lo, hi = np.searchsorted(freqs, f * (1 - window)), np.searchsorted(freqs, f * (1 + window))
    if hi - lo < 3:
        return f, "out_of_range"
    # Power averaged over ±AVG_FRAMES frames: one frame's noise is too rough to pick a peak in.
    a0, a1 = max(0, j - AVG_FRAMES), min(len(times), j + AVG_FRAMES + 1)
    row = 10 * np.log10(np.mean(10 ** (db[a0:a1] / 10), axis=0))
    k = lo + int(np.argmax(row[lo:hi]))
    beyond_lo = row[max(0, lo - 3):lo]
    beyond_hi = row[hi:hi + 3]
    rising_out = (k - lo < 2 and len(beyond_lo) and beyond_lo.max() > row[k]) or (
        hi - 1 - k < 2 and len(beyond_hi) and beyond_hi.max() > row[k]
    )
    if rising_out:
        return f, "edge"
    b0, b1 = np.searchsorted(freqs, f * (1 - BG_WIN)), np.searchsorted(freqs, f * (1 + BG_WIN))
    if row[k] - float(np.median(row[b0:b1])) < PEAK_DB:
        # Nothing stands out in the window: the brightest bin is noise, not the ridge.
        return f, "no_peak"
    off = 0.0
    if 0 < k < len(row) - 1:
        a, b, c = row[k - 1], row[k], row[k + 1]
        den = a - 2 * b + c
        off = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den != 0 else 0.0
    return float(freqs[k] + off * (freqs[1] - freqs[0])), None


def snap(db, times, freqs, eye, window=SNAP_WIN, wide=SNAP_WIDE):
    """Each eye (t, f) → {t, f_eye, f, flags}: f moved to the brightest bin within ±`window`
    (`_peak`). A mark whose ridge lies beyond that window (`edge`: eye readings on steep curves run
    up to ~5 % off) gets a second pass within ±`wide`, flagged `wide_snap`; still beyond it, it
    keeps its eye pitch, flagged `edge`. Then a mark whose correction (f / f_eye) differs from its
    neighbours' median correction by more than JUMP_MAX keeps its eye pitch (`snap_rejected`): a
    consistent eye bias passes, an isolated jump onto another tone doesn't. Marks off the
    spectrogram are `out_of_range` and not snapped. The marker confirms every snap on the
    overlay."""
    out = []
    for t, f in eye:
        m = {"t": float(t), "f_eye": float(f), "f": float(f), "flags": []}
        g, flag = _peak(db, times, freqs, t, f, window)
        if flag in ("edge", "no_peak"):
            g, flag = _peak(db, times, freqs, t, f, wide)
            if flag is None:
                m["flags"].append("wide_snap")
        if flag:
            m["flags"].append(flag)
        else:
            m["f"] = g
        out.append(m)
    ok = [i for i, m in enumerate(out) if not {"out_of_range", "edge", "no_peak"} & set(m["flags"])]
    corr = {i: np.log(out[i]["f"] / out[i]["f_eye"]) for i in ok}
    for n, i in enumerate(ok):
        nb = [corr[ok[n - 1]]] if n > 0 else []
        nb += [corr[ok[n + 1]]] if n + 1 < len(ok) else []
        if nb and abs(corr[i] - float(np.median(nb))) > np.log1p(JUMP_MAX):
            out[i]["flags"].append("snap_rejected")
    for i in ok:
        if "snap_rejected" in out[i]["flags"]:
            out[i]["f"] = out[i]["f_eye"]
    return out


UNSNAPPED = {"edge", "no_peak", "snap_rejected", "out_of_range"}
MIN_CLEAN = 4


def truth_points(snapped):
    """The (t, f) a truth curve is built from: only marks that snapped onto a peak (a `wide_snap`
    counts; `edge`, `no_peak`, `snap_rejected` and `out_of_range` marks keep the eye's pitch,
    which on a steep stretch can be several percent off, so they are left to interpolation).
    With fewer than MIN_CLEAN clean marks every mark is kept and the curve is flagged
    `few_clean_marks`. Returns (points, curve flags)."""
    clean = [(m["t"], m["f"]) for m in snapped if not UNSNAPPED & set(m["flags"])]
    if len(clean) >= MIN_CLEAN:
        return clean, []
    keep = [(m["t"], m["f"]) for m in snapped if "out_of_range" not in m["flags"]]
    return keep, ["few_clean_marks"]


def densify(points, hop=0.125):
    """Monotone piecewise-cubic (Fritsch–Carlson) through `points` (t, f), sampled every `hop`."""
    p = sorted((float(t), float(f)) for t, f in points)
    t = np.array([q[0] for q in p])
    f = np.array([q[1] for q in p])
    if len(t) < 2:
        return [(float(a), float(b)) for a, b in zip(t, f)]
    h = np.diff(t)
    s = np.diff(f) / h
    m = np.empty(len(t))
    m[0], m[-1] = s[0], s[-1]
    for i in range(1, len(t) - 1):
        if s[i - 1] * s[i] <= 0:
            m[i] = 0.0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / s[i - 1] + w2 / s[i])
    out = []
    for tq in np.arange(t[0], t[-1] + 1e-9, hop):
        i = min(int(np.searchsorted(t, tq, side="right")) - 1, len(t) - 2)
        u = (tq - t[i]) / h[i]
        h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
        h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
        out.append((float(tq), float(h00 * f[i] + h10 * h[i] * m[i] + h01 * f[i + 1] + h11 * h[i] * m[i + 1])))
    return out


def model(t, f0, v, d, t0, c=C):
    """Receive-time constant-velocity Doppler (deliberately not geomedia's emission-time model)."""
    x = v * (t - t0)
    return f0 / (1 + (v / c) * x / np.sqrt(d * d + x * x))


def fit_cv(points, c=C):
    """{f0, v, d, t0, rms_pct, flags}: least squares of `model`. Start: the best of a small grid
    of (v, d, t0) — f₀ solved in closed form for each — then damped Gauss–Newton (Levenberg–
    Marquardt; a step is taken only when it lowers the error) inside physical bounds. A partial
    pass (CPA near an end) is fitted without running off. `implausible` when v > 300 m/s or
    d < 50 m; `cpa_outside_curve` when the fitted CPA lies beyond the traced points."""
    t = np.array([p[0] for p in points], dtype=float)
    f = np.array([p[1] for p in points], dtype=float)
    o = np.argsort(t)
    t, f = t[o], f[o]
    lo_b = np.array([0.5 * f.min(), 5.0, 20.0, t[0] - 300.0])
    hi_b = np.array([2.0 * f.max(), 350.0, 30000.0, t[-1] + 300.0])

    def shape(v, d, t0):
        return model(t, 1.0, v, d, t0, c=c)

    def with_f0(v, d, t0):
        g = shape(v, d, t0)
        return np.array([float(np.dot(f, g) / np.dot(g, g)), v, d, t0])

    def sse(p):
        r = f - model(t, *p, c=c)
        return float(np.dot(r, r))

    slope = np.gradient(f, t)
    t_steep = t[int(np.argmin(slope))]
    span = t[-1] - t[0]
    starts = [
        with_f0(v, d, t0)
        for v in (40.0, 70.0, 120.0, 200.0)
        for d in (300.0, 800.0, 2000.0, 5000.0)
        for t0 in (t_steep, t[0] + 0.5 * span, t[-1], t[-1] + 0.2 * span, t[0], t[0] - 0.2 * span)
    ]
    p = min(starts, key=sse)
    lam, e = 1e-3, sse(p)
    for _ in range(200):
        r = f - model(t, *p, c=c)
        jac = np.empty((len(t), 4))
        for i in range(4):
            dp = np.zeros(4)
            dp[i] = 1e-6 * max(1.0, abs(p[i]))
            jac[:, i] = (model(t, *(p + dp), c=c) - model(t, *p, c=c)) / dp[i]
        a = jac.T @ jac
        g = jac.T @ r
        improved = False
        for _ in range(12):
            try:
                step = np.linalg.solve(a + lam * np.diag(np.diag(a) + 1e-12), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            q = np.clip(p + step, lo_b, hi_b)
            eq = sse(q)
            if eq < e:
                p, e, lam, improved = q, eq, max(lam / 3, 1e-9), True
                break
            lam *= 10
        if not improved or e == 0:
            break
    rms = float(np.sqrt(np.mean((f / model(t, *p, c=c) - 1) ** 2)) * 100)
    flags = ["implausible"] if p[1] > 300 or p[2] < 50 else []
    if not t[0] <= p[3] <= t[-1]:
        # CPA beyond the traced curve: speed and distance are poorly pinned (a partial pass).
        flags.append("cpa_outside_curve")
    return {"f0": float(p[0]), "v": float(p[1]), "d": float(p[2]), "t0": float(p[3]), "rms_pct": rms, "flags": flags}


def write_marks(path, file, verdict, curves, extra=None):
    """One file's marks (schema in docs/superpowers/plans/2026-11-06-real-curve-test.md)."""
    doc = {
        "file": file,
        "verdict": verdict,
        "marked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "before_discovery": True,
        "curves": curves,
        "no_curve_stretches": [],
    }
    doc.update(extra or {})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def aircraft_runs(labels, row_s=ROW_S):
    """Consecutive "1" rows of one label column as (start s, end s)."""
    runs, start = [], None
    for i, v in enumerate(list(labels) + ["0"]):
        if v.strip() == "1" and start is None:
            start = i
        elif v.strip() != "1" and start is not None:
            runs.append((start * row_s, i * row_s))
            start = None
    return runs


def uncovered(runs, marks_doc):
    """The aircraft runs that no curve's points and no no-curve stretch overlap."""
    spans = [(min(p[0] for p in c["points"]), max(p[0] for p in c["points"]))
             for c in marks_doc.get("curves", []) if c.get("points")]
    spans += [tuple(s) for s in marks_doc.get("no_curve_stretches", [])]
    return [r for r in runs if not any(a < r[1] and b > r[0] for a, b in spans)]


def _cases(manifest):
    with open(manifest, "rb") as f:
        return tomllib.load(f)["case"]


def _resolve(media_dir, p):
    return os.path.join(media_dir, p[len("media:"):]) if p.startswith("media:") else p


def check(media_dir, manifest):
    """Problems: missing or unreadable marks, bad verdicts, uncovered aircraft runs."""
    import csv

    problems, n = [], 0
    for c in _cases(manifest):
        p = _resolve(media_dir, c["marks"])
        n += 1
        try:
            with open(p) as f:
                doc = json.load(f)
        except (OSError, ValueError) as e:
            problems.append(f"{c['id']}: {e}")
            continue
        if doc.get("verdict") not in ("curves", "no_curve"):
            problems.append(f"{c['id']}: verdict {doc.get('verdict')!r}")
        if "labels_column" in c:
            with open(_resolve(media_dir, c["labels_csv"]), newline="") as f:
                rows = list(csv.reader(f))[1:]
            col = [r[c["labels_column"]] for r in rows if r]
            for r in uncovered(aircraft_runs(col), doc):
                problems.append(f"{c['id']}: aircraft run {r[0]:.0f}–{r[1]:.0f} s not marked")
    return n, problems


def hash_all(media_dir, manifest, out):
    with open(out, "w") as f:
        for c in _cases(manifest):
            p = _resolve(media_dir, c["marks"])
            with open(p, "rb") as g:
                f.write(f"{hashlib.sha256(g.read()).hexdigest()}  {c['marks']}\n")


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        n, probs = check(sys.argv[2], sys.argv[3])
        for p in probs:
            print(p)
        print(f"{n} files, {len(probs)} errors")
        sys.exit(1 if probs else 0)
    if sys.argv[1] == "--hash":
        hash_all(sys.argv[2], sys.argv[3], sys.argv[4])
        print("written", sys.argv[4])
