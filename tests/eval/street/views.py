"""Plain views of a WAV for blind marking (numpy + the marking tool's helpers; never geomedia).

  views.py WAV OUTDIR

The WAV is mono 16 kHz, made from the recording with
`ffmpeg -i IN -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le OUT.wav` (`wavio` reads WAV only).

Writes strip-NNN.png (log-frequency spectrogram, 120-s strips, each bin's whole-file median
removed; white at the 99.5th percentile, black 25 dB below), levels.png (octave-band levels, 1-s
frames, each band's median removed) and levels.tsv (the same numbers, 5-s means).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aerosonic"))
import numpy as np  # noqa: E402
import spec  # noqa: E402
import wavio  # noqa: E402

wav, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
sr, x = wavio.read(wav)
x = x.astype(np.float64)
dur = len(x) / sr

# ---- spectrogram strips, whole-file median removed -------------------------------------------
F_MAX = 3400.0
times, freqs, db = spec.stft(x, sr, 0.0, f_max=F_MAX, remove_median=True)
STRIP = 120.0
k = 0
t = 0.0
while t < dur:
    t1 = min(dur, t + STRIP)
    w, h = spec.overview_size(t1 - t)
    img = spec._canvas(w, h)
    pw = w - spec.ML - spec.MR
    col_t = t + (np.arange(pw) + 0.5) / spec.OV_PX_S
    jj = np.clip(np.searchsorted(times, col_t), 0, len(times) - 1)
    row_f = 30.0 * (F_MAX / 30.0) ** (1 - (np.arange(spec.OV_H) + 0.5) / spec.OV_H)
    qq = np.clip(np.searchsorted(freqs, row_f), 0, len(freqs) - 1)
    patch = db[np.ix_(jj, qq)]
    # Robust scale: white at the 99.5th percentile (clicks saturate instead of blacking out the
    # strip), black 25 dB below it.
    hi = float(np.percentile(patch, 99.5))
    g = (np.clip((patch - (hi - 25.0)) / 25.0, 0, 1) * 255).T.astype(np.uint8)
    img[spec.MT:spec.MT + spec.OV_H, spec.ML:spec.ML + pw] = g[:, :, None]
    for f in (50, 100, 200, 400, 800, 1600, 3200):
        y = spec.MT + int(spec.OV_H * (1 - np.log(f / 30.0) / np.log(F_MAX / 30.0)))
        img[y, spec.ML:spec.ML + pw] = (0, 120, 255)
        spec.text(img, 4, y - 5, str(f))
    for s5 in range(int(np.ceil(t / 5.0)), int(t1 / 5.0) + 1):
        x_ = spec.ML + int((5 * s5 - t) * spec.OV_PX_S)
        if spec.ML <= x_ < spec.ML + pw:
            img[spec.MT:spec.MT + spec.OV_H, x_] = (0, 90, 0) if s5 % 2 else (0, 170, 0)
            if s5 % 2 == 0:
                spec.text(img, x_ - 8, spec.MT + spec.OV_H + 8, f"{5 * s5}s")
    spec.write_png(os.path.join(outdir, f"strip-{int(t):03d}.png"), img)
    k += 1
    t += STRIP

# ---- octave-band levels ------------------------------------------------------------------------
FR = 1.0
n = int(FR * sr)
nfr = int(len(x) // n)
win = np.hanning(n)
fq = np.fft.rfftfreq(n, 1 / sr)
bands = [(c / np.sqrt(2), c * np.sqrt(2), f"{c}") for c in (63, 125, 250, 500, 1000, 2000, 4000)]
bands.append((100.0, 1600.0, "mid"))
lev = np.empty((nfr, len(bands)))
for j in range(nfr):
    p = np.abs(np.fft.rfft(x[j * n:(j + 1) * n] * win)) ** 2
    for b, (lo, hi, _) in enumerate(bands):
        sel = (fq >= lo) & (fq < hi)
        lev[j, b] = 10 * np.log10(p[sel].sum() + 1e-20)
rel = lev - np.median(lev, axis=0, keepdims=True)

with open(os.path.join(outdir, "levels.tsv"), "w") as fh:
    fh.write("t0\t" + "\t".join(b[2] for b in bands) + "\n")
    for a in range(0, nfr, 5):
        m = rel[a:a + 5].mean(axis=0)
        fh.write(f"{a}\t" + "\t".join(f"{v:+.1f}" for v in m) + "\n")

PX_S = 3
LANE = 110  # px per band lane, ±? dB scaled to lane
W = spec.ML + nfr * PX_S + spec.MR
H = spec.MT + LANE * len(bands) + spec.MB
img = spec._canvas(W, H)
DB_SPAN = 30.0  # lane spans −10 … +20 dB around the median
for b, (_, _, name) in enumerate(bands):
    top = spec.MT + b * LANE
    y0 = top + int(LANE * 20 / DB_SPAN)  # the median line
    img[y0, spec.ML:W - spec.MR] = (60, 60, 60)
    y10 = top + int(LANE * 10 / DB_SPAN)
    img[y10, spec.ML:W - spec.MR] = (40, 40, 90)
    img[top, spec.ML:W - spec.MR] = (0, 120, 255)
    spec.text(img, 4, top + 4, name)
    prev = None
    for j in range(nfr):
        v = np.clip(rel[j, b], -10, 20)
        y = top + int(LANE * (20 - v) / DB_SPAN)
        y = min(max(y, top + 1), top + LANE - 1)
        x_ = spec.ML + j * PX_S
        if prev is not None:
            lo_, hi_ = sorted((prev, y))
            img[lo_:hi_ + 1, x_] = (255, 220, 0)
        img[y, x_:x_ + PX_S] = (255, 220, 0)
        prev = y
for s in range(0, nfr + 1, 10):
    x_ = spec.ML + s * PX_S
    if x_ < W - spec.MR:
        img[spec.MT:H - spec.MB, x_] = (0, 90, 0) if s % 30 else (0, 170, 0)
        if s % 30 == 0:
            spec.text(img, x_ - 8, H - spec.MB + 8, f"{s}s")
spec.write_png(os.path.join(outdir, "levels.png"), img)
print(f"{dur:.2f} s at {sr} Hz; {k} strips; {nfr} level frames")
