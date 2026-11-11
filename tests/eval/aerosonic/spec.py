"""The plain marking spectrogram and its images. Stdlib + numpy only; never geomedia.

stft: 0.5 s Hann window, 0.125 s hop, zero-padded ×4 (display and sub-bin snapping), dB; for display
each bin's median over the segment is removed, for snapping not. overview_png: log-frequency 30 Hz–3.4 kHz with labelled
gridlines; zoom_png: a linear crop with gridlines every 1 s and every 1 % of its centre
frequency, optionally overlaying point sets (eye marks, snapped marks).
"""
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # siblings, also under python -I
import wavio  # noqa: E402

WIN_S, HOP_S, PAD = 0.5, 0.125, 4
F_MAX = 3600.0


def stft(x, sr, t_offset=0.0, f_max=F_MAX, remove_median=True):
    """(times s, freqs Hz, dB[frame, bin]) of `x`; times are absolute (`t_offset` + centre).
    `remove_median`: subtract each bin's median over the segment (for display: it flattens
    stationary hum; snapping uses the plain dB, where a long steady tone is still a peak)."""
    n = int(WIN_S * sr)
    h = int(HOP_S * sr)
    nfft = n * PAD
    nfr = max(0, (len(x) - n) // h + 1)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    keep = freqs <= f_max
    win = np.hanning(n)
    db = np.empty((nfr, int(keep.sum())), dtype=np.float32)
    for j in range(nfr):
        s = np.fft.rfft(x[j * h:j * h + n] * win, nfft)[keep]
        db[j] = 10 * np.log10(np.abs(s) ** 2 + 1e-20)
    if nfr and remove_median:
        db -= np.median(db, axis=0, keepdims=True)
    times = t_offset + (np.arange(nfr) * h + n / 2) / sr
    return times, freqs[keep], db


def segment(path, t0=None, t1=None, pad_s=0.0):
    """(sr, samples, t_start) of the file between t0 − pad_s and t1 + pad_s (clipped)."""
    sr, x = wavio.read(path)
    a = 0 if t0 is None else max(0, int((t0 - pad_s) * sr))
    b = len(x) if t1 is None else min(len(x), int((t1 + pad_s) * sr))
    return sr, x[a:b], a / sr


# ---- images ----------------------------------------------------------------------------------

FONT = {  # 3×5 glyphs, rows top to bottom
    "0": ["111", "101", "101", "101", "111"], "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"], "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"], "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"], "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"], "9": ["111", "101", "111", "001", "111"],
    ".": ["000", "000", "000", "000", "010"], "-": ["000", "000", "111", "000", "000"],
    "s": ["000", "111", "110", "011", "111"], "k": ["100", "101", "110", "101", "101"],
    " ": ["000", "000", "000", "000", "000"],
}


def text(img, x, y, s, colour=(255, 255, 0), scale=2):
    for ch in s:
        g = FONT.get(ch, FONT[" "])
        for r, row in enumerate(g):
            for c, bit in enumerate(row):
                if bit == "1":
                    img[y + r * scale:y + (r + 1) * scale, x + c * scale:x + (c + 1) * scale] = colour
        x += 4 * scale


def write_png(path, rgb):
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def grey(db):
    """0–255 scale of a dB patch: the patch's maximum → white, TOP_DB below it (or 2 MAD below the
    median, if higher) → black, linear in dB between. Only a strong ridge's very centre reaches
    white (not a band several percent wide), while stretches tens of dB fainter stay visible."""
    med = np.median(db)
    mad = np.median(np.abs(db - med)) * 1.4826 + 1e-6
    hi = float(np.max(db))
    lo = max(med - 2 * mad, hi - TOP_DB)
    if hi <= lo:
        return np.zeros_like(db)
    return np.clip((db - lo) / (hi - lo), 0, 1) * 255


TOP_DB = 30.0  # dB shown below a patch's maximum
ML, MR, MT, MB = 70, 10, 10, 30  # margins
OV_PX_S, OV_H = 16, 720
ZOOM_W, ZOOM_H = 1200, 800


def overview_size(duration_s):
    return (ML + int(round(duration_s * OV_PX_S)) + MR, MT + OV_H + MB)


def zoom_size():
    return (ML + ZOOM_W + MR, MT + ZOOM_H + MB)


def _canvas(w, h):
    return np.zeros((h, w, 3), dtype=np.uint8)


def overview_png(wav, out, f_max=3400.0, t0=None, t1=None):
    """Log-frequency overview (30 Hz–f_max) of the file, or of [t0, t1]."""
    sr, x, ts = segment(wav, t0, t1)
    dur = len(x) / sr
    w, h = overview_size(dur)
    img = _canvas(w, h)
    times, freqs, db = stft(x, sr, ts, f_max=f_max)
    pw = w - ML - MR
    if len(times):
        col_t = ts + (np.arange(pw) + 0.5) / OV_PX_S
        jj = np.clip(np.searchsorted(times, col_t), 0, len(times) - 1)
        row_f = 30.0 * (f_max / 30.0) ** (1 - (np.arange(OV_H) + 0.5) / OV_H)
        qq = np.clip(np.searchsorted(freqs, row_f), 0, len(freqs) - 1)
        g = grey(db[np.ix_(jj, qq)]).T.astype(np.uint8)
        img[MT:MT + OV_H, ML:ML + pw] = g[:, :, None]
    for f in (50, 100, 200, 400, 800, 1600, 3200):
        if f >= f_max:
            continue
        y = MT + int(OV_H * (1 - np.log(f / 30.0) / np.log(f_max / 30.0)))
        img[y, ML:ML + pw] = (0, 120, 255)
        text(img, 4, y - 5, str(f))
    k0 = int(np.ceil(ts / 5.0))
    for k in range(k0, int((ts + dur) / 5.0) + 1):
        x_ = ML + int((5 * k - ts) * OV_PX_S)
        if ML <= x_ < ML + pw:
            img[MT:MT + OV_H, x_] = (0, 90, 0) if k % 2 else (0, 170, 0)
            if k % 2 == 0:
                text(img, x_ - 8, MT + OV_H + 8, f"{5 * k}s")
    write_png(out, img)


def zoom_png(wav, out, t0, t1, f0, f1, overlay=None):
    """Linear crop [t0, t1] × [f0, f1]; gridlines every 1 s and every 1 % of the centre
    frequency; `overlay` point lists drawn in blue, red, green."""
    sr, x, ts = segment(wav, t0, t1, pad_s=1.0)
    w, h = zoom_size()
    img = _canvas(w, h)
    times, freqs, db = stft(x, sr, ts, f_max=min(F_MAX, f1 * 1.05))
    if len(times):
        col_t = t0 + (np.arange(ZOOM_W) + 0.5) / ZOOM_W * (t1 - t0)
        jj = np.clip(np.searchsorted(times, col_t), 0, len(times) - 1)
        row_f = f1 - (np.arange(ZOOM_H) + 0.5) / ZOOM_H * (f1 - f0)
        qq = np.clip(np.searchsorted(freqs, row_f), 0, len(freqs) - 1)
        img[MT:MT + ZOOM_H, ML:ML + ZOOM_W] = grey(db[np.ix_(jj, qq)]).T.astype(np.uint8)[:, :, None]
    fc = 0.5 * (f0 + f1)
    step = 0.01 * fc
    k = int(np.ceil(f0 / step))
    while k * step <= f1:
        y = MT + int((f1 - k * step) / (f1 - f0) * ZOOM_H)
        if MT <= y < MT + ZOOM_H:
            img[y, ML:ML + ZOOM_W] = (0, 80, 160)
            text(img, 2, y - 5, f"{k * step:.1f}")
        k += 1
    for s in range(int(np.ceil(t0)), int(t1) + 1):
        x_ = ML + int((s - t0) / (t1 - t0) * ZOOM_W)
        if ML <= x_ < ML + ZOOM_W:
            img[MT:MT + ZOOM_H, x_] = (0, 150, 0) if s % 5 == 0 else (0, 70, 0)
            if s % 5 == 0:
                text(img, x_ - 8, MT + ZOOM_H + 8, f"{s}s")
    colours = [(60, 140, 255), (255, 60, 60), (60, 255, 60)]
    for pts, col in zip(overlay or [], colours):
        for t, f in pts:
            x_ = ML + int((t - t0) / (t1 - t0) * ZOOM_W)
            y = MT + int((f1 - f) / (f1 - f0) * ZOOM_H)
            if ML + 3 <= x_ < ML + ZOOM_W - 3 and MT + 3 <= y < MT + ZOOM_H - 3:
                img[y - 3:y + 4, x_] = col
                img[y, x_ - 3:x_ + 4] = col
    write_png(out, img)
