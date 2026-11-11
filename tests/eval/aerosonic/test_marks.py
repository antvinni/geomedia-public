"""Marking tools on synthetic signals: plain spectrogram, snap, densify, fit, coverage check."""
import json
import os
import struct
import tempfile
import unittest

import numpy as np

import marks
import spec
import wavio

SR = 8000
C = 343.0


def cv(t, f0, v, d, t0):
    x = v * (t - t0)
    return f0 / (1 + (v / C) * x / np.sqrt(d * d + x * x))


def tones(seconds, parts, noise=0.02, seed=3):
    """Noise plus tones: each part is (f(t) callable, amplitude, (t0, t1))."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = noise * rng.standard_normal(len(t))
    for f, a, (a0, a1) in parts:
        ph = 2 * np.pi * np.cumsum(f(t)) / SR
        on = (t >= a0) & (t < a1)
        x[on] += a * np.sin(ph[on])
    return x


class Snap(unittest.TestCase):
    def spec_of(self, x):
        return spec.stft(x, SR, remove_median=False)

    def test_snap_moves_to_the_tone(self):
        x = tones(30, [(lambda t: np.full_like(t, 120.7), 0.2, (0, 30))])
        times, freqs, db = self.spec_of(x)
        out = marks.snap(db, times, freqs, [(t, 118.5) for t in (5.0, 10.0, 15.0, 20.0)])
        for m in out:
            self.assertNotIn("snap_rejected", m["flags"])
            self.assertLess(abs(m["f"] / 120.7 - 1), 0.003, m)

    def test_snap_rejects_a_jump(self):
        # The true tone at 120 Hz; a stronger unrelated tone 1.8 % up only around t = 15 s.
        x = tones(30, [(lambda t: np.full_like(t, 120.0), 0.1, (0, 30)),
                       (lambda t: np.full_like(t, 122.2), 0.5, (13, 17))])
        times, freqs, db = self.spec_of(x)
        out = marks.snap(db, times, freqs, [(t, 120.5) for t in (5.0, 10.0, 15.0, 20.0, 25.0)])
        bad = out[2]
        self.assertIn("snap_rejected", bad["flags"])
        self.assertEqual(bad["f"], bad["f_eye"])
        for m in out[:2] + out[3:]:
            self.assertNotIn("snap_rejected", m["flags"])

    def test_an_edge_mark_gets_a_wider_second_pass(self):
        # The tone is 5 % above the eye marks (eye reading on a steep curve): inside ±2 % the
        # brightest bin is the window's edge, so a ±6 % second pass finds it, flagged.
        x = tones(30, [(lambda t: np.full_like(t, 210.0), 0.3, (0, 30))])
        times, freqs, db = self.spec_of(x)
        out = marks.snap(db, times, freqs, [(t, 200.0) for t in (5.0, 10.0, 15.0)])
        for m in out:
            self.assertIn("wide_snap", m["flags"], m)
            self.assertNotIn("edge", m["flags"], m)
            self.assertLess(abs(m["f"] / 210.0 - 1), 0.003, m)

    def test_a_peak_beyond_the_wide_window_is_not_snapped(self):
        # 9 % away: beyond ±6 % too — the mark keeps its eye pitch, flagged `edge`.
        x = tones(30, [(lambda t: np.full_like(t, 218.0), 0.3, (0, 30))])
        times, freqs, db = self.spec_of(x)
        out = marks.snap(db, times, freqs, [(t, 200.0) for t in (5.0, 10.0, 15.0)])
        for m in out:
            self.assertTrue({"edge", "no_peak"} & set(m["flags"]), m)
            self.assertEqual(m["f"], m["f_eye"])

    def test_out_of_range_marks_are_flagged(self):
        x = tones(10, [(lambda t: np.full_like(t, 200.0), 0.2, (0, 10))])
        times, freqs, db = self.spec_of(x)
        out = marks.snap(db, times, freqs, [(5.0, 200.0), (25.0, 200.0), (5.0, 0.0), (5.0, 9000.0)])
        self.assertNotIn("out_of_range", out[0]["flags"])
        for m in out[1:]:
            self.assertIn("out_of_range", m["flags"])
            self.assertEqual(m["f"], m["f_eye"])


class Curves(unittest.TestCase):
    def test_densify_stays_on_a_doppler_curve(self):
        ts = np.linspace(5, 55, 10)
        pts = [(t, cv(t, 100, 60, 500, 30)) for t in ts]
        dense = marks.densify(pts)
        self.assertGreater(len(dense), 300)
        for t, f in dense:
            self.assertLess(abs(f / cv(t, 100, 60, 500, 30) - 1), 0.008, (t, f))

    def test_fit_recovers_geometry(self):
        rng = np.random.default_rng(5)
        ts = np.linspace(-10, 70, 40)
        pts = [(t, cv(t, 120, 70, 900, 30) * (1 + 0.003 * rng.uniform(-1, 1))) for t in ts]
        fit = marks.fit_cv(pts)
        self.assertLess(abs(fit["f0"] / 120 - 1), 0.01, fit)
        self.assertLess(abs(fit["v"] / 70 - 1), 0.15, fit)
        self.assertLess(abs(fit["d"] / 900 - 1), 0.15, fit)
        self.assertLess(abs(fit["t0"] - 30), 1.0, fit)
        self.assertNotIn("implausible", fit["flags"])

    def test_fit_survives_a_partial_pass(self):
        # A long approach plateau and the start of the fall, CPA near the end, 0.4 % noise:
        # the fit must stay finite and close, not run off (an undamped Gauss–Newton from the
        # plateau/tail guess diverged on 12 of 40 such cases).
        rng = np.random.default_rng(0)
        t0, v, d = rng.uniform(50, 60), rng.uniform(45, 70), rng.uniform(400, 900)
        ts = np.linspace(2, 58, 120)
        pts = [(t, cv(t, 240, v, d, t0) * (1 + 0.004 * rng.standard_normal())) for t in ts]
        fit = marks.fit_cv(pts)
        for k in ("f0", "v", "d", "t0"):
            self.assertTrue(np.isfinite(fit[k]), fit)
        self.assertLess(abs(fit["f0"] / 240 - 1), 0.03, fit)
        self.assertLess(fit["rms_pct"], 0.6, fit)

    def test_fit_flags_a_cpa_outside_the_curve(self):
        ts = np.linspace(0, 30, 60)
        fit = marks.fit_cv([(t, cv(t, 150, 60, 800, 50)) for t in ts])
        self.assertIn("cpa_outside_curve", fit["flags"])
        fit = marks.fit_cv([(t, cv(t, 150, 60, 800, 15)) for t in ts])
        self.assertNotIn("cpa_outside_curve", fit["flags"])

    def test_fit_flags_implausible(self):
        ts = np.linspace(0, 40, 30)
        fit = marks.fit_cv([(t, cv(t, 150, 60, 20, 20)) for t in ts])
        self.assertIn("implausible", fit["flags"])


class TruthPoints(unittest.TestCase):
    def test_only_cleanly_snapped_marks_make_the_truth(self):
        sn = [{"t": float(t), "f_eye": 100.0, "f": 100.0 - t, "flags": fl}
              for t, fl in [(0, []), (1, ["edge"]), (2, []), (3, ["snap_rejected"]), (4, []),
                            (5, ["wide_snap"]), (6, ["no_peak"]), (7, [])]]
        pts, flags = marks.truth_points(sn)
        self.assertEqual([t for t, _ in pts], [0.0, 2.0, 4.0, 5.0, 7.0])
        self.assertEqual(flags, [])

    def test_too_few_clean_marks_keep_all_and_say_so(self):
        sn = [{"t": float(t), "f_eye": 100.0, "f": 100.0, "flags": ["edge"] if t % 2 else []}
              for t in range(5)]
        sn[0]["flags"] = ["no_peak"]
        pts, flags = marks.truth_points(sn)
        self.assertEqual(len(pts), 5)
        self.assertEqual(flags, ["few_clean_marks"])


class Render(unittest.TestCase):
    def png_size(self, path):
        with open(path, "rb") as f:
            head = f.read(24)
        self.assertEqual(head[:8], b"\x89PNG\r\n\x1a\n")
        return struct.unpack(">II", head[16:24])

    def test_short_clip_renders(self):
        d = tempfile.mkdtemp()
        w = os.path.join(d, "short.wav")
        wavio.write(w, SR, tones(18, [(lambda t: np.full_like(t, 300.0), 0.2, (0, 18))]))
        out = os.path.join(d, "o.png")
        spec.overview_png(w, out)
        self.assertEqual(self.png_size(out), spec.overview_size(18.0))

    def test_zoom_with_overlay_renders(self):
        d = tempfile.mkdtemp()
        w = os.path.join(d, "z.wav")
        wavio.write(w, SR, tones(20, [(lambda t: np.full_like(t, 300.0), 0.2, (0, 20))]))
        out = os.path.join(d, "z.png")
        spec.zoom_png(w, out, 2, 18, 280, 320, overlay=[[(5, 299)], [(5, 300)]])
        self.assertEqual(self.png_size(out), spec.zoom_size())


def png_pixels(path):
    import zlib
    with open(path, "rb") as f:
        b = f.read()
    pos, idat = 8, b""
    while pos < len(b):
        n = struct.unpack(">I", b[pos:pos + 4])[0]
        ty = b[pos + 4:pos + 8]
        if ty == b"IHDR":
            w, h = struct.unpack(">II", b[pos + 8:pos + 16])
        if ty == b"IDAT":
            idat += b[pos + 8:pos + 8 + n]
        pos += 12 + n
    return np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w * 3 + 1)[:, 1:].reshape(h, w, 3)


class Display(unittest.TestCase):
    def test_a_strong_tone_peaks_on_its_own_row(self):
        # Two strong tone bursts: in the zoom crop each must be brightest within ±0.3 % of its
        # frequency, not one saturated band several percent wide.
        d = tempfile.mkdtemp()
        w = os.path.join(d, "t.wav")
        t = np.arange(20 * SR) / SR
        x = 0.01 * np.random.default_rng(1).standard_normal(len(t))
        on = (t > 8) & (t < 12)
        x[on] += 0.3 * np.sin(2 * np.pi * 100 * t[on]) + 0.3 * np.sin(2 * np.pi * 150 * t[on])
        wavio.write(w, SR, x)
        out = os.path.join(d, "z.png")
        spec.zoom_png(w, out, 2, 18, 85, 165)
        img = png_pixels(out).astype(int)
        col = img[spec.MT:spec.MT + spec.ZOOM_H, spec.ML + spec.ZOOM_W // 2 + 5]
        g = np.where((col[:, 0] == col[:, 1]) & (col[:, 1] == col[:, 2]), col[:, 0], -1)
        row_f = 165 - (np.arange(spec.ZOOM_H) + 0.5) / spec.ZOOM_H * 80
        for f in (100.0, 150.0):
            near = np.abs(row_f / f - 1) < 0.05
            r = int(np.argmax(np.where(near, g, -1)))
            self.assertLess(abs(row_f[r] / f - 1), 0.003, (f, row_f[r]))
            # 2 % away the ridge has clearly faded (a saturated display stays white there).
            for side in (0.98, 1.02):
                k = int(np.argmin(np.abs(row_f - f * side)))
                self.assertLess(g[k], g[r] - 30, (f, side, g[k], g[r]))


class Coverage(unittest.TestCase):
    def test_check_lists_uncovered_runs(self):
        labels = ["0"] * 4 + ["1"] * 3 + ["0", "ignore"] + ["1"] * 2 + ["0"] * 3
        runs = marks.aircraft_runs(labels)
        self.assertEqual(runs, [(20.0, 35.0), (45.0, 55.0)])
        m = {"curves": [{"points": [[22.0, 100.0], [33.0, 95.0]]}], "no_curve_stretches": []}
        self.assertEqual(marks.uncovered(runs, m), [(45.0, 55.0)])
        m["no_curve_stretches"] = [[43.0, 57.0]]
        self.assertEqual(marks.uncovered(runs, m), [])

    def test_write_marks_schema(self):
        p = os.path.join(tempfile.mkdtemp(), "m.json")
        marks.write_marks(p, "a.wav", "no_curve", [], {"no_curve_stretches": [[0, 5]]})
        with open(p) as f:
            j = json.load(f)
        self.assertEqual(j["verdict"], "no_curve")
        self.assertTrue(j["before_discovery"])
        self.assertIn("marked_utc", j)


if __name__ == "__main__":
    unittest.main()
