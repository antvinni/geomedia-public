"""Label-column → recording mapping check (synthetic files only)."""
import os
import tempfile
import unittest

import numpy as np

import envmap
import wavio

SR = 8000


def tone_wav(path, seconds, tone=None):
    """Noise, plus a 300 Hz tone over `tone` = (t0, t1) if given."""
    rng = np.random.default_rng(1)
    x = 0.01 * rng.standard_normal(int(seconds * SR))
    if tone:
        t = np.arange(len(x)) / SR
        on = (t >= tone[0]) & (t < tone[1])
        x[on] += 0.3 * np.sin(2 * np.pi * 300 * t[on])
    wavio.write(path, SR, x)


class EnvMap(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        env = os.path.join(self.dir, "env")
        os.makedirs(env)
        # a.wav: aircraft-like tone 20–40 s; b.wav: tone 40–55 s. 12 five-second rows each.
        tone_wav(os.path.join(env, "a.wav"), 60, (20, 40))
        tone_wav(os.path.join(env, "b.wav"), 60, (40, 55))
        self.env = env

    def labels(self, cols):
        p = os.path.join(self.dir, "map.csv")
        with open(p, "w") as f:
            f.write(",".join(str(i) for i in range(len(cols))) + "\n")
            for r in range(12):
                f.write(",".join(c[r] for c in cols) + "\n")
        return p

    def col(self, on):
        # "1" for rows inside `on` (row r covers [5r, 5r+5)), "0 " (stray space, as in the
        # dataset) elsewhere.
        return ["1" if on[0] <= 5 * r < on[1] else "0 " for r in range(12)]

    def test_mapping_check_accepts_the_right_columns(self):
        p = self.labels([self.col((20, 40)), self.col((40, 55))])
        diag = envmap.check_mapping(self.env, p, min_gap_db=3.0)
        self.assertEqual([d[1] for d in diag], ["a.wav", "b.wav"])
        self.assertTrue(all(d[2] > 3.0 for d in diag))

    def test_mapping_check_rejects_shuffled_columns(self):
        p = self.labels([self.col((40, 55)), self.col((20, 40))])
        with self.assertRaises(ValueError):
            envmap.check_mapping(self.env, p, min_gap_db=3.0)

    def test_mapping_check_rejects_a_wrong_length(self):
        tone_wav(os.path.join(self.env, "b.wav"), 30, (10, 20))
        p = self.labels([self.col((20, 40)), self.col((40, 55))])
        with self.assertRaises(ValueError):
            envmap.check_mapping(self.env, p, min_gap_db=3.0)


if __name__ == "__main__":
    unittest.main()
