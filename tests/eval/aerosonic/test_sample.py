"""The seeded sample and the manifest (on the real sample_meta.csv; skipped if not fetched)."""
import csv
import os
import tempfile
import tomllib
import unittest
from collections import Counter

import sample

MEDIA = os.environ.get(
    "GEOMEDIA_MEDIA", os.path.expanduser("~/Library/Application Support/geomedia/media")
)
META = os.path.join(MEDIA, "aerosonic", "sample_meta.csv")


@unittest.skipUnless(os.path.exists(META), "AeroSonicDB not fetched")
class Sample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(META, newline="") as f:
            cls.rows = list(csv.DictReader(f))

    def test_counts(self):
        s = sample.draw(self.rows, sample.SEED)
        c = Counter(r["stratum"] for r in s)
        self.assertEqual(
            c,
            {"piston-all": 36, "rotor-all": 5, "low-turboprop": 8, "low-turbofan": 12,
             "far-turboprop": 40, "far-turbofan": 40},
        )

    def test_deterministic_and_unique(self):
        a, b = sample.draw(self.rows, sample.SEED), sample.draw(self.rows, sample.SEED)
        self.assertEqual([r["filename"] for r in a], [r["filename"] for r in b])
        self.assertEqual(len({r["filename"] for r in a}), 141)
        self.assertNotEqual(
            [r["filename"] for r in sample.draw(self.rows, sample.SEED + 1)],
            [r["filename"] for r in a],
        )

    def test_only_aircraft_clips(self):
        self.assertTrue(all(r["class"] == "1" for r in sample.draw(self.rows, sample.SEED)))

    def test_manifest_lists_every_case(self):
        s = sample.draw(self.rows, sample.SEED)
        out = os.path.join(tempfile.mkdtemp(), "m.toml")
        env = [f"env{k}.wav" for k in range(6)]
        sample.write_manifest(s, env, out)
        with open(out, "rb") as f:
            cases = tomllib.load(f)["case"]
        self.assertEqual(len(cases), 147)
        self.assertEqual(len({c["id"] for c in cases}), 147)
        self.assertTrue(all(c["kind"] == "curves" for c in cases))
        cont = [c for c in cases if "labels_column" in c]
        self.assertEqual([c["labels_column"] for c in cont], list(range(6)))
        clip = next(c for c in cases if "labels_column" not in c)
        self.assertTrue(clip["media"].startswith("media:aerosonic/audio/"))
        self.assertTrue(clip["marks"].startswith("media:aerosonic/marks/"))
        self.assertEqual(len(clip["audible"]), 2)
        self.assertTrue(all(c["marks_hashes"] == "aerosonic/marks.sha256" for c in cases))


if __name__ == "__main__":
    unittest.main()
