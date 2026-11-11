"""5-s label rows from blind marks, from the phone clip's marks, and from ADS-B passes."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # siblings, also under python -I
import labels  # noqa: E402


class Rows(unittest.TestCase):
    def test_midpoint_rule(self):
        # Rows [0,5) [5,10) [10,15) [15,20): midpoints 2.5, 7.5, 12.5, 17.5. The stretch is
        # half-open, so 12.5 is outside [0, 12.5).
        rows = labels.rows_from_stretches(20.0, aircraft=[(0.0, 12.5)], not_aircraft=[], ignore=[])
        self.assertEqual(rows, ["1", "1", "0", "0"])

    def test_ignore_wins(self):
        rows = labels.rows_from_stretches(
            20.0, aircraft=[(0.0, 20.0)], not_aircraft=[(0.0, 20.0)], ignore=[(5.0, 10.0)]
        )
        self.assertEqual(rows, ["1", "", "1", "1"])

    def test_last_row_runs_past_the_end(self):
        self.assertEqual(len(labels.rows_from_stretches(603.4, [], [], [])), 121)
        self.assertEqual(len(labels.rows_from_stretches(600.0, [], [], [])), 120)

    def test_phone_marks_mapping(self):
        marks = {
            "aircraft_like": [
                {"t0": 50, "t1": 60, "confidence": "medium"},
                {"t0": 100, "t1": 110, "confidence": "low"},
            ],
            "cant_tell": [{"t0": 70, "t1": 80}],
            "not_aircraft": [
                {"t0": 0, "t1": 21, "kind": "handling"},
                {"t0": 120, "t1": 130, "kind": "engine"},
            ],
        }
        a, n, i = labels.phone_marks_stretches(marks)
        rows = labels.rows_from_stretches(140.0, a, n, i)
        want = ["0"] * 28
        for k in (0, 1, 2, 3):  # 0–21 s: ignore, although marked not-aircraft
            want[k] = ""
        for k in (10, 11):  # 50–60 s medium aircraft-like
            want[k] = "1"
        for k in (14, 15, 20, 21):  # can't tell; low-confidence aircraft-like
            want[k] = ""
        self.assertEqual(rows, want)

    def test_from_passes_bands(self):
        passes = [
            {"icao": "a", "slant_m": 2000.0, "t_audio": 50.0},
            {"icao": "b", "slant_m": 5000.0, "t_audio": 200.0},
            {"icao": "c", "slant_m": 9000.0, "t_audio": 280.0},
        ]
        a, n, i = labels.passes_stretches(passes)
        rows = labels.rows_from_stretches(300.0, a, n, i)
        self.assertEqual(len(rows), 60)
        for k, v in enumerate(rows):
            if 6 <= k <= 13:
                self.assertEqual(v, "1", k)
            elif k <= 21 or 28 <= k <= 51:
                self.assertEqual(v, "", k)
            else:
                self.assertEqual(v, "0", k)

    def test_from_marks_schema(self):
        marks = {
            "aircraft": [[10, 20]],
            "not_aircraft": [[0, 10, "bus"]],
            "ignore": [[15, 20, "faint"]],
        }
        a, n, i = labels.marks_stretches(marks)
        self.assertEqual(labels.rows_from_stretches(25.0, a, n, i), ["0", "0", "1", "", "0"])

    def test_csv_has_a_header_and_one_row_per_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "l.csv")
            labels.write_csv(p, ["1", "", "0"])
            with open(p) as f:
                self.assertEqual(f.read(), "label\n1\n\n0\n")

    def test_results_case_lookup(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "results.json")
            with open(p, "w") as f:
                json.dump({"schema_version": 1, "cases": [
                    {"id": "x", "expected_passes": None},
                    {"id": "oysterbay", "expected_passes": [{"slant_m": 1.0, "t_audio": 9.0}]},
                ]}, f)
            self.assertEqual(labels.case_passes(p, "oysterbay"), [{"slant_m": 1.0, "t_audio": 9.0}])
            with self.assertRaises(SystemExit):
                labels.case_passes(p, "x")


if __name__ == "__main__":
    unittest.main()
