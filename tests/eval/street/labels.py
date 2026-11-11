"""5-s stage-1 label rows (the `labels_csv` format geomedia's detect cases read) for the street
recordings. Stdlib only; never geomedia: the labels must not come from the tool they judge.

  labels.py from-marks MARKS DURATION_S OUT          blind marks (Piccadilly)
  labels.py phone-marks MARKS DURATION_S OUT         the marks of the phone clip
  labels.py from-passes RESULTS_JSON CASE_ID DURATION_S OUT   ADS-B passes (Oyster Bay)

Rows: header `label`, then one row per 5 s, ceil(duration / 5) rows (the last runs past the
end). A row takes the stretch containing its midpoint; stretches are half-open [t0, t1).
Cells: `1` aircraft, `0` not aircraft, empty = ignore. Priority: ignore > aircraft > not;
unmarked rows are `0`.

Marks schema for `from-marks`:
  {"aircraft": [[t0, t1], …], "not_aircraft": [[t0, t1, "kind"], …], "ignore": [[t0, t1, "why"], …]}
"""
import json
import math
import sys

ROW_S = 5.0
# Oyster Bay from ADS-B: aircraft within ±AIR_S of a close pass's CPA, ignore within ±TAIL_S of
# any pass up to FAR_M (a pass's audible tails and a far pass are not silence).
NEAR_M, FAR_M = 3000.0, 8000.0
AIR_S, TAIL_S = 20.0, 60.0
# The phone clip: the first 21 s are handling noise over whatever else was there.
PHONE_IGNORE = [(0.0, 21.0)]


def _inside(t, stretches):
    return any(a <= t < b for a, b in stretches)


def rows_from_stretches(duration_s, aircraft, not_aircraft, ignore):
    """One cell per 5-s row. `not_aircraft` is accepted for symmetry: unmarked rows are `0`
    anyway, so it changes nothing."""
    del not_aircraft
    n = math.ceil(duration_s / ROW_S - 1e-9)
    out = []
    for k in range(n):
        mid = k * ROW_S + ROW_S / 2
        if _inside(mid, ignore):
            out.append("")
        elif _inside(mid, aircraft):
            out.append("1")
        else:
            out.append("0")
    return out


def _pairs(items):
    return [(float(x[0]), float(x[1])) for x in items]


def marks_stretches(marks):
    """(aircraft, not_aircraft, ignore) from the blind-marks schema."""
    return (
        _pairs(marks.get("aircraft", [])),
        _pairs(marks.get("not_aircraft", [])),
        _pairs(marks.get("ignore", [])),
    )


def phone_marks_stretches(marks):
    """The phone clip's marks: aircraft-like at medium confidence → aircraft; at low
    confidence, every can't-tell stretch, and 0–21 s → ignore; not-aircraft → not."""
    air, ign = [], list(PHONE_IGNORE)
    for m in marks.get("aircraft_like", []):
        (ign if m.get("confidence") == "low" else air).append((float(m["t0"]), float(m["t1"])))
    for m in marks.get("cant_tell", []):
        ign.append((float(m["t0"]), float(m["t1"])))
    nots = [(float(m["t0"]), float(m["t1"])) for m in marks.get("not_aircraft", [])]
    return air, nots, ign


def passes_stretches(passes):
    """(aircraft, not, ignore) from expected passes ({slant_m, t_audio})."""
    air, ign = [], []
    for p in passes:
        s, t = float(p["slant_m"]), float(p["t_audio"])
        if s <= NEAR_M:
            air.append((t - AIR_S, t + AIR_S))
        if s <= FAR_M:
            ign.append((t - TAIL_S, t + TAIL_S))
    # Ignore wins over aircraft, so the close passes' own ±20 s must not be in `ign`: cut
    # each tail window around the aircraft windows it contains.
    cut = []
    for a, b in ign:
        pieces = [(a, b)]
        for c, d in air:
            nxt = []
            for x, y in pieces:
                if d <= x or c >= y:
                    nxt.append((x, y))
                    continue
                if x < c:
                    nxt.append((x, c))
                if d < y:
                    nxt.append((d, y))
            pieces = nxt
        cut.extend(pieces)
    return air, [], cut


def write_csv(path, rows):
    with open(path, "w") as f:
        f.write("label\n")
        for r in rows:
            f.write(r + "\n")


def case_passes(results_json, case_id):
    with open(results_json) as f:
        doc = json.load(f)
    for c in doc["cases"]:
        if c.get("id") == case_id:
            ps = c.get("expected_passes")
            if ps is None:
                sys.exit(f"{case_id}: no expected_passes in {results_json}")
            return ps
    sys.exit(f"{case_id}: not in {results_json}")


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    cmd = argv[1]
    if cmd in ("from-marks", "phone-marks") and len(argv) == 5:
        with open(argv[2]) as f:
            marks = json.load(f)
        a, n, i = (marks_stretches if cmd == "from-marks" else phone_marks_stretches)(marks)
        dur, out = float(argv[3]), argv[4]
    elif cmd == "from-passes" and len(argv) == 6:
        a, n, i = passes_stretches(case_passes(argv[2], argv[3]))
        dur, out = float(argv[4]), argv[5]
    else:
        sys.exit(__doc__)
    rows = rows_from_stretches(dur, a, n, i)
    write_csv(out, rows)
    print(f"{out}: {len(rows)} rows, {rows.count('1')} aircraft, {rows.count('')} ignore, "
          f"{rows.count('0')} not")


if __name__ == "__main__":
    main(sys.argv)
