# AeroSonicDB real-audio curve test — tools

Data: AeroSonicDB YPAD-0523 v1.1.1 — B. Downward, Zenodo record 8371595, **CC BY-NC 4.0**
(non-commercial evaluation only). Audio, marks and images stay in the kept-data media
folder (`media:aerosonic/`), never in git. Design:
`docs/superpowers/specs/2026-11-06-real-curve-test-design.md`; plan:
`docs/superpowers/plans/2026-11-06-real-curve-test.md`.

These tools never import geomedia or call its binary: the truth must not come from the tool it
judges.

| File | What it does |
|---|---|
| `fetch.sh` | downloads the record into the media folder, checks every md5, unpacks the WAVs, creates `.venv` (numpy 2.5.4) |
| `wavio.py` | minimal WAV reader/writer |
| `envmap.py` | checks which 1-hour recording is which label column (6 × 6 level matrix) |
| `sample.py` | the seeded sample (`sample.csv`) and the manifest (`../aerosonic.toml`), from metadata only |
| `spec.py` | the plain marking spectrogram: overview and zoomed crops with gridlines and overlays |
| `marks.py` | snap eye marks (±2 %, sub-bin), densify, fit, write marks JSON; `--check`, `--hash` |
| `marks.sha256` | the committed hashes of every marks file, written before discovery ran on them |

Rebuild: `zsh tests/eval/aerosonic/fetch.sh`, then
`tests/eval/aerosonic/.venv/bin/python -I tests/eval/aerosonic/sample.py "$MEDIA"`.
Tests: `tests/eval/aerosonic/.venv/bin/python -I -m unittest discover -s tests/eval/aerosonic`.
