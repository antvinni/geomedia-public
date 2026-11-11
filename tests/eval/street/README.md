# Street recordings: stage-1 labels

Design: `docs/superpowers/specs/2026-11-09-street-sounds-design.md`.

Media, marks and labels stay in the kept-data media folder (`media:phone/`, `media:street/`,
`media:oysterbay/`), never in git. Only these scripts, the generated manifest
(`../street-detect.toml`) and the sha256 list are committed. The scripts never import geomedia
or call its binary on the audio: the labels must not come from the tool they judge.

| File | What it does |
|---|---|
| `views.py` | plain views of a 16-kHz mono WAV for blind marking: spectrogram strips and octave levels |
| `labels.py` | 5-s label rows from blind marks, from the phone clip's marks, or from ADS-B passes (`expected_passes` in an eval's `results.json`) |
| `make_manifest.py` | writes `../street-detect.toml`: the street cases plus every AeroSonicDB detect case |
| `labels.sha256` | hashes of the labels, the Piccadilly marks and audio, written before any stage-1 rule ran on them |

Tests: `tests/eval/aerosonic/.venv/bin/python -I -m unittest discover -s tests/eval/street`.
