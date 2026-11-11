# geomedia

Estimate where (and when) an audio or video recording was made, from the file's metadata and the sounds in it.

**Write-up:** [Can a passing plane tell where a recording was made?](https://antvinni.pages.dev/blog/geomedia/), a blog post about the project: what it does, what worked and what did not.

## Status

This is a research project, not a finished tool. The code is published as it stood when the write-up was written.

What works:

- **The metadata report** (`geomedia meta`).
- **The simulator** (`geomedia sim`): synthetic flyover recordings with a known answer, used by the tests.
- **The location step, once the aircraft are known.** Given a propeller's Doppler curve and the flight tracks, it places colsto's Oyster Bay recording 276 m from the published position. From the timing and loudness of the passes alone, with no curve, 240 m. Oyster Bay was the development case, so these numbers show that the method can work. They are not a test result.

What does not:

- **Finding the aircraft in a real recording.** On the Oyster Bay hour the search found, on its own, 0 of the 20 passes that the flight tracks put within 5 km. Most jets leave no usable curve, only broadband roar, and street noise can pass for an aircraft.
- **The one held-out recording**, a 10-minute phone clip: the answer was "not located". The clip was later used to tune the first stage, so it cannot serve as a test again.
- **One known wrong answer.** Oyster Bay with its clock set 30 s wrong comes out "located" 3,054 m away. The project's rule was that a "located" answer is never more than 3 km off. This is not fixed.

## Usage

```sh
cargo install --path crates/gm-cli      # installs the `geomedia` binary (Rust 1.99.1, pinned in rust-toolchain.toml)

geomedia meta clip.mov                  # summary in the terminal; meta.html and meta.json go to the managed run folder
geomedia meta clip.mov --out report/    # also copy the report here
geomedia meta clip.mov --json - --no-report   # JSON (schema v1) on stdout only
geomedia sim tests/fixtures/sim/two_props.toml --out sim/   # synthetic flyover: audio.wav + truth.json
geomedia events sim/audio.wav [--event 15:00-16:40]          # aircraft Doppler passes: speed, CPA distance/time, f0
geomedia sources                                             # every data source, licence, cost, cache status
geomedia preflight clip.webm --area "Oyster Bay, NY" --when 2026-09-08T20:17:29Z±5s [--yes]   # what a locate run needs
geomedia locate clip.webm --area "Oyster Bay, NY" --when 2026-09-08T20:17:29Z±5s [--yes]      # where it was recorded (exit 0 located, 3 not)
geomedia data                                                # what is kept on disk (downloads, runs, media) and where
geomedia data clean runs                                     # delete kept items; lists them and asks first
geomedia eval tests/eval/sim-suite.toml                      # run locate over a manifest of cases with a known answer
```

Every command has `--help`.

`geomedia meta` reads:
- container and stream details
- GPS location (QuickTime `©xyz`, Apple `mdta` keys, EXIF)
- creation time (`mvhd`, tags, Broadcast WAV `bext`, EXIF)
- device, software and encoder
- platform traces, such as an ffmpeg re-mux

The format is detected from the file's bytes, not its extension. A file with nothing embedded prints `metadata: empty`.

**Every embedded value is untrusted.** Metadata can be stripped, rewritten or forged, so geomedia reports it as a hint and never applies it as truth.

Supported: MP4/MOV/M4A/3GP, WAV/BWF, MP3, FLAC, Ogg, Matroska (audio), AIFF, JPEG, PNG and HEIF.

## Methods

1. **Metadata.** Container, EXIF and QuickTime GPS and creation-time tags, plus encoder traces. These are always treated as untrusted hints.
2. **Aircraft Doppler.** The pitch of a passing aircraft's tone drops as it goes by. The shape of that drop gives its speed, its closest distance and the moment it was closest. `locate` matches these against historical ADS-B tracks: a place fits when the aircraft that flew near it explain the curves under one recording clock.
3. **Timing and loudness.** An aircraft with no usable tone still gets louder and then quieter. `locate` matches when the passes were heard, and how loud they were, against the same tracks.

`locate` needs to be told roughly where and when: an area (`--area`) and a start time (`--when`). It answers "located", "ambiguous" or "inconclusive", and only "located" exits with 0.

A fourth method, thunder matched against lightning-strike catalogues, was planned and not built.

## Repository

| Crate | What it holds |
|---|---|
| `gm-core` | geodesy, local frames and time |
| `gm-media` | format detection and embedded metadata |
| `gm-doppler` | the Doppler forward model, audio decoding and finding aircraft in audio |
| `gm-sim` | the flyover simulator |
| `gm-sources` | the data-source registry, the cache and the readers for each source |
| `gm-search` | the search for the place and time that fit |
| `gm-report` | terminal, JSON and HTML reports |
| `gm-cli` | the `geomedia` binary |

`cargo test` runs the tests. Test fixtures are regenerated with `tests/fixtures/make.sh` (needs ffmpeg and python3).

`tests/eval/` holds the evaluation manifests for `geomedia eval`. The simulator manifests (`sim-suite.toml`, `sim-broadband.toml`) run as they are. The others name recordings as `media:…`, files in a local media folder that is not part of this repository, so they document what was measured and do not run from a fresh clone.

Some comments and manifests point to design notes under `docs/`. Those notes are not part of this repository.

## Credits

- Method inspiration: colsto, ["I discovered a sound that leaks your exact location"](https://www.youtube.com/watch?v=UytYrhYLOKI).
- Science: emission-time Doppler for aircraft acoustics (Ferguson, *J. Acoust. Soc. Am.*, 1992); atmospheric absorption per ISO 9613-1.

## Data sources

Data keeps its own licence. Derived caches are never committed.

| Data | Source | Licence |
|---|---|---|
| ADS-B | [adsb.lol](https://adsb.lol) | ODbL 1.0 |
| Aircraft types | [tar1090-db](https://github.com/wiedehopf/tar1090-db) | see upstream |
| Terrain | Copernicus GLO-30 DEM (© DLR e.V., © Airbus Defence and Space GmbH, provided under COPERNICUS by the EU and ESA) | Copernicus licence |
| Geoid | NGA EGM2008 | public domain |
| Weather | [Open-Meteo](https://open-meteo.com/) | CC BY 4.0 (non-commercial free tier) |
| Place names | OpenStreetMap Nominatim | ODbL |

A few test files are taken from other people's data and keep that data's terms:

| Files | From | Terms |
|---|---|---|
| `tests/eval/aerosonic/sample.csv`, `tests/eval/aerosonic.toml`, `tests/eval/aerosonic-detect.toml`, and the AeroSonicDB cases in `tests/eval/street-detect.toml` (clip names and labels) | AeroSonicDB YPAD-0523 v1.1.1, B. Downward, [Zenodo 8371595](https://zenodo.org/records/8371595) | CC BY-NC 4.0 |
| `tests/eval/livermore.toml` (clip names) | "Aeroplane data full analysis and raw files" (smartphone recordings at Livermore), Praveen Pathak, [Zenodo 23424278](https://zenodo.org/records/23424278) | CC BY 4.0 |

No audio from these datasets is in the repository.

The street-walk case in `tests/eval/street-detect.toml` and the fixture `tests/fixtures/ridges/street_vehicle.json` hold labels and measured tone frequencies from someone else's public walking video. Nothing of the video itself is in the repository.

## Responsible use

This tool can place a recording, and so the person who made it, on a map. Use it to verify footage, to research the privacy leak itself, or with the recordist's consent. Do not use it to find, track or expose people.

## Licence

The code and text written for this project are licensed under either of

- the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)), or
- the MIT licence ([LICENSE-MIT](LICENSE-MIT)),

at your option.

This covers the project's own work only. The data and the test files listed under [Data sources](#data-sources) keep their own terms. The files taken from AeroSonicDB, for example, are CC BY-NC 4.0: no commercial use.

Unless you state otherwise, any contribution you intentionally submit for inclusion in this project, as defined in the Apache-2.0 licence, is dual licensed as above, with no additional terms or conditions.
