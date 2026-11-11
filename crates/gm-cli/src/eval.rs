//! `geomedia eval`: run `locate` over ground-truth cases and report honest metrics — coverage,
//! accuracy of located answers, and (first) confident wrong answers.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use gm_core::{Geodetic, geodesic_distance_m};
use gm_search::Verdict;
use serde::{Deserialize, Serialize};

use crate::locate::{Args, locate};

/// A located answer this far from the truth is a confident wrong answer.
pub const WRONG_M: f64 = 3000.0;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    case: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    kind: String,
    scenario: Option<String>,
    media: Option<String>,
    events: Option<String>,
    #[serde(default)]
    area: String,
    #[serde(default)]
    when: String,
    truth: Option<[f64; 2]>,
    #[serde(default)]
    truth_note: String,
    /// `kind = "ridges"`: the annotated frequency–time curve (`time_utc,frequency_hz`).
    truth_csv: Option<String>,
    /// The blind leg's clock window when `when` is wide (`locate --blind-when`).
    blind_when: Option<String>,
    /// Wrong-day control: ADS-B from this many days away (`locate --traffic-shift-days`).
    traffic_shift_days: Option<i64>,
    /// `kind = "curves"`: the file's hand marks (JSON), and its labels.
    marks: Option<String>,
    engine: Option<String>,
    site: Option<i64>,
    stratum: Option<String>,
    /// The dataset's "aircraft audible" window (s) of a clip.
    audible: Option<[f64; 2]>,
    /// A continuous recording's 5-s labels: the CSV and its column.
    labels_csv: Option<String>,
    labels_column: Option<usize>,
    /// The committed sha256 list of all marks files (marking happened before discovery ran).
    marks_hashes: Option<String>,
    /// `kind = "detect"`: tuning or test split; the clip's audible window covers > 70 % of it.
    split: Option<String>,
    floor_limited: Option<bool>,
    /// Locate cases: skip the timing-and-loudness leg, or (dev) use only it.
    no_envelope: Option<bool>,
    envelope_only: Option<bool>,
}

/// A ridge-truth curve's alignment must stand this far (robust z) above the other offsets, or the
/// clip is excluded (and counted as such).
const ALIGN_Z_MIN: f64 = 5.0;

/// `kind = "ridges"`: discovery only, scored against the clip's annotated curve, aligned by its
/// own guide image (no location, no ADS-B).
fn run_ridges(c: &Case, base: &Path) -> Result<Row> {
    let started = Instant::now();
    let media = resolve(
        base,
        c.media.as_deref().context("ridges case needs `media`")?,
    );
    let csv = resolve(
        base,
        c.truth_csv
            .as_deref()
            .context("ridges case needs `truth_csv`")?,
    );
    let audio = gm_doppler::audio::decode(&media)
        .with_context(|| format!("cannot read audio from {}", media.display()))?;
    let pts = gm_search::ridge_truth::load_truth_csv(&csv).map_err(|e| anyhow::anyhow!(e))?;
    let img = gm_doppler::audio::guide::guide_image(&audio);
    let events =
        gm_doppler::audio::discover(&audio, &gm_doppler::audio::EventParams::default(), &[]);
    let (verdict_kind, verdict, ridge) = match gm_search::ridge_truth::align(&img, &pts) {
        Some((off, z)) if z >= ALIGN_Z_MIN => {
            let truth = vec![gm_search::ridge_truth::TruthRidge {
                id: c.id.clone(),
                points: pts.iter().map(|&(t, f)| (t + off, f)).collect(),
            }];
            let d = gm_search::ridge_truth::ridge_detection(&events, &truth);
            (
                if d.found > 0 {
                    "ridge_found"
                } else {
                    "ridge_missed"
                },
                format!("aligned at {off:.2} s (z {z:.1}); {} events", events.len()),
                Some(d),
            )
        }
        Some((off, z)) => (
            "excluded",
            format!("alignment too weak (offset {off:.2} s, z {z:.1} < {ALIGN_Z_MIN})"),
            None,
        ),
        None => ("excluded", "curve can't be aligned".into(), None),
    };
    Ok(Row {
        id: c.id.clone(),
        verdict_kind: verdict_kind.into(),
        verdict,
        error_m: None,
        matched: vec![],
        delta_s: None,
        seconds: started.elapsed().as_secs_f64(),
        truth_note: c.truth_note.clone(),
        detection: None,
        ridge,
        ..Default::default()
    })
}

#[derive(Serialize, Default)]
struct Row {
    id: String,
    verdict_kind: String,
    verdict: String,
    error_m: Option<f64>,
    matched: Vec<String>,
    delta_s: Option<f64>,
    seconds: f64,
    truth_note: String,
    /// Which passes the tracks say the recording contains (≤ 5 km at the truth), and which the
    /// events found.
    detection: Option<gm_search::bench::Detection>,
    /// Every pass the tracks put within 8 km of the truth, with its audio time: what
    /// stage-1 labels for a recording with ADS-B are made from.
    expected_passes: Option<Vec<gm_search::bench::ExpectedPass>>,
    /// `kind = "ridges"`: the annotated curve found or not.
    ridge: Option<gm_search::ridge_truth::RidgeDetection>,
    /// A wrong-day control (ADS-B from another day).
    wrong_day: bool,
    /// The blind leg's own verdict, track-first's (with its error and p), and which the
    /// combined answer came from.
    blind_kind: Option<String>,
    tf_kind: Option<String>,
    tf_error_m: Option<f64>,
    tf_p: Option<f64>,
    source: Option<String>,
    /// Aircraft contributing at the truth's coarse cell (track-first diagnostic).
    tf_truth_aircraft: Option<usize>,
    /// `kind = "curves"`: the score, the marks file's sha256, and whether it matches the
    /// committed list.
    curves: Option<crate::curvetest::CurveScore>,
    marks_sha256: Option<String>,
    marks_hash_ok: Option<bool>,
    #[serde(skip)]
    curve_info: Option<crate::curvetest::CaseInfo>,
    /// `kind = "detect"`: stage-1 detection against the labels.
    detect: Option<crate::detecteval::DetectScore>,
    /// Timing and loudness: its verdict, distance from truth to the nearest region cell (0
    /// inside), whether truth lies in the region; aircraft sounds found and those with no curve.
    env_kind: Option<String>,
    env_error_m: Option<f64>,
    env_in_region: Option<bool>,
    env_p: Option<f64>,
    passes: Option<usize>,
    passes_no_curve: Option<usize>,
    #[serde(skip)]
    detect_info: Option<crate::detecteval::DetectInfo>,
}

/// Is `sha` the hash `list` (lines of `<sha256>  <marks path>`) records for `key`?
fn hash_listed(list: &str, key: &str, sha: &str) -> bool {
    list.lines().any(|l| {
        let mut it = l.split_whitespace();
        it.next() == Some(sha) && it.next() == Some(key)
    })
}

/// `kind = "curves"`: discovery on the file, scored against its hand marks. The marks are read
/// (and hashed) before the audio is decoded, so a missing marks file never runs discovery.
fn run_curves(c: &Case, base: &Path, pass_gate: bool) -> Result<Row> {
    use sha2::{Digest, Sha256};
    let started = Instant::now();
    let mp = resolve(
        base,
        c.marks.as_deref().context("curves case needs `marks`")?,
    );
    let bytes = std::fs::read(&mp).with_context(|| format!("marks not found: {}", mp.display()))?;
    let marks: crate::curvetest::Marks = serde_json::from_slice(&bytes)
        .with_context(|| format!("marks unreadable: {}", mp.display()))?;
    let sha: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let hash_ok = match &c.marks_hashes {
        Some(h) => {
            let list = std::fs::read_to_string(resolve(base, h)).unwrap_or_default();
            Some(hash_listed(
                &list,
                c.marks.as_deref().unwrap_or_default(),
                &sha,
            ))
        }
        None => None,
    };
    let labels = match (&c.labels_csv, c.labels_column) {
        (Some(p), Some(k)) => Some(crate::curvetest::labels_from_csv(&resolve(base, p), k)?),
        _ => None,
    };
    let media = resolve(
        base,
        c.media.as_deref().context("curves case needs `media`")?,
    );
    let audio = gm_doppler::audio::decode(&media)
        .with_context(|| format!("cannot read audio from {}", media.display()))?;
    let mut events =
        gm_doppler::audio::discover(&audio, &gm_doppler::audio::EventParams::default(), &[]);
    if pass_gate {
        let det = gm_doppler::audio::detect(&audio, &gm_doppler::audio::DetectParams::default());
        let log = gm_doppler::audio::passlog::build(&det, &events);
        events = gm_doppler::audio::passlog::gate_to_passes(&events, &log);
    }
    let score = crate::curvetest::score(
        &events,
        &marks,
        labels.as_deref(),
        c.audible.map(|a| (a[0], a[1])),
    );
    let found = score.per_curve.iter().filter(|x| x.found).count();
    Ok(Row {
        id: c.id.clone(),
        verdict_kind: "curves".into(),
        verdict: format!(
            "{found}/{} marked curves found; {} events",
            score.per_curve.len(),
            events.len()
        ),
        seconds: started.elapsed().as_secs_f64(),
        truth_note: c.truth_note.clone(),
        curves: Some(score),
        marks_sha256: Some(sha),
        marks_hash_ok: hash_ok,
        curve_info: Some(crate::curvetest::CaseInfo {
            engine: c.engine.clone(),
            stratum: c.stratum.clone(),
            site: c.site,
            continuous: labels.is_some(),
        }),
        ..Default::default()
    })
}

/// The labels of a detect case (continuous recordings).
fn detect_labels(c: &Case, base: &Path) -> Result<Option<Vec<crate::curvetest::Label>>> {
    match (&c.labels_csv, c.labels_column) {
        (Some(p), Some(k)) => Ok(Some(crate::curvetest::labels_from_csv(
            &resolve(base, p),
            k,
        )?)),
        _ => Ok(None),
    }
}

fn detect_info(c: &Case, continuous: bool) -> crate::detecteval::DetectInfo {
    crate::detecteval::DetectInfo {
        split: c.split.clone().unwrap_or_else(|| "all".into()),
        engine: c.engine.clone(),
        continuous,
    }
}

/// `kind = "detect"`: stage 1 (find aircraft sounds) on the file, scored against its labels or
/// audible window. No ADS-B, no location.
fn run_detect(c: &Case, base: &Path, params: &gm_doppler::audio::DetectParams) -> Result<Row> {
    let started = Instant::now();
    let labels = detect_labels(c, base)?;
    let media = resolve(
        base,
        c.media.as_deref().context("detect case needs `media`")?,
    );
    let audio = gm_doppler::audio::decode(&media)
        .with_context(|| format!("cannot read audio from {}", media.display()))?;
    let d = gm_doppler::audio::detect(&audio, params);
    let score = crate::detecteval::score_detection(
        &d,
        labels.as_deref(),
        c.audible.map(|a| (a[0], a[1])),
        c.floor_limited.unwrap_or(false),
    );
    Ok(Row {
        id: c.id.clone(),
        verdict_kind: "detect".into(),
        verdict: format!("{} passes, {} rejected", d.passes.len(), d.rejected.len()),
        seconds: started.elapsed().as_secs_f64(),
        truth_note: c.truth_note.clone(),
        detect: Some(score),
        detect_info: Some(detect_info(c, labels.is_some())),
        ..Default::default()
    })
}

/// `eval --detect-tune`: every grid point of `detecttune::tune_grid` judged on the split's detect
/// cases, against the fixed `original` row and today's defaults (`current`), with the keep rule.
/// Each case is decoded once and kept at the 8-kHz analysis rate; `prepare()` runs once per
/// distinct `prepare_key`.
pub fn detect_tune(manifest: &Path, split: &str) -> Result<()> {
    use crate::detecttune::{TuneTotals, hf_summary, keeps, prepare_key, table, tune_grid};
    use gm_doppler::audio::{
        DetectParams,
        detect::{detect_prepared, prepare},
    };
    let text = std::fs::read_to_string(manifest)
        .with_context(|| format!("cannot read {}", manifest.display()))?;
    let m: Manifest = toml::from_str(&text)
        .with_context(|| format!("{} is not a valid eval manifest", manifest.display()))?;
    let base = manifest.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut cases = Vec::new();
    for c in m
        .case
        .iter()
        .filter(|c| c.kind == "detect" && c.split.as_deref() == Some(split))
    {
        let labels = detect_labels(c, &base)?;
        if labels.is_none() && c.floor_limited.unwrap_or(false) {
            continue; // A9: floor-limited clips stay out of tuning.
        }
        let media = resolve(
            &base,
            c.media.as_deref().context("detect case needs `media`")?,
        );
        let audio = gm_doppler::audio::decode(&media)
            .with_context(|| format!("cannot read audio from {}", media.display()))?;
        // The detector's analysis rate: decimate once, not per preparation.
        let audio = if audio.sample_rate > DETECT_RATE {
            gm_doppler::audio::decimate(&audio, DETECT_RATE)
        } else {
            audio
        };
        eprintln!("decoded {}", gm_report::sanitize_terminal(&c.id));
        cases.push((c, labels, audio));
    }
    let grid = tune_grid(&DetectParams::default());
    let mut totals: Vec<Option<TuneTotals>> = vec![None; grid.len()];
    let keys: Vec<String> = grid.iter().map(|g| prepare_key(&g.1)).collect();
    let mut seen = std::collections::HashSet::new();
    // `hf_db` of every pass under `current` (grid entry 1), by class, for the absorption cue.
    let mut hf = Vec::new();
    for key in keys.into_iter().filter(|k| seen.insert(k.clone())) {
        let members: Vec<usize> = (0..grid.len())
            .filter(|&i| prepare_key(&grid[i].1) == key)
            .collect();
        eprintln!("preparing for {key} ({} grid points)", members.len());
        let mut scores = vec![Vec::with_capacity(cases.len()); members.len()];
        for (c, labels, audio) in &cases {
            let prep = prepare(audio, &grid[members[0]].1);
            for (j, &gi) in members.iter().enumerate() {
                let d = detect_prepared(&prep, &grid[gi].1);
                if gi == 1 {
                    hf.extend(
                        crate::detecteval::pass_classes(
                            &d,
                            labels.as_deref(),
                            c.audible.map(|a| (a[0], a[1])),
                        )
                        .into_iter()
                        .map(|(class, v, t)| (c.id.clone(), class, v, t)),
                    );
                }
                scores[j].push(crate::detecteval::score_detection(
                    &d,
                    labels.as_deref(),
                    c.audible.map(|a| (a[0], a[1])),
                    c.floor_limited.unwrap_or(false),
                ));
            }
        }
        for (j, &gi) in members.iter().enumerate() {
            let rows: Vec<(&str, &crate::detecteval::DetectScore, bool)> = cases
                .iter()
                .zip(&scores[j])
                .map(|((c, labels, _), s)| (c.id.as_str(), s, labels.is_some()))
                .collect();
            totals[gi] = Some(TuneTotals::of(&rows));
        }
    }
    let totals: Vec<TuneTotals> = totals.into_iter().map(Option::unwrap_or_default).collect();
    let rows: Vec<(String, TuneTotals, bool)> = grid
        .iter()
        .zip(&totals)
        .enumerate()
        .map(|(i, ((name, _), t))| {
            (
                name.clone(),
                t.clone(),
                i >= 2 && keeps(&totals[0], &totals[1], t),
            )
        })
        .collect();
    print!("{}", table(&rows));
    print!("{}", hf_summary(&hf));
    Ok(())
}

/// The stage-1 detector's analysis rate (its octave bands reach 3.2 kHz).
const DETECT_RATE: u32 = 8000;

fn resolve(base: &Path, p: &str) -> PathBuf {
    // `media:<path>`: a recording in the kept data's media folder, whatever the machine.
    if let Some(rest) = p.strip_prefix("media:") {
        return gm_sources::paths::media_dir().join(rest);
    }
    let p = match p.strip_prefix("~/") {
        Some(rest) => std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_default()
            .join(rest),
        None => PathBuf::from(p),
    };
    if p.is_absolute() { p } else { base.join(p) }
}

fn dist(a: (f64, f64), b: &Geodetic) -> f64 {
    geodesic_distance_m(
        Geodetic {
            lat_deg: a.0,
            lon_deg: a.1,
            h_m: 0.0,
        },
        Geodetic { h_m: 0.0, ..*b },
    )
}

fn run_case(c: &Case, base: &Path, out: &Path) -> Result<Row> {
    let started = Instant::now();
    let dir = out.join("cases").join(&c.id);
    std::fs::create_dir_all(&dir)?;
    let (file, sim_truth, truth) = match c.kind.as_str() {
        "sim" => {
            let scen = resolve(
                base,
                c.scenario.as_deref().context("sim case needs `scenario`")?,
            );
            let s = gm_sim::load(&scen).with_context(|| format!("scenario {}", scen.display()))?;
            let r = gm_sim::render(&s)?;
            let wav = dir.join("audio.wav");
            gm_sim::write_wav(&wav, s.sample_rate, &r.samples)?;
            let tp = dir.join("truth.json");
            std::fs::write(&tp, serde_json::to_vec(&r.truth)?)?;
            let truth = c
                .truth
                .map_or((r.truth.observer.lat_deg, r.truth.observer.lon_deg), |t| {
                    (t[0], t[1])
                });
            (wav, Some(tp), truth)
        }
        "media" => {
            let m = resolve(
                base,
                c.media.as_deref().context("media case needs `media`")?,
            );
            anyhow::ensure!(m.exists(), "media not found: {}", m.display());
            let t = c.truth.context("media case needs `truth = [lat, lon]`")?;
            (m, None, (t[0], t[1]))
        }
        k => anyhow::bail!("unknown case kind {k:?} (expected sim or media)"),
    };
    let args = Args {
        file,
        area: c.area.clone(),
        when: c.when.clone(),
        yes: false,
        windows: vec![],
        out: Some(dir.clone()),
        json: None,
        sim_truth,
        events: c.events.as_deref().map(|e| resolve(base, e)),
        debug_cell: None,
        weather: crate::locate::Weather::OpenMeteo,
        guided: true,
        track_first: true,
        blind_when: c.blind_when.clone(),
        traffic_shift_days: c.traffic_shift_days.unwrap_or(0),
        truth: Some(truth),
        envelope: !c.no_envelope.unwrap_or(false),
        envelope_only: c.envelope_only.unwrap_or(false),
        // As `geomedia locate` runs by default.
        pass_gate: false,
    };
    let res = locate(&args)?;
    // The case folder keeps the full record (pass log, rejections, envelope), as `locate` does.
    let (text, html) = crate::locate::documents(&res, &args)?;
    std::fs::write(dir.join("locate.json"), text)?;
    std::fs::write(dir.join("locate.html"), html)?;
    let o = &res.outcome;
    let obs = gm_core::geodetic_to_ecef(gm_core::Geodetic {
        lat_deg: truth.0,
        lon_deg: truth.1,
        h_m: (res.ground)(truth.0, truth.1) + gm_search::params::MIC_H,
    });
    let passes = gm_search::bench::expected_passes(
        &res.aircraft,
        obs,
        res.duration_s,
        &gm_search::atmos::Atmosphere::still(),
        8000.0,
    );
    let detection = Some(gm_search::bench::detection(
        &res.events,
        &passes,
        gm_search::bench::TOL_S,
        res.duration_s,
    ));
    let (kind, error_m) = match &o.verdict {
        Verdict::Located => ("located", o.best.as_ref().map(|b| dist(truth, &b.geo))),
        Verdict::AmbiguousMirror => {
            let d = [o.best.as_ref(), o.mirror.as_ref()]
                .into_iter()
                .flatten()
                .map(|r| dist(truth, &r.geo))
                .fold(f64::INFINITY, f64::min);
            ("ambiguous_mirror", d.is_finite().then_some(d))
        }
        Verdict::AmbiguousStream => (
            "ambiguous_stream",
            o.best.as_ref().map(|b| dist(truth, &b.geo)),
        ),
        Verdict::Inconclusive(_) => ("inconclusive", None),
    };
    let short = |v: &Verdict| match v {
        Verdict::Located => "located",
        Verdict::AmbiguousMirror => "ambiguous_mirror",
        Verdict::AmbiguousStream => "ambiguous_stream",
        Verdict::Inconclusive(_) => "inconclusive",
    };
    let tf = res.track_first.as_ref();
    let env = res.envelope.as_ref();
    let env_err = env.and_then(|e| {
        e.region
            .iter()
            .map(|&c| {
                dist(
                    truth,
                    &Geodetic {
                        lat_deg: c.0,
                        lon_deg: c.1,
                        h_m: 0.0,
                    },
                )
            })
            .reduce(f64::min)
    });
    let env_kind = env.map(|e| match &e.verdict {
        gm_search::envelope::EnvVerdict::Located { .. } => "located".to_string(),
        gm_search::envelope::EnvVerdict::Band { mirror: true, .. } => "band_mirror".into(),
        gm_search::envelope::EnvVerdict::Band { .. } => "band".into(),
        gm_search::envelope::EnvVerdict::Ambiguous { .. } => "ambiguous".into(),
        gm_search::envelope::EnvVerdict::Inconclusive { .. } => "inconclusive".into(),
    });
    let log = res.pass_log.as_ref();
    Ok(Row {
        env_kind,
        // A 1-km cell: truth within ~710 m of a region cell's centre lies inside it.
        env_in_region: env_err.map(|d| d <= 710.0),
        env_error_m: env_err.map(|d| if d <= 710.0 { 0.0 } else { d }),
        env_p: env.map(|e| e.p),
        passes: log.map(|l| l.passes.len()),
        passes_no_curve: log.map(|l| {
            l.passes
                .iter()
                .filter(|r| matches!(r.curve, gm_doppler::audio::Curve::None { .. }))
                .count()
        }),
        wrong_day: c.traffic_shift_days.is_some_and(|d| d != 0),
        blind_kind: Some(short(&res.blind.verdict).into()),
        tf_kind: tf.map(|t| short(&t.outcome.verdict).into()),
        tf_error_m: tf
            .filter(|t| t.outcome.verdict == Verdict::Located)
            .and_then(|t| t.outcome.best.as_ref())
            .map(|b| dist(truth, &b.geo)),
        tf_p: tf.and_then(|t| t.p),
        source: Some(format!("{:?}", res.source)),
        tf_truth_aircraft: tf.and_then(|t| t.truth_probe),
        id: c.id.clone(),
        verdict_kind: kind.into(),
        verdict: format!("{:?}", o.verdict),
        error_m,
        matched: o.matched.iter().map(|(_, i)| i.clone()).collect(),
        delta_s: (kind != "inconclusive").then_some(o.delta_s),
        seconds: started.elapsed().as_secs_f64(),
        truth_note: c.truth_note.clone(),
        detection,
        expected_passes: Some(passes),
        ridge: None,
        ..Default::default()
    })
}

/// Case ids name directories under `--out/cases/`: plain names only, each used once.
fn check_ids(cases: &[Case]) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    for c in cases {
        let ok =
            c.id.chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_alphanumeric())
                && c.id
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || "._-".contains(ch));
        anyhow::ensure!(
            ok,
            "case id {:?} must be letters, digits, '.', '_' or '-' (starting with a letter or digit)",
            c.id
        );
        anyhow::ensure!(
            seen.insert(c.id.as_str()),
            "case id {:?} is used more than once",
            c.id
        );
    }
    Ok(())
}

pub fn run(
    manifest: &Path,
    out: Option<PathBuf>,
    split: Option<&str>,
    pass_gate: bool,
) -> Result<()> {
    let text = std::fs::read_to_string(manifest)
        .with_context(|| format!("cannot read {}", manifest.display()))?;
    let m: Manifest = toml::from_str(&text)
        .with_context(|| format!("{} is not a valid eval manifest", manifest.display()))?;
    check_ids(&m.case)?;
    let base = manifest.parent().unwrap_or(Path::new(".")).to_path_buf();
    // Case folders live in a managed run folder; results.json and summary.md are also copied
    // to `--out` when given.
    let run = crate::runs::new_run("eval", Some(manifest))?;
    let mut rows = Vec::new();
    let detect_params = gm_doppler::audio::DetectParams::default();
    for c in m
        .case
        .iter()
        .filter(|c| split.is_none_or(|sp| c.split.as_deref() == Some(sp)))
    {
        eprintln!("eval {} …", gm_report::sanitize_terminal(&c.id));
        let res = match c.kind.as_str() {
            "ridges" => run_ridges(c, &base),
            "curves" => run_curves(c, &base, pass_gate),
            "detect" => run_detect(c, &base, &detect_params),
            _ => run_case(c, &base, &run),
        };
        let row = res.unwrap_or_else(|e| Row {
            id: c.id.clone(),
            verdict_kind: "error".into(),
            verdict: format!("{e:#}"),
            error_m: None,
            matched: vec![],
            delta_s: None,
            seconds: 0.0,
            truth_note: c.truth_note.clone(),
            ..Default::default()
        });
        rows.push(row);
    }
    let summary = summarize(&rows);
    let out = out.as_deref();
    crate::runs::write_both(
        &run,
        out,
        "results.json",
        &serde_json::to_vec_pretty(&serde_json::json!({"schema_version": 1, "cases": rows}))?,
    )?;
    crate::runs::write_both(&run, out, "summary.md", summary.as_bytes())?;
    let located = rows.iter().filter(|r| r.verdict_kind == "located").count();
    crate::runs::finish_run(
        &run,
        &crate::runs::RunInfo {
            kind: "eval".into(),
            file: manifest.display().to_string(),
            verdict: format!("{located} of {} located", rows.len()),
            ..Default::default()
        },
    )?;
    print!("{summary}");
    eprintln!(
        "run: {}",
        gm_report::sanitize_terminal(&run.display().to_string())
    );
    Ok(())
}

fn summarize(rows: &[Row]) -> String {
    let n = rows.len();
    let count = |k: &str| rows.iter().filter(|r| r.verdict_kind == k).count();
    let located: Vec<f64> = rows
        .iter()
        .filter(|r| r.verdict_kind == "located")
        .filter_map(|r| r.error_m)
        .collect();
    let wrong: Vec<&Row> = rows
        .iter()
        .filter(|r| r.verdict_kind == "located" && r.error_m.is_some_and(|e| e > WRONG_M))
        .collect();
    let mut s = String::new();
    let _ = writeln!(s, "# geomedia eval\n");
    let _ = writeln!(
        s,
        "confident wrong: {} (located but more than {:.0} km off){}",
        wrong.len(),
        WRONG_M / 1000.0,
        if wrong.is_empty() {
            String::new()
        } else {
            format!(
                " — {}",
                wrong
                    .iter()
                    .map(|r| r.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    );
    let _ = writeln!(
        s,
        "located: {} of {n} · ambiguous: {} · inconclusive: {} · errors: {}",
        located.len(),
        count("ambiguous_mirror") + count("ambiguous_stream"),
        count("inconclusive"),
        count("error")
    );
    let echoes = rows
        .iter()
        .filter(|r| {
            r.wrong_day && r.verdict_kind == "located" && r.error_m.is_some_and(|e| e <= 1000.0)
        })
        .count();
    if rows.iter().any(|r| r.wrong_day) {
        let _ = writeln!(
            s,
            "route echoes (wrong-day traffic located within 1 km): {echoes} of {} wrong-day controls",
            rows.iter().filter(|r| r.wrong_day).count()
        );
    }
    if rows.iter().any(|r| r.env_kind.is_some()) {
        let _ = writeln!(
            s,
            "\n## Timing and loudness\n\n| case | aircraft sounds (no curve) | verdict | p | truth in region | distance to region |\n|---|---|---|---|---|---|"
        );
        for r in rows
            .iter()
            .filter(|r| r.env_kind.is_some() || r.passes.is_some())
        {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} | {} |",
                r.id,
                r.passes.map_or("—".into(), |n| format!(
                    "{n} ({})",
                    r.passes_no_curve.unwrap_or(0)
                )),
                r.env_kind.as_deref().unwrap_or("not run"),
                r.env_p.map_or("—".into(), |p| format!("{p:.3}")),
                r.env_in_region
                    .map_or("—", |b| if b { "yes" } else { "no" }),
                r.env_error_m
                    .map_or("—".into(), |d| format!("{:.1} km", d / 1000.0)),
            );
        }
    }
    if rows.iter().any(|r| r.tf_kind.is_some()) {
        let tf_loc: Vec<f64> = rows
            .iter()
            .filter(|r| r.tf_kind.as_deref() == Some("located"))
            .filter_map(|r| r.tf_error_m)
            .collect();
        let tf_wrong = tf_loc.iter().filter(|&&e| e > WRONG_M).count();
        let bl = rows
            .iter()
            .filter(|r| r.blind_kind.as_deref() == Some("located"))
            .count();
        let mut v = tf_loc.clone();
        v.sort_by(f64::total_cmp);
        let _ = writeln!(
            s,
            "track-first alone: located {} (confident wrong {tf_wrong}{}) · blind leg alone: located {bl}",
            tf_loc.len(),
            v.get(v.len() / 2)
                .map_or(String::new(), |m| format!(", median {m:.0} m"))
        );
    }
    if !located.is_empty() {
        let mut v = located.clone();
        v.sort_by(f64::total_cmp);
        let med = if v.len() % 2 == 1 {
            v[v.len() / 2]
        } else {
            0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
        };
        let hit = |r: f64| v.iter().filter(|e| **e <= r).count();
        let _ = writeln!(
            s,
            "located accuracy: median {med:.0} m · ≤0.5 km {} · ≤3 km {} · ≤10 km {}",
            hit(500.0),
            hit(3000.0),
            hit(10_000.0)
        );
    }
    let _ = writeln!(
        s,
        "\n| case | verdict | error | matched | clock | note |\n|---|---|---|---|---|---|"
    );
    for r in rows {
        let err = r.error_m.map_or("—".into(), |e| format!("{e:.0} m"));
        let clock = r.delta_s.map_or("—".into(), |d| format!("{d:+.2} s"));
        let verdict = if r.verdict_kind == "error" {
            format!("error: {}", r.verdict.replace('|', "/"))
        } else {
            r.verdict_kind.clone()
        };
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} |",
            r.id,
            verdict,
            err,
            r.matched.join(", "),
            clock,
            r.truth_note
        );
    }
    // Detection: passes the tracks say each recording contains (≤ 5 km at the truth, arrival
    // inside the recording), found by discovery (and with guided events) within ±TOL_S, geometry agreeing.
    let _ = writeln!(
        s,
        "\n## Detection (expected passes found: discovery / with guided)\n\n| case | ≤1 km | 1–2 km | 2–3 km | 3–5 km | 5–8 km | by class | unmatched events |\n|---|---|---|---|---|---|---|---|"
    );
    let mut tot = [(0usize, 0usize, 0usize); 5];
    for r in rows {
        let Some(d) = &r.detection else { continue };
        let cell = |i: usize| {
            let (_, e, f, g) = d.by_slant[i];
            tot[i].0 += e;
            tot[i].1 += f;
            tot[i].2 += g;
            if e == 0 {
                "—".to_string()
            } else {
                format!("{f}/{g} of {e}")
            }
        };
        let bins: Vec<String> = (0..5).map(cell).collect();
        let classes = d
            .by_class
            .iter()
            .map(|(c, e, f)| format!("{c} {f}/{e}"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} | {} | {} of {} discovered, {} of {} guided (±3 s windows cover {:.0} %) |",
            r.id,
            bins[0],
            bins[1],
            bins[2],
            bins[3],
            bins[4],
            classes,
            d.unmatched_discovered,
            d.discovered,
            d.unmatched_guided,
            d.guided,
            100.0 * d.chance_coverage
        );
    }
    let fmt = |t: (usize, usize, usize)| {
        if t.0 == 0 {
            "—".to_string()
        } else {
            format!("{}/{} of {}", t.1, t.2, t.0)
        }
    };
    let _ = writeln!(
        s,
        "| **all** | {} | {} | {} | {} | {} | | |",
        fmt(tot[0]),
        fmt(tot[1]),
        fmt(tot[2]),
        fmt(tot[3]),
        fmt(tot[4])
    );
    // Every missed pass with its nearest event: tells a detector miss from a scoring miss.
    for r in rows {
        let Some(d) = &r.detection else { continue };
        if d.missed.is_empty() {
            continue;
        }
        let _ = writeln!(s, "\n{} missed:", r.id);
        for m in &d.missed {
            let near = match &m.nearest {
                Some(n) => format!(
                    "nearest Δtc {:+.1} s, d×{:.2}, v×{:.2} ({})",
                    n.dt_s,
                    n.d_ratio,
                    n.v_ratio,
                    match n.reason {
                        gm_search::bench::MissReason::Time => "time",
                        gm_search::bench::MissReason::Geometry => "geometry",
                        gm_search::bench::MissReason::TimeAndGeometry => "time and geometry",
                        gm_search::bench::MissReason::Taken => "taken by another pass",
                    }
                ),
                None => format!(
                    "no event within {:.0} s",
                    gm_search::bench::NEAR_MISS_REACH_S
                ),
            };
            let _ = writeln!(
                s,
                "- missed {} {:.0} m @ {:.1} s: {near}",
                m.icao, m.slant_m, m.t_audio
            );
        }
    }
    // Ridge truth (annotated curves): discovery alone, no location.
    let ridge_rows: Vec<&Row> = rows
        .iter()
        .filter(|r| r.ridge.is_some() || r.verdict_kind == "excluded")
        .collect();
    if !ridge_rows.is_empty() {
        let found = rows
            .iter()
            .filter(|r| r.verdict_kind == "ridge_found")
            .count();
        let scored = rows.iter().filter(|r| r.ridge.is_some()).count();
        let excluded = rows.iter().filter(|r| r.verdict_kind == "excluded").count();
        let unmatched: usize = rows
            .iter()
            .filter_map(|r| r.ridge.as_ref())
            .map(|d| d.unmatched_events)
            .sum();
        let _ = writeln!(
            s,
            "\n## Detection (ridge truth)\n\nfound {found} of {scored} aligned curves; {excluded} excluded (alignment too weak); {unmatched} unmatched events\n"
        );
        for r in ridge_rows {
            let _ = writeln!(s, "- {}: {} — {}", r.id, r.verdict_kind, r.verdict);
        }
    }
    let curve_rows: Vec<(
        &str,
        &crate::curvetest::CurveScore,
        &crate::curvetest::CaseInfo,
    )> = rows
        .iter()
        .filter_map(|r| Some((r.id.as_str(), r.curves.as_ref()?, r.curve_info.as_ref()?)))
        .collect();
    let detect_rows: Vec<(
        &str,
        &crate::detecteval::DetectScore,
        &crate::detecteval::DetectInfo,
    )> = rows
        .iter()
        .filter_map(|r| Some((r.id.as_str(), r.detect.as_ref()?, r.detect_info.as_ref()?)))
        .collect();
    if !detect_rows.is_empty() {
        s.push_str(&crate::detecteval::summarize_detection(&detect_rows));
    }
    if !curve_rows.is_empty() {
        s.push_str(&crate::curvetest::summarize(&curve_rows));
        let bad: Vec<&str> = rows
            .iter()
            .filter(|r| r.marks_hash_ok == Some(false))
            .map(|r| r.id.as_str())
            .collect();
        let unchecked = rows
            .iter()
            .filter(|r| r.curves.is_some() && r.marks_hash_ok.is_none())
            .count();
        let _ = writeln!(
            s,
            "
marks hashes: {} mismatch the committed list{}{}",
            bad.len(),
            if bad.is_empty() {
                String::new()
            } else {
                format!(" — {}", bad.join(", "))
            },
            if unchecked > 0 {
                format!("; {unchecked} not checked (no list)")
            } else {
                String::new()
            }
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, kind: &str, err: Option<f64>, wrong_day: bool) -> Row {
        Row {
            id: id.into(),
            verdict_kind: kind.into(),
            error_m: err,
            wrong_day,
            ..Default::default()
        }
    }

    fn curves_case(marks: &str, media: &str) -> Case {
        toml::from_str(&format!(
            "id = \"x\"\nkind = \"curves\"\nmedia = \"{media}\"\nmarks = \"{marks}\"\n"
        ))
        .unwrap()
    }

    #[test]
    fn eval_case_folder_holds_the_pass_log() {
        // The broadband simulator case through run_case: the case folder must hold the full
        // locate document (passes, rejected, envelope), not only the summary counts.
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/eval");
        let c: Case = toml::from_str(
            "id = \"bb\"\nkind = \"sim\"\nscenario = \"../fixtures/sim/broadband/single.toml\"\narea = \"40.8840,-73.5110~12km\"\nwhen = \"2026-09-08T20:10:00Z±60s\"\n",
        )
        .unwrap();
        let out = tempfile::tempdir().unwrap();
        let row = run_case(&c, &base, out.path()).unwrap();
        assert_eq!(row.passes, Some(1));
        let doc: serde_json::Value = serde_json::from_slice(
            &std::fs::read(out.path().join("cases/bb/locate.json")).expect("locate.json written"),
        )
        .unwrap();
        assert!(
            doc["passes"].as_array().is_some_and(|p| p.len() == 1),
            "{doc}"
        );
        assert!(doc["rejected"].is_array());
        assert!(
            doc["envelope"]["verdict"].is_object(),
            "{}",
            doc["envelope"]
        );
        assert!(out.path().join("cases/bb/locate.html").exists());
    }

    #[test]
    fn missing_marks_is_an_error_row() {
        // Neither file exists: the marks are read first, so the error names them and no audio
        // is ever decoded.
        let d = tempfile::tempdir().unwrap();
        let e = run_curves(&curves_case("missing.json", "no-such.wav"), d.path(), false)
            .err()
            .expect("a missing marks file is an error");
        assert!(format!("{e:#}").contains("marks not found"), "{e:#}");
    }

    #[test]
    fn a_changed_marks_file_fails_its_hash() {
        let list = "aaaa  media:aerosonic/marks/a.json\nbbbb  media:aerosonic/marks/b.json\n";
        assert!(hash_listed(list, "media:aerosonic/marks/a.json", "aaaa"));
        // Edited after the list was committed: a different hash.
        assert!(!hash_listed(list, "media:aerosonic/marks/a.json", "cccc"));
        // Another file's hash doesn't vouch for this one.
        assert!(!hash_listed(list, "media:aerosonic/marks/a.json", "bbbb"));
    }

    #[test]
    fn wrong_day_located_far_counts_as_wrong_and_near_as_route_echo() {
        let rows = vec![
            row("far", "located", Some(5000.0), true),
            row("echo", "located", Some(500.0), true),
            row("good", "located", Some(300.0), false),
        ];
        let s = summarize(&rows);
        assert!(s.contains("confident wrong: 1"), "{s}");
        assert!(
            s.contains("route echoes (wrong-day traffic located within 1 km): 1"),
            "{s}"
        );
    }
}
