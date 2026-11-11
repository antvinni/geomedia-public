//! `kind = "curves"`: blind discovery scored against Doppler curves marked by hand on a plain
//! spectrogram (AeroSonicDB; docs/superpowers/specs/2026-11-06-real-curve-test-design.md). An
//! event finds a marked curve when it follows it (gm-search `ridge_truth`'s rule: half its
//! points within 1 % of the curve or of a harmonic). In a continuous recording every event is
//! placed by its CPA time in the dataset's 5-s labels: on silence it is a false curve.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};
use gm_doppler::audio::Event;
use gm_search::ridge_truth::{TruthRidge, best_fraction, best_overlap_fraction, follows_ratio};
use serde::{Deserialize, Serialize};

/// One file's marks (written by tests/eval/aerosonic/marks.py).
#[derive(Debug, Clone, Deserialize)]
pub struct Marks {
    pub verdict: String,
    #[serde(default)]
    pub curves: Vec<MarkCurve>,
    #[serde(default)]
    pub no_curve_stretches: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MarkCurve {
    pub id: String,
    pub clarity: String,
    #[serde(default)]
    pub above_band: bool,
    pub points: Vec<[f64; 2]>,
    pub fit: Option<MarkFit>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MarkFit {
    pub f0: f64,
    pub v: f64,
    pub d: f64,
    pub t0: f64,
    /// `implausible`, `cpa_outside_curve` (a partial pass): the distance isn't trusted.
    #[serde(default)]
    pub flags: Vec<String>,
}

impl MarkFit {
    /// The fitted CPA distance, unless the fit is flagged.
    pub fn distance(&self) -> Option<f64> {
        self.flags.is_empty().then_some(self.d)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelKind {
    Silence,
    Aircraft,
    Ignore,
}

/// One 5-s row of a continuous recording's labels.
#[derive(Debug, Clone, Copy)]
pub struct Label {
    pub t0: f64,
    pub t1: f64,
    pub kind: LabelKind,
}

/// Column `column` of environment_class_mappings.csv (0 / 1 / ignore per 5 s).
pub fn labels_from_csv(path: &Path, column: usize) -> Result<Vec<Label>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().skip(1).enumerate() {
        let cell = line
            .split(',')
            .nth(column)
            .with_context(|| format!("{}: row {} has no column {column}", path.display(), i + 2))?
            .trim();
        let kind = match cell {
            "1" => LabelKind::Aircraft,
            "0" => LabelKind::Silence,
            _ => LabelKind::Ignore,
        };
        out.push(Label {
            t0: i as f64 * 5.0,
            t1: (i + 1) as f64 * 5.0,
            kind,
        });
    }
    Ok(out)
}

/// Discovery's fit against the marked curve's own fit (f₀ at the matched harmonic ratio).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct GeomDiff {
    pub dt0: f64,
    pub dv_rel: f64,
    pub dd_rel: f64,
    pub df0_rel: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CurveResult {
    pub id: String,
    pub clarity: String,
    pub above_band: bool,
    /// The marked curve's fitted CPA distance (m).
    pub d_m: Option<f64>,
    pub found: bool,
    pub ratio: Option<f64>,
    pub geom: Option<GeomDiff>,
    /// Missed: the closest event's best fraction of points on the curve, and at which ratio.
    pub nearest: Option<(f64, f64)>,
    /// Missed: over the part of an event inside the curve's span (covering ≥ half of it), the
    /// best fraction of those points on the curve, and at which ratio. High when an event
    /// follows the curve but runs past a shorter truth.
    pub nearest_in_span: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CurveScore {
    pub verdict: String,
    pub per_curve: Vec<CurveResult>,
    pub events: usize,
    /// Clips: events in a clip marked "no curve visible" (maybe a tone the marker missed).
    pub events_in_no_curve: usize,
    /// Clips: events matching no marked curve, and of those, outside the audible window.
    pub unmatched: usize,
    pub unmatched_outside_audible: usize,
    /// Continuous: events in silence rows (false curves, with their CPA times), the seconds of
    /// silence, events in aircraft rows matching no curve, events in ignore rows.
    pub false_events: usize,
    pub false_tc: Vec<f64>,
    pub silence_s: f64,
    pub unmatched_in_aircraft: usize,
    pub excluded_ignore: usize,
}

fn truth(c: &MarkCurve) -> TruthRidge {
    TruthRidge {
        id: c.id.clone(),
        points: c.points.iter().map(|p| (p[0], p[1])).collect(),
    }
}

/// Score `events` against `marks`; `labels` for a continuous recording, `audible` for a clip.
pub fn score(
    events: &[Event],
    marks: &Marks,
    labels: Option<&[Label]>,
    audible: Option<(f64, f64)>,
) -> CurveScore {
    let mut used = vec![false; events.len()];
    let mut per_curve = Vec::new();
    for c in &marks.curves {
        let r = truth(c);
        let hit = (0..events.len()).find_map(|i| {
            if used[i] {
                None
            } else {
                follows_ratio(&events[i], &r).map(|ratio| (i, ratio))
            }
        });
        let (found, ratio, geom, nearest, nearest_in_span) = match hit {
            Some((i, ratio)) => {
                used[i] = true;
                let e = &events[i].fit;
                // A flagged fit (partial pass, implausible) isn't a reference for geometry.
                let geom = c
                    .fit
                    .as_ref()
                    .filter(|m| m.distance().is_some())
                    .map(|m| GeomDiff {
                        dt0: e.tc - m.t0,
                        dv_rel: e.v / m.v - 1.0,
                        dd_rel: e.d / m.d - 1.0,
                        df0_rel: e.f0 / (ratio * m.f0) - 1.0,
                    });
                (true, Some(ratio), geom, None, None)
            }
            None => {
                let near = events
                    .iter()
                    .map(|e| best_fraction(e, &r))
                    .max_by(|a, b| a.0.total_cmp(&b.0));
                let in_span = events
                    .iter()
                    .filter_map(|e| best_overlap_fraction(e, &r))
                    .max_by(|a, b| a.0.total_cmp(&b.0));
                (false, None, None, near, in_span)
            }
        };
        per_curve.push(CurveResult {
            id: c.id.clone(),
            clarity: c.clarity.clone(),
            above_band: c.above_band,
            d_m: c.fit.as_ref().and_then(MarkFit::distance),
            found,
            ratio,
            geom,
            nearest,
            nearest_in_span,
        });
    }
    let mut s = CurveScore {
        verdict: marks.verdict.clone(),
        per_curve,
        events: events.len(),
        ..Default::default()
    };
    match labels {
        Some(rows) => {
            s.silence_s = rows
                .iter()
                .filter(|l| l.kind == LabelKind::Silence)
                .map(|l| l.t1 - l.t0)
                .sum();
            for (e, _) in events.iter().zip(&used).filter(|(_, u)| !**u) {
                let tc = e.fit.tc;
                match rows
                    .iter()
                    .find(|l| tc >= l.t0 && tc < l.t1)
                    .map(|l| l.kind)
                {
                    Some(LabelKind::Silence) => {
                        s.false_events += 1;
                        s.false_tc.push(tc);
                    }
                    // Inside a stretch marked "no curve visible": like a no-curve clip.
                    Some(LabelKind::Aircraft)
                        if marks
                            .no_curve_stretches
                            .iter()
                            .any(|r| tc >= r[0] && tc <= r[1]) =>
                    {
                        s.events_in_no_curve += 1
                    }
                    Some(LabelKind::Aircraft) => s.unmatched_in_aircraft += 1,
                    _ => s.excluded_ignore += 1,
                }
            }
        }
        None => {
            for (e, _) in events.iter().zip(&used).filter(|(_, u)| !**u) {
                if marks.verdict == "no_curve" {
                    s.events_in_no_curve += 1;
                } else {
                    s.unmatched += 1;
                    if audible.is_some_and(|(a, b)| e.fit.tc < a || e.fit.tc > b) {
                        s.unmatched_outside_audible += 1;
                    }
                }
            }
        }
    }
    s
}

/// What a case is, for the summary tables.
#[derive(Debug, Clone, Default)]
pub struct CaseInfo {
    pub engine: Option<String>,
    pub stratum: Option<String>,
    pub site: Option<i64>,
    pub continuous: bool,
}

fn frac(n: usize, d: usize) -> String {
    if d == 0 {
        "—".into()
    } else {
        format!("{n}/{d}")
    }
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(v[v.len() / 2])
}

/// The curves section of the eval summary.
pub fn summarize(rows: &[(&str, &CurveScore, &CaseInfo)]) -> String {
    let mut s = String::new();
    let clips: Vec<_> = rows.iter().filter(|r| !r.2.continuous).collect();
    let cont: Vec<_> = rows.iter().filter(|r| r.2.continuous).collect();
    let _ = writeln!(
        s,
        "\n## Real-audio Doppler curves (marked by hand; clip stage only on 60-s clips)\n"
    );
    let _ = writeln!(
        s,
        "| stratum | clips | no curve visible | clear found | faint found | above 1.6 kHz found | events in no-curve clips | unmatched events (outside audible) |\n|---|---|---|---|---|---|---|---|"
    );
    let mut strata: Vec<String> = clips.iter().filter_map(|r| r.2.stratum.clone()).collect();
    strata.sort();
    strata.dedup();
    strata.push("all".into());
    for st in &strata {
        let sel: Vec<_> = clips
            .iter()
            .filter(|r| st == "all" || r.2.stratum.as_deref() == Some(st.as_str()))
            .collect();
        let curves: Vec<&CurveResult> = sel.iter().flat_map(|r| r.1.per_curve.iter()).collect();
        let count = |clar: &str, above: bool| {
            let c: Vec<_> = curves
                .iter()
                .filter(|c| c.above_band == above && (above || c.clarity == clar))
                .collect();
            frac(c.iter().filter(|c| c.found).count(), c.len())
        };
        let _ = writeln!(
            s,
            "| {st} | {} | {} | {} | {} | {} | {} | {} ({}) |",
            sel.len(),
            sel.iter().filter(|r| r.1.verdict == "no_curve").count(),
            count("clear", false),
            count("faint", false),
            count("", true),
            sel.iter().map(|r| r.1.events_in_no_curve).sum::<usize>(),
            sel.iter().map(|r| r.1.unmatched).sum::<usize>(),
            sel.iter()
                .map(|r| r.1.unmatched_outside_audible)
                .sum::<usize>(),
        );
    }
    // By the marked curve's fitted CPA distance (clip stage band only).
    let in_band: Vec<&CurveResult> = clips
        .iter()
        .flat_map(|r| r.1.per_curve.iter())
        .filter(|c| !c.above_band)
        .collect();
    let bins = [
        (0.0, 1000.0, "≤1 km"),
        (1000.0, 2000.0, "1–2 km"),
        (2000.0, 3000.0, "2–3 km"),
        (3000.0, f64::INFINITY, ">3 km"),
    ];
    let _ = write!(s, "\n| marked curves by fitted distance |");
    for b in &bins {
        let _ = write!(s, " {} |", b.2);
    }
    let _ = write!(s, " no fit |\n|---|");
    for _ in 0..=bins.len() {
        let _ = write!(s, "---|");
    }
    let _ = write!(s, "\n| found |");
    for (a, b, _) in &bins {
        let c: Vec<_> = in_band
            .iter()
            .filter(|c| c.d_m.is_some_and(|d| d > *a && d <= *b))
            .collect();
        let _ = write!(
            s,
            " {} |",
            frac(c.iter().filter(|c| c.found).count(), c.len())
        );
    }
    let nf: Vec<_> = in_band.iter().filter(|c| c.d_m.is_none()).collect();
    let _ = writeln!(
        s,
        " {} |",
        frac(nf.iter().filter(|c| c.found).count(), nf.len())
    );
    // Engine types: how often a curve exists at all.
    let mut engines: Vec<String> = clips.iter().filter_map(|r| r.2.engine.clone()).collect();
    engines.sort();
    engines.dedup();
    let _ = write!(s, "\nclips with no visible curve, by engine:");
    for en in &engines {
        let sel: Vec<_> = clips
            .iter()
            .filter(|r| r.2.engine.as_deref() == Some(en.as_str()))
            .collect();
        let _ = write!(
            s,
            " {en} {}",
            frac(
                sel.iter().filter(|r| r.1.verdict == "no_curve").count(),
                sel.len()
            )
        );
    }
    let _ = writeln!(s);
    let mut sites: Vec<i64> = clips.iter().filter_map(|r| r.2.site).collect();
    sites.sort_unstable();
    sites.dedup();
    let _ = write!(s, "marked curves found, by site:");
    for st in sites {
        let c: Vec<&CurveResult> = clips
            .iter()
            .filter(|r| r.2.site == Some(st))
            .flat_map(|r| r.1.per_curve.iter())
            .collect();
        let _ = write!(
            s,
            " {st}: {}",
            frac(c.iter().filter(|c| c.found).count(), c.len())
        );
    }
    let _ = writeln!(s);
    // Near misses: a missed curve whose closest event had ≥ 30 % of its points on it.
    let missed: Vec<&CurveResult> = rows
        .iter()
        .flat_map(|r| r.1.per_curve.iter())
        .filter(|c| !c.found)
        .collect();
    let near = missed
        .iter()
        .filter(|c| c.nearest.is_some_and(|n| n.0 >= 0.3))
        .count();
    let outrun = missed
        .iter()
        .filter(|c| c.nearest_in_span.is_some_and(|n| n.0 >= 0.5))
        .count();
    let _ = writeln!(
        s,
        "missed curves: {} ({near} near misses: an event with ≥ 30 % of its points on the curve; \
         {outrun} followed by an event wherever both exist, over ≥ half the curve, which then runs \
         past it)",
        missed.len()
    );
    // Geometry of found curves.
    let found: Vec<&GeomDiff> = rows
        .iter()
        .flat_map(|r| r.1.per_curve.iter())
        .filter_map(|c| c.geom.as_ref())
        .collect();
    let m = |f: &dyn Fn(&GeomDiff) -> f64| median(found.iter().map(|g| f(g).abs()).collect());
    if let (Some(t), Some(v), Some(d), Some(f)) = (
        m(&|g| g.dt0),
        m(&|g| g.dv_rel),
        m(&|g| g.dd_rel),
        m(&|g| g.df0_rel),
    ) {
        let _ = writeln!(
            s,
            "found curves' geometry vs the marked fit (median |difference|, {} curves): t₀ {t:.1} s · v {:.0} % · d {:.0} % · f₀ {:.1} %",
            found.len(),
            100.0 * v,
            100.0 * d,
            100.0 * f
        );
    }
    // Continuous recordings: the full pipeline.
    if !cont.is_empty() {
        let _ = writeln!(
            s,
            "\n| continuous recording | marked curves found | false curves (silence) | silence h | false / h (upper bound) | unmatched in aircraft rows |\n|---|---|---|---|---|---|"
        );
        let (mut fe, mut sh) = (0usize, 0.0f64);
        for r in &cont {
            let c = r.1;
            fe += c.false_events;
            sh += c.silence_s / 3600.0;
            let _ = writeln!(
                s,
                "| {} | {} | {} | {:.2} | {:.1} | {} |",
                r.0,
                frac(
                    c.per_curve.iter().filter(|x| x.found).count(),
                    c.per_curve.len()
                ),
                c.false_events,
                c.silence_s / 3600.0,
                if c.silence_s > 0.0 {
                    c.false_events as f64 / (c.silence_s / 3600.0)
                } else {
                    0.0
                },
                c.unmatched_in_aircraft
            );
        }
        let _ = writeln!(
            s,
            "\nfalse curves on silence: {fe} in {sh:.2} h ({:.1} per hour, an upper bound: a faint aircraft discovery finds there counts as false)",
            if sh > 0.0 { fe as f64 / sh } else { 0.0 }
        );
        let listed: Vec<String> = cont
            .iter()
            .flat_map(|r| {
                r.1.false_tc
                    .iter()
                    .map(move |t| format!("{} @ {t:.1} s", r.0))
            })
            .collect();
        if !listed.is_empty() {
            let _ = writeln!(s, "to listen to: {}", listed.join(", "));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    // The street labels (tests/eval/street/labels.py) write an ignore row as an empty cell, which
    // in a one-column CSV is a blank line: it must stay a row, or every later row shifts by 5 s.
    #[test]
    fn blank_cells_are_ignore_rows_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("l.csv");
        std::fs::write(&p, "label\n1\n\n0\n\n").unwrap();
        let rows = labels_from_csv(&p, 0).unwrap();
        let kinds: Vec<_> = rows.iter().map(|r| r.kind).collect();
        assert_eq!(
            kinds,
            [
                LabelKind::Aircraft,
                LabelKind::Ignore,
                LabelKind::Silence,
                LabelKind::Ignore
            ]
        );
        assert_eq!(rows[2].t0, 10.0);
    }

    /// An event whose points follow f(t) over [t0, t1] every 0.25 s, with fit (tc, v, d, f0).
    fn event(t0: f64, t1: f64, f: impl Fn(f64) -> f64, fit: (f64, f64, f64, f64)) -> Event {
        let pts: Vec<(f64, f64)> = (0..=((t1 - t0) / 0.25) as usize)
            .map(|k| {
                let t = t0 + k as f64 * 0.25;
                (t, f(t))
            })
            .collect();
        serde_json::from_value(serde_json::json!({
            "id": 1, "t_start": t0, "t_end": t1,
            "fit": {"v": fit.1, "d": fit.2, "tc": fit.0, "f0": fit.3, "rms_hz": 0.1,
                    "sigma_v": 1.0, "sigma_d": 1.0, "sigma_tc": 0.1, "n": pts.len()},
            "harmonic": 1, "f0_fundamental": fit.3, "ridges": 1, "mean_snr_db": 10.0,
            "points": pts,
        }))
        .unwrap()
    }

    fn marks(verdict: &str, curves: serde_json::Value) -> Marks {
        serde_json::from_value(serde_json::json!({"verdict": verdict, "curves": curves})).unwrap()
    }

    fn falling(t: f64) -> f64 {
        130.0 - 0.5 * (t - 10.0)
    }

    fn one_curve() -> Marks {
        let pts: Vec<[f64; 2]> = (0..=80)
            .map(|k| [10.0 + k as f64 * 0.25, falling(10.0 + k as f64 * 0.25)])
            .collect();
        marks(
            "curves",
            serde_json::json!([{"id": "c1", "clarity": "clear", "points": pts,
                                "fit": {"f0": 125.0, "v": 60.0, "d": 800.0, "t0": 20.0}}]),
        )
    }

    #[test]
    fn found_when_the_event_follows_the_curve() {
        let e = event(12.0, 28.0, falling, (20.5, 66.0, 880.0, 125.0));
        let s = score(&[e], &one_curve(), None, Some((5.0, 35.0)));
        assert!(s.per_curve[0].found);
        assert_eq!(s.per_curve[0].ratio, Some(1.0));
        assert_eq!(s.unmatched, 0);
    }

    #[test]
    fn found_on_a_harmonic() {
        let e = event(12.0, 28.0, |t| 2.0 * falling(t), (20.0, 60.0, 800.0, 250.0));
        let s = score(&[e], &one_curve(), None, None);
        assert!(s.per_curve[0].found);
        assert_eq!(s.per_curve[0].ratio, Some(2.0));
    }

    #[test]
    fn a_missed_curve_reports_its_nearest_event() {
        let e = event(
            12.0,
            28.0,
            |t| falling(t) * 1.03,
            (20.0, 60.0, 800.0, 129.0),
        );
        let s = score(&[e], &one_curve(), None, Some((5.0, 35.0)));
        assert!(!s.per_curve[0].found);
        assert!(s.per_curve[0].nearest.is_some());
        assert_eq!(s.unmatched, 1);
        assert_eq!(s.unmatched_outside_audible, 0);
    }

    #[test]
    fn events_in_a_no_curve_clip_are_not_false() {
        let e = event(12.0, 28.0, falling, (20.0, 60.0, 800.0, 125.0));
        let s = score(&[e], &marks("no_curve", serde_json::json!([])), None, None);
        assert_eq!(s.events_in_no_curve, 1);
        assert_eq!((s.unmatched, s.false_events), (0, 0));
    }

    fn labels(kinds: &[LabelKind]) -> Vec<Label> {
        kinds
            .iter()
            .enumerate()
            .map(|(i, &kind)| Label {
                t0: 5.0 * i as f64,
                t1: 5.0 * (i + 1) as f64,
                kind,
            })
            .collect()
    }

    #[test]
    fn events_in_silence_are_false_and_counted_per_hour() {
        use LabelKind::*;
        // 120 rows of silence (600 s) then one aircraft row; two events in silence.
        let mut k = vec![Silence; 120];
        k.push(Aircraft);
        let l = labels(&k);
        let ev = [
            event(10.0, 30.0, falling, (20.0, 60.0, 800.0, 125.0)),
            event(
                300.0,
                320.0,
                |t| 200.0 - 0.3 * (t - 300.0),
                (310.0, 60.0, 800.0, 195.0),
            ),
        ];
        let s = score(&ev, &marks("curves", serde_json::json!([])), Some(&l), None);
        assert_eq!(s.false_events, 2);
        assert!((s.silence_s - 600.0).abs() < 1e-9);
        let row = (
            "env",
            &s,
            &CaseInfo {
                continuous: true,
                ..Default::default()
            },
        );
        assert!(summarize(&[row]).contains("12.0 per hour"));
    }

    #[test]
    fn events_in_ignore_are_excluded() {
        use LabelKind::*;
        let l = labels(&[Silence, Ignore, Aircraft, Silence]);
        let ev = [
            event(5.0, 10.0, falling, (7.0, 60.0, 800.0, 125.0)),
            event(10.0, 15.0, falling, (12.0, 60.0, 800.0, 125.0)),
        ];
        let s = score(&ev, &marks("curves", serde_json::json!([])), Some(&l), None);
        assert_eq!(
            (s.excluded_ignore, s.unmatched_in_aircraft, s.false_events),
            (1, 1, 0)
        );
    }

    #[test]
    fn events_in_a_no_curve_stretch_are_not_unmatched() {
        use LabelKind::*;
        let l = labels(&[Silence, Aircraft, Aircraft, Silence]);
        let m: Marks = serde_json::from_value(serde_json::json!({
            "verdict": "curves", "curves": [], "no_curve_stretches": [[5.0, 15.0]]
        }))
        .unwrap();
        let ev = [event(5.0, 12.0, falling, (8.0, 60.0, 800.0, 125.0))];
        let s = score(&ev, &m, Some(&l), None);
        assert_eq!((s.events_in_no_curve, s.unmatched_in_aircraft), (1, 0));
    }

    #[test]
    fn a_flagged_fit_gives_no_distance() {
        let pts: Vec<[f64; 2]> = (0..=40)
            .map(|k| [10.0 + k as f64 * 0.5, falling(10.0 + k as f64 * 0.5)])
            .collect();
        let m = marks(
            "curves",
            serde_json::json!([{"id": "c1", "clarity": "clear", "points": pts,
                                "fit": {"f0": 125.0, "v": 60.0, "d": 800.0, "t0": 80.0,
                                        "flags": ["cpa_outside_curve"]}}]),
        );
        let s = score(&[], &m, None, None);
        assert_eq!(s.per_curve[0].d_m, None);
        // An unflagged fit keeps its distance.
        assert_eq!(
            score(&[], &one_curve(), None, None).per_curve[0].d_m,
            Some(800.0)
        );
    }

    #[test]
    fn geometry_difference_uses_the_matched_ratio() {
        let e = event(12.0, 28.0, |t| 2.0 * falling(t), (21.0, 66.0, 880.0, 255.0));
        let s = score(&[e], &one_curve(), None, None);
        let g = s.per_curve[0].geom.unwrap();
        assert!((g.dt0 - 1.0).abs() < 1e-9);
        assert!((g.dv_rel - 0.1).abs() < 1e-9);
        assert!((g.dd_rel - 0.1).abs() < 1e-9);
        assert!((g.df0_rel - 0.02).abs() < 1e-9, "{}", g.df0_rel);
    }

    #[test]
    fn geometry_skips_a_flagged_fit() {
        let pts: Vec<[f64; 2]> = (0..=80)
            .map(|k| [10.0 + k as f64 * 0.25, falling(10.0 + k as f64 * 0.25)])
            .collect();
        let m = marks(
            "curves",
            serde_json::json!([{"id": "c1", "clarity": "clear", "points": pts,
                                "fit": {"f0": 125.0, "v": 60.0, "d": 800.0, "t0": 80.0,
                                        "flags": ["cpa_outside_curve"]}}]),
        );
        let e = event(12.0, 28.0, falling, (20.0, 60.0, 800.0, 125.0));
        let s = score(&[e], &m, None, None);
        assert!(s.per_curve[0].found);
        assert!(s.per_curve[0].geom.is_none());
    }

    #[test]
    fn nearest_reports_span_overlap() {
        // On the curve over all of its span (10–30 s), far off it outside: fewer than half of
        // the event's points are on the curve, so it is missed, but it follows the curve
        // wherever both exist.
        let long = event(
            0.0,
            60.0,
            |t| {
                if (10.0..=30.0).contains(&t) {
                    falling(t)
                } else {
                    300.0
                }
            },
            (20.0, 60.0, 800.0, 125.0),
        );
        let s = score(&[long], &one_curve(), None, None);
        let c = &s.per_curve[0];
        assert!(!c.found);
        assert!(c.nearest.unwrap().0 < 0.5);
        let (frac, ratio) = c.nearest_in_span.unwrap();
        assert!((frac - 1.0).abs() < 1e-9, "{frac}");
        assert_eq!(ratio, 1.0);
        // An event overlapping only 2 s of the 20-s curve says nothing about it.
        let short = event(
            28.0,
            60.0,
            |t| if t <= 30.0 { falling(t) } else { 300.0 },
            (20.0, 60.0, 800.0, 125.0),
        );
        let s = score(&[short], &one_curve(), None, None);
        assert!(s.per_curve[0].nearest_in_span.is_none());
    }
}
