//! Stage-1 detection scored against labels: a continuous recording's 5-s aircraft/silence rows,
//! or a clip's "aircraft audible" window.

use std::fmt::Write as _;

use gm_doppler::audio::Detection;
use serde::Serialize;

use crate::curvetest::{Label, LabelKind};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct RowCounts {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub tn: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DetectScore {
    /// Continuous recordings: 5-s rows covered by a pass vs their labels (ignore rows left out).
    pub rows: Option<RowCounts>,
    /// Clips: does a pass overlap the audible window?
    pub clip_hit: Option<bool>,
    pub passes: usize,
    /// Passes overlapping no aircraft row (the PR #24 measure).
    pub false_passes: usize,
    /// Of those, the passes on ignore rows only (no aircraft row and no not-aircraft row: a
    /// can't-tell stretch, a far pass): not strictly false.
    pub passes_on_ignored: usize,
    pub silence_s: f64,
    /// Labelled aircraft runs (consecutive aircraft rows) with a pass on them.
    pub runs_found: usize,
    pub runs: usize,
    /// Per labelled row: (the highest smoothed excess in it, is it an aircraft row) — for AUC.
    pub row_scores: Vec<(f64, bool)>,
    /// The clip's audible window covers > 70 % of it: little background to compare against.
    pub floor_limited: bool,
}

#[derive(Clone, Debug, Default)]
pub struct DetectInfo {
    pub split: String,
    pub engine: Option<String>,
    pub continuous: bool,
}

/// Score `d` against `labels` (continuous recording) or `audible` (clip).
pub fn score_detection(
    d: &Detection,
    labels: Option<&[Label]>,
    audible: Option<(f64, f64)>,
    floor_limited: bool,
) -> DetectScore {
    let overlaps = |a0: f64, a1: f64, b0: f64, b1: f64| a0 < b1 && a1 > b0;
    let mut s = DetectScore {
        passes: d.passes.len(),
        floor_limited,
        ..Default::default()
    };
    if let Some((a0, a1)) = audible {
        s.clip_hit = Some(
            d.passes
                .iter()
                .any(|p| overlaps(p.t_start, p.t_end, a0, a1)),
        );
    }
    let Some(rows) = labels else {
        return s;
    };
    let env = &d.envelope;
    // The highest smoothed excess inside [t0, t1).
    let peak_in = |t0: f64, t1: f64| -> f64 {
        env.mid_db
            .iter()
            .enumerate()
            .filter(|(k, _)| {
                let t = env.t0 + *k as f64 * env.dt;
                t >= t0 && t < t1
            })
            .map(|(_, v)| f64::from(*v))
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let mut c = RowCounts::default();
    for r in rows {
        let covered = d
            .passes
            .iter()
            .any(|p| overlaps(p.t_start, p.t_end, r.t0, r.t1));
        match r.kind {
            LabelKind::Aircraft => {
                if covered {
                    c.tp += 1;
                } else {
                    c.fn_ += 1;
                }
            }
            LabelKind::Silence => {
                s.silence_s += r.t1 - r.t0;
                if covered {
                    c.fp += 1;
                } else {
                    c.tn += 1;
                }
            }
            LabelKind::Ignore => continue,
        }
        let v = peak_in(r.t0, r.t1);
        s.row_scores.push((
            if v.is_finite() { v } else { f64::MIN },
            r.kind == LabelKind::Aircraft,
        ));
    }
    s.rows = Some(c);
    let on = |p: &gm_doppler::audio::Pass, kind: LabelKind| {
        rows.iter()
            .any(|r| r.kind == kind && overlaps(p.t_start, p.t_end, r.t0, r.t1))
    };
    let false_ones: Vec<_> = d
        .passes
        .iter()
        .filter(|p| !on(p, LabelKind::Aircraft))
        .collect();
    s.false_passes = false_ones.len();
    s.passes_on_ignored = false_ones
        .iter()
        .filter(|p| !on(p, LabelKind::Silence))
        .count();
    // Labelled runs: consecutive aircraft rows.
    let mut runs: Vec<(f64, f64)> = Vec::new();
    for r in rows.iter().filter(|r| r.kind == LabelKind::Aircraft) {
        match runs.last_mut() {
            Some(last) if (r.t0 - last.1).abs() < 1e-9 => last.1 = r.t1,
            _ => runs.push((r.t0, r.t1)),
        }
    }
    s.runs = runs.len();
    s.runs_found = runs
        .iter()
        .filter(|&&(a, b)| d.passes.iter().any(|p| overlaps(p.t_start, p.t_end, a, b)))
        .count();
    s
}

/// What lies under a pass: an aircraft row, only ignore rows, a not-aircraft row (strictly
/// false); for a clip, its audible window or not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PassClass {
    Aircraft,
    OnIgnored,
    StrictFalse,
    ClipHit,
    ClipOther,
}

/// Each pass of `d` with its class, its `hf_db` and its start time.
pub fn pass_classes(
    d: &Detection,
    labels: Option<&[Label]>,
    audible: Option<(f64, f64)>,
) -> Vec<(PassClass, Option<f64>, f64)> {
    let overlaps = |a0: f64, a1: f64, b0: f64, b1: f64| a0 < b1 && a1 > b0;
    d.passes
        .iter()
        .map(|p| {
            let on = |kind: LabelKind| {
                labels.is_some_and(|rows| {
                    rows.iter()
                        .any(|r| r.kind == kind && overlaps(p.t_start, p.t_end, r.t0, r.t1))
                })
            };
            let class = match (labels, audible) {
                (Some(_), _) if on(LabelKind::Aircraft) => PassClass::Aircraft,
                (Some(_), _) if on(LabelKind::Silence) => PassClass::StrictFalse,
                (Some(_), _) => PassClass::OnIgnored,
                (None, Some((a0, a1))) if overlaps(p.t_start, p.t_end, a0, a1) => {
                    PassClass::ClipHit
                }
                (None, _) => PassClass::ClipOther,
            };
            (class, p.hf_db, p.t_start)
        })
        .collect()
}

/// Rank AUC of `scores` (score, positive?); ties count half.
pub fn auc(scores: &[(f64, bool)]) -> Option<f64> {
    let pos: Vec<f64> = scores.iter().filter(|s| s.1).map(|s| s.0).collect();
    let neg: Vec<f64> = scores.iter().filter(|s| !s.1).map(|s| s.0).collect();
    if pos.is_empty() || neg.is_empty() {
        return None;
    }
    let mut all: Vec<(f64, bool)> = scores.to_vec();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Average ranks over ties.
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && all[j + 1].0 == all[i].0 {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        rank_sum += r * all[i..=j].iter().filter(|x| x.1).count() as f64;
        i = j + 1;
    }
    let (n1, n0) = (pos.len() as f64, neg.len() as f64);
    Some((rank_sum - n1 * (n1 + 1.0) / 2.0) / (n1 * n0))
}

/// The detection tables: per split, rows (AUC, precision, recall), labelled runs found, false
/// passes per silent hour; clips found per engine type (all, and without floor-limited clips).
pub fn summarize_detection(rows: &[(&str, &DetectScore, &DetectInfo)]) -> String {
    let mut s = String::new();
    let frac = |a: usize, b: usize| {
        if b == 0 {
            "—".to_string()
        } else {
            format!("{a}/{b}")
        }
    };
    let mut splits: Vec<&str> = rows.iter().map(|r| r.2.split.as_str()).collect();
    splits.sort_unstable();
    splits.dedup();
    let cont: Vec<_> = rows.iter().filter(|r| r.2.continuous).collect();
    if !cont.is_empty() {
        let _ = writeln!(
            s,
            "\n## Stage 1: aircraft sounds in the labelled hours (5-s rows)\n\n| split | hours | row AUC | precision | recall | F1 | labelled runs found | false passes | per silent hour |\n|---|---|---|---|---|---|---|---|---|"
        );
        for sp in &splits {
            let rs: Vec<_> = cont.iter().filter(|r| r.2.split == *sp).collect();
            if rs.is_empty() {
                continue;
            }
            let mut c = RowCounts::default();
            let (mut fp, mut sil, mut runs, mut found) = (0usize, 0.0f64, 0usize, 0usize);
            let mut scores = Vec::new();
            for r in &rs {
                if let Some(rc) = r.1.rows {
                    c.tp += rc.tp;
                    c.fp += rc.fp;
                    c.fn_ += rc.fn_;
                    c.tn += rc.tn;
                }
                fp += r.1.false_passes;
                sil += r.1.silence_s;
                runs += r.1.runs;
                found += r.1.runs_found;
                scores.extend_from_slice(&r.1.row_scores);
            }
            let prec = c.tp as f64 / (c.tp + c.fp).max(1) as f64;
            let rec = c.tp as f64 / (c.tp + c.fn_).max(1) as f64;
            let f1 = if prec + rec > 0.0 {
                2.0 * prec * rec / (prec + rec)
            } else {
                0.0
            };
            let _ = writeln!(
                s,
                "| {sp} | {} | {} | {prec:.3} | {rec:.3} | {f1:.3} | {} | {fp} | {:.1} |",
                rs.len(),
                auc(&scores).map_or("—".into(), |a| format!("{a:.3}")),
                frac(found, runs),
                if sil > 0.0 {
                    fp as f64 / (sil / 3600.0)
                } else {
                    0.0
                }
            );
        }
    }
    let clips: Vec<_> = rows.iter().filter(|r| r.1.clip_hit.is_some()).collect();
    if !clips.is_empty() {
        let _ = writeln!(
            s,
            "\n## Stage 1: labelled aircraft clips (diagnostic; floor-limited = audible > 70 % of the clip)\n\n| split | engine | found (all) | found (not floor-limited) |\n|---|---|---|---|"
        );
        let mut engines: Vec<String> = clips
            .iter()
            .map(|r| r.2.engine.clone().unwrap_or_else(|| "?".into()))
            .collect();
        engines.sort();
        engines.dedup();
        for sp in &splits {
            for e in &engines {
                let cs: Vec<_> = clips
                    .iter()
                    .filter(|r| {
                        r.2.split == *sp && r.2.engine.clone().unwrap_or_else(|| "?".into()) == *e
                    })
                    .collect();
                if cs.is_empty() {
                    continue;
                }
                let hits = cs.iter().filter(|c| c.1.clip_hit == Some(true)).count();
                let nf: Vec<_> = cs.iter().filter(|c| !c.1.floor_limited).collect();
                let nf_hits = nf.iter().filter(|c| c.1.clip_hit == Some(true)).count();
                let _ = writeln!(
                    s,
                    "| {sp} | {e} | {} | {} |",
                    frac(hits, cs.len()),
                    frac(nf_hits, nf.len())
                );
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use gm_doppler::audio::{Envelope, Pass};

    fn pass(t0: f64, t1: f64) -> Pass {
        Pass {
            id: 1,
            t_start: t0,
            t_end: t1,
            t_peak: (t0 + t1) / 2.0,
            excess_db: 6.0,
            score: 5.0,
            flags: vec![],
            hf_db: None,
        }
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

    /// A detection with these passes and a flat smoothed envelope of `level` dB (0.5-s frames).
    fn det(passes: Vec<Pass>, seconds: f64, level: f32) -> Detection {
        let n = (seconds / 0.5) as usize;
        Detection {
            passes,
            envelope: Envelope {
                t0: 0.5,
                dt: 0.5,
                frame_s: 1.0,
                mid_db: vec![level; n],
                raw_mid_db: vec![level; n],
                bands_db: vec![[level; 6]; n],
            },
            ..Default::default()
        }
    }

    #[test]
    fn passes_are_classed_by_what_lies_under_them() {
        use LabelKind::*;
        let l = labels(&[Aircraft, Ignore, Ignore, Silence]);
        let d = det(
            vec![pass(0.0, 4.0), pass(5.0, 15.0), pass(15.5, 19.0)],
            20.0,
            1.0,
        );
        let c: Vec<PassClass> = pass_classes(&d, Some(&l), None)
            .into_iter()
            .map(|x| x.0)
            .collect();
        assert_eq!(
            c,
            [
                PassClass::Aircraft,
                PassClass::OnIgnored,
                PassClass::StrictFalse
            ]
        );
        let d = det(vec![pass(12.0, 18.0), pass(40.0, 60.0)], 60.0, 1.0);
        let c: Vec<PassClass> = pass_classes(&d, None, Some((10.0, 20.0)))
            .into_iter()
            .map(|x| x.0)
            .collect();
        assert_eq!(c, [PassClass::ClipHit, PassClass::ClipOther]);
    }

    #[test]
    fn pass_on_ignored_rows_only_is_not_strictly_false() {
        use LabelKind::*;
        // Rows: A · · S (· = ignore). One pass lies on the two ignore rows only (5–15 s), one on
        // an ignore row and the silence row (10–20 s): both are false passes in the PR #24
        // sense, but only the second is strictly false.
        let l = labels(&[Aircraft, Ignore, Ignore, Silence]);
        let s = score_detection(
            &det(vec![pass(5.0, 15.0), pass(10.0, 20.0)], 20.0, 1.0),
            Some(&l),
            None,
            false,
        );
        assert_eq!(s.false_passes, 2);
        assert_eq!(s.passes_on_ignored, 1);
    }

    #[test]
    fn row_counts_follow_the_labels() {
        use LabelKind::*;
        // Rows: S A A S S; a pass over 6–16 s overlaps rows 1, 2 (aircraft) and 3 (silence).
        let l = labels(&[Silence, Aircraft, Aircraft, Silence, Silence]);
        let s = score_detection(
            &det(vec![pass(6.0, 16.0)], 25.0, 1.0),
            Some(&l),
            None,
            false,
        );
        // Rows 1, 2, 3 overlap the pass: tp 2 (rows 1, 2), fp 1 (row 3), tn 2 (rows 0, 4).
        assert_eq!(
            s.rows,
            Some(RowCounts {
                tp: 2,
                fp: 1,
                fn_: 0,
                tn: 2
            })
        );
        assert_eq!((s.runs, s.runs_found, s.false_passes), (1, 1, 0));
        assert!((s.silence_s - 15.0).abs() < 1e-9);
    }

    #[test]
    fn a_pass_in_silence_is_false() {
        use LabelKind::*;
        let l = labels(&[Silence, Silence, Silence, Aircraft, Aircraft]);
        let s = score_detection(
            &det(vec![pass(1.0, 12.0)], 25.0, 1.0),
            Some(&l),
            None,
            false,
        );
        assert_eq!(s.false_passes, 1);
        assert_eq!((s.runs, s.runs_found), (1, 0));
        assert_eq!(s.rows.unwrap().fn_, 2);
    }

    #[test]
    fn clip_hit_needs_overlap_with_audible() {
        let hit = score_detection(
            &det(vec![pass(10.0, 30.0)], 60.0, 1.0),
            None,
            Some((25.0, 50.0)),
            false,
        );
        assert_eq!(hit.clip_hit, Some(true));
        let miss = score_detection(
            &det(vec![pass(0.0, 20.0)], 60.0, 1.0),
            None,
            Some((25.0, 50.0)),
            true,
        );
        assert_eq!(miss.clip_hit, Some(false));
        assert!(miss.floor_limited);
    }

    #[test]
    fn ignore_rows_are_excluded() {
        use LabelKind::*;
        let l = labels(&[Ignore, Aircraft, Ignore, Silence]);
        let s = score_detection(
            &det(vec![pass(0.0, 20.0)], 20.0, 1.0),
            Some(&l),
            None,
            false,
        );
        let r = s.rows.unwrap();
        assert_eq!(r.tp + r.fp + r.fn_ + r.tn, 2);
        assert_eq!(s.row_scores.len(), 2);
    }

    #[test]
    fn summary_reports_auc_per_split() {
        use LabelKind::*;
        // A perfectly separating score: aircraft rows high, silence rows low.
        let l = labels(&[Silence, Aircraft, Silence, Aircraft]);
        let mut d = det(vec![pass(5.0, 10.0), pass(15.0, 20.0)], 20.0, 0.0);
        for (k, v) in d.envelope.mid_db.iter_mut().enumerate() {
            let t = 0.5 + 0.5 * k as f64;
            *v = if (5.0..10.0).contains(&t) || (15.0..20.0).contains(&t) {
                8.0
            } else {
                0.0
            };
        }
        let s = score_detection(&d, Some(&l), None, false);
        let info = DetectInfo {
            split: "test".into(),
            engine: None,
            continuous: true,
        };
        let text = summarize_detection(&[("env", &s, &info)]);
        assert!(text.contains("| test |"), "{text}");
        assert!(text.contains("1.000"), "AUC should be 1.000: {text}");
    }
}
